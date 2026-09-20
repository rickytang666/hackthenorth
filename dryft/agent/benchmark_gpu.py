"""One fresh process per engine/shape; matched prompts, profiling and replay.

Run with the pinned challenge dependencies and a local checkpoint. No network
or submission calls are made. Results are diagnostic, not official scores.
"""
import argparse
import gc
from functools import partial
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-dir', type=Path, required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--prompt-length', type=int, required=True)
    parser.add_argument('--output-length', type=int, default=32)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--prompts', type=Path,
                        help='JSON list of token batches; first is warmup, then samples')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--native-rope', action='store_true', help='Disable cached RoPE for an ablation')
    parser.add_argument('--keep-decode-mask', action='store_true', help='Retain unused masks for an ablation')
    parser.add_argument('--fuse-residual-norm', action='store_true',
                        help='Opt in to the experimental residual-normalization kernel')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.prompt_length, args.output_length, args.samples) < 1:
        parser.error('shape dimensions and samples must be positive')
    return args


def main():
    args = arguments()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required; no benchmark was run.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.engine_dir.resolve()))
    import engine as engine_module
    if args.native_rope or args.keep_decode_mask:
        engine_module.DecodeState = partial(
            engine_module.DecodeState, cache_rotary=not args.native_rope,
            omit_unused_masks=not args.keep_decode_mask)
    Engine = engine_module.Engine

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    if args.prompts:
        prompts = json.loads(args.prompts.read_text())
        source = 'provided token batches'
    else:
        generator = torch.Generator().manual_seed(args.seed)
        prompts = torch.randint(config.vocab_size,
                                (args.samples + 1, args.batch, args.prompt_length),
                                generator=generator).tolist()
        source = 'synthetic random token IDs; not representative natural-language performance'
    if len(prompts) != args.samples + 1 or any(
        len(batch) != args.batch or any(
            len(row) != args.prompt_length or any(
                type(token) is not int or not 0 <= token < config.vocab_size for token in row
            ) for row in batch
        ) for batch in prompts
    ):
        raise ValueError('prompts must contain one warmup plus samples, all matching the shape')

    digest = hashlib.sha256()
    for path in sorted(args.engine_dir.rglob('*.py')):
        digest.update(path.relative_to(args.engine_dir).as_posix().encode())
        digest.update(path.read_bytes())
    report = dict(engine_sha256=digest.hexdigest(), prompt_source=source,
                  prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                  shape=[args.batch, args.prompt_length, args.output_length],
                  cache_rotary=not args.native_rope, omit_unused_masks=not args.keep_decode_mask,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(), samples=[])
    start = time.perf_counter()
    engine = Engine(args.model_path)
    report['fuse_residual_norm'] = args.fuse_residual_norm
    if args.fuse_residual_norm:
        for layer in engine.model.model.layers:
            if not hasattr(layer, 'fuse_residual_norm'):
                raise ValueError('engine does not support residual-normalization experiment')
            layer.fuse_residual_norm = True
    list(engine.generate(prompts[0], args.output_length))
    torch.cuda.synchronize()
    report['load_warmup_seconds'] = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    generated = []
    with torch.inference_mode():
        for prompt in prompts[1:]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs, arrivals = [], []
            for tokens in engine.generate(prompt, args.output_length):
                arrivals.append(time.perf_counter() - start)
                if len(tokens) != args.batch:
                    raise ValueError('wrong number of tokens per yield')
                outputs.append(list(tokens))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if len(outputs) != args.output_length:
                raise ValueError('wrong number of yields')
            generated.append(outputs)
            report['samples'].append(dict(
                total_seconds=elapsed, ttft_seconds=arrivals[0],
                tpot_seconds=(arrivals[-1] - arrivals[0]) / (args.output_length - 1)
                if args.output_length > 1 else None))
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        elapsed = [sample['total_seconds'] for sample in report['samples']]
        report['tokens_per_second'] = args.batch * args.output_length / statistics.median(elapsed)
        report['sample_range_over_median'] = (max(elapsed) - min(elapsed)) / statistics.median(elapsed)
        if args.profile:
            # Profile separately: instrumentation must not contaminate timings.
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True) as prof:
                list(engine.generate(prompts[1], args.output_length))
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
            args.output.with_suffix('.profile.txt').write_text(
                prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=60))
            del prof

    # Remove candidate/graph allocations before loading the independent native model.
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    native = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').eval().to('cuda')
    failures, checked, worst = 0, 0, 0.0
    with torch.inference_mode():
        for prompt, output in zip(prompts[1:], generated):
            emitted = torch.tensor(output, device='cuda').T.contiguous()
            prefix = torch.tensor(prompt, device='cuda')
            sequence = torch.cat((prefix, emitted[:, :-1]), dim=1)
            logits = native(sequence, use_cache=False,
                            logits_to_keep=args.output_length).logits.float()
            deficit = logits.amax(-1) - logits.gather(-1, emitted[..., None]).squeeze(-1)
            failures += ((deficit > 2.0) | ~torch.isfinite(deficit)).sum().item()
            checked += deficit.numel()
            worst = max(worst, deficit.max().item())
            del logits, deficit, sequence, emitted, prefix
    report['teacher_forced'] = dict(checked_tokens=checked, failed_positions=failures,
                                   worst_logit_deficit=worst)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
