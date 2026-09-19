"""Cohere pre-flight. Run this before spending a paid H100 minute.

A run that adapts nothing looks exactly like a run that is learning slowly, and
you only find out three hours later. This asserts the wiring instead.

Five checks, in order:
  1. print the real module names on the pinned revision
  2. attach LoRA to the top N encoder blocks plus the decoder
  3. freeze every lower encoder parameter, then prove it
  4. one batch through the collator, one backward pass, finite loss
  5. record examples per second and peak memory, to set max_steps from

    uv run python -m train.cohere.preflight --dry-run     # steps 1 to 3 only
    uv run python -m train.cohere.preflight               # all five
"""

import argparse
import re
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor

from train.cohere.data import SpeechSeq2SeqCollator, TorgoDataset
from train.cohere.model import MODEL_ID, load
TOP_ENCODER_BLOCKS = 6
LORA_R = 8

# Verified against revision b1eacc26 by running this script. The source plan's
# names (q_proj, k_proj, v_proj, o_proj, fc1, fc2) do not exist on this model:
# the encoder is a Fast-Conformer and the decoder uses NeMo-style sub-layers.
#   encoder.layers.{0..47}.self_attn.{linear_q,linear_k,linear_v,linear_out}
#   encoder.layers.{0..47}.feed_forward{1,2}.{linear1,linear2}
#   transf_decoder._decoder.layers.{0..7}.{first,second}_sub_layer.
#       {query_net,key_net,value_net,out_projection}
#   transf_decoder._decoder.layers.{0..7}.third_sub_layer.{dense_in,dense_out}
ENCODER_SUFFIXES = ["linear_q", "linear_k", "linear_v", "linear_out", "linear1", "linear2"]
DECODER_SUFFIXES = ["query_net", "key_net", "value_net", "out_projection",
                    "dense_in", "dense_out"]

ENCODER_PREFIX = "encoder.layers."
DECODER_PREFIX = "transf_decoder."

_ENCODER_BLOCK = re.compile(r"^encoder\.layers\.(\d+)\.")


def encoder_block_index(name: str) -> int | None:
    """Layer index if this module lives in an encoder block, else None.

    The paths start at `encoder.`, not `.encoder.`. Matching on the latter
    silently classifies every encoder tensor as non-encoder, which is how a run
    ends up adapting nothing while looking fine.
    """
    match = _ENCODER_BLOCK.search(name)
    return int(match.group(1)) if match else None


def target_modules(model, top_n: int) -> list[str]:
    """Full module names for the top N encoder blocks plus the whole decoder.

    Naming them explicitly, rather than by suffix, avoids building adapters on
    all 48 encoder blocks and then freezing 42 of them.
    """
    names = []
    indices = {
        i for i in (encoder_block_index(n) for n, _ in model.named_modules())
        if i is not None
    }
    cutoff = max(indices) - top_n + 1
    for name, module in model.named_modules():
        if list(module.children()):
            continue
        leaf = name.rsplit(".", 1)[-1]
        index = encoder_block_index(name)
        if index is not None and index >= cutoff and leaf in ENCODER_SUFFIXES:
            names.append(name)
        elif name.startswith(DECODER_PREFIX) and leaf in DECODER_SUFFIXES:
            names.append(name)
    return names


def report_modules(model) -> dict:
    """Step 1. The names in any plan are guesses until this prints them."""
    leaves = [n for n, m in model.named_modules() if not list(m.children())]
    encoder_blocks = {i for i in (encoder_block_index(n) for n in leaves) if i is not None}
    decoder_leaves = [n for n in leaves if n.startswith(DECODER_PREFIX)]
    total = sum(p.numel() for p in model.parameters())
    enc = sum(p.numel() for n, p in model.named_parameters() if n.startswith("encoder."))
    dec = sum(p.numel() for n, p in model.named_parameters() if n.startswith(DECODER_PREFIX))

    print(f"\n[1] leaf modules {len(leaves)}")
    if not encoder_blocks:
        raise SystemExit("    no encoder blocks matched; fix encoder_block_index")
    print(f"    encoder blocks {len(encoder_blocks)} "
          f"(indices {min(encoder_blocks)}..{max(encoder_blocks)})")
    print(f"    decoder leaf modules {len(decoder_leaves)}")
    print(f"    parameters: encoder {enc/1e9:.2f}B ({100*enc/total:.1f}%), "
          f"decoder {dec/1e9:.3f}B ({100*dec/total:.1f}%), total {total/1e9:.2f}B")
    print("    This split is why the LoRA reaches into the encoder: adapting the "
          "decoder alone touches a few percent of the model.")

    found = {n.rsplit(".", 1)[-1] for n in leaves}
    missing = [s for s in ENCODER_SUFFIXES + DECODER_SUFFIXES if s not in found]
    if missing:
        raise SystemExit(f"\n    names not on this revision: {missing}")
    return {"encoder_blocks": encoder_blocks}


def apply_lora(model, encoder_blocks: set[int], top_n: int = TOP_ENCODER_BLOCKS):
    """Steps 2 and 3. Adapt the top blocks plus the decoder, then prove it."""
    cutoff = max(encoder_blocks) - top_n + 1
    targets = target_modules(model, top_n)
    if not targets:
        raise SystemExit("no target modules resolved; the name lists are wrong")

    model = get_peft_model(model, LoraConfig(
        r=LORA_R,
        lora_alpha=2 * LORA_R,
        lora_dropout=0.05,
        bias="none",
        target_modules=targets,
    ))

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    def block_of(n):
        return encoder_block_index(n.replace("base_model.model.", ""))

    leaked = [n for n in trainable if (i := block_of(n)) is not None and i < cutoff]
    assert not leaked, f"lower encoder blocks are trainable: {leaked[:5]}"
    assert n_trainable > 0, "nothing is trainable; LoRA did not attach"

    top_encoder = sum(1 for n in trainable if block_of(n) is not None)
    in_decoder = sum(1 for n in trainable if DECODER_PREFIX in n)
    assert top_encoder > 0, "no encoder tensors trainable; the encoder was the whole point"

    print(f"\n[2] LoRA r={LORA_R} on {len(targets)} modules: "
          f"encoder blocks {cutoff}..{max(encoder_blocks)} plus the whole decoder")
    print(f"[3] trainable {n_trainable:,} / {n_total:,} = {100*n_trainable/n_total:.3f}%")
    print(f"    {top_encoder} trainable tensors in top encoder blocks, "
          f"{in_decoder} in the decoder")
    print(f"    no trainable tensor below encoder block {cutoff}: verified")
    return model


def backward_pass(model, processor, device: str, batch_size: int = 2) -> None:
    """Steps 4 and 5."""
    dataset = TorgoDataset("torgo_dys_train.jsonl")
    collate = SpeechSeq2SeqCollator(processor)
    batch = collate([dataset[i] for i in range(batch_size)])
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

    model.train().to(device)
    began = time.perf_counter()
    out = model(**batch)
    loss = out.loss
    assert loss is not None, "model returned no loss; check the label key"
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()
    elapsed = time.perf_counter() - began

    grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients reached any trainable parameter"

    print(f"\n[4] loss {loss.item():.4f}, finite. gradients on {len(grads)} tensors")
    print(f"[5] {batch_size / elapsed:.2f} examples/s on {device}"
          f" ({elapsed:.1f}s for {batch_size})")
    if device == "cuda":
        print(f"    peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print(f"    dataset: {len(dataset)} training clips")
    print("\n    Set max_steps from this rate on the H100, not from the plan's 400.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="steps 1 to 3 only")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--top-blocks", type=int, default=TOP_ENCODER_BLOCKS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device {device}, dtype {dtype}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = load(dtype)

    info = report_modules(model)
    model = apply_lora(model, info["encoder_blocks"], args.top_blocks)

    if args.dry_run:
        print("\ndry run: skipped the backward pass")
        return
    backward_pass(model, processor, device, args.batch_size)


if __name__ == "__main__":
    main()
