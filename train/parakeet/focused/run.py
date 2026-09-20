"""Controlled adapter/last-two-block comparison; fixed greedy decoding only."""
import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

from contract import manifest
from contract.evaluate import score
from contract.normalize import normalize
from train.parakeet.decode import check_manifest_hash_scope, transcribe_batch, resolve_nemo_checkpoint
from train.parakeet.focused.data import BalancedDraws, SEED


def metrics(rows):
    return {s:score([r for r in rows if r['speaker_id']==s]) for s in sorted({r['speaker_id'] for r in rows})}


def acceptable(current, baseline):
    return all(current[s]['wer'] <= baseline[s]['wer']+tol+1e-9 for s,tol in [('M01',.05),('F04',.02),('M04',.05)])


def selection(current):
    return (current['F01']['wer'],current['M01']['wer'])


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from nemo.collections.asr.models import ASRModel
    from nemo.core import adapter_mixins
    from omegaconf import OmegaConf, open_dict
    args=argparse.ArgumentParser();args.add_argument('variant',choices=['baseline','adapter','partial','final_frozen']);variant=args.parse_args().variant
    max_steps=int(os.environ.get('FOCUSED_MAX_STEPS','600'))
    if max_steps < 1:raise ValueError('FOCUSED_MAX_STEPS must be positive')
    model_id=os.environ.get('FOCUSED_MODEL_ID','nvidia/parakeet-tdt-0.6b-v2')
    normalize_targets=os.environ.get('FOCUSED_NORMALIZE_TARGETS','0')=='1'
    out=Path(os.environ['BT_CHECKPOINT_DIR'])/'results'/variant;out.mkdir(parents=True,exist_ok=True)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    check_manifest_hash_scope('contract/MANIFEST_HASHES','train-dev')
    root=manifest.data_root();receipt=json.loads((root/'split_receipt.json').read_text())
    for name,info in receipt.items():
        assert hashlib.sha256((root/'manifests'/name).read_bytes()).hexdigest()==info['sha256']
    (out/'split_receipt.json').write_text(json.dumps(receipt,indent=2))
    base=str(resolve_nemo_checkpoint(os.environ['PARAKEET_MODEL_PATH']))
    cfg=ASRModel.restore_from(base,return_config=True)
    if variant in ('adapter','partial'):
        with open_dict(cfg):cfg.encoder._target_=adapter_mixins.get_registered_adapter(cfg.encoder._target_).adapter_class_path
    model=ASRModel.restore_from(base,override_config_path=cfg).cuda().eval()
    with open_dict(model.cfg.decoding):model.cfg.decoding.strategy='greedy_batch'
    model.change_decoding_strategy(model.cfg.decoding)
    model.spec_augmentation=None

    def evaluate(name,tag):
        model.eval()
        rows=manifest.load(name,resolve=True);predictions=[]
        with torch.inference_mode():
            for i in range(0,len(rows),16):
                batch=rows[i:i+16];start=time.perf_counter()
                hyps=transcribe_batch(model,[r['audio_filepath'] for r in batch],len(batch))
                assert len(hyps)==len(batch)
                ms=(time.perf_counter()-start)*1000/len(batch)
                predictions.extend({**r,'prediction':h,'model_id':model_id+'-'+variant,'latency_ms':ms} for r,h in zip(batch,hyps))
        (out/f'{tag}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        result=metrics(predictions);(out/f'{tag}_scores.json').write_text(json.dumps(result,indent=2))
        print('FOCUSED_EVAL '+json.dumps({'variant':variant,'tag':tag,'wer':{s:v['wer'] for s,v in result.items()}}),flush=True)
        return result

    if variant in ('baseline','final_frozen'):
        for name in (['dev','smoke'] if variant=='baseline' else ['test']):evaluate(f'focused_{name}.jsonl',name)
        return

    baseline=json.loads((out.parent/'baseline/dev_scores.json').read_text())
    # Fresh adapters, identical seed in both arms; never load the old contaminated adapter.
    adapter_cfg=OmegaConf.create({'_target_':'nemo.collections.common.parts.adapter_modules.LinearAdapter','in_features':int(model.cfg.encoder.d_model),'dim':32,'activation':'swish','norm_position':'pre','dropout':.1})
    model.add_adapter(name='encoder:focused',cfg=adapter_cfg)
    model.set_enabled_adapters(enabled=True);model.cuda()
    model.freeze();model.unfreeze_enabled_adapters()
    adapter_names={n for n,p in model.named_parameters() if p.requires_grad}
    assert adapter_names and all(n.startswith('encoder.') for n in adapter_names)
    n_layers=len(model.encoder.layers);assert n_layers>=2
    pretrained_names=set()
    if variant=='partial':
        for i in range(n_layers-2,n_layers):
            for suffix,p in model.encoder.layers[i].named_parameters():
                name=f'encoder.layers.{i}.{suffix}'
                if name not in adapter_names:pretrained_names.add(name)
    enabled=adapter_names|pretrained_names
    def train_mode():
        model.eval()  # Keep frozen dropout and all BN running statistics fixed.
        for n,p in model.named_parameters():p.requires_grad_(n in enabled)
        if variant=='partial':
            for layer in model.encoder.layers[-2:]:layer.train()
        for n,module in model.named_modules():
            if 'adapter_layer' in n:module.train()
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        assert not any(p.requires_grad for p in model.decoder.parameters())
        assert not any(p.requires_grad for p in model.joint.parameters())
    train_mode()
    params=dict(model.named_parameters())
    groups=[{'params':[params[n] for n in sorted(adapter_names)],'lr':3e-5,'label':'adapters'}]
    if pretrained_names:groups.append({'params':[params[n] for n in sorted(pretrained_names)],'lr':3e-6,'label':'last_two_blocks'})
    optimizer=torch.optim.AdamW(groups,weight_decay=0.01)
    counts={'adapter':sum(params[n].numel() for n in adapter_names),'pretrained':sum(params[n].numel() for n in pretrained_names)}
    config={'variant':variant,'seed':SEED,'counts':counts,'trainable_names':sorted(enabled),'adapter_lr':3e-5,'pretrained_lr':3e-6,'max_steps':max_steps,'model_id':model_id,'encoder_layers':n_layers,'normalize_targets':normalize_targets,'eval_interval':50,'patience':5,'selection':'lexicographic F01 then M01 development WER','guardrails':{'M01':.05,'F04':.02,'M04':.05},'greedy':True,'fresh_base':True}
    (out/'recipe.json').write_text(json.dumps(config,indent=2));print('TRAINABLE '+json.dumps(counts),flush=True)
    def frozen_hash():
        h=hashlib.sha256()
        for n,p in model.named_parameters():
            if n not in enabled:h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    initial_frozen_hash=frozen_hash()
    # In-memory best state avoids restoring another model in this NeMo process.
    def snapshot():return {n:t.detach().cpu().clone() for n,t in model.state_dict().items() if n in enabled or n in dict(model.named_buffers())}
    initial=evaluate('focused_dev.jsonl','step_0000_dev')
    assert all(abs(initial[s]['wer']-baseline[s]['wer'])<1e-9 for s in baseline),'Fresh adapter must reproduce frozen WER'
    best=selection(initial);best_state=snapshot();best_step=0;stale=0;history=[]
    sampler=BalancedDraws(manifest.load('focused_train.jsonl',resolve=True));draw_counts={s:0 for s in sampler.speakers}
    def batch():
        rows=[sampler.draw() for _ in range(4)];signals=[];targets=[]
        for r in rows:
            audio,sr=sf.read(r['audio_filepath'],dtype='float32');assert sr==16000 and audio.ndim==1
            signals.append(torch.from_numpy(audio));targets.append(torch.tensor(model.tokenizer.text_to_ids(normalize(r['text']) if normalize_targets else r['text']),dtype=torch.long));draw_counts[r['speaker_id']]+=1
        return (pad_sequence(signals,batch_first=True).cuda(),torch.tensor([len(x) for x in signals],device='cuda'),pad_sequence(targets,batch_first=True).cuda(),torch.tensor([len(x) for x in targets],device='cuda'))
    for step in range(1,max_steps+1):
        train_mode();optimizer.zero_grad(set_to_none=True);loss_sum=0.
        # Four microbatches of four utterances; same sample stream in both arms.
        for micro in range(4):
            signal,length,target,target_length=batch()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                encoded,encoded_len=model(input_signal=signal,input_signal_length=length)
                decoded,target_length,_=model.decoder(targets=target,target_length=target_length)
                if model.joint.fuse_loss_wer:
                    loss,_,_,_=model.joint(encoder_outputs=encoded,decoder_outputs=decoded,encoder_lengths=encoded_len,transcripts=target,transcript_lengths=target_length,compute_wer=False)
                else:
                    logits=model.joint(encoder_outputs=encoded,decoder_outputs=decoded)
                    loss=model.loss(log_probs=logits.float(),targets=target,input_lengths=encoded_len,target_lengths=target_length)
            assert torch.isfinite(loss),'Non-finite loss'
            (loss/4).backward();loss_sum+=loss.item()/4
        torch.nn.utils.clip_grad_norm_([params[n] for n in enabled],1.)
        # Warm up pretrained and adapter rates together; maintain 10:1 ratio.
        scale=min(1.,step/25)
        for g,lr in zip(optimizer.param_groups,[3e-5,3e-6]):g['lr']=lr*scale
        optimizer.step()
        if step%10==0:print('TRAIN_STEP '+json.dumps({'variant':variant,'step':step,'loss':loss_sum}),flush=True)
        if step%50==0:
            current=evaluate('focused_dev.jsonl',f'step_{step:04d}_dev')
            eligible=acceptable(current,baseline);key=selection(current)
            improved=eligible and key<best
            if improved:best=key;best_state=snapshot();best_step=step;stale=0
            else:stale+=1
            history.append({'step':step,'loss':loss_sum,'wer':{s:v['wer'] for s,v in current.items()},'eligible':eligible,'best_step':best_step})
            (out/'history.json').write_text(json.dumps(history,indent=2))
            if stale>=5:break
    model.load_state_dict(best_state,strict=False);model.eval()
    assert frozen_hash()==initial_frozen_hash,'Frozen weights changed'
    for name in ['dev','smoke','test']:evaluate(f'focused_{name}.jsonl',f'best_{name}')
    model.save_to(str(out/'best.nemo'))
    (out/'selection.json').write_text(json.dumps({'best_step':best_step,'last_step':step,'best_dev_key':best,'frozen_weights_unchanged':True,'draw_counts':draw_counts},indent=2))
    print('FOCUSED_COMPLETE '+json.dumps({'variant':variant,'best_step':best_step,'last_step':step}),flush=True)

if __name__=='__main__':main()
