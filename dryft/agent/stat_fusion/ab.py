"""Full-generation A/B of single-partition attention direct stores."""
import sys,time,statistics
START=time.monotonic()
sys.path.insert(0,'/research/engine');sys.path.insert(0,'/research/prompts')
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer
from engine import Engine
import attention as attention_module
import fused_layer,projections
from kernels import attention as candidate
import attention_baseline as baseline
from prompts import prompts,corpora,emit

@torch.inference_mode()
def main():
    # Exercise both kernel paths, including multi-query verification and graph
    # replay with changing inputs/positions. Non-power-of-two capacities too.
    torch.manual_seed(715)
    for batch,capacity,queries in ((1,31,1),(4,63,4),(16,319,1),(16,319,4),(32,511,4),(16,639,1)):
        q=torch.randn(batch,32,queries,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(batch,8,capacity,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        pos=torch.arange(capacity-queries,capacity,device='cuda')
        candidate.grouped_attention(q,k,v,pos)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):actual=candidate.grouped_attention(q,k,v,pos)
        for start in (0,capacity//2,capacity-queries):
            pos.copy_(torch.arange(start,start+queries,device='cuda'))
            q.normal_();k.normal_();v.normal_();graph.replay()
            expected=baseline.grouped_attention(q,k,v,pos)
            assert torch.equal(actual,expected),(batch,capacity,queries,float((actual-expected).abs().max()))
        emit('kernel_parity',batch=batch,capacity=capacity,queries=queries,exact=True)
    projections.calibrate=lambda *args,**kwargs:None
    path=snapshot_download('Qwen/Qwen3-4B-Instruct-2507',revision='cdbee75f17c01a7cc42f958dc650907174af0554',local_files_only=True)
    engine=Engine(path)
    tokenizer=AutoTokenizer.from_pretrained(path);streams=corpora(tokenizer)
    reference=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
    def mode(enabled):
        module=candidate if enabled else baseline
        attention_module.prologue_attention=module.prologue_attention
        attention_module.grouped_attention=module.grouped_attention
        fused_layer.grouped_attention=module.grouped_attention
    def generate(inputs,length):
        torch.cuda.synchronize();t=time.perf_counter();out=list(engine.generate(inputs,length));torch.cuda.synchronize()
        assert len(out)==length and all(len(s)==len(inputs) for s in out)
        return out,time.perf_counter()-t
    reports=[]
    for shape,seed in [((16,256,64),715),((32,256,64),716),((16,512,128),717),((4,2048,32),718)]:
        if time.monotonic()-START>105:break
        projections._choices[('rope_attention',shape[0])]=('fused',)
        inputs=prompts(tokenizer,streams,shape,seed)[0]
        states={};times={False:[],True:[]};outputs={}
        for enabled in times:
            mode(enabled);engine.state=None
            generate(inputs,shape[2]);states[enabled]=engine.state
        for repeat in range(5):
            for enabled in ([False,True] if repeat%2==0 else [True,False]):
                mode(enabled);engine.state=states[enabled]
                outputs[enabled],seconds=generate(inputs,shape[2]);times[enabled].append(seconds)
        same=outputs[False]==outputs[True];failed=0;worst=0.
        for row,prompt in enumerate(inputs):
            tokens=[step[row] for step in outputs[True]]
            logits=reference(torch.tensor([prompt+tokens],device='cuda'),use_cache=False,logits_to_keep=len(tokens)+1).logits[0,:-1].float()
            deficit=logits.amax(-1)-logits.gather(-1,torch.tensor(tokens,device='cuda')[:,None]).squeeze(-1)
            failed+=int((deficit>2).sum());worst=max(worst,float(deficit.max()))
        record=dict(shape=shape,seed=seed,samples={str(k):v for k,v in times.items()},speedup=statistics.median(times[False])/statistics.median(times[True]),identical_tokens=same,checked=shape[0]*shape[2],failed=failed,worst=worst)
        reports.append(record);emit('paired',**record,elapsed=time.monotonic()-START)
        assert same and failed==0
    emit('complete',reports=reports,elapsed=time.monotonic()-START,scope='five alternating timings per shape; fixed projection choices; own-prefix teacher-forced correctness')
if __name__=='__main__':main()
