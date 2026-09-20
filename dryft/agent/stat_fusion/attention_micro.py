import sys,time,json,statistics
from types import SimpleNamespace
START=time.monotonic()
sys.path.insert(0,'/research/engine')
import torch
from kernels.attention import prologue_attention
from attention_ops import attention
from micro import capture,measure,emit

@torch.inference_mode()
def main():
    torch.manual_seed(713)
    for batch,capacity in ((4,2079),(8,639),(16,639),(16,1151),(32,319),(16,319)):
        qkv=torch.randn(batch,1,6144,device='cuda',dtype=torch.bfloat16)
        norm=SimpleNamespace(weight=torch.ones(128,device='cuda',dtype=torch.bfloat16),variance_epsilon=1e-6)
        angle=torch.randn(128,device='cuda',dtype=torch.bfloat16)
        cos,sin=angle.cos(),angle.sin()
        caches=[(torch.randn(batch,8,capacity,128,device='cuda',dtype=torch.bfloat16),torch.randn(batch,8,capacity,128,device='cuda',dtype=torch.bfloat16)) for _ in range(8)]
        pos=torch.tensor([capacity-2],device='cuda',dtype=torch.int64)
        def run(fn):return [fn(qkv,norm,norm,cos,sin,k,v,pos,8,4,4096,5120) for k,v in caches]
        bg,bo=capture(lambda:run(prologue_attention))
        for chunk in (512,1024,2048):
            if time.monotonic()-START>110:break
            cg,co=capture(lambda:run(lambda *args:attention(*args,max_chunk=chunk)))
            bt=[];ct=[]
            for repeat in range(5):
                for graph,target in ((bg,bt),(cg,ct))[::1 if repeat%2==0 else -1]:target.append(measure(graph)/len(caches))
            diff=0.;relative=0.;different=0
            for position in (capacity//2,capacity-2,capacity-1):
                pos.fill_(position);qkv.normal_();bg.replay();cg.replay()
                for b,c in zip(bo,co):
                    delta=b.float()-c.float()
                    diff=max(diff,float(delta.abs().max()));relative=max(relative,float(delta.norm()/b.float().norm()))
                    different+=int((b!=c).sum())
            emit('attention',batch=batch,capacity=capacity,max_chunk=chunk,
                baseline_us=statistics.median(bt),candidate_us=statistics.median(ct),
                speedup=statistics.median(bt)/statistics.median(ct),max_abs=diff,relative_l2=relative,different_values=different)
            del cg,co
        del bg,bo,caches
    emit('complete',elapsed=time.monotonic()-START)
if __name__=='__main__':main()
