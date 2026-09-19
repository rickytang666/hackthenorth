"""Bounded BF16 down/residual -> next norm/QKV producer-consumer experiment."""
import sys,time,statistics,json
START=time.monotonic()
sys.path.insert(0,'/research/engine')
import torch
from huggingface_hub import snapshot_download
from engine import Engine
from kernels.tile import tile_projection
from kernels.rmsnorm import rms_norm
from ops import project

def emit(kind,**kw):print(json.dumps(dict(kind=kind,**kw)),flush=True)
def capture(fn):
    fn();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    for _ in range(3):graph.replay()
    return graph,out

def measure(graph):
    a,b=torch.cuda.Event(True),torch.cuda.Event(True)
    a.record()
    for _ in range(10):graph.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)*100

@torch.inference_mode()
def main():
    torch.manual_seed(712)
    path=snapshot_download('Qwen/Qwen3-4B-Instruct-2507',revision='cdbee75f17c01a7cc42f958dc650907174af0554',local_files_only=True)
    engine=Engine(path)
    layers=[l.original for l in engine.model.model.layers]
    weights=[]
    for i in range(4,12):
        down=layers[i].mlp.down_proj
        qkv=layers[i+1].self_attn.qkv
        norm=layers[i+1].input_layernorm
        weights.append((down._column(),qkv._column(),norm.weight,norm.variance_epsilon))
    emit('working_set',layers=len(weights),weight_bytes=sum((d.numel()+q.numel())*2 for d,q,_,_ in weights))
    records=[]
    dt=(32,128,2,5)
    for batch in (4,16,32):
        x=torch.randn((batch,9728),device='cuda',dtype=torch.bfloat16)
        residual=torch.randn((batch,2560),device='cuda',dtype=torch.bfloat16)
        qt=(64,128,4,5) if batch<=16 else (32,128,2,5)
        def baseline():
            outputs=[]
            for d,q,g,eps in weights:
                h=tile_projection(x,d,batch,dt,residual=residual)
                y=tile_projection(rms_norm(h,g,eps),q,batch,qt)
                outputs.append((h,y))
            return outputs
        bg,bo=capture(baseline)
        for candidate_tile in (qt,(64,64,4,4),(128,64,4,3)):
            if time.monotonic()-START>110:break
            def candidate():
                outputs=[]
                for d,q,g,eps in weights:
                    h,stats,_=project(x,d,dt,residual=residual,write_stats=True)
                    y,_,_=project(h,q,candidate_tile,gain=g,eps=eps,stats=stats)
                    outputs.append((h,y,stats))
                return outputs
            cg,co=capture(candidate)
            base_times=[];candidate_times=[]
            for repeat in range(5):
                for graph,target in ((bg,base_times),(cg,candidate_times))[::1 if repeat%2==0 else -1]:
                    target.append(measure(graph)/len(weights))
            relative=0.;max_abs=0.;differences=0;positions=0
            for scale in (.25,1.,4.):
                x.normal_(std=scale);residual.normal_(std=scale)
                bg.replay();cg.replay()
                for (bh,by),(ch,cy,stats) in zip(bo,co):
                    assert torch.equal(bh,ch),'residual BF16 output changed'
                    expected=ch.float().square().reshape(batch,-1,dt[0]).sum(-1)
                    torch.testing.assert_close(stats,expected,rtol=1e-6,atol=1e-5)
                    delta=(by.float()-cy.float())
                    relative=max(relative,float(delta.norm()/by.float().norm()))
                    max_abs=max(max_abs,float(delta.abs().max()))
                    differences+=int((by!=cy).sum());positions+=by.numel()
            h,stats,producer=project(x,weights[0][0],dt,residual=residual,write_stats=True)
            _,_,consumer=project(h,weights[0][1],candidate_tile,gain=weights[0][2],stats=stats)
            base=statistics.median(base_times);cand=statistics.median(candidate_times)
            record=dict(batch=batch,tile=candidate_tile,baseline_us=base,candidate_us=cand,speedup=base/cand,
                baseline_samples_us=base_times,candidate_samples_us=candidate_times,
                max_relative_l2=relative,max_abs=max_abs,different_values=differences,compared_values=positions,
                producer_registers=producer.n_regs,producer_spills=producer.n_spills,
                consumer_registers=consumer.n_regs,consumer_spills=consumer.n_spills,
                normalized_tensor_bytes=batch*2560*2,partial_stats_bytes=stats.numel()*4)
            records.append(record);emit('measurement',**record)
            del cg,co
        del bg,bo
    emit('complete',elapsed=time.monotonic()-START,records=records)
if __name__=='__main__':main()
