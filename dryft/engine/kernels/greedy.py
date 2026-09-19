"""BF16 vocabulary projection with tile-local greedy selection.

All vocabulary rows are evaluated. Projection sums round to BF16 before
comparison; equal maxima choose the lowest vocabulary id. Inputs must be
contiguous BF16 tensors and the tied weight remains in its original layout.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _head_tiles(X,W,VAL,IDX,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                COLUMN:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    c=tl.program_id(1)*BN+tl.arange(0,BN)
    kr=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(K//BK):
        k=start*BK+kr
        x=tl.load(X+r[:,None]*K+k[None,:],r[:,None]<M,0)
        if COLUMN:
            w=tl.load(W+k[:,None]*N+c[None,:],c[None,:]<N,0)
        else:
            w=tl.load(W+c[None,:]*K+k[:,None],c[None,:]<N,0)
        acc=tl.dot(x,w,acc)
    logits=tl.where(c[None,:]<N,acc.to(tl.bfloat16).to(tl.float32),float('-inf'))
    maximum=tl.max(logits,1)
    index=tl.min(tl.where(logits==maximum[:,None],c[None,:],2147483647),1)
    tiles=tl.cdiv(N,BN)
    tl.store(VAL+r*tiles+tl.program_id(1),maximum,r<M)
    tl.store(IDX+r*tiles+tl.program_id(1),index,r<M)


@triton.jit
def _head_finish(VAL,IDX,Y,T:tl.constexpr,BT:tl.constexpr):
    t=tl.arange(0,BT)
    v=tl.load(VAL+tl.program_id(0)*T+t,t<T,float('-inf'))
    i=tl.load(IDX+tl.program_id(0)*T+t,t<T,2147483647)
    maximum=tl.max(v,0)
    index=tl.min(tl.where(v==maximum,i,2147483647),0)
    tl.store(Y+tl.program_id(0),index)


def greedy_token(x,w):
    m=x.numel()//x.shape[-1]
    k=x.shape[-1]
    if m > 32:
        return torch.nn.functional.linear(x,w).argmax(-1)
    n=w.shape[0]
    column=False
    bm,bn,bk,warps,stages=(16,64,128,4,5) if m <= 16 else (16,256,64,4,3)
    tiles=triton.cdiv(n,bn)
    values=torch.empty((m,tiles),device=x.device,dtype=torch.float32)
    indices=torch.empty((m,tiles),device=x.device,dtype=torch.int32)
    out=torch.empty((m,),device=x.device,dtype=torch.int64)
    _head_tiles[(triton.cdiv(m,bm),tiles)](x,w,values,indices,m,n,k,column,bm,bn,bk,
        num_warps=warps,num_stages=stages)
    _head_finish[(m,)](values,indices,out,tiles,triton.next_power_of_2(tiles),num_warps=4)
    return out.reshape(x.shape[:-1])
