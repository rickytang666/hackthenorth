"""Batch-one BF16 down projection with FP32 split reduction and residual add."""
import torch
import triton
import triton.language as tl


@triton.jit
def _vector_parts(X,W,P,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.program_id(1)*BK+tl.arange(0,BK)
    x=tl.load(X+k,k<K,0).to(tl.float32)
    w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
    value=tl.sum(w*x[None,:],1)
    tl.store(P+tl.program_id(1)*N+n,value,n<N)


@triton.jit
def _vector_finish(P,R,Y,N:tl.constexpr,S:tl.constexpr,BS:tl.constexpr,BLOCK:tl.constexpr):
    n=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    s=tl.arange(0,BS)
    value=tl.sum(tl.load(P+s[:,None]*N+n[None,:],(s[:,None]<S)&(n[None,:]<N),0),0)
    value=value.to(tl.bfloat16).to(tl.float32)
    residual=tl.load(R+n,n<N,0).to(tl.float32)
    tl.store(Y+n,value+residual,n<N)


def down_residual(x,w,residual):
    n,k=w.shape
    bn,bk,warps=4,1024,4
    parts=triton.cdiv(k,bk)
    partial=torch.empty((parts,n),device=x.device,dtype=torch.float32)
    out=torch.empty_like(residual)
    _vector_parts[(triton.cdiv(n,bn),parts)](x,w,partial,n,k,bn,bk,num_warps=warps,enable_fp_fusion=False)
    _vector_finish[(triton.cdiv(n,256),)](partial,residual,out,n,parts,triton.next_power_of_2(parts),256,num_warps=4)
    return out


