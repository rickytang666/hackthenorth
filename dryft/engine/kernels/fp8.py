"""FP8 decode MLP: cuBLAS matmul and fused per-row/channel scaling."""
import torch
import triton
import triton.language as tl

@triton.jit
def quant_x(X,Q,S,SMOOTH,USE_SMOOTH:tl.constexpr,M:tl.constexpr,P:tl.constexpr,K:tl.constexpr,R:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,R)
    x=tl.load(X+row*K+col,(row<M)&(col<K),0).to(tl.float32)
    if USE_SMOOTH:
        x = x / tl.load(SMOOTH+col,col<K,1.)
    scale=tl.maximum(tl.max(tl.abs(x),0),1.e-8)/448.
    q=tl.minimum(tl.maximum(x/scale,-448.),448.)
    tl.store(Q+row*K+col,q,col<K);tl.store(S+row,scale)

@triton.jit
def epilogue(A,XS,WS,Y,M:tl.constexpr,N:tl.constexpr,P:tl.constexpr,TRANS:tl.constexpr,SWIGLU:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    width=N//2 if SWIGLU else N
    offsets=col*P+row if TRANS else row*N+col
    a=tl.load(A+offsets,col<width,0).to(tl.float32)
    xs=tl.load(XS+row);ws=tl.load(WS+col,col<width,0)
    rounded=(a*xs*ws).to(tl.bfloat16).to(tl.float32)
    if SWIGLU:
        offs=(col+width)*P+row if TRANS else row*N+col+width
        up=tl.load(A+offs,col<width,0).to(tl.float32)
        us=tl.load(WS+col+width,col<width,0)
        up=(up*xs*us).to(tl.bfloat16).to(tl.float32)
        rounded=(rounded*tl.sigmoid(rounded)).to(tl.bfloat16).to(tl.float32)*up
    tl.store(Y+row*width+col,rounded,col<width)


def fp8(x,w,s,one,swiglu=False,smooth=None):
    shape=x.shape[:-1]
    x=x.reshape(-1,x.shape[-1])
    m,k=x.shape;n=w.shape[0];p=m
    q=torch.empty((p,k),device='cuda',dtype=torch.float8_e4m3fn);xs=torch.empty(p,device='cuda')
    quant_x[(p,)](x,q,xs,x if smooth is None else smooth,smooth is not None,m,p,k,triton.next_power_of_2(k),num_warps=4)
    # Scalar scale=1 selects cuBLAS; apply row/channel scales before BF16 rounding
    # in the epilogue, avoiding PyTorch's large-M rowwise CUTLASS dispatch.
    acc=torch._scaled_mm(q,w.T,scale_a=one,scale_b=one,out_dtype=torch.float32,use_fast_accum=False)
    width=n//2 if swiglu else n
    out=torch.empty((m,width),device='cuda',dtype=torch.bfloat16)
    epilogue[(m,triton.cdiv(width,256))](acc,xs,s,out,m,n,p,False,swiglu,256,num_warps=4,enable_fp_fusion=False)
    return out.reshape(*shape,width)


@triton.jit
def _pack(W, SMOOTH, RMS, Q, SCALE, K:tl.constexpr, R:tl.constexpr, SEARCH:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,R)
    smooth=tl.load(SMOOTH+col,col<K,1.)
    rms=tl.load(RMS+col,col<K,0.) / smooth
    w=tl.load(W+row*K+col,col<K,0.).to(tl.float32)*smooth
    base=tl.maximum(tl.max(tl.abs(w),0),1.e-8)/448.
    best_error=float('inf')
    best_scale=base
    best_q=tl.full((R,),0,tl.float32)
    for i in range(5 if SEARCH else 1):
        factor=0.8+i*0.1 if SEARCH else 1.
        scale=base*factor
        q=tl.minimum(tl.maximum(w/scale,-448.),448.).to(tl.float8e4nv)
        delta=(q.to(tl.float32)*scale-w)*rms
        error=tl.sum(delta*delta,0)
        better=error<best_error
        best_error=tl.minimum(best_error,error)
        best_scale=tl.where(better,scale,best_scale)
        best_q=tl.where(better,q.to(tl.float32),best_q)
    tl.store(Q+row*K+col,best_q,col<K)
    tl.store(SCALE+row,best_scale)

def pack(weight, smooth, rms, search):
    n,k=weight.shape
    q=torch.empty_like(weight,dtype=torch.float8_e4m3fn)
    scales=torch.empty(n,device=weight.device,dtype=torch.float32)
    _pack[(n,)](weight,smooth,rms,q,scales,k,triton.next_power_of_2(k),search,num_warps=4)
    return q,scales
