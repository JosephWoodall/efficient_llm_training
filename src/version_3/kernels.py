import torch
import triton
import triton.language as tl

@triton.jit
def ternary_gemm_kernel(
    a_ptr, w_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        # Weights are stored as int8: {-1, 0, 1}
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        
        # Triton supports integer dot which uses integer arithmetic units.
        # Bypasses standard FPUs for inference speedup.
        accumulator += tl.dot(a, w)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk
        
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def ternary_matmul(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    a: (M, K) int8 activations
    w: (K, N) int8 ternary weights {-1, 0, 1}
    Returns: (M, N) int32 results
    """
    assert a.shape[1] == w.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert w.is_contiguous(), "Matrix W must be contiguous"
    M, K = a.shape
    K, N = w.shape
    
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    
    ternary_gemm_kernel[grid](
        a, w, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return c
