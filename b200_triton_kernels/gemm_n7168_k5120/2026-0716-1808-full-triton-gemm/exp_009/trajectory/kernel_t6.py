import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N_dim,
    K_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N_dim, BLOCK_N)
    num_tiles = num_tiles_m * num_tiles_n
    
    for tile_id in range(start_pid, num_tiles, tile_stride):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        
        m_offset = pid_m * BLOCK_M
        n_offset = pid_n * BLOCK_N
        
        if m_offset >= M:
            break
            
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        offs_m = m_offset + tl.arange(0, BLOCK_M)
        offs_n = n_offset + tl.arange(0, BLOCK_N)
        
        for k0 in range(0, triton.cdiv(K_dim, BLOCK_K)):
            k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
            
            mask_m = (offs_m[:, None] < M) & (k[None, :] < K_dim)
            a_tile = tl.load(A_ptr + offs_m[:, None] * K_dim + k[None, :] * 1, mask=mask_m, other=0.0)
            
            mask_n = (offs_n[None, :] < N_dim) & (k[:, None] < K_dim)
            b_tile = tl.load(B_ptr + offs_n[None, :] * K_dim + k[:, None] * 1, mask=mask_n, other=0.0)
            
            acc = tl.dot(a_tile, b_tile, acc)
        
        mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N_dim)
        out_ptr = C_ptr + offs_m[:, None] * N_dim + offs_n[None, :] * 1
        tl.store(out_ptr, acc.to(tl.bfloat16), mask=mask_out)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N_dim = B.shape[0] 
    K_dim = B.shape[1]
    
    block_m = 128
    block_n = 128
    block_k = 64
    
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    num_tiles_m = triton.cdiv(M, block_m)
    num_tiles_n = triton.cdiv(N_dim, block_n)
    num_tiles = num_tiles_m * num_tiles_n
    
    grid = (min(num_sms, num_tiles),)
    
    _gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N_dim,
        K_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )


if __name__ == "__main__":
    M_val = 8192
    N_val = 7168
    K_val = 5120
    
    A = torch.randn((M_val, K_val), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((N_val, K_val), dtype=torch.bfloat16, device="cuda")
    C = torch.empty((M_val, N_val), dtype=torch.bfloat16, device="cuda")
    
    run(A, B, C)
    
    C_ref = torch.matmul(A, B.T)
    print(f"Max difference: {torch.max(torch.abs(C - C_ref))}")
    print(f"Mean difference: {torch.mean(torch.abs(C - C_ref))}")