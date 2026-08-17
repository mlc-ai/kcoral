import torch
import triton
import triton.language as tl

# Set allocator for device-created descriptors as required by Triton's TMA implementation
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    configs = []
    
    # 1. Persistent Warp-Specialized configs (Hardware accelerated persistent grid)
    for block_m, block_n, block_k, stages, warps in [
        (128, 256, 128, 2, 8),
        (256, 128, 128, 2, 8),
        (128, 128, 128, 3, 8),
        (128, 256, 64, 3, 8),
        (256, 128, 64, 3, 8),
    ]:
        for gm in [4, 8]:
            configs.append(triton.Config(
                {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': gm, 'PERSISTENT': True, 'WARP_SPECIALIZE': True},
                num_stages=stages, num_warps=warps, num_ctas=1
            ))

    # 2. Standard software pipelined grid with cluster grouping (lets hardware scheduler handle latency)
    for block_m, block_n, block_k, stages, warps, ctas in [
        (128, 256, 128, 2, 8, 1),
        (256, 128, 128, 2, 8, 1),
        (128, 128, 128, 3, 8, 1),
        (128, 256, 128, 2, 8, 2),
        (256, 128, 128, 2, 8, 2),
        (128, 256, 128, 2, 8, 4),
        (256, 128, 128, 2, 8, 4),
        (128, 256, 128, 2, 8, 8),
        (256, 128, 128, 2, 8, 8),
        (128, 256, 64, 3, 8, 2),
    ]:
        for gm in [4, 8]:
            configs.append(triton.Config(
                {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': gm, 'PERSISTENT': False, 'WARP_SPECIALIZE': False},
                num_stages=stages, num_warps=warps, num_ctas=ctas
            ))

    return configs


@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K']
)
@triton.jit
def _unified_tma_gemm(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_bn, stride_cm,
    N: tl.constexpr, K: tl.constexpr, NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    PERSISTENT: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # TMA descriptors natively map directly to hardware WGMMA acceleration without layout conversions
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, 1], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, 1], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_k_tiles = K // BLOCK_K
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n

    if PERSISTENT:
        start_pid = tl.program_id(0)
        # Deeply integrated Hopper persistent iteration sequence (WARP_SPECIALIZE=True)
        for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False, warp_specialize=WARP_SPECIALIZE):
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
            
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m
            
            offset_m = pid_m * BLOCK_M
            offset_n = pid_n * BLOCK_N
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

            for k_tile in range(num_k_tiles):
                offset_k = k_tile * BLOCK_K
                a = a_desc.load([offset_m, offset_k])
                b = b_desc.load([offset_n, offset_k])
                # Hardware handles b.T implicitly through its columnar WGMMA instruction operand mapping 
                acc = tl.dot(a, b.T, acc)

            c_desc.store([offset_m, offset_n], acc.to(dtype))
            
    else:
        pid = tl.program_id(0)
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)

        c_desc.store([offset_m, offset_n], acc.to(dtype))


_NUM_SMS_CACHE = {}

def run(A, B, C):
    """
    Computes C = A @ B.T directly into the supplied C tensor.
    Exploits Hopper's TMA and WGMMA natively using exact shapes.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    device = A.device
    if device not in _NUM_SMS_CACHE:
        _NUM_SMS_CACHE[device] = torch.cuda.get_device_properties(device).multi_processor_count
    NUM_SMS = _NUM_SMS_CACHE[device]
    
    def grid_fn(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = triton.cdiv(N, META['BLOCK_N'])
        num_tiles = num_pid_m * num_pid_n
        if META['PERSISTENT']:
            return (min(NUM_SMS, num_tiles),)
        else:
            return (num_tiles,)
            
    _unified_tma_gemm[grid_fn](
        A, B, C,
        M,
        A.stride(0), B.stride(0), C.stride(0),
        N=N, K=K, NUM_SMS=NUM_SMS
    )