#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ void atomicAddBf162(__nv_bfloat162* address, __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    atomicAdd(address, val);
#else
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;
    do {
        assumed = old;
        float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&assumed));
        float2 f_val = __bfloat1622float2(val);
        f.x += f_val.x; f.y += f_val.y;
        __nv_bfloat162 sum = __floats2bfloat162_rn(f.x, f.y);
        old = atomicCAS(address_as_uint, assumed, *reinterpret_cast<unsigned int*>(&sum));
    } while (assumed != old);
#endif
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t trans_a, uint32_t trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (trans_a << 15);
    d |= (trans_b << 16);
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_MxNxK_cg1(
    uint32_t tmem_C, 
    __nv_bfloat16* smem_A, int A_k_stride, uint32_t lbo_A, uint32_t sbo_A,
    __nv_bfloat16* smem_B, int B_k_stride, uint32_t lbo_B, uint32_t sbo_B,
    int M, int N, int K_total,
    uint32_t idesc, bool accumulate)
{
    #pragma unroll
    for (int k = 0; k < K_total / 16; ++k) {
        uint64_t desc_A = make_smem_desc_sm100_fn(smem_A + k * 16 * A_k_stride, lbo_A, sbo_A);
        uint64_t desc_B = make_smem_desc_sm100_fn(smem_B + k * 16 * B_k_stride, lbo_B, sbo_B);
        uint32_t acc = (k == 0 && !accumulate) ? 0 : 1;
        umma_f16_cg1_fn(tmem_C, desc_A, desc_B, idesc, acc);
    }
}

__device__ __forceinline__ void write_swizzled_128B(__nv_bfloat16* smem, int row, int col, __nv_bfloat16 val) {
    int x_chunk = col / 8;
    int swizzled_x_chunk = (row % 8) ^ x_chunk;
    int swizzled_col = swizzled_x_chunk * 8 + (col % 8);
    smem[row * 64 + swizzled_col] = val;
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled_128B(const __nv_bfloat16* smem, int row, int col) {
    int x_chunk = col / 8;
    int swizzled_x_chunk = (row % 8) ^ x_chunk;
    int swizzled_col = swizzled_x_chunk * 8 + (col % 8);
    return smem[row * 64 + swizzled_col];
}

__device__ __forceinline__ void zero_oob_smem(__nv_bfloat16* smem_L, __nv_bfloat16* smem_R, int valid_rows, int total_rows) {
    if (valid_rows >= total_rows) return;
    int oob_rows = total_rows - valid_rows;
    if (oob_rows <= 0) return;
    
    for (int idx = threadIdx.x; idx < oob_rows * 128; idx += blockDim.x) {
        int r = valid_rows + idx / 128;
        int c = idx % 128;
        __nv_bfloat16* target = (c < 64) ? smem_L : smem_R;
        write_swizzled_128B(target, r, c % 64, __float2bfloat16(0.0f));
    }
}

__global__ void ComputeDKernel(const __nv_bfloat162* O, const __nv_bfloat162* dO, float* D, int B, int H, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H * S) {
        float sum = 0.0f;
        int offset = idx * 64; 
        for (int k = 0; k < 64; ++k) {
            __nv_bfloat162 o_val = O[offset + k];
            __nv_bfloat162 do_val = dO[offset + k];
            float2 o_f2 = __bfloat1622float2(o_val);
            float2 do_f2 = __bfloat1622float2(do_val);
            sum += o_f2.x * do_f2.x + o_f2.y * do_f2.y;
        }
        D[idx] = sum;
    }
}

__device__ __forceinline__ void compute_and_store_P_dS(
    uint32_t tmem_S, uint32_t tmem_dP,
    __nv_bfloat16* smem_PT_L, __nv_bfloat16* smem_PT_R,
    __nv_bfloat16* smem_dST_L, __nv_bfloat16* smem_dST_R,
    __nv_bfloat16* smem_dS,
    const float* L, const float* D,
    int i_start, int j_start, int S_seq, float scale) 
{
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t row = warp_id * 32 + lane_id; 
    int global_i = i_start + row;
    
    float l_val = (global_i < S_seq) ? L[global_i] : 0.0f;
    float d_val = (global_i < S_seq) ? D[global_i] : 0.0f;
    
    for (uint32_t c = 0; c < 64; c += 4) {
        uint32_t col_S = tmem_S + c;
        uint32_t col_dP = tmem_dP + c;
        
        uint32_t s0, s1, s2, s3;
        uint32_t dp0, dp1, dp2, dp3;
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(s0), "=r"(s1), "=r"(s2), "=r"(s3) : "r"(col_S));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(dp0), "=r"(dp1), "=r"(dp2), "=r"(dp3) : "r"(col_dP));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float fs[4] = { __uint_as_float(s0), __uint_as_float(s1), __uint_as_float(s2), __uint_as_float(s3) };
        float fdp[4] = { __uint_as_float(dp0), __uint_as_float(dp1), __uint_as_float(dp2), __uint_as_float(dp3) };
        
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int col = c + k; 
            int global_j = j_start + col;
            
            bool valid = (global_j <= global_i && global_i < S_seq && global_j < S_seq);
            float p = valid ? expf(fs[k] * scale - l_val) : 0.0f;
            float ds = valid ? p * (fdp[k] - d_val) * scale : 0.0f;
            
            __nv_bfloat16 bp = __float2bfloat16(p);
            __nv_bfloat16 bds = __float2bfloat16(ds);
            
            if (row < 64) {
                write_swizzled_128B(smem_PT_L, col, row, bp);
                write_swizzled_128B(smem_dST_L, col, row, bds);
            } else {
                write_swizzled_128B(smem_PT_R, col, row - 64, bp);
                write_swizzled_128B(smem_dST_R, col, row - 64, bds);
            }
            write_swizzled_128B(smem_dS, row, col, bds);
        }
    }
}

__device__ __forceinline__ void atomicAdd_dQ(
    uint32_t tmem_dQ_L, uint32_t tmem_dQ_R,
    float* dQ_global,
    int i_start, int S_seq)
{
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t row = warp_id * 32 + lane_id; 
    int global_i = i_start + row;
    
    for (uint32_t c = 0; c < 64; c += 4) {
        uint32_t col = tmem_dQ_L + c;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        if (global_i < S_seq) {
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            float* out_ptr = &dQ_global[global_i * 128 + c];
            atomicAdd(out_ptr + 0, f0);
            atomicAdd(out_ptr + 1, f1);
            atomicAdd(out_ptr + 2, f2);
            atomicAdd(out_ptr + 3, f3);
        }
    }
    
    for (uint32_t c = 0; c < 64; c += 4) {
        uint32_t col = tmem_dQ_R + c;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        if (global_i < S_seq) {
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            float* out_ptr = &dQ_global[global_i * 128 + 64 + c];
            atomicAdd(out_ptr + 0, f0);
            atomicAdd(out_ptr + 1, f1);
            atomicAdd(out_ptr + 2, f2);
            atomicAdd(out_ptr + 3, f3);
        }
    }
}

__device__ __forceinline__ void tmem_to_smem_64x64(uint32_t tmem_col, __nv_bfloat16* smem) {
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t row = warp_id * 32 + lane_id; 
    
    for (uint32_t c = 0; c < 64; c += 4) {
        uint32_t col = tmem_col + c;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        if (warp_id < 2) {
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            write_swizzled_128B(smem, row, c + 0, __float2bfloat16(f0));
            write_swizzled_128B(smem, row, c + 1, __float2bfloat16(f1));
            write_swizzled_128B(smem, row, c + 2, __float2bfloat16(f2));
            write_swizzled_128B(smem, row, c + 3, __float2bfloat16(f3));
        } else {
            (void)r0; (void)r1; (void)r2; (void)r3;
        }
    }
}

__global__ void __launch_bounds__(128) FABackwardKernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    float* __restrict__ dQ_float,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S, float scale)
{
    setmaxnreg_inc_sync_fn<256>();

    extern __shared__ __align__(1024) char smem_buf[];
    __nv_bfloat16* smem_Q_L  = (__nv_bfloat16*)smem_buf;         
    __nv_bfloat16* smem_Q_R  = smem_Q_L + 8192;                  
    __nv_bfloat16* smem_dO_L = smem_Q_R + 8192;                  
    __nv_bfloat16* smem_dO_R = smem_dO_L + 8192;                 
    __nv_bfloat16* smem_K_L  = smem_dO_R + 8192;                 
    __nv_bfloat16* smem_K_R  = smem_K_L + 4096;                  
    __nv_bfloat16* smem_V_L  = smem_K_R + 4096;                  
    __nv_bfloat16* smem_V_R  = smem_V_L + 4096;                  

    __nv_bfloat16* smem_PT_L  = smem_V_R + 4096;                 
    __nv_bfloat16* smem_PT_R  = smem_PT_L + 4096;                
    __nv_bfloat16* smem_dST_L = smem_PT_R + 4096;                
    __nv_bfloat16* smem_dST_R = smem_dST_L + 4096;               
    __nv_bfloat16* smem_dS    = smem_dST_R + 4096;               
    
    uint64_t* mbar_Q = (uint64_t*)(smem_dS + 8192);
    uint64_t* mbar_KV = mbar_Q + 1;
    uint64_t* mbar_mma = mbar_KV + 1;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV, 1);
        init_smem_barrier_fn(mbar_mma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    __shared__ uint32_t tmem_addr_smem;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_addr_smem, 512); 
    }
    __syncthreads();
    uint32_t tmem_addr = tmem_addr_smem;
    
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int j_block = blockIdx.x;
    
    int j_start = j_block * 64;
    
    size_t head_offset_D = (b_idx * gridDim.y + h_idx) * (size_t)S;
    const float* L_head = L + head_offset_D;
    const float* D_head = D + head_offset_D;
    
    int c2_offset = (b_idx * gridDim.y + h_idx) * S;
    float* dQ_head = dQ_float + c2_offset * 128;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_KV, 16384 * 2);
        tma_load_2d_fn(&tma_K, mbar_KV, smem_K_L, 0, c2_offset + j_start);
        tma_load_2d_fn(&tma_K, mbar_KV, smem_K_R, 64, c2_offset + j_start);
        tma_load_2d_fn(&tma_V, mbar_KV, smem_V_L, 0, c2_offset + j_start);
        tma_load_2d_fn(&tma_V, mbar_KV, smem_V_R, 64, c2_offset + j_start);
    }
    mbarrier_wait_fn(mbar_KV, 0);
    
    int valid_j = (S > j_start) ? S - j_start : 0;
    zero_oob_smem(smem_K_L, smem_K_R, valid_j, 64);
    zero_oob_smem(smem_V_L, smem_V_R, valid_j, 64);
    
    __syncthreads();
    fence_async_shared_fn();
    
    uint32_t idesc_S  = make_instr_desc_fn(128, 64, 0, 0);
    uint32_t idesc_dP = make_instr_desc_fn(128, 64, 0, 0);
    uint32_t idesc_dV = make_instr_desc_fn(64, 64, 0, 1);
    uint32_t idesc_dK = make_instr_desc_fn(64, 64, 0, 1);
    uint32_t idesc_dQ = make_instr_desc_fn(128, 64, 0, 1);
    
    uint32_t tmem_S    = 0;
    uint32_t tmem_dP   = 64;
    uint32_t tmem_dV_L = 128;
    uint32_t tmem_dV_R = 192;
    uint32_t tmem_dK_L = 256;
    uint32_t tmem_dK_R = 320;
    uint32_t tmem_dQ_L = 384;
    uint32_t tmem_dQ_R = 448;
    
    int num_i_blocks = (S + 127) / 128;
    int start_i_block = j_start / 128;
    
    int mma_phase = 0;
    
    for (int i_block = start_i_block; i_block < num_i_blocks; ++i_block) {
        int i_start = i_block * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768 * 2);
            tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q_L, 0, c2_offset + i_start);
            tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q_R, 64, c2_offset + i_start);
            tma_load_2d_fn(&tma_dO, mbar_Q, smem_dO_L, 0, c2_offset + i_start);
            tma_load_2d_fn(&tma_dO, mbar_Q, smem_dO_R, 64, c2_offset + i_start);
        }
        mbarrier_wait_fn(mbar_Q, (i_block - start_i_block) & 1); 
        
        int valid_i = (S > i_start) ? S - i_start : 0;
        zero_oob_smem(smem_Q_L, smem_Q_R, valid_i, 128);
        zero_oob_smem(smem_dO_L, smem_dO_R, valid_i, 128);
        
        __syncthreads();
        fence_async_shared_fn();
        
        tcgen05_fence_before_fn();
        
        bool accum_dV = (i_block != start_i_block);
        bool accum_dK = (i_block != start_i_block);
        
        if (threadIdx.x == 0) {
            umma_MxNxK_cg1(tmem_S, smem_Q_L, 1, 1, 1024, smem_K_L, 1, 1, 1024, 128, 64, 64, idesc_S, false);
            umma_MxNxK_cg1(tmem_S, smem_Q_R, 1, 1, 1024, smem_K_R, 1, 1, 1024, 128, 64, 64, idesc_S, true);
            
            umma_MxNxK_cg1(tmem_dP, smem_dO_L, 1, 1, 1024, smem_V_L, 1, 1, 1024, 128, 64, 64, idesc_dP, false);
            umma_MxNxK_cg1(tmem_dP, smem_dO_R, 1, 1, 1024, smem_V_R, 1, 1, 1024, 128, 64, 64, idesc_dP, true);
            
            uint32_t mbar_a = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_a));
        }
        
        mbarrier_wait_fn(mbar_mma, mma_phase & 1);
        mma_phase++;
        tcgen05_fence_after_fn();
        
        compute_and_store_P_dS(tmem_S, tmem_dP, smem_PT_L, smem_PT_R, smem_dST_L, smem_dST_R, smem_dS, L_head, D_head, i_start, j_start, S, scale);
        
        __syncthreads();
        fence_proxy_async_fn();
        
        tcgen05_fence_before_fn();
        
        if (threadIdx.x == 0) {
            umma_MxNxK_cg1(tmem_dV_L, smem_PT_L, 1, 1, 1024, smem_dO_L, 64, 2048, 1024, 64, 64, 64, idesc_dV, accum_dV);
            umma_MxNxK_cg1(tmem_dV_L, smem_PT_R, 1, 1, 1024, smem_dO_L + 64*64, 64, 2048, 1024, 64, 64, 64, idesc_dV, true);
            
            umma_MxNxK_cg1(tmem_dV_R, smem_PT_L, 1, 1, 1024, smem_dO_R, 64, 2048, 1024, 64, 64, 64, idesc_dV, accum_dV);
            umma_MxNxK_cg1(tmem_dV_R, smem_PT_R, 1, 1, 1024, smem_dO_R + 64*64, 64, 2048, 1024, 64, 64, 64, idesc_dV, true);
            
            umma_MxNxK_cg1(tmem_dK_L, smem_dST_L, 1, 1, 1024, smem_Q_L, 64, 2048, 1024, 64, 64, 64, idesc_dK, accum_dK);
            umma_MxNxK_cg1(tmem_dK_L, smem_dST_R, 1, 1, 1024, smem_Q_L + 64*64, 64, 2048, 1024, 64, 64, 64, idesc_dK, true);
            
            umma_MxNxK_cg1(tmem_dK_R, smem_dST_L, 1, 1, 1024, smem_Q_R, 64, 2048, 1024, 64, 64, 64, idesc_dK, accum_dK);
            umma_MxNxK_cg1(tmem_dK_R, smem_dST_R, 1, 1, 1024, smem_Q_R + 64*64, 64, 2048, 1024, 64, 64, 64, idesc_dK, true);
            
            umma_MxNxK_cg1(tmem_dQ_L, smem_dS, 1, 1, 1024, smem_K_L, 64, 2048, 1024, 128, 64, 64, idesc_dQ, false);
            umma_MxNxK_cg1(tmem_dQ_R, smem_dS, 1, 1, 1024, smem_K_R, 64, 2048, 1024, 128, 64, 64, idesc_dQ, false);
            
            uint32_t mbar_a = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_a));
        }
        
        mbarrier_wait_fn(mbar_mma, mma_phase & 1);
        mma_phase++;
        tcgen05_fence_after_fn();
        
        atomicAdd_dQ(tmem_dQ_L, tmem_dQ_R, dQ_head, i_start, S);
        __syncthreads(); 
    }
    
    tmem_to_smem_64x64(tmem_dK_L, smem_K_L); 
    tmem_to_smem_64x64(tmem_dK_R, smem_K_R); 
    tmem_to_smem_64x64(tmem_dV_L, smem_V_L); 
    tmem_to_smem_64x64(tmem_dV_R, smem_V_R); 
    
    __syncthreads();
    
    for (int step = 0; step < 64; ++step) {
        int lin = threadIdx.x + step * 128;
        if (lin < 64 * 128) {
            int r = lin / 128;
            int c = lin % 128;
            if (j_start + r < S) {
                __nv_bfloat16 val_dK = (c < 64) ? read_swizzled_128B(smem_K_L, r, c) : read_swizzled_128B(smem_K_R, r, c - 64);
                __nv_bfloat16 val_dV = (c < 64) ? read_swizzled_128B(smem_V_L, r, c) : read_swizzled_128B(smem_V_R, r, c - 64);
                
                dK[c2_offset * 128 + (j_start + r) * 128 + c] = val_dK;
                dV[c2_offset * 128 + (j_start + r) * 128 + c] = val_dV;
            }
        }
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, 512);
    }
}

__global__ void CastDQKernel(const float* dQ_float, __nv_bfloat16* dQ, int elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        dQ[idx] = __float2bfloat16(dQ_float[idx]);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t total_S, uint32_t smem_rows) {
    cuuint64_t globalDim[2] = {128, total_S};
    cuuint64_t globalStrides[1] = {128 * 2};
    cuuint32_t boxDim[2] = {64, smem_rows};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = 128; 
    
    float scale = 1.0f / sqrtf((float)d);
    
    float* d_D = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_D, B * H * S * sizeof(float), stream));
    
    int threads_D = 128;
    int blocks_D = (B * H * S + threads_D - 1) / threads_D;
    ComputeDKernel<<<blocks_D, threads_D, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(O.data_ptr()),
        static_cast<const __nv_bfloat162*>(dO.data_ptr()),
        d_D, B, H, S
    );
    
    float* dQ_float = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_float, B * H * S * 128 * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_float, 0, B * H * S * 128 * sizeof(float), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), B * H * S, 128);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), B * H * S, 64);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), B * H * S, 64);
    create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), B * H * S, 128);
    
    int num_j_blocks = (S + 63) / 64;
    dim3 grid(num_j_blocks, H, B);
    dim3 block(128); 
    
    int smem_size = 144 * 1024 + 64; 
    CUDA_CHECK(cudaFuncSetAttribute(FABackwardKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    FABackwardKernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        d_D,
        dQ_float,
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, scale
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_D, stream));
    
    int elements = B * H * S * 128;
    int threads_cast = 256;
    int blocks_cast = (elements + threads_cast - 1) / threads_cast;
    CastDQKernel<<<blocks_cast, threads_cast, 0, stream>>>(dQ_float, static_cast<__nv_bfloat16*>(dQ.data_ptr()), elements);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(dQ_float, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda