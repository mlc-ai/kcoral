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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
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

__device__ __forceinline__ void compute_P_and_dS(
    uint32_t tmem_S, uint32_t tmem_dP,
    __nv_bfloat16* smem_P, __nv_bfloat16* smem_dS,
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
        
        float fs0 = __uint_as_float(s0);
        float fs1 = __uint_as_float(s1);
        float fs2 = __uint_as_float(s2);
        float fs3 = __uint_as_float(s3);
        
        float fdp0 = __uint_as_float(dp0);
        float fdp1 = __uint_as_float(dp1);
        float fdp2 = __uint_as_float(dp2);
        float fdp3 = __uint_as_float(dp3);
        
        int global_j0 = j_start + c + 0;
        int global_j1 = j_start + c + 1;
        int global_j2 = j_start + c + 2;
        int global_j3 = j_start + c + 3;
        
        float p0 = (global_j0 <= global_i && global_i < S_seq && global_j0 < S_seq) ? expf(fs0 * scale - l_val) : 0.0f;
        float p1 = (global_j1 <= global_i && global_i < S_seq && global_j1 < S_seq) ? expf(fs1 * scale - l_val) : 0.0f;
        float p2 = (global_j2 <= global_i && global_i < S_seq && global_j2 < S_seq) ? expf(fs2 * scale - l_val) : 0.0f;
        float p3 = (global_j3 <= global_i && global_i < S_seq && global_j3 < S_seq) ? expf(fs3 * scale - l_val) : 0.0f;
        
        float ds0 = p0 * (fdp0 - d_val) * scale;
        float ds1 = p1 * (fdp1 - d_val) * scale;
        float ds2 = p2 * (fdp2 - d_val) * scale;
        float ds3 = p3 * (fdp3 - d_val) * scale;
        
        uint32_t idx = row * 64 + c;
        smem_P[idx + 0] = __float2bfloat16(p0);
        smem_P[idx + 1] = __float2bfloat16(p1);
        smem_P[idx + 2] = __float2bfloat16(p2);
        smem_P[idx + 3] = __float2bfloat16(p3);
        
        smem_dS[idx + 0] = __float2bfloat16(ds0);
        smem_dS[idx + 1] = __float2bfloat16(ds1);
        smem_dS[idx + 2] = __float2bfloat16(ds2);
        smem_dS[idx + 3] = __float2bfloat16(ds3);
    }
}

__device__ __forceinline__ void atomicAdd_dQ(
    uint32_t tmem_dQ,
    __nv_bfloat16* dQ_global,
    int i_start, int S_seq)
{
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t row = warp_id * 32 + lane_id;
    int global_i = i_start + row;
    
    for (uint32_t c = 0; c < 128; c += 4) {
        uint32_t col = tmem_dQ + c;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        if (global_i < S_seq) {
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            __nv_bfloat162 b01 = __floats2bfloat162_rn(f0, f1);
            __nv_bfloat162 b23 = __floats2bfloat162_rn(f2, f3);
            
            __nv_bfloat162* out_ptr = (__nv_bfloat162*)&dQ_global[global_i * 128 + c];
            atomicAddBf162(out_ptr + 0, b01);
            atomicAddBf162(out_ptr + 1, b23);
        }
    }
}

__device__ __forceinline__ void tmem_to_smem_64x128(uint32_t tmem_base_col, __nv_bfloat16* smem, uint32_t smem_stride_cols) {
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t row = warp_id * 32 + lane_id;
    for (uint32_t c = 0; c < 128; c += 4) {
        uint32_t col = tmem_base_col + c;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        if (warp_id < 2) {
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            uint32_t idx = row * smem_stride_cols + c;
            smem[idx + 0] = __float2bfloat16(f0);
            smem[idx + 1] = __float2bfloat16(f1);
            smem[idx + 2] = __float2bfloat16(f2);
            smem[idx + 3] = __float2bfloat16(f3);
        }
    }
}

__global__ void FABackwardKernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* __restrict__ L,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dQ,
    int S, float scale)
{
    setmaxnreg_inc_sync_fn<256>();

    extern __shared__ __align__(1024) char smem_buf[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_buf; 
    __nv_bfloat16* smem_dO = smem_Q + 16384;          
    __nv_bfloat16* smem_K = smem_dO + 16384;          
    __nv_bfloat16* smem_V = smem_K + 8192;            
    __nv_bfloat16* smem_P = smem_V + 8192;            
    __nv_bfloat16* smem_dS = smem_P + 8192;           

    uint64_t* mbar_Q = (uint64_t*)(smem_buf + 131072);
    uint64_t* mbar_KV = mbar_Q + 1;
    uint64_t* mbar_mma = mbar_KV + 1;
    
    uint32_t tmem_addr;
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV, 1);
        init_smem_barrier_fn(mbar_mma, 1);
        tmem_alloc_fn(&tmem_addr, 512); 
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int j_block = blockIdx.x;
    
    int j_start = j_block * 64;
    
    size_t head_offset_D = (b_idx * gridDim.y + h_idx) * (size_t)S;
    const float* L_head = L + head_offset_D;
    const float* D_head = D + head_offset_D;
    
    int c2_offset = (b_idx * gridDim.y + h_idx) * S;
    __nv_bfloat16* dQ_head = dQ + c2_offset * 128;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_KV, 16384 * 2);
        tma_load_3d_fn(&tma_K, mbar_KV, smem_K, 0, 0, c2_offset + j_start);
        tma_load_3d_fn(&tma_V, mbar_KV, smem_V, 0, 0, c2_offset + j_start);
    }
    mbarrier_wait_fn(mbar_KV, 0);
    fence_async_shared_fn();
    
    uint32_t idesc_S = make_instr_desc_fn(128, 64, 0, 0);
    uint32_t idesc_dP = make_instr_desc_fn(128, 64, 0, 0);
    uint32_t idesc_dV = make_instr_desc_fn(64, 128, 1, 0);
    uint32_t idesc_dK = make_instr_desc_fn(64, 128, 1, 0);
    uint32_t idesc_dQ = make_instr_desc_fn(128, 128, 0, 1);
    
    uint32_t tmem_dV = 0;
    uint32_t tmem_dK = 128;
    uint32_t tmem_S = 256;
    uint32_t tmem_dP = 320;
    uint32_t tmem_dQ = 384;
    
    int num_i_blocks = (S + 127) / 128;
    int start_i_block = j_start / 128;
    
    int mma_phase = 0;
    
    for (int i_block = start_i_block; i_block < num_i_blocks; ++i_block) {
        int i_start = i_block * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768 * 2);
            tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q, 0, 0, c2_offset + i_start);
            tma_load_3d_fn(&tma_dO, mbar_Q, smem_dO, 0, 0, c2_offset + i_start);
        }
        mbarrier_wait_fn(mbar_Q, i_block - start_i_block); 
        fence_async_shared_fn();
        
        tcgen05_fence_before_fn();
        
        uint32_t accum_dV = (i_block == start_i_block) ? 0 : 1;
        uint32_t accum_dK = (i_block == start_i_block) ? 0 : 1;
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_Q_k = make_smem_desc_sm100_fn(smem_Q + k * 16, 1, 1024);
                uint64_t desc_dO_k = make_smem_desc_sm100_fn(smem_dO + k * 16, 1, 1024);
                uint64_t desc_K_k = make_smem_desc_sm100_fn(smem_K + k * 16, 1, 1024);
                uint64_t desc_V_k = make_smem_desc_sm100_fn(smem_V + k * 16, 1, 1024);
                
                uint32_t acc = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_S, desc_Q_k, desc_K_k, idesc_S, acc);
                umma_f16_cg1_fn(tmem_dP, desc_dO_k, desc_V_k, idesc_dP, acc);
            }
            
            uint32_t mbar_a = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_a));
        }
        
        mbarrier_wait_fn(mbar_mma, mma_phase);
        mma_phase++;
        
        tcgen05_fence_after_fn();
        
        compute_P_and_dS(tmem_S, tmem_dP, smem_P, smem_dS, L_head, D_head, i_start, j_start, S, scale);
        
        __syncthreads();
        fence_proxy_async_fn();
        
        tcgen05_fence_before_fn();
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_P_k = make_smem_desc_sm100_fn(smem_P + k * 16 * 64, 16384, 1024);
                uint64_t desc_dO_k = make_smem_desc_sm100_fn(smem_dO + k * 16, 1, 1024);
                uint64_t desc_dS_k = make_smem_desc_sm100_fn(smem_dS + k * 16 * 64, 16384, 1024);
                uint64_t desc_Q_k = make_smem_desc_sm100_fn(smem_Q + k * 16, 1, 1024);
                
                uint32_t acc = (k == 0) ? accum_dV : 1;
                umma_f16_cg1_fn(tmem_dV, desc_P_k, desc_dO_k, idesc_dV, acc);
                umma_f16_cg1_fn(tmem_dK, desc_dS_k, desc_Q_k, idesc_dK, (k == 0) ? accum_dK : 1);
            }
            
            for (int k = 0; k < 4; ++k) {
                uint64_t desc_dS_k2 = make_smem_desc_sm100_fn(smem_dS + k * 16, 1, 1024);
                uint64_t desc_K_k2 = make_smem_desc_sm100_fn(smem_K + k * 16 * 128, 8192, 1024);
                
                uint32_t acc = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_dQ, desc_dS_k2, desc_K_k2, idesc_dQ, acc);
            }
            
            uint32_t mbar_a = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_a));
        }
        
        mbarrier_wait_fn(mbar_mma, mma_phase);
        mma_phase++;
        
        tcgen05_fence_after_fn();
        
        atomicAdd_dQ(tmem_dQ, dQ_head, i_start, S);
        __syncthreads(); 
    }
    
    tmem_to_smem_64x128(tmem_dK, smem_K, 128); 
    tmem_to_smem_64x128(tmem_dV, smem_V, 128); 
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tma_store_3d_fn(&tma_dK, smem_K, 0, 0, c2_offset + j_start);
        tma_store_3d_fn(&tma_dV, smem_V, 0, 0, c2_offset + j_start);
        tma_store_commit_fn();
    }
    tma_store_wait_fn<0>();
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_addr, 512);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t total_S, uint32_t smem_rows, 
                                     CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {64, 2, total_S};
    cuuint64_t globalStrides[2] = {64 * 2, 128 * 2};
    cuuint32_t boxDim[3] = {64, 2, smem_rows};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
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
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * 128 * sizeof(__nv_bfloat16), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO, tma_dK, tma_dV;
    create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), B * H * S, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), B * H * S, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), B * H * S, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_dO, dO.data_ptr(), B * H * S, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_dK, dK.data_ptr(), B * H * S, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_dV, dV.data_ptr(), B * H * S, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    
    int num_j_blocks = (S + 63) / 64;
    dim3 grid(num_j_blocks, H, B);
    dim3 block(128); 
    
    int smem_size = 131072 + 64; 
    CUDA_CHECK(cudaFuncSetAttribute(FABackwardKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    FABackwardKernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, tma_dK, tma_dV,
        static_cast<const float*>(L.data_ptr()),
        d_D,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S, scale
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_D, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda