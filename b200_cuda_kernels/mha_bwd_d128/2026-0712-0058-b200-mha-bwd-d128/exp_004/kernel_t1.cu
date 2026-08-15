#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

// --------------------------------------------------------------------------------------

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

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, %4;"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ uint64_t make_smem_desc_128B(void* smem_ptr, bool major, uint32_t dim) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    
    uint32_t SBO = 128 * 8; // 1024
    if (major) {
        d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    } else {
        d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
        uint32_t LBO = (dim / 8) * SBO;
        d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    }
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

template<int M_major, int N_major>
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = make_instr_desc_fn(M, N);
    d |= ((uint32_t)M_major << 15);
    d |= ((uint32_t)N_major << 16);
    return d;
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void write_swizzled_128B(__nv_bfloat16* smem, uint32_t row, uint32_t col, __nv_bfloat16 val) {
    uint32_t x = col / 8;
    uint32_t rem = col % 8;
    uint32_t swizzled_x = (row % 8) ^ x;
    smem[row * 128 + swizzled_x * 8 + rem] = val;
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled_128B(const __nv_bfloat16* smem, uint32_t row, uint32_t col) {
    uint32_t x = col / 8;
    uint32_t rem = col % 8;
    uint32_t swizzled_x = (row % 8) ^ x;
    return smem[row * 128 + swizzled_x * 8 + rem];
}

// --------------------------------------------------------------------------------------

__global__ void flash_attn_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const float* L, int32_t S, float attn_scale)
{
    extern __shared__ __align__(128) uint8_t smem_pool[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_pool + 32768);
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_pool + 65536);
    __nv_bfloat16* smem_dO = (__nv_bfloat16*)(smem_pool + 98304);
    __nv_bfloat16* smem_O = (__nv_bfloat16*)(smem_pool + 131072);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 163840);
    __nv_bfloat16* smem_dS = (__nv_bfloat16*)(smem_pool + 196608);
    float* smem_D = (float*)(smem_pool + 229376);
    uint64_t* mbar = (uint64_t*)(smem_pool + 230400);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t c_ST[128], c_PT[128], c_dPT[128], c_dST[128], c_dV[128], c_dK[128], c_dQT[128];
    if (threadIdx.x == 0) {
        tmem_alloc_fn(c_ST, 128);
        tmem_alloc_fn(c_PT, 128);
        tmem_alloc_fn(c_dPT, 128);
        tmem_alloc_fn(c_dST, 128);
        tmem_alloc_fn(c_dV, 128);
        tmem_alloc_fn(c_dK, 128);
        tmem_alloc_fn(c_dQT, 128);
    }
    __syncthreads();

    uint32_t kv_tile = blockIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t kv_off = kv_tile * 128;
    uint32_t row_h_offset = head_idx * S;
    
    uint32_t k_chunks[8] = {0, 16, 32, 48, 64, 80, 96, 112};

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 2 * 32768);
        tma_load_2d_fn(&tma_K, mbar, smem_K, 0, row_h_offset + kv_off);
        tma_load_2d_fn(&tma_V, mbar, smem_V, 0, row_h_offset + kv_off);
    }
    
    mbarrier_wait_fn(mbar, 0);
    
    uint32_t num_Q_tiles = (S + 127) / 128;
    uint32_t tid = threadIdx.x;
    
    for (uint32_t q_tile = 0; q_tile < num_Q_tiles; ++q_tile) {
        uint32_t q_off = q_tile * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 3 * 32768);
            tma_load_2d_fn(&tma_Q, mbar, smem_Q, 0, row_h_offset + q_off);
            tma_load_2d_fn(&tma_O, mbar, smem_O, 0, row_h_offset + q_off);
            tma_load_2d_fn(&tma_dO, mbar, smem_dO, 0, row_h_offset + q_off);
        }
        
        if (threadIdx.x < 128) {
            uint32_t q_idx = q_off + threadIdx.x;
            float sum = 0;
            if (q_idx < S) {
                for (uint32_t i = 0; i < 128; i++) {
                    uint32_t x = i / 8;
                    uint32_t rem = i % 8;
                    uint32_t swizzled_x = (threadIdx.x % 8) ^ x;
                    float o_val = __bfloat162float(smem_O[threadIdx.x * 128 + swizzled_x * 8 + rem]);
                    float do_val = __bfloat162float(smem_dO[threadIdx.x * 128 + swizzled_x * 8 + rem]);
                    sum += o_val * do_val;
                }
            }
            smem_D[threadIdx.x] = sum;
        }
        
        int phase_to_wait = (q_tile + 1) % 2;
        mbarrier_wait_fn(mbar, phase_to_wait);
        
        uint32_t idesc_S = make_instr_desc_fn<0, 0>(256, 256);
        uint32_t idesc_dP = make_instr_desc_fn<0, 0>(256, 256);
        
        for (uint32_t i = 0; i < 8; i++) {
            uint64_t desc_K[i] = make_smem_desc_128B((char*)smem_K + i * 32, true, 128);
            uint64_t desc_Q[i] = make_smem_desc_128B((char*)smem_Q + i * 32, true, 128);
            uint32_t c_ST_i = c_ST[0] + i;
            uint32_t accum_S = (i == 0) ? 0 : 1;
            umma_f16_cg2_fn(c_ST_i, desc_K[i], desc_Q[i], idesc_S, accum_S);
            
            uint64_t desc_V[i] = make_smem_desc_128B((char*)smem_V + i * 32, true, 128);
            uint64_t desc_dO[i] = make_smem_desc_128B((char*)smem_dO + i * 32, true, 128);
            uint32_t c_dPT_i = c_dPT[0] + i;
            uint32_t accum_dP = (i == 0) ? 0 : 1;
            umma_f16_cg2_fn(c_dPT_i, desc_V[i], desc_dO[i], idesc_dP, accum_dP);
        }
        
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase_to_wait);
        tcgen05_fence_after_fn();
        
        for (uint32_t c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(c_ST[0] + c));
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            uint32_t global_q_idx0 = q_off + tid;
            uint32_t global_q_idx1 = q_off + tid + 1;
            uint32_t global_kv_idx0 = kv_off + tid;
            uint32_t global_kv_idx1 = kv_off + tid + 1;
            
            float lse0 = L[row_h_offset + global_q_idx0] * sqrt(128.0f);
            float lse1 = L[row_h_offset + global_q_idx1] * sqrt(128.0f);
            
            float s0 = f0 * attn_scale;
            float p0 = fast_exp2f_fn((s0 - lse0) * 1.4426950408889634f);
            if (global_q_idx0 >= S || global_kv_idx0 >= S) p0 = 0;
            
            float s1 = f1 * attn_scale;
            float p1 = fast_exp2f_fn((s1 - lse1) * 1.4426950408889634f);
            if (global_q_idx1 >= S || global_kv_idx1 >= S) p1 = 0;
            
            float s2 = f2 * attn_scale;
            float p2 = fast_exp2f_fn((s2 - lse0) * 1.4426950408889634f);
            if (global_q_idx0 >= S || global_kv_idx2 >= S) p2 = 0;
            
            float s3 = f3 * attn_scale;
            float p3 = fast_exp2f_fn((s3 - lse1) * 1.4426950408889634f);
            if (global_q_idx1 >= S || global_kv_idx3 >= S) p3 = 0;
            
            write_swizzled_128B(smem_P, tid, c, p0);
            write_swizzled_128B(smem_P, tid + 1, c, p1);
            write_swizzled_128B(smem_P, tid, c + 1, p2);
            write_swizzled_128B(smem_P, tid + 1, c + 1, p3);
        }
        
        for (uint32_t c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(c_dPT[0] + c));
            float dp0 = __uint_as_float(r0);
            float dp1 = __uint_as_float(r1);
            float dp2 = __uint_as_float(r2);
            float dp3 = __uint_as_float(r3);
            
            float p0 = __bfloat162float(read_swizzled_128B(smem_P, tid, c));
            float p1 = __bfloat162float(read_swizzled_128B(smem_P, tid + 1, c));
            float p2 = __bfloat162float(read_swizzled_128B(smem_P, tid, c + 1));
            float p3 = __bfloat162float(read_swizzled_128B(smem_P, tid + 1, c + 1));
            
            float ds0 = p0 * (dp0 - smem_D[tid]);
            float ds1 = p1 * (dp1 - smem_D[tid + 1]);
            float ds2 = p2 * (dp2 - smem_D[tid]);
            float ds3 = p3 * (dp3 - smem_D[tid + 1]);
            
            write_swizzled_128B(smem_dS, tid, c, ds0);
            write_swizzled_128B(smem_dS, tid + 1, c, ds1);
            write_swizzled_128B(smem_dS, tid, c + 1, ds2);
            write_swizzled_128B(smem_dS, tid + 1, c + 1, ds3);
        }
        
        __syncthreads();
        fence_async_shared_fn();
        
        uint32_t idesc_dV = make_instr_desc_fn<0, 1>(256, 256);
        uint32_t idesc_dK = make_instr_desc_fn<0, 1>(256, 256);
        
        for (uint32_t i = 0; i < 8; i++) {
            uint64_t desc_P_K[i] = make_smem_desc_128B((char*)smem_P + i * 32, true, 128);
            uint64_t desc_dO_MN[i] = make_smem_desc_128B((char*)smem_dO + i * 32 * 128, false, 128);
            uint32_t c_dV_i = c_dV[0] + i;
            umma_f16_cg2_fn(c_dV_i, desc_P_K[i], desc_dO_MN[i], idesc_dV, 1);
            
            uint64_t desc_dS_K[i] = make_smem_desc_128B((char*)smem_dS + i * 32, true, 128);
            uint64_t desc_Q_MN[i] = make_smem_desc_128B((char*)smem_Q + i * 32 * 128, false, 128);
            uint32_t c_dK_i = c_dK[0] + i;
            umma_f16_cg2_fn(c_dK_i, desc_dS_K[i], desc_Q_MN[i], idesc_dK, 1);
        }
        
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase_to_wait);
        tcgen05_fence_after_fn();
        
        __syncthreads(); 
        
        for (uint32_t i = 0; i < 128; i += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(c_dST + i));
            float ds0 = __uint_as_float(r0);
            float ds1 = __uint_as_float(r1);
            float ds2 = __uint_as_float(r2);
            float ds3 = __uint_as_float(r3);
            
            if (global_q_idx0 < S && global_kv_idx0 < S) {
                __nv_bfloat162 val0 = {__float2bfloat16(ds0), __float2bfloat16(ds1)};
                uint32_t base_idx = head_idx * S * 128 + (q_off + tid) * 128;
                atomicAdd((__nv_bfloat2*)(dQ + base_idx + i), val0);
            }
            if (global_q_idx1 < S && global_kv_idx1 < S) {
                __nv_bfloat162 val1 = {__float2bfloat16(ds2), __float2bfloat16(ds3)};
                uint32_t base_idx = head_idx * S * 128 + (q_off + tid + 1) * 128;
                atomicAdd((__nv_bfloat2*)(dQ + base_idx + i), val1);
            }
        }
    } 
    
    __syncthreads();
    
    for (uint32_t i = 0; i < 128; i += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(c_dV + i));
        if (kv_off + tid < S) {
            uint32_t base_idx_v = head_idx * S * 128 + (kv_off + tid) * 128;
            *reinterpret_cast<uint32_t*>(dV + base_idx_v + i) = pack_bf16_fn(r0, r1);
            *reinterpret_cast<uint32_t*>(dV + base_idx_v + i + 2) = pack_bf16_fn(r2, r3);
        }
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(c_dK + i));
        if (kv_off + tid < S) {
            uint32_t base_idx_k = head_idx * S * 128 + (kv_off + tid) * 128;
            *reinterpret_cast<uint32_t*>(dK + base_idx_k + i) = pack_bf16_fn(r0, r1);
            *reinterpret_cast<uint32_t*>(dK + base_idx_k + i + 2) = pack_bf16_fn(r2, r3);
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(c_ST[0], 128);
        tmem_dealloc_fn(c_PT[0], 128);
        tmem_dealloc_fn(c_dPT[0], 128);
        tmem_dealloc_fn(c_dST[0], 128);
        tmem_dealloc_fn(c_dV[0], 128);
        tmem_dealloc_fn(c_dK[0], 128);
        tmem_dealloc_fn(c_dQT[0], 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    float attn_scale = 1.0f / sqrt(128.0f);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 230408 + 256;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, flash_attn_bwd_kernel, tma_Q, tma_K, tma_V, tma_O, tma_dO, dQ.data_ptr(), dK.data_ptr(), dV.data_ptr(), L.data_ptr(), S, attn_scale));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);