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

namespace mha_with_lse_d128 {

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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tm_load_4x(uint32_t addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], [%2], %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void cp_128x128b(uint32_t tmem_addr, uint64_t smem_desc) {
    asm volatile("tcgen05.cp.cta_group::2.128x128b [%0], %1;" :: "r"(tmem_addr), "l"(smem_desc) : "memory");
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

template<uint32_t M, uint32_t N, uint32_t A_major, uint32_t B_major>
__device__ __forceinline__ uint32_t make_idesc() {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (A_major << 15);   
    d |= (B_major << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__global__ __launch_bounds__(128, 1) void mha_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V, const __grid_constant__ CUtensorMap tma_Q, const __grid_constant__ CUtensorMap tma_K, const __grid_constant__ CUtensorMap tma_V, const __grid_constant__ CUtensorMap tma_O, float* __restrict__ LSE, int S) {
    int bh = blockIdx.x;
    int num_q_tiles = (S + 127) / 128;
    int q_tile = blockIdx.y * 2 + (cluster_rank_fn() & 1);
    int tid = threadIdx.x;
    int q_start = q_tile * 128;
    
    if (q_start >= S) return;
    
    int b_idx = bh / 48;
    int h_idx = bh % 48;
    
    extern __shared__ __align__(1024) char smem_raw[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem_raw + 0);
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_raw + 32768);
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_raw + 65536);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_raw + 98304); 
    
    uint64_t* bar_Q = (uint64_t*)(smem_raw + 131072);
    uint64_t* bar_K = (uint64_t*)(smem_raw + 131080);
    uint64_t* bar_V = (uint64_t*)(smem_raw + 131088);

    __shared__ uint32_t tmem_Q[1];
    __shared__ uint32_t tmem_K[1];
    __shared__ uint32_t tmem_V[1];
    __shared__ uint32_t tmem_S[1];
    
    if (tid == 0) {
        tmem_alloc_fn(tmem_Q, 128);
        tmem_alloc_fn(tmem_K, 128);
        tmem_alloc_fn(tmem_V, 128);
        tmem_alloc_fn(tmem_S, 128);
        
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    setmaxnreg_inc_sync_fn<256>();
    
    uint64_t desc_Q_0 = make_smem_desc_sm100_fn(smem_Q, 1, 1024);
    uint64_t desc_Q_1 = make_smem_desc_sm100_fn(smem_Q + 8192, 1, 1024);
    uint64_t desc_K_0 = make_smem_desc_sm100_fn(smem_K, 1, 1024);
    uint64_t desc_K_1 = make_smem_desc_sm100_fn(smem_K + 8192, 1, 1024);
    uint64_t desc_V_0 = make_smem_desc_sm100_fn(smem_V, 16384, 1024);
    uint64_t desc_V_1 = make_smem_desc_sm100_fn(smem_V + 8192, 16384, 1024);
    uint64_t desc_P_0 = make_smem_desc_sm100_fn(smem_P, 1, 1024);
    uint64_t desc_P_1 = make_smem_desc_sm100_fn(smem_P + 8192, 1, 1024);
    
    uint32_t idesc_QK = make_idesc<128, 128, 0, 0>();
    uint32_t idesc_PV = make_idesc<128, 128, 0, 1>();
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
        tma_load_4d_fn(&tma_Q, bar_Q, smem_Q, 0, q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, bar_Q, smem_Q + 8192, 64, q_start, h_idx, b_idx);
    }
    
    cp_128x128b(*tmem_Q, desc_Q_0);
    cp_128x128b(*tmem_Q + 2048, desc_Q_1);
    
    mbarrier_wait_fn(bar_Q, 0);
    
    float O_acc[128] __attribute__((aligned(16))) = {0};
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    int phase_k = 0;
    int phase_v = 0;
    int phase_q = 1;
    int num_k_tiles = (S + 127) / 128;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int k_start = k_tile * 128;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
            tma_load_4d_fn(&tma_K, bar_K, smem_K, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, bar_K, smem_K + 8192, 64, k_start, h_idx, b_idx);
            
            mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
            tma_load_4d_fn(&tma_V, bar_V, smem_V, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, bar_V, smem_V + 8192, 64, k_start, h_idx, b_idx);
        }
        
        mbarrier_wait_fn(bar_K, phase_k);
        mbarrier_wait_fn(bar_V, phase_v);
        fence_proxy_async_fn();
        __syncthreads();
        
        cp_128x128b(*tmem_K, desc_K_0);
        cp_128x128b(*tmem_K + 2048, desc_K_1);
        cp_128x128b(*tmem_V, desc_V_0);
        cp_128x128b(*tmem_V + 2048, desc_V_1);
        
        if (tid == 0) {
            umma_f16_cg2_fn(*tmem_S, *tmem_Q, *tmem_K, idesc_QK, 0);
            for(int i = 1; i < 4; ++i) {
                umma_f16_cg2_fn(*tmem_S, *tmem_Q + i * 2, *tmem_K + i * 2, idesc_QK, 1);
            }
            for(int i = 0; i < 4; ++i) {
                umma_f16_cg2_fn(*tmem_S, *tmem_Q + 2048 + i * 2, *tmem_K + 2048 + i * 2, idesc_QK, 1);
            }
        }
        umma_commit_2sm_fn(bar_Q);
        mbarrier_wait_fn(bar_Q, phase_q);
        phase_q ^= 1;
        
        float S_acc[128] __attribute__((aligned(16))) = {0};
        for(int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tm_load_4x(*tmem_S + col, &r0, &r1, &r2, &r3);
            S_acc[col] = __uint_as_float(r0);
            S_acc[col+1] = __uint_as_float(r1);
            S_acc[col+2] = __uint_as_float(r2);
            S_acc[col+3] = __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        
        float m_curr = -INFINITY;
        for(int i = 0; i < 128; ++i) {
            if (k_start + i >= S) {
                S_acc[i] = -INFINITY;
            } else {
                S_acc[i] *= 0.08838834764f; // 1/sqrt(128)
            }
            m_curr = fmaxf(m_curr, S_acc[i]);
        }
        
        float m_new = fmaxf(m_prev, m_curr);
        float l_new = 0.0f;
        for(int i = 0; i < 128; ++i) {
            float p = 0.0f;
            if (k_start + i < S) {
                p = expf((S_acc[i] - m_new) * 1.44269504089f);
            }
            l_new += p;
            S_acc[i] = p;
        }
        l_new += l_prev * expf((m_prev - m_new) * 1.44269504089f);
        float scale_o = expf((m_prev - m_new) * 1.44269504089f);
        l_prev = l_new;
        m_prev = m_new;
        
        __syncthreads(); 
        
        for(int i = 0; i < 128; i += 2) {
            int cx = i / 8;
            int scx = cx ^ (tid % 8);
            __nv_bfloat16 bf0 = __float2bfloat16(S_acc[i]);
            __nv_bfloat16 bf1 = __float2bfloat16(S_acc[i+1]);
            uint32_t val = ((uint32_t)*(uint16_t*)&bf1 << 16) | *(uint16_t*)&bf0;
            uint32_t* smem_P_u32 = (uint32_t*)smem_P;
            smem_P_u32[tid * 64 + scx * 4 + (i % 8)/2] = val;
            
            __nv_bfloat16 bf2 = __float2bfloat16(S_acc[i + 64]);
            __nv_bfloat16 bf3 = __float2bfloat16(S_acc[i + 65]);
            uint32_t val2 = ((uint32_t)*(uint16_t*)&bf3 << 16) | *(uint16_t*)&bf2;
            uint32_t* smem_P_u32_1 = (uint32_t*)(smem_P + 8192);
            smem_P_u32_1[tid * 64 + scx * 4 + (i % 8)/2] = val2;
        }
        
        __syncthreads(); 
        
        cp_128x128b(*tmem_K, desc_P_0);
        cp_128x128b(*tmem_K + 2048, desc_P_1);
        
        if (scale_o > 0.0f) {
            for(int i = 0; i < 128; ++i) {
                O_acc[i] *= scale_o;
            }
        }
        
        if (tid == 0) {
            umma_f16_cg2_fn(*tmem_S, *tmem_K, *tmem_V, idesc_PV, 0);
            for(int i = 1; i < 4; ++i) {
                umma_f16_cg2_fn(*tmem_S, *tmem_K + i * 2, *tmem_V + i * 8, idesc_PV, 1);
            }
            
            for(int i = 0; i < 4; ++i) {
                umma_f16_cg2_fn(*tmem_S + 64, *tmem_K + i * 2, *tmem_V + 2048 + i * 8, idesc_PV, 1);
            }
            
            for(int i = 0; i < 4; ++i) {
                umma_f16_cg2_fn(*tmem_S, *tmem_K + 2048 + i * 2, *tmem_V + i * 8, idesc_PV, 1);
            }
            
            for(int i = 0; i < 4; ++i) {
                umma_f16_cg2_fn(*tmem_S + 64, *tmem_K + 2048 + i * 2, *tmem_V + 2048 + i * 8, idesc_PV, 1);
            }
        }
        umma_commit_2sm_fn(bar_K);
        mbarrier_wait_fn(bar_K, phase_k ^ 1);
        
        for(int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tm_load_4x(*tmem_S + col, &r0, &r1, &r2, &r3);
            O_acc[col] += __uint_as_float(r0);
            O_acc[col+1] += __uint_as_float(r1);
            O_acc[col+2] += __uint_as_float(r2);
            O_acc[col+3] += __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        
        __syncthreads();
        phase_k ^= 1;
        phase_v ^= 1;
    }
    
    __syncthreads();
    
    __nv_bfloat16* smem_O = smem_V; 
    for(int i = 0; i < 128; i += 2) {
        int cx = i / 8;
        int scx = cx ^ (tid % 8);
        __nv_bfloat16 bf0 = __float2bfloat16(O_acc[i] / l_prev);
        __nv_bfloat16 bf1 = __float2bfloat16(O_acc[i+1] / l_prev);
        uint32_t val = ((uint32_t)*(uint16_t*)&bf1 << 16) | *(uint16_t*)&bf0;
        uint32_t* smem_O_u32 = (uint32_t*)smem_O;
        smem_O_u32[tid * 64 + scx * 4 + (i % 8)/2] = val;
        
        __nv_bfloat16 bf2 = __float2bfloat16(O_acc[i + 64] / l_prev);
        __nv_bfloat16 bf3 = __float2bfloat16(O_acc[i + 65] / l_prev);
        uint32_t val2 = ((uint32_t)*(uint16_t*)&bf3 << 16) | *(uint16_t*)&bf2;
        uint32_t* smem_O_u32_1 = (uint32_t*)(smem_O + 8192);
        smem_O_u32_1[tid * 64 + scx * 4 + (i % 8)/2] = val2;
    }
    
    fence_async_shared_fn();
    __syncthreads();
    
    if (tid == 0) {
        tma_store_4d_fn(&tma_O, smem_O, 0, q_start, h_idx, b_idx);
        tma_store_4d_fn(&tma_O, smem_O + 8192, 64, q_start, h_idx, b_idx);
        tma_store_commit_fn();
    }
    
    if (q_start + tid < S) {
        LSE[bh * S + q_start + tid] = m_prev + logf(l_prev);
    }
    
    tma_store_wait_fn<0>();
    __syncthreads();
    
    if (tid == 0) {
        tmem_dealloc_fn(*tmem_Q, 128);
        tmem_dealloc_fn(*tmem_K, 128);
        tmem_dealloc_fn(*tmem_V, 128);
        tmem_dealloc_fn(*tmem_S, 128);
    }
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_d, uint64_t gmem_s, uint64_t gmem_h, uint64_t gmem_b, uint32_t smem_d, uint32_t smem_s, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {gmem_d, gmem_s, gmem_h, gmem_b};
    cuuint64_t globalStrides[3] = {gmem_d * 2, gmem_d * gmem_s * 2, gmem_d * gmem_s * gmem_h * 2};
    cuuint32_t boxDim[4] = {smem_d, smem_s, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    create_tma_4d_descriptor_2B(&tma_Q, (void*)Q_data, D, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor_2B(&tma_K, (void*)K_data, D, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor_2B(&tma_V, (void*)V_data, D, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor_2B(&tma_O, (void*)O_data, D, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    
    int num_q_tiles = (S + 127) / 128;
    int cluster_size = 2;
    dim3 grid(B * H, (num_q_tiles + cluster_size - 1) / cluster_size);
    dim3 block(128);
    
    int smem_size = 131104;
    CUDA_CHECK(cudaFuncSetAttribute(
        mha_with_lse_d128::mha_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
        
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_with_lse_d128::mha_kernel, Q_data, K_data, V_data, tma_Q, tma_K, tma_V, tma_O, LSE_data, S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace mha_with_lse_d128