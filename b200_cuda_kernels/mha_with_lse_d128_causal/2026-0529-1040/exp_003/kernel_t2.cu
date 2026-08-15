#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                         \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n",         \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                 :: "r"(r0),"r"(r1),"r"(r2),"r"(r3), "r"(col) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_wait_st_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major_none(void* smem_ptr, uint32_t M) {
    uint32_t sbo = 128;
    uint32_t lbo = (M / 8) * sbo; 
    return make_smem_desc_sm100_fn(smem_ptr, lbo, sbo);
}

__device__ __forceinline__ uint64_t make_smem_desc_n_major_none(void* smem_ptr, uint32_t K) {
    uint32_t lbo = 128;
    uint32_t sbo = (K / 8) * lbo; 
    return make_smem_desc_sm100_fn(smem_ptr, lbo, sbo);
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool a_n_major, bool b_n_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((a_n_major ? 1u : 0u) << 15);
    d |= ((b_n_major ? 1u : 0u) << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__launch_bounds__(128, 1)
__global__ void fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE,
    int S) {
    
    extern __shared__ __align__(128) uint8_t smem_base[];
    __nv_bfloat16* smem_Q  = (__nv_bfloat16*)smem_base;
    __nv_bfloat16* smem_K0 = smem_Q + 16384;
    __nv_bfloat16* smem_V0 = smem_K0 + 16384;
    __nv_bfloat16* smem_K1 = smem_V0 + 16384;
    __nv_bfloat16* smem_V1 = smem_K1 + 16384;
    __nv_bfloat16* smem_P  = smem_V1 + 16384;
    
    uint64_t* mbar_tma_Q  = (uint64_t*)(smem_base + 196608);
    uint64_t* mbar_tma_K0 = (uint64_t*)(smem_base + 196608 + 8);
    uint64_t* mbar_tma_V0 = (uint64_t*)(smem_base + 196608 + 16);
    uint64_t* mbar_tma_K1 = (uint64_t*)(smem_base + 196608 + 24);
    uint64_t* mbar_tma_V1 = (uint64_t*)(smem_base + 196608 + 32);
    uint64_t* mbar_umma   = (uint64_t*)(smem_base + 196608 + 40);
    uint32_t* tmem_base_ptr = (uint32_t*)(smem_base + 196608 + 48);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_tma_Q, 1);
        init_smem_barrier_fn(mbar_tma_K0, 1);
        init_smem_barrier_fn(mbar_tma_V0, 1);
        init_smem_barrier_fn(mbar_tma_K1, 1);
        init_smem_barrier_fn(mbar_tma_V1, 1);
        init_smem_barrier_fn(mbar_umma, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(tmem_base_ptr, 256);
    }
    __syncthreads();
    
    uint32_t tmem_base = *tmem_base_ptr;
    uint32_t tmem_S = tmem_base;
    uint32_t tmem_O = tmem_base + 128;

    int m_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_tma_Q, 128 * 128 * 2);
        tma_load_4d_fn(&tma_Q, mbar_tma_Q, smem_Q, 0, m_block * 128, h, b);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_tma_K0, 128 * 128 * 2);
        tma_load_4d_fn(&tma_K, mbar_tma_K0, smem_K0, 0, 0, h, b);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_tma_V0, 128 * 128 * 2);
        tma_load_4d_fn(&tma_V, mbar_tma_V0, smem_V0, 0, 0, h, b);
    }

    mbarrier_wait_fn(mbar_tma_Q, 0);

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    int phase_K[2] = {0, 0};
    int phase_umma = 0;
    
    uint64_t* mbar_K_arr[2] = {mbar_tma_K0, mbar_tma_K1};
    uint64_t* mbar_V_arr[2] = {mbar_tma_V0, mbar_tma_V1};
    __nv_bfloat16* smem_K_arr[2] = {smem_K0, smem_K1};
    __nv_bfloat16* smem_V_arr[2] = {smem_V0, smem_V1};

    uint32_t idesc_qk = make_instr_desc_fn(128, 128, false, false);
    uint32_t idesc_pv = make_instr_desc_fn(128, 128, false, true);

    for (int n_block = 0; n_block <= m_block; ++n_block) {
        int buf_idx = n_block % 2;
        int next_buf_idx = (n_block + 1) % 2;

        if (n_block + 1 <= m_block) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_K_arr[next_buf_idx], 128 * 128 * 2);
                tma_load_4d_fn(&tma_K, mbar_K_arr[next_buf_idx], smem_K_arr[next_buf_idx], 0, (n_block + 1) * 128, h, b);
                
                mbarrier_arrive_and_expect_tx_fn(mbar_V_arr[next_buf_idx], 128 * 128 * 2);
                tma_load_4d_fn(&tma_V, mbar_V_arr[next_buf_idx], smem_V_arr[next_buf_idx], 0, (n_block + 1) * 128, h, b);
            }
        }

        mbarrier_wait_fn(mbar_K_arr[buf_idx], phase_K[buf_idx]);
        mbarrier_wait_fn(mbar_V_arr[buf_idx], phase_K[buf_idx]);

        // Sync to guarantee all threads finish their prior reads,
        // and threads observe the buffer loads completed
        __syncthreads();

        __nv_bfloat16* cur_K = smem_K_arr[buf_idx];
        __nv_bfloat16* cur_V = smem_V_arr[buf_idx];

        for (int k_chunk = 0; k_chunk < 8; ++k_chunk) {
            uint64_t desc_a = make_smem_desc_k_major_none(smem_Q + k_chunk * 16, 128);
            uint64_t desc_b = make_smem_desc_k_major_none(cur_K + k_chunk * 16, 128);
            uint32_t accum = (n_block == 0 && k_chunk == 0) ? 0 : 1;
            if (threadIdx.x == 0) {
                umma_f16_cg1_fn(tmem_S, desc_a, desc_b, idesc_qk, accum);
            }
        }
        
        if (threadIdx.x == 0) {
            tcgen05_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;

        float m_curr = -INFINITY;
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0) * 0.0883883f;
            float f1 = __uint_as_float(r1) * 0.0883883f;
            float f2 = __uint_as_float(r2) * 0.0883883f;
            float f3 = __uint_as_float(r3) * 0.0883883f;
            
            uint32_t q_pos = m_block * 128 + threadIdx.x;
            uint32_t k_pos = n_block * 128 + col;
            
            if (k_pos + 0 > q_pos || k_pos + 0 >= S) f0 = -INFINITY;
            if (k_pos + 1 > q_pos || k_pos + 1 >= S) f1 = -INFINITY;
            if (k_pos + 2 > q_pos || k_pos + 2 >= S) f2 = -INFINITY;
            if (k_pos + 3 > q_pos || k_pos + 3 >= S) f3 = -INFINITY;
            
            m_curr = fmaxf(m_curr, f0);
            m_curr = fmaxf(m_curr, f1);
            m_curr = fmaxf(m_curr, f2);
            m_curr = fmaxf(m_curr, f3);
        }

        float m_new = fmaxf(m_prev, m_curr);
        float scale = 0.0f;
        if (m_new != -INFINITY) {
            scale = fast_exp2f_fn((m_prev - m_new) * 1.44269504f);
        }
        float l_new = l_prev * scale;

        // Convergent scale across the warp prevents divergence on tcgen05 intrinsics
        if (n_block > 0) {
            for (uint32_t col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_O + col, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float f0 = __uint_as_float(r0) * scale;
                float f1 = __uint_as_float(r1) * scale;
                float f2 = __uint_as_float(r2) * scale;
                float f3 = __uint_as_float(r3) * scale;
                
                tmem_store_4x_fn(tmem_O + col, __float_as_uint(f0), __float_as_uint(f1), __float_as_uint(f2), __float_as_uint(f3));
            }
            tmem_wait_st_fn();
        }

        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0) * 0.0883883f;
            float f1 = __uint_as_float(r1) * 0.0883883f;
            float f2 = __uint_as_float(r2) * 0.0883883f;
            float f3 = __uint_as_float(r3) * 0.0883883f;
            
            uint32_t q_pos = m_block * 128 + threadIdx.x;
            uint32_t k_pos = n_block * 128 + col;
            
            if (k_pos + 0 > q_pos || k_pos + 0 >= S) f0 = -INFINITY;
            if (k_pos + 1 > q_pos || k_pos + 1 >= S) f1 = -INFINITY;
            if (k_pos + 2 > q_pos || k_pos + 2 >= S) f2 = -INFINITY;
            if (k_pos + 3 > q_pos || k_pos + 3 >= S) f3 = -INFINITY;
            
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            if (m_new != -INFINITY) {
                p0 = (f0 == -INFINITY) ? 0.0f : fast_exp2f_fn((f0 - m_new) * 1.44269504f);
                p1 = (f1 == -INFINITY) ? 0.0f : fast_exp2f_fn((f1 - m_new) * 1.44269504f);
                p2 = (f2 == -INFINITY) ? 0.0f : fast_exp2f_fn((f2 - m_new) * 1.44269504f);
                p3 = (f3 == -INFINITY) ? 0.0f : fast_exp2f_fn((f3 - m_new) * 1.44269504f);
            }
            
            l_new += p0 + p1 + p2 + p3;
            
            smem_P[threadIdx.x * 128 + col + 0] = __float2bfloat16(p0);
            smem_P[threadIdx.x * 128 + col + 1] = __float2bfloat16(p1);
            smem_P[threadIdx.x * 128 + col + 2] = __float2bfloat16(p2);
            smem_P[threadIdx.x * 128 + col + 3] = __float2bfloat16(p3);
        }

        // Wait for all threads to deposit softmax results before the UMMA fires
        __syncthreads();
        fence_proxy_async_fn();
        
        for (int k_chunk = 0; k_chunk < 8; ++k_chunk) {
            uint64_t desc_a = make_smem_desc_k_major_none(smem_P + k_chunk * 16, 128);
            uint64_t desc_b = make_smem_desc_n_major_none(cur_V + k_chunk * 2048, 128); 
            uint32_t accum = (n_block == 0 && k_chunk == 0) ? 0 : 1;
            if (threadIdx.x == 0) {
                umma_f16_cg1_fn(tmem_O, desc_a, desc_b, idesc_pv, accum);
            }
        }
        
        if (threadIdx.x == 0) {
            tcgen05_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;

        m_prev = m_new;
        l_prev = l_new;
        phase_K[buf_idx] ^= 1;
    }

    float out_scale = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O + col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        float f0 = __uint_as_float(r0) * out_scale;
        float f1 = __uint_as_float(r1) * out_scale;
        float f2 = __uint_as_float(r2) * out_scale;
        float f3 = __uint_as_float(r3) * out_scale;
        
        smem_P[threadIdx.x * 128 + col + 0] = __float2bfloat16(f0);
        smem_P[threadIdx.x * 128 + col + 1] = __float2bfloat16(f1);
        smem_P[threadIdx.x * 128 + col + 2] = __float2bfloat16(f2);
        smem_P[threadIdx.x * 128 + col + 3] = __float2bfloat16(f3);
    }

    __syncthreads();
    tma_store_fence_fn();
    if (threadIdx.x == 0) {
        tma_store_4d_fn(&tma_O, smem_P, 0, m_block * 128, h, b);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    if (m_block * 128 + threadIdx.x < S) {
        LSE[(b * gridDim.y + h) * S + m_block * 128 + threadIdx.x] = m_prev + logf(l_prev);
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

namespace tvm_ffi_example_cuda {

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
    uint32_t smem_dim0, uint32_t smem_dim1) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {smem_dim0, smem_dim1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 128, 128));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 128, 128));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 128, 128));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_O, O.data_ptr(), D, S, H, B, 128, 128));
    
    int64_t num_m_blocks = (S + 127) / 128;
    dim3 grid(num_m_blocks, H, B);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_size = 196608 + 256; 
    CUDA_CHECK(cudaFuncSetAttribute((const void*)fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    fwd_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_O, static_cast<float*>(LSE.data_ptr()), S);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda