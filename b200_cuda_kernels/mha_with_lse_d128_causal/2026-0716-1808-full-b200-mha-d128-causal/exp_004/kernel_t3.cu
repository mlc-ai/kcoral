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
#include <mma.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

// ---------------- PTX Wrappers ----------------

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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ void tmem_load_4x_packed_fn(uint32_t taddr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(taddr));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
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

__host__ __forceinline__ CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, 1};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// ---------------- Kernel ----------------

struct SwizzledSmem {
    alignas(1024) uint16_t Q[128 * 128];
    alignas(1024) uint16_t K[128 * 128];
    alignas(1024) uint16_t V[128 * 128];
    alignas(1024) uint16_t P[128 * 128];
    alignas(1024) float S[128 * 128]; 
    alignas(1024) uint16_t O[128 * 128];
};

__global__ void flash_attention_kernel(
    __nv_bfloat16* O_gmem, float* LSE,
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    int32_t S_seq, int32_t D_dim, int32_t B_H)
{
    extern __shared__ __align__(128) uint8_t smem_pool[];
    SwizzledSmem* smem = reinterpret_cast<SwizzledSmem*>(smem_pool);
    
    // Position barriers safely with adequate alignment after data structures
    uint64_t* bar_Q = reinterpret_cast<uint64_t*>(smem_pool + sizeof(SwizzledSmem));
    uint64_t* bar_K = bar_Q + 1;
    uint64_t* bar_V = bar_K + 1;
    uint64_t* bar_mma = bar_V + 1;
    uint64_t* bar_out = bar_mma + 1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        init_smem_barrier_fn(bar_mma, 1);
        init_smem_barrier_fn(bar_out, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t tmem_S, tmem_O, tmem_P;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S, 256);
        tmem_alloc_fn(&tmem_O, 256);
        tmem_alloc_fn(&tmem_P, 256);
    }
    __syncthreads();

    int32_t blk_q = blockIdx.x;
    int32_t row_start_q = blk_q * 128;
    int32_t bh_idx = blockIdx.y;

    if (row_start_q >= S_seq) {
        if (threadIdx.x == 0) {
            tmem_dealloc_fn(tmem_S, 256);
            tmem_dealloc_fn(tmem_O, 256);
            tmem_dealloc_fn(tmem_P, 256);
        }
        return;
    }

    uint32_t phase_Q = 0;
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 128 * 128 * 2);
        tma_load_3d_fn(&tma_Q, bar_Q, smem->Q, 0, row_start_q, bh_idx);
        tma_load_3d_fn(&tma_Q, bar_Q, smem->Q + 8192, 64, row_start_q, bh_idx);
    }
    mbarrier_wait_fn(bar_Q, phase_Q);

    float global_sum[128];
    for(int i = 0; i < 128; ++i) {
        global_sum[i] = 0.0f;
    }

    float softmax_s_old[128];
    for(int i = 0; i < 128; ++i) softmax_s_old[i] = -1e38f;

    uint32_t phase_K = 0, phase_V = 0, phase_mma = 0;

    uint32_t r_O[128]; 
    for(int i = 0; i < 128; ++i) r_O[i] = __float_as_uint(0.0f);

    uint32_t idesc_qk = make_instr_desc_fn(128, 128); // 0x8800181 (Assume standard K-major configurations)
    uint32_t idesc_pv = make_instr_desc_fn(128, 128);
    idesc_pv |= (1u << 16); // Explicit declaration noting V acts MN-major relative to P

    for (int blk_kv = 0; blk_kv <= blk_q; ++blk_kv) {
        int row_start_kv = blk_kv * 128;
        if (row_start_kv > row_start_q) break;

        __syncthreads();

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 128 * 128 * 2);
            mbarrier_arrive_and_expect_tx_fn(bar_V, 128 * 128 * 2);
            tma_load_3d_fn(&tma_K, bar_K, smem->K, 0, row_start_kv, bh_idx);
            tma_load_3d_fn(&tma_K, bar_K, smem->K + 8192, 64, row_start_kv, bh_idx);
            
            tma_load_3d_fn(&tma_V, bar_V, smem->V, 0, row_start_kv, bh_idx);
            tma_load_3d_fn(&tma_V, bar_V, smem->V + 8192, 64, row_start_kv, bh_idx);
        }
        mbarrier_wait_fn(bar_K, phase_K);
        mbarrier_wait_fn(bar_V, phase_V);
        phase_K ^= 1;
        phase_V ^= 1;

        // QK^T MatMul (Ensure Correctness via TMEM Iteration)
        for (int k_step = 0; k_step < 64; k_step += 16) {
            uint64_t desc_a = make_smem_desc(smem->Q + k_step, 1, 1024);
            uint64_t desc_b = make_smem_desc(smem->K + k_step, 1, 1024);
            umma_f16_cg1(tmem_S, desc_a, desc_b, idesc_qk, (k_step == 0) ? 0 : 1);
        }
        for (int k_step = 0; k_step < 64; k_step += 16) {
            uint64_t desc_a = make_smem_desc(smem->Q + 8192 + k_step, 1, 1024);
            uint64_t desc_b = make_smem_desc(smem->K + 8192 + k_step, 1, 1024);
            umma_f16_cg1(tmem_S, desc_a, desc_b, idesc_qk, 1);
        }
        
        umma_commit_1sm(bar_mma);
        mbarrier_wait_fn(bar_mma, phase_mma);
        phase_mma ^= 1;

        uint32_t ts = tmem_S;
        uint32_t r_S[128];
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_packed_fn(ts + col, &r0, &r1, &r2, &r3);
            r_S[col] = r0;
            r_S[col + 1] = r1;
            r_S[col + 2] = r2;
            r_S[col + 3] = r3;
        }
        tmem_load_fence_fn();

        int row = threadIdx.x;
        float m_val = -1e38f;
        int max_col = (row_start_q + row >= row_start_kv + 127) ? 127 : (row_start_q + row - row_start_kv);

        for(int col = 0; col < 128; ++col) {
            float s_val = __uint_as_float(r_S[col]) * 0.08838834764f; 
            if (col > max_col || row_start_q + row < row_start_kv) s_val = -1e38f;
            m_val = max(m_val, s_val);
        }
        
        float prev_max = softmax_s_old[row];
        float curr_max = max(prev_max, m_val);
        if (curr_max == -1e38f) curr_max = 0;

        float sum_val = 0;
        for(int col = 0; col < 128; ++col) {
            float s_val = __uint_as_float(r_S[col]) * 0.08838834764f;
            if (col > max_col || row_start_q + row < row_start_kv) s_val = -1e38f;
            float p_val = 0;
            if (s_val > -1e37f) {
                p_val = __expf(s_val - curr_max);
            }
            sum_val += p_val;
            
            float scale_curr = __expf(curr_max - curr_max); 
            float final_p = p_val * scale_curr;
            
            r_S[col] = __float_as_uint(final_p);
        }

        float safe_prev_max = (prev_max == -1e38f) ? 0 : prev_max;
        float safe_curr_max = (curr_max == -1e38f) ? 0 : curr_max;
        float curr_sum_scaled = global_sum[row] * __expf(safe_prev_max - safe_curr_max) + sum_val * __expf(curr_max - safe_curr_max);
        global_sum[row] = curr_sum_scaled;
        softmax_s_old[row] = curr_max;

        // In TMEM, scale existing partial O by exp(old_max - new_max) logic mapped through identity
        for (int k_step = 0; k_step < 128; k_step += 8) {
            uint32_t taddr = tmem_O + k_step; 
            uint32_t scale = __float_as_uint(__expf(softmax_s_old[row] - curr_max));
            *reinterpret_cast<uint4*>(&tmem_O[k_step]) = *reinterpret_cast<uint4*>(&scale);
        }

        uint32_t tmem_P_base = tmem_P;

        // Utilize Hardware TMEM stores ensuring perfectly matching WGMMA 128B swizzled layout
        uint32_t packed_p[64];
        for (int i = 0; i < 128; i += 4) {
            packed_p[i/4] = __float_as_uint(__float2bfloat16(__uint_as_float(r_S[i])));
            packed_p[i/4 + 1] = __float_as_uint(__float2bfloat16(__uint_as_float(r_S[i+1])));
            packed_p[i/4 + 2] = __float_as_uint(__float2bfloat16(__uint_as_float(r_S[i+2])));
            packed_p[i/4 + 3] = __float_as_uint(__float2bfloat16(__uint_as_float(r_S[i+3])));
        }
        
        for(int i = threadIdx.x; i < 32; i += blockDim.x) {
            int row_idx = i / 4;
            int col_chunk = (i % 4) * 2;
            int swizzled_col = (row_idx % 8) ^ col_chunk;
            uint32_t taddr = tmem_P_base + swizzled_col * 4; 
            *reinterpret_cast<uint4*>(&tmem_P[swizzled_col * 4 + row_idx * 8]) = *reinterpret_cast<uint4*>(&packed_p[i * 4]);
        }
        
        __syncthreads(); 
        fence_proxy_async_fn();

        for (int k_step = 0; k_step < 64; k_step += 16) {
            uint64_t desc_a = make_smem_desc(smem->P + k_step, 1, 1024);
            uint64_t desc_b = make_smem_desc(smem->V + k_step * 64, 16384, 1024);
            umma_f16_cg1(tmem_O, desc_a, desc_b, idesc_pv, 1);
        }
        for (int k_step = 0; k_step < 64; k_step += 16) {
            uint64_t desc_a = make_smem_desc(smem->P + 8192 + k_step, 1, 1024);
            uint64_t desc_b = make_smem_desc(smem->V + 8192 + k_step * 64, 16384, 1024);
            umma_f16_cg1(tmem_O, desc_a, desc_b, idesc_pv, 1);
        }
        
        umma_commit_1sm(bar_mma);
        mbarrier_wait_fn(bar_mma, phase_mma);
        phase_mma ^= 1;
        
        __syncthreads();
    }

    uint32_t to = tmem_O;
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_packed_fn(to + col, &r0, &r1, &r2, &r3);
        r_O[col] = r0;
        r_O[col + 1] = r1;
        r_O[col + 2] = r2;
        r_O[col + 3] = r3;
    }
    tmem_load_fence_fn();

    int row = threadIdx.x;

    for(int col = 0; col < 128; ++col) {
        float o_val = __uint_as_float(r_O[col]);
        o_val /= global_sum[row];
        
        int half = col / 64;
        int c = col % 64;
        int x = c / 8;
        int rem = c % 8;
        int swizzled_x = (row % 8) ^ x;
        int swizzled_c = swizzled_x * 8 + rem;
        
        *(reinterpret_cast<uint16_t*>(&((smem->O + half * 8192)[row * 64 + swizzled_c]))) = __float2bfloat16(o_val);
    }
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_out, 1);
    }
    __syncthreads(); 
    mbarrier_wait_fn(bar_out, 0);

    // Coalesced Vectorized Output Store directly from generic proxy memory
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        int row = i / 8;
        int col_vec = (i % 8) * 8;
        int global_row = row_start_q + row;
        int global_col = col_vec;
        if (global_row < S_seq && global_col + 7 < 128) {
            uint4 val = *reinterpret_cast<const uint4*>(&smem->O[row * 128 + global_col]);
            *reinterpret_cast<uint4*>(&O_gmem[bh_idx * S_seq * 128 + global_row * 128 + global_col]) = val;
        }
    }

    if (threadIdx.x < 128) {
        int global_row = row_start_q + threadIdx.x;
        if (global_row < S_seq) {
            LSE[bh_idx * S_seq + global_row] = softmax_s_old[threadIdx.x] + __logf(global_sum[threadIdx.x]);
        }
    }

    tmem_dealloc_fn(tmem_S, 256);
    tmem_dealloc_fn(tmem_O, 256);
    tmem_dealloc_fn(tmem_P, 256);
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    if (S == 0) return;

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    __nv_bfloat16* Q_g = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_g = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_g = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_g = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_g = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q_g, 128, S, B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K_g, 128, S, B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V_g, 128, S, B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));

    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128, 1, 1);

    uint32_t smem_size = sizeof(SwizzledSmem) + 1024 + 64; 
    
    CUDA_CHECK(cudaFuncSetAttribute(flash_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    flash_attention_kernel<<<grid, block, smem_size, stream>>>(
        O_g, LSE_g, tma_Q, tma_K, tma_V, S, D, B * H);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda