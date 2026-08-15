#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_cuda_mha {

// -------------------------------------------------------------------------
// SM100 Helper Inline Assembly Functions
// -------------------------------------------------------------------------

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
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
                 :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)swizzle << 61;
    return d;
}

// -------------------------------------------------------------------------
// Kernel
// -------------------------------------------------------------------------

__global__ void __launch_bounds__(128) mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* __restrict__ LSE_ptr,
    int S, int H, int B
) {
    setmaxnreg_inc_sync_fn<256>(); // Boost available registers per thread

    // SMEM layouts: Q, K, V, P + Control Barrier + TMEM pointers
    extern __shared__ __align__(128) uint8_t smem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem);          // 32 KB
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem + 32768);  // 32 KB
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem + 65536);  // 32 KB
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem + 98304);  // 32 KB

    uint64_t* mbar_Q   = (uint64_t*)(smem + 131072);
    uint64_t* mbar_K   = mbar_Q + 1;
    uint64_t* mbar_V   = mbar_K + 1;
    uint64_t* mbar_mma = mbar_V + 1;
    uint32_t* tmem_addr_smem = (uint32_t*)(mbar_mma + 1);

    int batch_idx = blockIdx.z;
    int head_idx  = blockIdx.y;
    int block_s   = blockIdx.x; // Block-level S-dimension offset

    // Initialize Barriers 
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_mma, 1);
        fence_smem_barrier_init_fn();
    }
    
    // Allocate Tensor Memory (Requires all threads in WP0)
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(tmem_addr_smem, 256);
    }
    __syncthreads();

    uint32_t tmem_base = *tmem_addr_smem;
    uint32_t tmem_O = tmem_base;
    uint32_t tmem_S = tmem_base + 128;

    // Build SM100 Shared Memory Descriptors (No Swizzle defaults)
    // LBO/SBO mappings correspond uniquely to their mathematical dimension alignment:
    uint64_t desc_Q_base = make_smem_desc_sm100_fn(smem_Q, 16, 2048, 0);   // K-Major A Matrix
    uint64_t desc_K_base = make_smem_desc_sm100_fn(smem_K, 256, 128, 0);   // K-Major B Matrix 
    uint64_t desc_P_base = make_smem_desc_sm100_fn(smem_P, 16, 2048, 0);   // K-Major A Matrix
    uint64_t desc_V_base = make_smem_desc_sm100_fn(smem_V, 128, 256, 0);   // MN-Major B Matrix

    // SM100 Instruction Descriptors
    uint32_t idesc_S = 0;
    idesc_S |= (1u << 4);  // C_format = FP32
    idesc_S |= (1u << 7);  // A_format = BF16
    idesc_S |= (1u << 10); // B_format = BF16
    idesc_S |= (0 << 15);  // Transpose A = 0 (K-Major)
    idesc_S |= (0 << 16);  // Transpose B = 0 (K-Major)
    idesc_S |= ((128 / 8) << 17); // n_dim = 16
    idesc_S |= ((128 / 16) << 24); // m_dim = 8

    uint32_t idesc_O = 0;
    idesc_O |= (1u << 4);  // C_format = FP32
    idesc_O |= (1u << 7);  // A_format = BF16
    idesc_O |= (1u << 10); // B_format = BF16
    idesc_O |= (0 << 15);  // Transpose A = 0 (K-Major)
    idesc_O |= (1 << 16);  // Transpose B = 1 (MN-Major)
    idesc_O |= ((128 / 8) << 17); // n_dim = 16
    idesc_O |= ((128 / 16) << 24); // m_dim = 8

    // Load static Q tile
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 128 * 128 * 2);
        tma_load_4d_fn(&tma_Q, mbar_Q, smem_Q, 0, block_s * 128, head_idx, batch_idx);
        mbarrier_wait_fn(mbar_Q, 0);
    }
    __syncthreads();

    // In-register partial results (rescaled incrementally)
    float O_reg[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) O_reg[i] = 0.0f;

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    const float scale_S = 0.0883883476f; // 1.0 / sqrt(128)

    int phase_K = 0, phase_V = 0, phase_mma = 0;

    // Multi-head attention inner iteration
    for (int s_idx = 0; s_idx < S; s_idx += 128) {
        
        // Dispatch TMA loads for K and V tiles
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 128 * 128 * 2);
            tma_load_4d_fn(&tma_K, mbar_K, smem_K, 0, s_idx, head_idx, batch_idx);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 128 * 128 * 2);
            tma_load_4d_fn(&tma_V, mbar_V, smem_V, 0, s_idx, head_idx, batch_idx);
            
            mbarrier_wait_fn(mbar_K, phase_K);
        }
        __syncthreads();
        
        // MM0: compute S = Q @ K.T
        if (threadIdx.x == 0) {
            tcgen05_fence_before_fn();
            for (int k = 0; k < 8; ++k) { // Traverse K dimension chunks 
                umma_f16_cg1_fn(tmem_S, desc_Q_base + k * 2, desc_K_base + k * 2, idesc_S, k == 0 ? 0 : 1);
            }
            tcgen05_commit_cg1_fn(mbar_mma);
            mbarrier_wait_fn(mbar_mma, phase_mma);
            tcgen05_fence_after_fn();
        }
        __syncthreads();

        // Safe softmax computation
        float m_curr = m_prev;
        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; ++i) {
                float val = __uint_as_float(r[i]) * scale_S;
                if (s_idx + c + i >= S) val = -INFINITY; // Mask sequence edges dynamically
                m_curr = max(m_curr, val);
            }
        }
        
        // Online rescaling factor
        float scale = fast_exp2f_fn((m_prev - m_curr) * 1.44269504f); // ln2 approx
        if (m_prev == -INFINITY) scale = 0.0f;
        
        // Propagate normalization scaling
        for (int c = 0; c < 128; ++c) {
            O_reg[c] *= scale;
        }
        l_prev *= scale;
        m_prev = m_curr;

        // Populate scaled values to P tile
        uint32_t smem_P_addr = (uint32_t)__cvta_generic_to_shared(smem_P);
        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; i += 2) {
                float p0 = 0.0f;
                if (s_idx + c + i < S) 
                    p0 = fast_exp2f_fn((__uint_as_float(r[i]) * scale_S - m_curr) * 1.44269504f);
                float p1 = 0.0f;
                if (s_idx + c + i + 1 < S) 
                    p1 = fast_exp2f_fn((__uint_as_float(r[i+1]) * scale_S - m_curr) * 1.44269504f);
                
                l_prev += p0 + p1;
                uint32_t bf16_pair = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                uint32_t dst_addr = smem_P_addr + threadIdx.x * 256 + (c + i) * 2;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(dst_addr), "r"(bf16_pair) : "memory");
            }
        }
        
        fence_proxy_async_fn();
        __syncthreads();

        // MM1: compute updated localized Output tile
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(mbar_V, phase_V);
            tcgen05_fence_before_fn();
            for (int k = 0; k < 8; ++k) { // Traverse K dimension chunks 
                umma_f16_cg1_fn(tmem_O, desc_P_base + k * 2, desc_V_base + k * 256, idesc_O, k == 0 ? 0 : 1);
            }
            tcgen05_commit_cg1_fn(mbar_mma);
            mbarrier_wait_fn(mbar_mma, phase_mma ^ 1);
            tcgen05_fence_after_fn();
        }
        __syncthreads();

        // Readout chunk from MM1 and accumulate into register
        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_O + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; ++i) {
                O_reg[c + i] += __uint_as_float(r[i]);
            }
        }
        __syncthreads(); // Avoid thread 0 clobbering TMEM on next cycle execution

        phase_K ^= 1;
        phase_V ^= 1;
    }

    // Apply Softmax standardizer
    for (int c = 0; c < 128; ++c) {
        O_reg[c] /= l_prev;
    }

    // Export completely localized O output to shared
    uint32_t smem_P_addr = (uint32_t)__cvta_generic_to_shared(smem_P);
    for (int c = 0; c < 128; c += 2) {
        uint32_t bf16_pair = pack_bf16_fn(__float_as_uint(O_reg[c]), __float_as_uint(O_reg[c+1]));
        uint32_t dst_addr = smem_P_addr + threadIdx.x * 256 + c * 2;
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(dst_addr), "r"(bf16_pair) : "memory");
    }
    
    tma_store_fence_fn();
    __syncthreads();

    // Export completed outputs and write via bulk store to Generic memory space
    if (threadIdx.x == 0) {
        tma_store_4d_fn(&tma_O, smem_P, 0, block_s * 128, head_idx, batch_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    // Store LogSumExp
    if (block_s * 128 + threadIdx.x < S) {
        float lse = m_prev + logf(l_prev);
        LSE_ptr[batch_idx * H * S + head_idx * S + block_s * 128 + threadIdx.x] = lse;
    }

    // Safely unwind allocations
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
    __syncthreads();
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t D, uint64_t S, uint64_t H, uint64_t B) {
    cuuint64_t globalDim[4] = {D, S, H, B};
    cuuint64_t globalStrides[3] = {D * 2, D * S * 2, D * S * H * 2};
    cuuint32_t boxDim[4] = {128, 128, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id)); 
    
    int B_size = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    create_tma_4d_descriptor_2B(&tma_Q, Q_ptr, D, S, H, B_size);
    create_tma_4d_descriptor_2B(&tma_K, K_ptr, D, S, H, B_size);
    create_tma_4d_descriptor_2B(&tma_V, V_ptr, D, S, H, B_size);
    create_tma_4d_descriptor_2B(&tma_O, O_ptr, D, S, H, B_size);

    int threads = 128;
    dim3 blocks((S + 127) / 128, H, B_size);
    int smem_size = 131108; // 4 block matrix allocations + sync resources

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    cudaFuncSetAttribute((void*)mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    mha_fwd_kernel<<<blocks, threads, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_O, LSE_ptr, S, H, B_size);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_cuda_mha::run);

} // namespace tvm_ffi_cuda_mha