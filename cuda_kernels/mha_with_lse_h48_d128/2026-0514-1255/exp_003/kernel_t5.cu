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
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                  \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ float add_round_down_fn(float x, float y) {
    float z;
    asm("add.rm.ftz.f32 %0, %1, %2;" : "=f"(z) : "f"(x), "f"(y));
    return z;
}

__device__ __forceinline__ float evaluate_polynomial_fn(float x) {
    float out = 0.077119089663028717041015625f;
    out = fmaf(out, x, 0.227564394474029541015625f);
    out = fmaf(out, x, 0.695146143436431884765625f);
    out = fmaf(out, x, 1.0f);
    return out;
}

__device__ __forceinline__ float combine_int_frac_ex2_fn(float x_rounded, float frac_ex2) {
    float out;
    asm("{\n\t"
        ".reg .s32 xri, fei, xre, oi;\n\t"
        "mov.b32 xri, %1;\n\t"
        "mov.b32 fei, %2;\n\t"
        "shl.b32 xre, xri, 23;\n\t"
        "add.s32 oi, xre, fei;\n\t"
        "mov.b32 %0, oi;\n\t"
        "}\n"
        : "=f"(out)
        : "f"(x_rounded), "f"(frac_ex2));
    return out;
}

__device__ __forceinline__ float ex2_emulation_fn(float x) {
    constexpr float MAGIC = 12582912.0f;          // 2^23 + 2^22
    constexpr float CLAMP_LO = -127.0f;
    float x_clamped      = fmaxf(x, CLAMP_LO);
    float x_rounded      = add_round_down_fn(x_clamped, MAGIC);
    float x_rounded_back = x_rounded - MAGIC;
    float x_frac         = x_clamped - x_rounded_back;
    float frac_ex2       = evaluate_polynomial_fn(x_frac);
    return combine_int_frac_ex2_fn(x_rounded, frac_ex2);
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)swizzle << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool trans_a, bool trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    if (trans_a) d |= (1u << 15);
    if (trans_b) d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void load_tmem_4(uint32_t col, uint32_t base_tmem, float* out) {
    uint32_t r0, r1, r2, r3;
    uint32_t addr = base_tmem + col; 
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    out[0] = __uint_as_float(r0);
    out[1] = __uint_as_float(r1);
    out[2] = __uint_as_float(r2);
    out[3] = __uint_as_float(r3);
}

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ void mha_fwd_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* out_LSE,
    int seq_len
) {
    int m_block = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;

    __nv_bfloat16* smem_q = (__nv_bfloat16*)(smem_pool + 0);
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem_pool + 32768);
    __nv_bfloat16* smem_v = (__nv_bfloat16*)(smem_pool + 65536);
    __nv_bfloat16* smem_p = (__nv_bfloat16*)(smem_pool + 98304);
    
    uint64_t* mbar_tma_ptr = (uint64_t*)(smem_pool + 131072);
    uint64_t* mbar_umma_ptr = (uint64_t*)(smem_pool + 131080);
    uint32_t* smem_tmem_s_ptr = (uint32_t*)(smem_pool + 131088);
    uint32_t* smem_tmem_o_ptr = (uint32_t*)(smem_pool + 131092);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_tma_ptr, 1);
        init_smem_barrier_fn(mbar_umma_ptr, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (threadIdx.x / 32 == 0) {
        tmem_alloc_cg1_fn(smem_tmem_s_ptr, 128);
        tmem_alloc_cg1_fn(smem_tmem_o_ptr, 128);
    }
    __syncthreads();
    uint32_t tmem_s = *smem_tmem_s_ptr;
    uint32_t tmem_o = *smem_tmem_o_ptr;

    int phase_tma = 0;
    int phase_umma = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_tma_ptr, 128 * 128 * 2);
        tma_load_4d_fn(&tma_Q, mbar_tma_ptr, smem_q, 0, m_block * 128, head_idx, batch_idx);
    }
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(mbar_tma_ptr, phase_tma);
    }
    __syncthreads();
    tcgen05_fence_after_fn();
    phase_tma ^= 1;

    uint32_t idesc_S = make_instr_desc_fn(128, 128, false, false);
    uint32_t idesc_O = make_instr_desc_fn(128, 128, false, true);

    float O_acc[128];
    for (int i = 0; i < 128; ++i) O_acc[i] = 0.0f;
    float old_max = -INFINITY;
    float sum_exp = 0.0f;

    int num_kv_blocks = (seq_len + 127) / 128;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma_ptr, 2 * 128 * 128 * 2);
            tma_load_4d_fn(&tma_K, mbar_tma_ptr, smem_k, 0, kv_block * 128, head_idx, batch_idx);
            tma_load_4d_fn(&tma_V, mbar_tma_ptr, smem_v, 0, kv_block * 128, head_idx, batch_idx);
        }
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(mbar_tma_ptr, phase_tma);
        }
        __syncthreads();
        tcgen05_fence_after_fn();
        phase_tma ^= 1;

        if (threadIdx.x == 0) {
            for (int i = 0; i < 8; ++i) {
                uint32_t accum = (i == 0) ? 0 : 1;
                uint64_t desc_a = make_smem_desc_sm100_fn(smem_q + i * 16, 16, 2048, 0);
                uint64_t desc_b = make_smem_desc_sm100_fn(smem_k + i * 16, 256, 16, 0);
                umma_f16_cg1_fn(tmem_s, desc_a, desc_b, idesc_S, accum);
            }
            umma_commit_cg1_fn(mbar_umma_ptr);
        }
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(mbar_umma_ptr, phase_umma);
        }
        __syncthreads();
        tcgen05_fence_after_fn();
        phase_umma ^= 1;

        float row_max = -INFINITY;
        for (int i = 0; i < 32; ++i) {
            float chunk[4];
            load_tmem_4(i * 4, tmem_s, chunk);
            for (int j = 0; j < 4; ++j) {
                int col = kv_block * 128 + i * 4 + j;
                float val = (col < seq_len) ? chunk[j] * 0.12751713468f : -INFINITY;
                row_max = fmaxf(row_max, val);
            }
        }

        float new_max = fmaxf(old_max, row_max);
        float scale = (old_max == -INFINITY) ? 0.0f : ex2_emulation_fn(old_max - new_max);
        for (int i = 0; i < 128; ++i) {
            O_acc[i] *= scale;
        }
        old_max = new_max;

        float row_sum = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float chunk[4];
            load_tmem_4(i * 4, tmem_s, chunk);
            for (int j = 0; j < 4; ++j) {
                int col = i * 4 + j;
                int global_col = kv_block * 128 + col;
                float val = (global_col < seq_len) ? chunk[j] * 0.12751713468f : -INFINITY;
                float p = (global_col < seq_len) ? ex2_emulation_fn(val - new_max) : 0.0f;
                row_sum += p;
                smem_p[threadIdx.x * 128 + col] = __float2bfloat16(p);
            }
        }
        sum_exp = sum_exp * scale + row_sum;

        __syncthreads();
        fence_async_shared_fn();

        if (threadIdx.x == 0) {
            for (int i = 0; i < 8; ++i) {
                uint32_t accum = (i == 0) ? 0 : 1;
                uint64_t desc_a = make_smem_desc_sm100_fn(smem_p + i * 16, 16, 2048, 0);
                uint64_t desc_b = make_smem_desc_sm100_fn(smem_v + i * 2048, 16, 256, 0); 
                umma_f16_cg1_fn(tmem_o, desc_a, desc_b, idesc_O, accum);
            }
            umma_commit_cg1_fn(mbar_umma_ptr);
        }
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(mbar_umma_ptr, phase_umma);
        }
        __syncthreads();
        tcgen05_fence_after_fn();
        phase_umma ^= 1;

        for (int i = 0; i < 32; ++i) {
            float chunk[4];
            load_tmem_4(i * 4, tmem_o, chunk);
            O_acc[i * 4 + 0] += chunk[0];
            O_acc[i * 4 + 1] += chunk[1];
            O_acc[i * 4 + 2] += chunk[2];
            O_acc[i * 4 + 3] += chunk[3];
        }
        __syncthreads();
    }

    float inv_sum = 1.0f / sum_exp;
    for (int col = 0; col < 128; ++col) {
        int idx = threadIdx.x * 128 + col;
        smem_q[idx] = __float2bfloat16(O_acc[col] * inv_sum);
    }
    
    __syncthreads();
    fence_async_shared_fn();

    if (threadIdx.x == 0) {
        tma_store_4d_fn(&tma_O, smem_q, 0, m_block * 128, head_idx, batch_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }

    int lse_idx = batch_idx * (gridDim.y * seq_len) + head_idx * seq_len + m_block * 128 + threadIdx.x;
    if (m_block * 128 + threadIdx.x < seq_len) {
        out_LSE[lse_idx] = (old_max + log2f(sum_exp)) * 0.69314718056f;
    }

    __syncthreads();
    if (threadIdx.x / 32 == 0) {
        tmem_dealloc_cg1_fn(tmem_s, 128);
        tmem_dealloc_cg1_fn(tmem_o, 128);
    }
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3, 
                                     uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3,
                                     CUtensorMapInterleave interleave) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides, boxDim, 
        elementStrides, interleave, CU_TENSOR_MAP_SWIZZLE_NONE, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, S, H, B, 128, 128, 1, 1, CU_TENSOR_MAP_INTERLEAVE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), 128, S, H, B, 128, 128, 1, 1, CU_TENSOR_MAP_INTERLEAVE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), 128, S, H, B, 128, 128, 1, 1, CU_TENSOR_MAP_INTERLEAVE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_O, O.data_ptr(), 128, S, H, B, 128, 128, 1, 1, CU_TENSOR_MAP_INTERLEAVE_NONE));
    
    int64_t seq_len = S;
    int blocks = (seq_len + 127) / 128;
    dim3 grid(blocks, H, B);
    dim3 block(128);
    
    int smem_bytes = 131072 + 128; 
    
    CUDA_CHECK(cudaFuncSetAttribute(
        mha_fwd_sm100_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_fwd_sm100_kernel, tma_Q, tma_K, tma_V, tma_O, static_cast<float*>(LSE.data_ptr()), seq_len));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda