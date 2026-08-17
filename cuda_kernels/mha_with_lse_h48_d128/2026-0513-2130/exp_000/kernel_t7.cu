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

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        globalAddress,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float fast_expf(float x) {
    return fast_exp2f_fn(x * 1.4426950408889634f);
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_none(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16; 
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)((addr >> 3) & 0x7) << 49;
    d |= (uint64_t)0 << 61;   // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_f16(uint32_t M, uint32_t N, bool transA, bool transB) {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    if (transA) d |= (1u << 15);
    if (transB) d |= (1u << 16);
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_gmem,
    float* __restrict__ LSE_gmem,
    int B, int H, int S) 
{
    setmaxnreg_inc_sync_fn<248>();
    
    int tid = threadIdx.x;
    
    __shared__ alignas(8) uint64_t mbar_q[1];
    __shared__ alignas(8) uint64_t mbar_kv_0[1];
    __shared__ alignas(8) uint64_t mbar_kv_1[1];
    __shared__ alignas(8) uint64_t mbar_mma[1];
    __shared__ alignas(4) uint32_t p_tmem_base[1];

    extern __shared__ __align__(128) char smem_raw[];
    
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)(smem_raw);
    __nv_bfloat16* K_smem[2] = { Q_smem + 16384, Q_smem + 49152 }; 
    __nv_bfloat16* V_smem[2] = { Q_smem + 32768, Q_smem + 65536 };
    __nv_bfloat16* P_smem = Q_smem + 81920; 

    if (tid == 0) {
        init_smem_barrier_fn(mbar_q, 1);
        init_smem_barrier_fn(mbar_kv_0, 1);
        init_smem_barrier_fn(mbar_kv_1, 1);
        init_smem_barrier_fn(mbar_mma, 1);
        fence_smem_barrier_init_fn();
    }
    if (tid < 32) {
        tmem_alloc_cg1_fn(p_tmem_base, 256);
    }
    __syncthreads();
    
    uint32_t tmem_s = *p_tmem_base;
    uint32_t tmem_o = tmem_s + 128;
    
    int block_row = blockIdx.z * H * S + blockIdx.y * S + blockIdx.x * 128;
    int kv_row_base = blockIdx.z * H * S + blockIdx.y * S;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_q, 128 * 128 * 2);
        tma_load_2d_fn(&tma_Q, mbar_q, Q_smem, 0, block_row);
    }
    mbarrier_wait_fn(mbar_q, 0);

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_kv_0, 128 * 128 * 2 * 2);
        tma_load_2d_fn(&tma_K, mbar_kv_0, K_smem[0], 0, kv_row_base);
        tma_load_2d_fn(&tma_V, mbar_kv_0, V_smem[0], 0, kv_row_base);
    }

    uint32_t idesc_qk = make_idesc_f16(128, 128, false, false);
    uint32_t idesc_pv = make_idesc_f16(128, 128, false, true);

    float m_i = -1e20f;
    float l_i = 0.0f;
    float O_acc[128] = {0};

    int num_n_blocks = (S + 127) / 128;
    int stage = 0;
    int phase_kv[2] = {0, 0};
    int phase_mma = 0;

    for (int n_block = 0; n_block < num_n_blocks; ++n_block) {
        int next_n = n_block + 1;
        uint64_t* cur_mbar_kv = (stage == 0) ? mbar_kv_0 : mbar_kv_1;
        uint64_t* next_mbar_kv = (stage == 0) ? mbar_kv_1 : mbar_kv_0;

        if (next_n < num_n_blocks && tid == 0) {
            int next_stage = (stage + 1) % 2;
            mbarrier_arrive_and_expect_tx_fn(next_mbar_kv, 128 * 128 * 2 * 2);
            tma_load_2d_fn(&tma_K, next_mbar_kv, K_smem[next_stage], 0, kv_row_base + next_n * 128);
            tma_load_2d_fn(&tma_V, next_mbar_kv, V_smem[next_stage], 0, kv_row_base + next_n * 128);
        }
        
        mbarrier_wait_fn(cur_mbar_kv, phase_kv[stage]);
        phase_kv[stage] ^= 1;

        __nv_bfloat16* cur_K = K_smem[stage];
        __nv_bfloat16* cur_V = V_smem[stage];

        if (tid == 0) {
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint64_t desc_a = make_smem_desc_none(Q_smem + k_step * 16, 16, 2048);
                uint64_t desc_b = make_smem_desc_none(cur_K + k_step * 16, 16, 2048);
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_s, desc_a, desc_b, idesc_qk, accum);
            }
            tcgen05_commit_cg1_fn(mbar_mma);
        }
        
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;

        int global_col_base = n_block * 128;
        float m_ij = -1e20f;
        
        for (int chunk = 0; chunk < 128; chunk += 32) {
            uint32_t r[32];
            for (int i = 0; i < 32; i += 4) {
                tmem_load_4x_fn(tmem_s + chunk + i, &r[i], &r[i+1], &r[i+2], &r[i+3]);
            }
            tmem_load_fence_fn();
            for (int i = 0; i < 32; ++i) {
                float s = __uint_as_float(r[i]) * 0.088388347f;
                if (global_col_base + chunk + i >= S) s = -1e20f;
                m_ij = max(m_ij, s);
            }
        }

        float m_new = max(m_i, m_ij);
        float alpha = fast_expf(m_i - m_new);
        float l_ij = 0.0f;

        for (int chunk = 0; chunk < 128; chunk += 32) {
            uint32_t r[32];
            for (int i = 0; i < 32; i += 4) {
                tmem_load_4x_fn(tmem_s + chunk + i, &r[i], &r[i+1], &r[i+2], &r[i+3]);
            }
            tmem_load_fence_fn();
            for (int i = 0; i < 32; ++i) {
                float s = __uint_as_float(r[i]) * 0.088388347f;
                if (global_col_base + chunk + i >= S) s = -1e20f;
                float p = fast_expf(s - m_new);
                l_ij += p;
                P_smem[tid * 128 + chunk + i] = __float2bfloat16(p);
            }
        }

        for (int i = 0; i < 128; ++i) {
            O_acc[i] *= alpha;
        }
        m_i = m_new;
        l_i = l_i * alpha + l_ij;

        __syncthreads();
        fence_proxy_async_fn();

        if (tid == 0) {
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint64_t desc_a = make_smem_desc_none(P_smem + k_step * 16, 16, 2048);
                uint64_t desc_b = make_smem_desc_none(cur_V + k_step * 16 * 128, 16, 256);
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_o, desc_a, desc_b, idesc_pv, accum);
            }
            tcgen05_commit_cg1_fn(mbar_mma);
        }

        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;

        for (int chunk = 0; chunk < 128; chunk += 32) {
            uint32_t r_o[32];
            for (int i = 0; i < 32; i += 4) {
                tmem_load_4x_fn(tmem_o + chunk + i, &r_o[i], &r_o[i+1], &r_o[i+2], &r_o[i+3]);
            }
            tmem_load_fence_fn();
            for (int i = 0; i < 32; ++i) {
                O_acc[chunk + i] += __uint_as_float(r_o[i]);
            }
        }

        __syncthreads();
        stage = (stage + 1) % 2;
    }

    int global_row = blockIdx.x * 128 + tid;

    __syncthreads();
    if (global_row < S) {
        float inv_l = 1.0f / l_i;
        for (int i = 0; i < 128; ++i) {
            P_smem[tid * 128 + i] = __float2bfloat16(O_acc[i] * inv_l);
        }
    }
    __syncthreads();

    int num_rows = S - blockIdx.x * 128;
    if (num_rows > 128) num_rows = 128;
    uint64_t base_gmem = (uint64_t)blockIdx.z * H * S * 128 + (uint64_t)blockIdx.y * S * 128 + (uint64_t)blockIdx.x * 128 * 128;
    
    for (int row = 0; row < num_rows; ++row) {
        O_gmem[base_gmem + row * 128 + tid] = P_smem[row * 128 + tid];
    }

    if (global_row < S) {
        LSE_gmem[(uint64_t)blockIdx.z * H * S + (uint64_t)blockIdx.y * S + global_row] = m_i + logf(l_i);
    }

    if (tid < 32) {
        tmem_dealloc_cg1_fn(*p_tmem_base, 256);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, (uint64_t)B*H*S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, (uint64_t)B*H*S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, (uint64_t)B*H*S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    int grid_x = (S + 127) / 128;
    int grid_y = H;
    int grid_z = B;
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int shared_mem_size = 197 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    
    mha_kernel<<<grid, block, shared_mem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        B, H, S);
        
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda