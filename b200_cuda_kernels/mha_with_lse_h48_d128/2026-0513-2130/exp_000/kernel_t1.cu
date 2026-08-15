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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_v_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16; 
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
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

__device__ __forceinline__ uint32_t swizzle_128B_addr_64(uint32_t row, uint32_t col) {
    uint32_t x_chunk = col / 8;
    uint32_t x_rem = col % 8;
    uint32_t swizzled_chunk = (row % 8) ^ x_chunk;
    uint32_t swizzled_col = swizzled_chunk * 8 + x_rem;
    return (row * 64 + swizzled_col) * 2;
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
    
    extern __shared__ __align__(128) char smem_raw[];
    char* smem_base = (char*)(((uintptr_t)smem_raw + 127) & ~127);
    
    uint64_t* mbar_q = (uint64_t*)(smem_base);
    uint64_t* mbar_kv_0 = (uint64_t*)(smem_base + 8);
    uint64_t* mbar_kv_1 = (uint64_t*)(smem_base + 16);
    uint64_t* mbar_mma = (uint64_t*)(smem_base + 24);
    uint32_t* p_tmem_base = (uint32_t*)(smem_base + 32);

    __nv_bfloat16* Q0_smem = (__nv_bfloat16*)(smem_base + 128);
    __nv_bfloat16* Q1_smem = Q0_smem + 128 * 64;
    __nv_bfloat16* buf0 = Q1_smem + 128 * 64;
    __nv_bfloat16* buf1 = buf0 + 4 * 128 * 64;
    
    __nv_bfloat16* K0_smem[2] = { buf0, buf1 };
    __nv_bfloat16* K1_smem[2] = { buf0 + 128*64, buf1 + 128*64 };
    __nv_bfloat16* V0_smem[2] = { buf0 + 2*128*64, buf1 + 2*128*64 };
    __nv_bfloat16* V1_smem[2] = { buf0 + 3*128*64, buf1 + 3*128*64 };
    
    __nv_bfloat16* P0_smem = buf1 + 4 * 128 * 64;
    __nv_bfloat16* P1_smem = P0_smem + 128 * 64;

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
    uint32_t tmem_o0 = tmem_s + 128;
    uint32_t tmem_o1 = tmem_s + 192;
    
    int block_row = blockIdx.z * H * S + blockIdx.y * S + blockIdx.x * 128;
    int kv_row_base = blockIdx.z * H * S + blockIdx.y * S;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_q, 128 * 64 * 2 * 2);
        tma_load_2d_fn(&tma_Q, mbar_q, Q0_smem, 0, block_row);
        tma_load_2d_fn(&tma_Q, mbar_q, Q1_smem, 64, block_row);
    }
    mbarrier_wait_fn(mbar_q, 0);

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_kv_0, 128 * 64 * 2 * 4);
        tma_load_2d_fn(&tma_K, mbar_kv_0, K0_smem[0], 0, kv_row_base);
        tma_load_2d_fn(&tma_K, mbar_kv_0, K1_smem[0], 64, kv_row_base);
        tma_load_2d_fn(&tma_V, mbar_kv_0, V0_smem[0], 0, kv_row_base);
        tma_load_2d_fn(&tma_V, mbar_kv_0, V1_smem[0], 64, kv_row_base);
    }

    uint32_t idesc_qk = make_idesc_f16(128, 128, false, false);
    uint32_t idesc_pv = make_idesc_f16(128, 64, false, true);

    float m_i = -1e20f;
    float l_i = 0.0f;
    float O0_acc[64] = {0};
    float O1_acc[64] = {0};

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
            mbarrier_arrive_and_expect_tx_fn(next_mbar_kv, 128 * 64 * 2 * 4);
            tma_load_2d_fn(&tma_K, next_mbar_kv, K0_smem[next_stage], 0, kv_row_base + next_n * 128);
            tma_load_2d_fn(&tma_K, next_mbar_kv, K1_smem[next_stage], 64, kv_row_base + next_n * 128);
            tma_load_2d_fn(&tma_V, next_mbar_kv, V0_smem[next_stage], 0, kv_row_base + next_n * 128);
            tma_load_2d_fn(&tma_V, next_mbar_kv, V1_smem[next_stage], 64, kv_row_base + next_n * 128);
        }
        
        mbarrier_wait_fn(cur_mbar_kv, phase_kv[stage]);
        phase_kv[stage] ^= 1;

        __nv_bfloat16* cur_K0 = K0_smem[stage];
        __nv_bfloat16* cur_K1 = K1_smem[stage];
        __nv_bfloat16* cur_V0 = V0_smem[stage];
        __nv_bfloat16* cur_V1 = V1_smem[stage];

        for (int k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_a = make_smem_desc_sm100_fn(Q0_smem + k_step * 16, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(cur_K0 + k_step * 16, 1024);
            uint32_t accum = (k_step == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_s, desc_a, desc_b, idesc_qk, accum);
        }
        for (int k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_a = make_smem_desc_sm100_fn(Q1_smem + k_step * 16, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(cur_K1 + k_step * 16, 1024);
            umma_f16_cg1_fn(tmem_s, desc_a, desc_b, idesc_qk, 1);
        }
        
        tcgen05_commit_cg1_fn(mbar_mma);
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;

        int global_col_base = n_block * 128;
        float m_ij = -1e20f;
        for (int col = 0; col < 128; col += 4) {
            uint32_t r[4];
            tmem_load_4x_fn(tmem_s + col, &r[0], &r[1], &r[2], &r[3]);
            for(int i=0; i<4; i++) {
                float s = __uint_as_float(r[i]) * 0.088388347f;
                if (global_col_base + col + i >= S) s = -1e20f;
                m_ij = max(m_ij, s);
            }
        }
        tmem_load_fence_fn();

        float m_new = max(m_i, m_ij);
        float alpha = fast_expf(m_i - m_new);
        float l_ij = 0.0f;

        for (int col = 0; col < 128; col += 4) {
            uint32_t r[4];
            tmem_load_4x_fn(tmem_s + col, &r[0], &r[1], &r[2], &r[3]);
            for(int i=0; i<4; i++) {
                float s = __uint_as_float(r[i]) * 0.088388347f;
                if (global_col_base + col + i >= S) s = -1e20f;
                float p = fast_expf(s - m_new);
                l_ij += p;
                int c = col + i;
                uint32_t addr;
                if (c < 64) {
                    addr = swizzle_128B_addr_64(tid, c);
                    *((__nv_bfloat16*)((char*)P0_smem + addr)) = __float2bfloat16(p);
                } else {
                    addr = swizzle_128B_addr_64(tid, c - 64);
                    *((__nv_bfloat16*)((char*)P1_smem + addr)) = __float2bfloat16(p);
                }
            }
        }
        tmem_load_fence_fn();

        for (int i = 0; i < 64; ++i) {
            O0_acc[i] *= alpha;
            O1_acc[i] *= alpha;
        }
        m_i = m_new;
        l_i = l_i * alpha + l_ij;

        __syncthreads();
        fence_proxy_async_fn();

        for (int k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_a = make_smem_desc_sm100_fn(P0_smem + k_step * 16, 1024);
            uint64_t desc_b = make_smem_desc_sm100_v_fn(cur_V0 + k_step * 16 * 64, 1024, 0);
            uint32_t accum = (k_step == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_o0, desc_a, desc_b, idesc_pv, accum);
        }
        for (int k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_a = make_smem_desc_sm100_fn(P1_smem + k_step * 16, 1024);
            uint64_t desc_b = make_smem_desc_sm100_v_fn(cur_V0 + 64 * 64 + k_step * 16 * 64, 1024, 0);
            umma_f16_cg1_fn(tmem_o0, desc_a, desc_b, idesc_pv, 1);
        }

        for (int k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_a = make_smem_desc_sm100_fn(P0_smem + k_step * 16, 1024);
            uint64_t desc_b = make_smem_desc_sm100_v_fn(cur_V1 + k_step * 16 * 64, 1024, 0);
            uint32_t accum = (k_step == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_o1, desc_a, desc_b, idesc_pv, accum);
        }
        for (int k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_a = make_smem_desc_sm100_fn(P1_smem + k_step * 16, 1024);
            uint64_t desc_b = make_smem_desc_sm100_v_fn(cur_V1 + 64 * 64 + k_step * 16 * 64, 1024, 0);
            umma_f16_cg1_fn(tmem_o1, desc_a, desc_b, idesc_pv, 1);
        }

        tcgen05_commit_cg1_fn(mbar_mma);
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;

        for (int col = 0; col < 64; col += 4) {
            uint32_t r[4];
            tmem_load_4x_fn(tmem_o0 + col, &r[0], &r[1], &r[2], &r[3]);
            for(int i=0; i<4; i++) O0_acc[col + i] += __uint_as_float(r[i]);
        }
        for (int col = 0; col < 64; col += 4) {
            uint32_t r[4];
            tmem_load_4x_fn(tmem_o1 + col, &r[0], &r[1], &r[2], &r[3]);
            for(int i=0; i<4; i++) O1_acc[col + i] += __uint_as_float(r[i]);
        }
        tmem_load_fence_fn();

        stage = (stage + 1) % 2;
    }

    int global_row = blockIdx.x * 128 + tid;
    if (global_row < S) {
        float inv_l = 1.0f / l_i;
        uint64_t base_idx = (uint64_t)blockIdx.z * H * S * 128 + blockIdx.y * S * 128 + global_row * 128;
        for (int i = 0; i < 64; ++i) {
            O_gmem[base_idx + i] = __float2bfloat16(O0_acc[i] * inv_l);
            O_gmem[base_idx + 64 + i] = __float2bfloat16(O1_acc[i] * inv_l);
        }
        LSE_gmem[blockIdx.z * H * S + blockIdx.y * S + global_row] = m_i + logf(l_i);
    }

    __syncthreads();
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
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    int grid_x = (S + 127) / 128;
    int grid_y = H;
    int grid_z = B;
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int shared_mem_size = 193 * 1024;
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