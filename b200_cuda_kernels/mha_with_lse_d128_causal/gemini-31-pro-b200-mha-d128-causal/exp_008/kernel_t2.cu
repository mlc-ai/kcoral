#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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
        const char* err_name;                                      \
        cuGetErrorName(_e, &err_name);                             \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_name, __FILE__, __LINE__);                     \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                     uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3,
                                     CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        4,
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
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

__global__ void __launch_bounds__(128, 1) mha_fwd_kernel(const __grid_constant__ CUtensorMap tma_Q,
                               const __grid_constant__ CUtensorMap tma_K,
                               const __grid_constant__ CUtensorMap tma_V,
                               __nv_bfloat16* __restrict__ O,
                               float* __restrict__ LSE,
                               int seq_len) {
    setmaxnreg_inc_sync_fn<248>();
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_start = blockIdx.x * 64;
    
    if (q_start >= seq_len) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    
    extern __shared__ __align__(128) char smem_dyn[];
    void* Q_smem = smem_dyn;
    void* K_smem = smem_dyn + 16384;
    void* V_smem = smem_dyn + 32768;
    void* P_smem = smem_dyn + 49152;
    uint64_t* mbar_tma = (uint64_t*)(smem_dyn + 57344);
    uint64_t* mbar_umma = (uint64_t*)(smem_dyn + 57352);
    
    __shared__ uint32_t tmem_addr;
    if (warp_id == 0) {
        tmem_alloc_cg1_fn(&tmem_addr, 256);
    }
    
    uint32_t phase_tma = 0;
    uint32_t phase_umma = 0;
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_tma, 1);
        init_smem_barrier_fn(mbar_umma, 1);
        fence_smem_barrier_init_fn();
        
        mbarrier_arrive_and_expect_tx_fn(mbar_tma, 16384);
        tma_load_4d_fn(&tma_Q, mbar_tma, Q_smem, 0, q_start, h, b);
    }
    __syncthreads();
    
    mbarrier_wait_fn(mbar_tma, phase_tma);
    phase_tma ^= 1;
    
    uint32_t tmem_base = tmem_addr;
    uint32_t tmem_S = tmem_base;
    uint32_t tmem_O = tmem_base + 64;
    
    uint32_t idesc_S = make_instr_desc_fn(64, 64, 0, 0);
    uint32_t idesc_O = make_instr_desc_fn(64, 128, 0, 1);
    
    float O_reg[128];
    for (int i = 0; i < 128; i++) O_reg[i] = 0.0f;
    float S_row[64];
    
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    for (int k_start = 0; k_start <= q_start && k_start < seq_len; k_start += 64) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma, 32768);
            tma_load_4d_fn(&tma_K, mbar_tma, K_smem, 0, k_start, h, b);
            tma_load_4d_fn(&tma_V, mbar_tma, V_smem, 0, k_start, h, b);
        }
        mbarrier_wait_fn(mbar_tma, phase_tma);
        phase_tma ^= 1;
        
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_a = make_smem_desc_sm100_fn((char*)Q_smem + k * 2, 1, 2048);
            uint64_t desc_b = make_smem_desc_sm100_fn((char*)K_smem + k * 2, 1, 2048);
            uint32_t accum = (k == 0) ? 0 : 1;
            if (tid == 0) umma_f16_cg1_fn(tmem_S, desc_a, desc_b, idesc_S, accum);
        }
        
        if (tid == 0) tcgen05_commit_cg1_fn(mbar_umma);
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        if (warp_id < 2) {
            int row = warp_id * 32 + lane;
            for (int c = 0; c < 64; c += 8) {
                uint32_t r[8];
                tmem_load_8x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                for (int i = 0; i < 8; i++) S_row[c + i] = __uint_as_float(r[i]);
            }
            tmem_load_fence_fn();
            
            int g_q = q_start + row;
            float max_val = -INFINITY;
            for (int c = 0; c < 64; c++) {
                int g_k = k_start + c;
                if (g_k > g_q || g_k >= seq_len || g_q >= seq_len) {
                    S_row[c] = -INFINITY;
                } else {
                    S_row[c] *= 0.0883883476f;
                }
                max_val = fmaxf(max_val, S_row[c]);
            }
            
            float m_new = fmaxf(m_i, max_val);
            float exp_diff = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new);
            l_i *= exp_diff;
            for (int c = 0; c < 128; c++) O_reg[c] *= exp_diff;
            
            float sum_val = 0.0f;
            for (int c = 0; c < 64; c += 8) {
                __nv_bfloat16 p[8];
                for (int i = 0; i < 8; i++) {
                    float val = (m_new == -INFINITY) ? 0.0f : expf(S_row[c + i] - m_new);
                    sum_val += val;
                    p[i] = __float2bfloat16(val);
                }
                uint32_t x = c / 8;
                uint32_t swizzled_x = (row % 8) ^ x;
                uint32_t offset = row * 128 + swizzled_x * 16;
                *(uint4*)((char*)P_smem + offset) = *(uint4*)p;
            }
            l_i += sum_val;
            m_i = m_new;
        }
        
        __syncthreads();
        fence_async_shared_fn();
        tcgen05_fence_after_fn();
        
        for (int kp = 0; kp < 64; kp += 16) {
            uint64_t desc_a = make_smem_desc_sm100_fn((char*)P_smem + kp * 2, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((char*)V_smem + kp * 256, 2048, 128);
            uint32_t accum = (kp == 0) ? 0 : 1;
            if (tid == 0) umma_f16_cg1_fn(tmem_O, desc_a, desc_b, idesc_O, accum);
        }
        
        if (tid == 0) tcgen05_commit_cg1_fn(mbar_umma);
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        if (warp_id < 2) {
            for (int c = 0; c < 128; c += 8) {
                uint32_t r[8];
                tmem_load_8x_fn(tmem_O + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                for (int i = 0; i < 8; i++) O_reg[c + i] += __uint_as_float(r[i]);
            }
            tmem_load_fence_fn();
        }
    }
    
    __nv_bfloat16* o_ptr = O + (int64_t)b * gridDim.y * seq_len * 128 + (int64_t)h * seq_len * 128;
    float* lse_ptr = LSE + (int64_t)b * gridDim.y * seq_len + (int64_t)h * seq_len;
    
    if (warp_id < 2) {
        int row = warp_id * 32 + lane;
        int g_q = q_start + row;
        if (g_q < seq_len) {
            lse_ptr[g_q] = m_i + logf(l_i);
        }
        
        float inv_l = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;
        for (int c = 0; c < 128; c += 8) {
            __nv_bfloat16 p[8];
            for (int i = 0; i < 8; i++) p[i] = __float2bfloat16(O_reg[c + i] * inv_l);
            *(uint4*)((char*)Q_smem + row * 256 + c * 2) = *(uint4*)p;
        }
    }
    
    __syncthreads();
    
    for (int i = tid; i < 64 * 128 / 8; i += 128) {
        int row = i / 16;
        int col = (i % 16) * 8;
        int g_q = q_start + row;
        if (g_q < seq_len) {
            *(uint4*)&o_ptr[g_q * 128 + col] = *(uint4*)((char*)Q_smem + row * 256 + col * 2);
        }
    }
    
    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    const void* q_data = Q.data_ptr();
    const void* k_data = K.data_ptr();
    const void* v_data = V.data_ptr();
    __nv_bfloat16* o_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_data = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, (void*)q_data, 128, S, H, B, 128, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, (void*)k_data, 128, S, H, B, 128, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, (void*)v_data, 128, S, H, B, 128, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    
    int64_t threads = 128;
    dim3 blocks((S + 63) / 64, H, B);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_size = 57360;
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_fwd_kernel<<<blocks, threads, smem_size, stream>>>(tma_Q, tma_K, tma_V, o_data, lse_data, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha