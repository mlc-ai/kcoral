#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

#define DRV_CHECK(call) do {                                       \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_name;                                      \
        cuGetErrorName(_e, &err_name);                             \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_name, __FILE__, __LINE__);                     \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (0u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_PV_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (1u << 16);  // b_major = 1
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint32_t swizzle_128B_64(uint32_t m, uint32_t k) {
    uint32_t chunk = k / 8;
    uint32_t offset = k % 8;
    uint32_t swizzled_chunk = chunk ^ (m % 8);
    return swizzled_chunk * 8 + offset;
}

__device__ __forceinline__ void write_P_smem(__nv_bfloat16* smem_P_part, int row, int col_in_part, float p0, float p1, float p2, float p3) {
    uint32_t b01 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
    uint32_t b23 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
    uint32_t sk0 = swizzle_128B_64(row, col_in_part);
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_P_part + row * 64 + sk0);
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(addr), "r"(b01), "r"(b23) : "memory");
}

__global__ void __launch_bounds__(128, 1) mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    void* __restrict__ O_out, 
    float* __restrict__ LSE_out, 
    int S_seq, 
    int H) 
{
    setmaxnreg_inc_sync_fn<240>();
    int batch_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int m = blockIdx.x * 128; 

    if (m >= S_seq) return;

    __shared__ __align__(1024) __nv_bfloat16 smem_Q[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_K[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_V[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_P[2][128 * 64];
    
    __shared__ __align__(8) uint64_t mbar_Q;
    __shared__ __align__(8) uint64_t mbar_K;
    __shared__ __align__(8) uint64_t mbar_V;
    __shared__ __align__(8) uint64_t mbar_umma_S;
    __shared__ __align__(8) uint64_t mbar_umma_PV;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_Q, 1);
        init_smem_barrier_fn(&mbar_K, 1);
        init_smem_barrier_fn(&mbar_V, 1);
        init_smem_barrier_fn(&mbar_umma_S, 1);
        init_smem_barrier_fn(&mbar_umma_PV, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_umma_S = 0, phase_umma_PV = 0;

    __shared__ uint32_t tmem_base;
    tmem_alloc_cg1_fn(&tmem_base, 256);

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q, 32768);
        tma_load_4d_fn(&tma_Q, &mbar_Q, smem_Q[0], 0, m, head_idx, batch_idx);
        tma_load_4d_fn(&tma_Q, &mbar_Q, smem_Q[1], 64, m, head_idx, batch_idx);
    }
    mbarrier_wait_fn(&mbar_Q, phase_Q);
    phase_Q ^= 1;

    float O_reg[128];
    for (int i = 0; i < 128; ++i) O_reg[i] = 0.0f;
    float m_reg = -1e20f;
    float sum_exp = 0.0f;

    uint32_t idesc_S = make_instr_desc_fn(128, 128);
    uint32_t idesc_PV = make_instr_desc_PV_fn(128, 64);

    for (int n = 0; n < S_seq; n += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K, 32768);
            tma_load_4d_fn(&tma_K, &mbar_K, smem_K[0], 0, n, head_idx, batch_idx);
            tma_load_4d_fn(&tma_K, &mbar_K, smem_K[1], 64, n, head_idx, batch_idx);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V, 32768);
            tma_load_4d_fn(&tma_V, &mbar_V, smem_V[0], 0, n, head_idx, batch_idx);
            tma_load_4d_fn(&tma_V, &mbar_V, smem_V[1], 64, n, head_idx, batch_idx);
        }
        mbarrier_wait_fn(&mbar_K, phase_K);
        phase_K ^= 1;
        
        if (threadIdx.x == 0) {
            for (int i = 0; i < 4; ++i) {
                uint64_t d_Q = make_smem_desc_sm100_fn(smem_Q[0] + i * 16, 1, 1024);
                uint64_t d_K = make_smem_desc_sm100_fn(smem_K[0] + i * 16, 1, 1024);
                uint32_t accum = (i == 0) ? 0 : 1;
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;"
                             :: "r"(tmem_base), "l"(d_Q), "l"(d_K), "r"(idesc_S), "r"(accum));
            }
            for (int i = 0; i < 4; ++i) {
                uint64_t d_Q = make_smem_desc_sm100_fn(smem_Q[1] + i * 16, 1, 1024);
                uint64_t d_K = make_smem_desc_sm100_fn(smem_K[1] + i * 16, 1, 1024);
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;"
                             :: "r"(tmem_base), "l"(d_Q), "l"(d_K), "r"(idesc_S), "r"(1));
            }
            umma_commit_cg1_fn(&mbar_umma_S);
        }
        mbarrier_wait_fn(&mbar_umma_S, phase_umma_S);
        phase_umma_S ^= 1;
        
        float m_new = m_reg;
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_base + col));
            float f0 = (n + col + 0 >= S_seq) ? -1e20f : (__uint_as_float(r0) * 0.08838834764f);
            float f1 = (n + col + 1 >= S_seq) ? -1e20f : (__uint_as_float(r1) * 0.08838834764f);
            float f2 = (n + col + 2 >= S_seq) ? -1e20f : (__uint_as_float(r2) * 0.08838834764f);
            float f3 = (n + col + 3 >= S_seq) ? -1e20f : (__uint_as_float(r3) * 0.08838834764f);
            m_new = fmaxf(m_new, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }
        tmem_load_fence_fn();
        
        float scale = fast_exp2f_fn((m_reg - m_new) * 1.44269504f);
        sum_exp *= scale;
        for (int i = 0; i < 128; ++i) O_reg[i] *= scale;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_base + col));
            float f0 = (n + col + 0 >= S_seq) ? -1e20f : (__uint_as_float(r0) * 0.08838834764f);
            float f1 = (n + col + 1 >= S_seq) ? -1e20f : (__uint_as_float(r1) * 0.08838834764f);
            float f2 = (n + col + 2 >= S_seq) ? -1e20f : (__uint_as_float(r2) * 0.08838834764f);
            float f3 = (n + col + 3 >= S_seq) ? -1e20f : (__uint_as_float(r3) * 0.08838834764f);
            
            float p0 = fast_exp2f_fn((f0 - m_new) * 1.44269504f);
            float p1 = fast_exp2f_fn((f1 - m_new) * 1.44269504f);
            float p2 = fast_exp2f_fn((f2 - m_new) * 1.44269504f);
            float p3 = fast_exp2f_fn((f3 - m_new) * 1.44269504f);
            
            sum_exp += p0 + p1 + p2 + p3;
            
            int part = (col < 64) ? 0 : 1;
            int col_in_part = col % 64;
            write_P_smem(smem_P[part], threadIdx.x, col_in_part, p0, p1, p2, p3);
        }
        tmem_load_fence_fn();
        m_reg = m_new;
        
        fence_proxy_async_fn();
        __syncthreads(); 
        
        mbarrier_wait_fn(&mbar_V, phase_V);
        phase_V ^= 1;
        
        if (threadIdx.x == 0) {
            for (int i = 0; i < 4; ++i) {
                uint64_t d_V = make_smem_desc_sm100_fn(smem_V[0] + i * 1024, 1, 1024);
                uint64_t d_P = make_smem_desc_sm100_fn(smem_P[0] + i * 16, 1, 1024);
                uint32_t accum = (i == 0) ? 0 : 1;
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;"
                             :: "r"(tmem_base + 128), "l"(d_P), "l"(d_V), "r"(idesc_PV), "r"(accum));
            }
            for (int i = 0; i < 4; ++i) {
                uint64_t d_V = make_smem_desc_sm100_fn(smem_V[1] + i * 1024, 1, 1024);
                uint64_t d_P = make_smem_desc_sm100_fn(smem_P[1] + i * 16, 1, 1024);
                uint32_t accum = (i == 0) ? 0 : 1;
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;"
                             :: "r"(tmem_base + 128 + 64), "l"(d_P), "l"(d_V), "r"(idesc_PV), "r"(accum));
            }
            umma_commit_cg1_fn(&mbar_umma_PV);
        }
        mbarrier_wait_fn(&mbar_umma_PV, phase_umma_PV);
        phase_umma_PV ^= 1;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_base + 128 + col));
            O_reg[col + 0] += __uint_as_float(r0);
            O_reg[col + 1] += __uint_as_float(r1);
            O_reg[col + 2] += __uint_as_float(r2);
            O_reg[col + 3] += __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        
        __syncthreads();
    }

    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < 128; ++i) {
        O_reg[i] *= inv_sum;
    }

    if (m + threadIdx.x < S_seq) {
        uint64_t row_idx = (uint64_t)batch_idx * H * S_seq * 128 + (uint64_t)head_idx * S_seq * 128 + (m + threadIdx.x) * 128;
        __nv_bfloat16* O_ptr = (__nv_bfloat16*)O_out + row_idx;
        for (int i = 0; i < 128; i += 2) {
            uint32_t packed = pack_bf16_fn(__float_as_uint(O_reg[i]), __float_as_uint(O_reg[i+1]));
            *(uint32_t*)(O_ptr + i) = packed;
        }
        
        uint64_t lse_idx = (uint64_t)batch_idx * H * S_seq + (uint64_t)head_idx * S_seq + (m + threadIdx.x);
        LSE_out[lse_idx] = m_reg + logf(sum_exp);
    }

    tmem_dealloc_cg1_fn(tmem_base, 256);
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
    uint32_t box0, uint32_t box1) 
{
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    void* o_ptr = O.data_ptr();
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_Q, q_ptr, D, S, H, B, 64, 128));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_K, k_ptr, D, S, H, B, 64, 128));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_V, v_ptr, D, S, H, B, 64, 128));

    dim3 grid((S + 127) / 128, B, H);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Pass TMA descriptors as values (mapped to __grid_constant__ correctly in signature)
    mha_fwd_kernel<<<grid, block, 0, stream>>>(tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S, H);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda