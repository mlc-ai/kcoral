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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
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
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint64_t make_smem_desc_linear_fn(void* smem_ptr, uint32_t sbo, uint32_t lbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transA, bool transB) {
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

__device__ __forceinline__ float ex2_mufu_fn(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ __launch_bounds__(128) void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE_ptr,
    int S
) {
    setmaxnreg_inc_sync_fn<256>();

    int b_h = blockIdx.y;
    int q_start = blockIdx.x * 128;
    int tid = threadIdx.x; 

    uint8_t* smem_Q = smem_pool;
    uint8_t* smem_K = smem_pool + 32768;
    uint8_t* smem_V = smem_pool + 65536;
    uint8_t* smem_P = smem_pool + 98304;
    uint8_t* smem_O_out = smem_P; 

    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 131072);
    uint64_t* mbar_K = (uint64_t*)(smem_pool + 131080);
    uint64_t* mbar_V = (uint64_t*)(smem_pool + 131088);
    uint64_t* mbar_P = (uint64_t*)(smem_pool + 131096);
    uint64_t* mbar_O_mma = (uint64_t*)(smem_pool + 131104);

    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_P, 1);
        init_smem_barrier_fn(mbar_O_mma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q, 0, q_start, b_h);
    }
    mbarrier_wait_fn(mbar_Q, 0);

    float m_i = -1e20f;
    float l_i = 0.0f;
    float O_acc[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) O_acc[i] = 0.0f;

    float scale_log2 = (1.0f / sqrtf(128.0f)) * 1.4426950408889634f;

    int phase_K = 0;
    int phase_P = 0;
    int phase_O_mma = 0;

    for (int kv_start = 0; kv_start < S; kv_start += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 32768);
            tma_load_3d_fn(&tma_K, mbar_K, smem_K, 0, kv_start, b_h);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 32768);
            tma_load_3d_fn(&tma_V, mbar_V, smem_V, 0, kv_start, b_h);
        }

        uint32_t tmem_P;
        if (tid == 0) tmem_alloc_cg1_fn(&tmem_P, 128);
        __syncthreads();

        mbarrier_wait_fn(mbar_K, phase_K);
        mbarrier_wait_fn(mbar_V, phase_K);

        if (tid == 0) {
            uint32_t idesc = make_instr_desc_fn(128, 128, false, false);
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_A = make_smem_desc_linear_fn(smem_Q + k * 2, 2048, 16); 
                uint64_t desc_B = make_smem_desc_linear_fn(smem_K + k * 2, 2048, 16);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_P, desc_A, desc_B, idesc, accum);
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_P)));
        }
        mbarrier_wait_fn(mbar_P, phase_P);
        phase_P ^= 1;

        float row_max = -1e20f;
        #pragma unroll
        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float val = __uint_as_float(r[i]) * scale_log2;
                int k_idx = kv_start + c + i;
                if (k_idx >= S) val = -1e20f;
                row_max = fmaxf(row_max, val);
            }
        }
        
        float m_new = fmaxf(m_i, row_max);
        float rescale = ex2_mufu_fn(m_i - m_new);
        
        #pragma unroll
        for (int c = 0; c < 128; ++c) {
            O_acc[c] *= rescale;
        }
        l_i *= rescale;
        
        #pragma unroll
        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            uint32_t packed[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float v0 = __uint_as_float(r[2*i]) * scale_log2;
                float v1 = __uint_as_float(r[2*i+1]) * scale_log2;
                int k_idx0 = kv_start + c + 2*i;
                int k_idx1 = kv_start + c + 2*i + 1;
                if (k_idx0 >= S) v0 = -1e20f;
                if (k_idx1 >= S) v1 = -1e20f;
                v0 = ex2_mufu_fn(v0 - m_new);
                v1 = ex2_mufu_fn(v1 - m_new);
                l_i += v0 + v1;
                packed[i] = pack_bf16_fn(*(uint32_t*)&v0, *(uint32_t*)&v1);
            }
            uint32_t* p_row = (uint32_t*)(smem_P + tid * 256);
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(&p_row[c / 2]), packed[0], packed[1], packed[2], packed[3]);
        }
        m_i = m_new;

        __syncthreads();
        if (tid == 0) tmem_dealloc_cg1_fn(tmem_P, 128);
        fence_async_shared_fn(); 
        
        uint32_t tmem_O_new;
        if (tid == 0) tmem_alloc_cg1_fn(&tmem_O_new, 128);
        __syncthreads();

        if (tid == 0) {
            uint32_t idesc = make_instr_desc_fn(128, 128, false, true); 
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_A = make_smem_desc_linear_fn(smem_P + k * 2, 2048, 16); 
                uint64_t desc_B = make_smem_desc_linear_fn(smem_V + k * 256, 16, 2048);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_new, desc_A, desc_B, idesc, accum);
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_O_mma)));
        }
        mbarrier_wait_fn(mbar_O_mma, phase_O_mma);
        phase_O_mma ^= 1;
        
        #pragma unroll
        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            O_acc[c+0] += __uint_as_float(r[0]);
            O_acc[c+1] += __uint_as_float(r[1]);
            O_acc[c+2] += __uint_as_float(r[2]);
            O_acc[c+3] += __uint_as_float(r[3]);
            O_acc[c+4] += __uint_as_float(r[4]);
            O_acc[c+5] += __uint_as_float(r[5]);
            O_acc[c+6] += __uint_as_float(r[6]);
            O_acc[c+7] += __uint_as_float(r[7]);
        }

        __syncthreads();
        if (tid == 0) tmem_dealloc_cg1_fn(tmem_O_new, 128);
        __syncthreads();
        phase_K ^= 1;
    }

    #pragma unroll
    for (int c = 0; c < 128; ++c) {
        O_acc[c] /= l_i;
    }

    __syncthreads();
    #pragma unroll
    for (int c = 0; c < 128; c += 8) {
        uint32_t packed[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            packed[i] = pack_bf16_fn(*(uint32_t*)&O_acc[c + 2*i], *(uint32_t*)&O_acc[c + 2*i + 1]);
        }
        uint32_t* p_row = (uint32_t*)(smem_O_out + tid * 256);
        st_shared_128_fn((uint32_t)__cvta_generic_to_shared(&p_row[c / 2]), packed[0], packed[1], packed[2], packed[3]);
    }
    __syncthreads();
    tma_store_fence_fn(); 

    if (tid == 0) {
        tma_store_3d_fn(&tma_O, smem_O_out, 0, q_start, b_h);
        tma_store_commit_fn();
    }
    tma_store_wait_fn<0>();

    int q_idx = q_start + tid;
    if (q_idx < S) {
        float lse_val = m_i * 0.6931471805599453f + logf(l_i); 
        LSE_ptr[b_h * S + q_idx] = lse_val;
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2,
                                     uint32_t box0, uint32_t box1, uint32_t box2) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B * H, 128, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B * H, 128, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B * H, 128, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_O, O.data_ptr(), D, S, B * H, 128, 128, 1));
    
    int num_blocks = (S + 127) / 128;
    dim3 grid(num_blocks, B * H);
    dim3 block(128);
    
    size_t smem_size = 4 * 32768 + 5 * 8; 
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_fwd_kernel, tma_Q, tma_K, tma_V, tma_O, static_cast<float*>(LSE.data_ptr()), (int)S));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha