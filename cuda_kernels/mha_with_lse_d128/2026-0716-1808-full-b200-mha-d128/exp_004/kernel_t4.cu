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

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    return d;
}

template <uint32_t M, uint32_t N, uint32_t A_MAJOR, uint32_t B_MAJOR>
__device__ __forceinline__ uint32_t make_instr_desc_fn() {
    uint32_t d = 0;
    d |= (1u << 4);     
    d |= (1u << 7);     
    d |= (1u << 10);    
    d |= (A_MAJOR << 15);   
    d |= (B_MAJOR << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint64_t desc_k_major_128b(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t desc_mn_major_128b(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 16384, 1024);
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t swizzle_128B_bf16(uint32_t row, uint32_t col) {
    uint32_t chunk_idx = col / 8;
    uint32_t swizzled_chunk_idx = (row % 8) ^ chunk_idx;
    return swizzled_chunk_idx * 8 + (col % 8);
}

__global__ __launch_bounds__(128, 1) void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_gmem,
    float* LSE_gmem,
    uint32_t S,
    uint32_t stride_O,
    uint32_t stride_BH)
{
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_Q_1 = smem_Q_0 + 128 * 64;
    __nv_bfloat16* smem_K_0 = smem_Q_1 + 128 * 64;
    __nv_bfloat16* smem_K_1 = smem_K_0 + 128 * 64;
    __nv_bfloat16* smem_V_0 = smem_K_1 + 128 * 64;
    __nv_bfloat16* smem_V_1 = smem_V_0 + 128 * 64;
    __nv_bfloat16* smem_P_0 = smem_V_1 + 128 * 64;
    __nv_bfloat16* smem_P_1 = smem_P_0 + 128 * 64;
    
    uint64_t* mbar = (uint64_t*)(smem_P_1 + 128 * 64); 
    
    __shared__ __align__(4) uint32_t tmem_c_pool[2];

    uint32_t tid = threadIdx.x;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_c_pool[0], 128);
        tmem_alloc_fn(&tmem_c_pool[1], 128);
    }
    __syncthreads();
    
    uint32_t tmem_c_0 = tmem_c_pool[0];
    uint32_t tmem_c_1 = tmem_c_pool[1];

    uint32_t s_block = blockIdx.x / 2;
    uint32_t ctaid = blockIdx.x % 2;
    uint32_t bh_idx = blockIdx.y;
    uint32_t q_offset = s_block * 128 + ctaid * 128;

    if (tid == 0) {
        init_smem_barrier_fn(mbar, 1);
    }
    __syncthreads();

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 2 * 128 * 64 * sizeof(__nv_bfloat16));
        tma_load_3d_fn(&tma_Q, mbar, smem_Q_0, 0, q_offset, bh_idx);
        tma_load_3d_fn(&tma_Q, mbar, smem_Q_1, 64, q_offset, bh_idx);
    }
    uint32_t phase = 0;
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;

    float global_max = -INFINITY;
    float global_sum = 0.0f;

    uint32_t idesc_QK = make_instr_desc_fn<128, 128, 0, 0>();
    uint32_t idesc_PV = make_instr_desc_fn<128, 64, 0, 1>();

    float O_0[64] = {0};
    float O_1[64] = {0};
    float scale = 0.08838834764f; // 1 / sqrt(128)

    for (uint32_t ks_offset = 0; ks_offset < S; ks_offset += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 4 * 128 * 64 * sizeof(__nv_bfloat16));
            tma_load_3d_fn(&tma_K, mbar, smem_K_0, 0, ks_offset, bh_idx);
            tma_load_3d_fn(&tma_K, mbar, smem_K_1, 64, ks_offset, bh_idx);
            tma_load_3d_fn(&tma_V, mbar, smem_V_0, 0, ks_offset, bh_idx);
            tma_load_3d_fn(&tma_V, mbar, smem_V_1, 64, ks_offset, bh_idx);
        }
        
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        fence_proxy_async_fn();

        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t d_Q0 = desc_k_major_128b((char*)smem_Q_0 + k * 2);
                uint64_t d_K0 = desc_k_major_128b((char*)smem_K_0 + k * 2);
                umma_f16_cg2_fn(tmem_c_0, d_Q0, d_K0, idesc_QK, (k == 0) ? 0 : 1);
                
                uint64_t d_Q1 = desc_k_major_128b((char*)smem_Q_1 + k * 2);
                uint64_t d_K1 = desc_k_major_128b((char*)smem_K_1 + k * 2);
                umma_f16_cg2_fn(tmem_c_0, d_Q1, d_K1, idesc_QK, 1);
            }
        }
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        float S_vals_0[64], S_vals_1[64];
        for (int col = 0; col < 64; col += 4) {
            uint32_t r[4];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((tid * 128 + col) + tmem_c_0));
            S_vals_0[col] = __uint_as_float(r[0]);
            S_vals_0[col+1] = __uint_as_float(r[1]);
            S_vals_0[col+2] = __uint_as_float(r[2]);
            S_vals_0[col+3] = __uint_as_float(r[3]);
            
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((tid * 128 + col + 64) + tmem_c_0));
            S_vals_1[col] = __uint_as_float(r[0]);
            S_vals_1[col+1] = __uint_as_float(r[1]);
            S_vals_1[col+2] = __uint_as_float(r[2]);
            S_vals_1[col+3] = __uint_as_float(r[3]);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float row_max = -INFINITY;
        for (int i = 0; i < 64; ++i) {
            if (ks_offset + i >= S) S_vals_0[i] = -INFINITY;
            else row_max = fmaxf(row_max, S_vals_0[i] * scale);
        }
        for (int i = 0; i < 64; ++i) {
            if (ks_offset + i + 64 >= S) S_vals_1[i] = -INFINITY;
            else row_max = fmaxf(row_max, S_vals_1[i] * scale);
        }
        
        float new_max = fmaxf(global_max, row_max);
        float rescale = fast_exp2f_fn((global_max - new_max) * 1.44269504089f);
        
        if (new_max - global_max < 0.001f && global_max > -INFINITY) {
            rescale = 1.0f;
            new_max = global_max;
        }
        
        global_sum *= rescale;
        for (int i = 0; i < 64; ++i) {
            O_0[i] *= rescale;
            O_1[i] *= rescale;
        }
        
        float row_sum = 0.0f;
        for (int i = 0; i < 64; ++i) {
            if (ks_offset + i >= S) {
                S_vals_0[i] = 0.0f;
            } else {
                S_vals_0[i] = fast_exp2f_fn((S_vals_0[i] * scale - new_max) * 1.44269504089f);
            }
            row_sum += S_vals_0[i];
        }
        for (int i = 0; i < 64; ++i) {
            if (ks_offset + i + 64 >= S) {
                S_vals_1[i] = 0.0f;
            } else {
                S_vals_1[i] = fast_exp2f_fn((S_vals_1[i] * scale - new_max) * 1.44269504089f);
            }
            row_sum += S_vals_1[i];
        }
        global_sum += row_sum;
        global_max = new_max;
        
        __syncthreads(); 
        
        for (int k = 0; k < 64; ++k) {
            uint32_t swizzled_k = swizzle_128B_bf16(tid, k);
            if (ks_offset + k >= S || q_offset + tid >= S) {
                smem_P_0[tid * 64 + swizzled_k] = __float2bfloat16(0.0f);
            } else {
                smem_P_0[tid * 64 + swizzled_k] = __float2bfloat16(S_vals_0[k]);
            }
            
            if (ks_offset + k + 64 >= S || q_offset + tid >= S) {
                smem_P_1[tid * 64 + swizzled_k] = __float2bfloat16(0.0f);
            } else {
                smem_P_1[tid * 64 + swizzled_k] = __float2bfloat16(S_vals_1[k]);
            }
        }
        
        __syncthreads();
        fence_proxy_async_fn();

        if (tid == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint64_t d_P = (k < 64) ? 
                    desc_k_major_128b((char*)smem_P_0 + k * 2) :
                    desc_k_major_128b((char*)smem_P_1 + (k - 64) * 2);
                    
                uint64_t d_V0 = desc_mn_major_128b((char*)smem_V_0 + k * 128);
                uint64_t d_V1 = desc_mn_major_128b((char*)smem_V_1 + k * 128);
                
                umma_f16_cg2_fn(tmem_c_1, d_P, d_V0, idesc_PV, (k == 0) ? 0 : 1);
                umma_f16_cg2_fn(tmem_c_1 + 64, d_P, d_V1, idesc_PV, (k == 0) ? 0 : 1);
            }
        }
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        __syncthreads();
    }

    float final_O_0[64], final_O_1[64];
    for (int col = 0; col < 64; col += 4) {
        uint32_t r[4];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((tid * 128 + col) + tmem_c_1));
        final_O_0[col] = __uint_as_float(r[0]);
        final_O_0[col+1] = __uint_as_float(r[1]);
        final_O_0[col+2] = __uint_as_float(r[2]);
        final_O_0[col+3] = __uint_as_float(r[3]);
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((tid * 128 + col + 64) + tmem_c_1));
        final_O_1[col] = __uint_as_float(r[0]);
        final_O_1[col+1] = __uint_as_float(r[1]);
        final_O_1[col+2] = __uint_as_float(r[2]);
        final_O_1[col+3] = __uint_as_float(r[3]);
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    __nv_bfloat16* O_gmem_bh = O_gmem + bh_idx * stride_BH;
    for (int col = 0; col < 64; col += 2) {
        float o0_0 = final_O_0[col] / global_sum;
        float o0_1 = final_O_0[col+1] / global_sum;
        __nv_bfloat16 o_0[2];
        o_0[0] = __float2bfloat16(o0_0);
        o_0[1] = __float2bfloat16(o0_1);
        
        float o1_0 = final_O_1[col] / global_sum;
        float o1_1 = final_O_1[col+1] / global_sum;
        __nv_bfloat16 o_1[2];
        o_1[0] = __float2bfloat16(o1_0);
        o_1[1] = __float2bfloat16(o1_1);
        
        uint32_t row_idx = q_offset + tid;
        if (row_idx < S) {
            *(uint32_t*)&O_gmem_bh[row_idx * stride_O + col] = *(uint32_t*)&o_0;
            *(uint32_t*)&O_gmem_bh[row_idx * stride_O + col + 64] = *(uint32_t*)&o_1;
        }
    }

    uint32_t row_idx = q_offset + tid;
    if (row_idx < S) {
        float* LSE_ptr = LSE_gmem + bh_idx * S + row_idx;
        *LSE_ptr = logf(global_sum) + global_max;
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_c_pool[0], 128);
        tmem_dealloc_fn(tmem_c_pool[1], 128);
    }
    __syncthreads();
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3); 

    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B * H, 64, 128, 1);
    create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B * H, 64, 128, 1);
    create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B * H, 64, 128, 1);

    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    dim3 grid(((S + 127) / 128) * 2, B * H, 1);
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 132000));

    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_kernel, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S, D, S * D));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda