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

__device__ __forceinline__ uint64_t make_smem_desc_swizzled(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 16384, 1024);
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void u32_to_float2(uint32_t val, float& f0, float& f1) {
    __nv_bfloat162 b2 = __bfloat1622bfloat162(__ushort2_as_bfloat162(*(uint16_t*)&val));
    f0 = __low2float(b2);
    f1 = __high2float(b2);
}

__device__ __forceinline__ void prefetch_all(uint32_t col, float (&vals)[128]) {
    uint32_t tmem_addr = *(uint32_t*)tmem_c_gmem;
    uint32_t my_row = tid % 128;
    uint32_t r[16];
    uint32_t col_base = (tmem_addr & 0xFFFF) | ((my_row << 16) & 0xFFFF0000);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
          "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
          "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
          "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(col_base));
    
    for(int i = 0; i < 8; ++i) {
        u32_to_float2(r[i], vals[i*2 + 0], vals[i*2 + 1]);
        u32_to_float2(r[i+8], vals[64 + i*2 + 0], vals[64 + i*2 + 1]);
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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

    uint32_t* tmem_c_gmem = (uint32_t*)(smem_V_1 + 128 * 64);
    uint32_t* tmem_c_h_gmem = tmem_c_gmem + 1;

    uint32_t tmem_c_0, tmem_c_1;
    uint32_t tmem_c_0_prev = 0, tmem_c_1_prev = 0;

    if (tid == 0) {
        tmem_alloc_fn(tmem_c_gmem, 128);
        tmem_alloc_fn(tmem_c_h_gmem, 128);
    }
    __syncthreads();
    
    tmem_c_0 = *tmem_c_gmem;
    tmem_c_1 = *tmem_c_h_gmem;

    uint32_t s_block = blockIdx.x;
    uint32_t bh_idx = blockIdx.y;
    uint32_t q_offset = s_block * 128;

    uint64_t* mbar = (uint64_t*)(smem_V_1 + 128 * 64 + 2); 
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar, 1);
    }
    __syncthreads();

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 2 * 128 * 64 * sizeof(__nv_bfloat16));
        tma_load_3d_fn(&tma_Q, mbar, smem_Q_0, 0, q_offset + cluster_rank() * 128 + tid, bh_idx);
        tma_load_3d_fn(&tma_Q, mbar, smem_Q_1, 64, q_offset + cluster_rank() * 128 + tid, bh_idx);
        
        if (0 <= S - 1) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 4 * 128 * 64 * sizeof(__nv_bfloat16));
            tma_load_3d_fn(&tma_K, mbar, smem_K_0, 0, 0, bh_idx);
            tma_load_3d_fn(&tma_K, mbar, smem_K_1, 64, 0, bh_idx);
            tma_load_3d_fn(&tma_V, mbar, smem_V_0, 0, 0, bh_idx);
            tma_load_3d_fn(&tma_V, mbar, smem_V_1, 64, 0, bh_idx);
        }
    }
    uint32_t phase = 0;
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;

    float global_max = -INFINITY;
    float global_sum = 0.0f;
    float exp_global_max = 1.0f;

    uint32_t idesc_QK = make_instr_desc_fn<128, 128, 0, 0>();
    uint32_t idesc_PV = make_instr_desc_fn<128, 64, 0, 1>();

    float O_0[64] = {0};
    float O_1[64] = {0};

    for (uint32_t ks_offset = 0; ks_offset < S; ks_offset += 128) {
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        fence_proxy_async_fn();

        if (tid == 0) {
            if (cluster_rank() == 0) {
                for (int k = 0; k < 64; k += 16) {
                    uint64_t d_Q0 = make_smem_desc_swizzled((char*)smem_Q_0 + k * 2);
                    uint64_t d_K0 = make_smem_desc_swizzled((char*)smem_K_0 + k * 2);
                    umma_f16_cg2_fn(tmem_c_0, d_Q0, d_K0, idesc_QK, (k == 0) ? 0 : 1);
                }
                for (int k = 0; k < 64; k += 16) {
                    uint64_t d_Q0 = make_smem_desc_swizzled((char*)smem_Q_0 + k * 2);
                    uint64_t d_K1 = make_smem_desc_swizzled((char*)smem_K_1 + k * 2);
                    umma_f16_cg2_fn(tmem_c_0, d_Q0, d_K1, idesc_QK, 1);
                }
            } else {
                for (int k = 0; k < 64; k += 16) {
                    uint64_t d_Q1 = make_smem_desc_swizzled((char*)smem_Q_1 + k * 2);
                    uint64_t d_K0 = make_smem_desc_swizzled((char*)smem_K_0 + k * 2);
                    umma_f16_cg2_fn(tmem_c_1, d_Q1, d_K0, idesc_QK, (k == 0) ? 0 : 1);
                }
                for (int k = 0; k < 64; k += 16) {
                    uint64_t d_Q1 = make_smem_desc_swizzled((char*)smem_Q_1 + k * 2);
                    uint64_t d_K1 = make_smem_desc_swizzled((char*)smem_K_1 + k * 2);
                    umma_f16_cg2_fn(tmem_c_1, d_Q1, d_K1, idesc_QK, 1);
                }
            }
        }
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        float S_vals[128];
        for (int col = 0; col < 128; col += 16) {
            prefetch_all(col, S_vals);
        }
        
        float block_max = -INFINITY;
        for (int i = 0; i < 128; ++i) {
            if (ks_offset + i >= S) S_vals[i] = -INFINITY;
            else block_max = fmaxf(block_max, S_vals[i]);
        }
        
        float block_exp_max = fmaxf(exp_global_max, fast_exp2f_fn((block_max - global_max) * 1.44269504089f));
        
        if (block_max > global_max) {
            exp_global_max *= fast_exp2f_fn((global_max - block_max) * 1.44269504089f);
            global_max = block_max;
        } else {
            exp_global_max = block_exp_max;
        }
        
        for (int i = 0; i < 128; ++i) {
            S_vals[i] = fast_exp2f_fn((S_vals[i] - global_max) * 1.44269504089f) * exp_global_max;
        }
        
        float block_sum = 0.0f;
        for (int i = 0; i < 128; ++i) {
            block_sum += S_vals[i];
        }
        global_sum = global_sum * exp_global_max + block_sum;
        
        __syncthreads(); 

        for (int c = 0; c < 128; c += 2) {
            float fc = S_vals[c];
            float fc1 = S_vals[c+1];
            __nv_bfloat16 p[2];
            p[0] = __float2bfloat16(fc);
            p[1] = __float2bfloat16(fc1);
            uint32_t packed_p = *(uint32_t*)&p;
            
            uint32_t swizzled_col_0 = (((tid % 128) & 7) ^ (c / 8)) * 8 + (c & 7);
            uint32_t swizzled_col_1 = swizzled_col_0 + 64;
            __nv_bfloat16* smem_P = smem_Q_0;
            __nv_bfloat16* smem_P_1 = smem_Q_1;
            *(uint32_t*)&smem_P[tid * 64 + swizzled_col_0] = packed_p;
            *(uint32_t*)&smem_P_1[tid * 64 + swizzled_col_1] = packed_p;
        }
        
        __syncthreads();
        fence_proxy_async_fn();

        if (tid == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint64_t d_P = make_smem_desc_swizzled((char*)smem_P + k * 2);
                if (cluster_rank() == 0) {
                    uint64_t d_V0 = make_smem_desc_mn_major((char*)smem_V_0 + k * 128);
                    umma_f16_cg2_fn(tmem_c_0, d_P, d_V0, idesc_PV, (k == 0) ? 0 : 1);
                } else {
                    uint64_t d_V1 = make_smem_desc_mn_major((char*)smem_V_1 + k * 128);
                    umma_f16_cg2_fn(tmem_c_1, d_P, d_V1, idesc_PV, (k == 0) ? 0 : 1);
                }
            }
        }
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (ks_offset + 128 < S && tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 4 * 8192 * 2);
            tma_load_3d_fn(&tma_K, mbar, smem_K_0, 0, ks_offset + 128, bh_idx);
            tma_load_3d_fn(&tma_K, mbar, smem_K_1, 64, ks_offset + 128, bh_idx);
            tma_load_3d_fn(&tma_V, mbar, smem_V_0, 0, ks_offset + 128, bh_idx);
            tma_load_3d_fn(&tma_V, mbar, smem_V_1, 64, ks_offset + 128, bh_idx);
        }

        for (int i = 0; i < 64; ++i) {
            O_0[i] *= exp_global_max;
            O_1[i] *= exp_global_max;
        }
    }

    for (int col = 0; col < 64; col += 16) {
        prefetch_all(col, O_0);
        prefetch_all(col, O_1);
    }

    for (int i = 0; i < 64; i += 2) {
        float o0_0 = O_0[i] / global_sum;
        float o0_1 = O_0[i+1] / global_sum;
        __nv_bfloat16 o_0[2];
        o_0[0] = __float2bfloat16(o0_0);
        o_0[1] = __float2bfloat16(o0_1);
        
        float o1_0 = O_1[i] / global_sum;
        float o1_1 = O_1[i+1] / global_sum;
        __nv_bfloat16 o_1[2];
        o_1[0] = __float2bfloat16(o1_0);
        o_1[1] = __float2bfloat16(o1_1);
        
        uint32_t row_idx = s_block * 128 + tid;
        if (row_idx < S) {
            *(uint32_t*)&O_gmem_bh[row_idx * stride_O + s_block * 128 + i] = *(uint32_t*)&o_0;
            *(uint32_t*)&O_gmem_bh[row_idx * stride_O + s_block * 128 + i + 64] = *(uint32_t*)&o_1;
        }
    }

    uint32_t row_idx = s_block * 128 + tid;
    if (row_idx < S) {
        float* LSE_ptr = LSE_gmem + bh_idx * S + row_idx;
        *LSE_ptr = log2f(global_sum) + global_max * 0.6931471805599453f;
    }

    if (tid == 0) {
        tmem_dealloc_fn(*tmem_c_gmem, 128);
        tmem_dealloc_fn(*tmem_c_h_gmem, 128);
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

    dim3 grid(((S + 127) / 128), B * H, 1);
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

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 100000));

    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_kernel, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S, D, S * D));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda