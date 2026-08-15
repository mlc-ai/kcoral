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
    return make_smem_desc_sm100_fn(smem_ptr, 16, 1024);
}

__device__ __forceinline__ uint64_t desc_mn_major_128b(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 16384, 1024);
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void read_row_128(uint32_t tmem_addr, uint32_t row, float (&vals)[128]) {
    uint32_t r[8];
    uint32_t col_base = (tmem_addr & 0xFFFF) | ((row << 16) & 0xFFFF0000);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
          "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(col_base));
    
    for(int i = 0; i < 8; ++i) {
        float4 tmp = __cvt_u32_to_float4(r[i]);
        vals[i*4 + 0] = tmp.x;
        vals[i*4 + 1] = tmp.y;
        vals[i*4 + 2] = tmp.z;
        vals[i*4 + 3] = tmp.w;
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void read_row_64(uint32_t tmem_addr, uint32_t row, float (&vals)[64]) {
    uint32_t r[8];
    uint32_t col_base = (tmem_addr & 0xFFFF) | ((row << 16) & 0xFFFF0000);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
          "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(col_base));
    
    for(int i = 0; i < 8; ++i) {
        float4 tmp = __cvt_u32_to_float4(r[i]);
        vals[i*4 + 0] = tmp.x;
        vals[i*4 + 1] = tmp.y;
        vals[i*4 + 2] = tmp.z;
        vals[i*4 + 3] = tmp.w;
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
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
    uint32_t stride_LSE)
{
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_Q_1 = smem_Q_0 + 128 * 64;
    __nv_bfloat16* smem_K_0 = smem_Q_1 + 128 * 64;
    __nv_bfloat16* smem_K_1 = smem_K_0 + 128 * 64;
    __nv_bfloat16* smem_V_0 = smem_K_1 + 128 * 64;
    __nv_bfloat16* smem_V_1 = smem_V_0 + 128 * 64;
    __nv_bfloat16* smem_P   = smem_V_1 + 128 * 64;

    uint32_t tmem_c_0, tmem_c_1_0, tmem_c_1_1;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_c_0, 128);
        tmem_alloc_fn(&tmem_c_1_0, 64);
        tmem_alloc_fn(&tmem_c_1_1, 64);
    }
    __syncthreads();

    uint32_t s_block = blockIdx.x;
    uint32_t bh_idx = blockIdx.y;
    uint32_t s_offset = s_block * 128;
    uint32_t q_offset = s_offset + (cluster_rank() * 128);

    uint64_t* mbar = (uint64_t*)(smem_P + 128 * 128); 
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 2 * 128 * 64 * sizeof(__nv_bfloat16));
        tma_load_3d_fn(&tma_Q, mbar, smem_Q_0, 0, q_offset, bh_idx);
        tma_load_3d_fn(&tma_Q, mbar, smem_Q_1, 64, q_offset, bh_idx);
    }
    uint32_t phase = 0;
    mbarrier_wait_fn(mbar, phase);

    float global_max = -INFINITY;
    float global_sum = 0.0f;

    uint32_t idesc_QK = make_instr_desc_fn<256, 128, 0, 0>();
    uint32_t idesc_PV = make_instr_desc_fn<256, 64, 0, 1>();

    for (uint32_t ks_offset = 0; ks_offset < S; ks_offset += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 4 * 128 * 64 * sizeof(__nv_bfloat16));
            tma_load_3d_fn(&tma_K, mbar, smem_K_0, 0, ks_offset, bh_idx);
            tma_load_3d_fn(&tma_K, mbar, smem_K_1, 64, ks_offset, bh_idx);
            tma_load_3d_fn(&tma_V, mbar, smem_V_0, 0, ks_offset, bh_idx);
            tma_load_3d_fn(&tma_V, mbar, smem_V_1, 64, ks_offset, bh_idx);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t d_Q0 = make_smem_desc_sm100_fn((char*)smem_Q_0 + k * 2, 16, 1024);
                uint64_t d_K0 = make_smem_desc_sm100_fn((char*)smem_K_0 + k * 2, 16, 1024);
                umma_f16_cg2_fn(tmem_c_0, d_Q0, d_K0, idesc_QK, (k == 0) ? 0 : 1);
                
                uint64_t d_Q1 = make_smem_desc_sm100_fn((char*)smem_Q_1 + k * 2, 16, 1024);
                uint64_t d_K1 = make_smem_desc_sm100_fn((char*)smem_K_1 + k * 2, 16, 1024);
                umma_f16_cg2_fn(tmem_c_0, d_Q1, d_K1, idesc_QK, 1);
            }
        }
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        float S_vals[128];
        read_row_128(tmem_c_0, threadIdx.x, S_vals);
        
        float local_max = -INFINITY;
        for (int i = 0; i < 128; ++i) {
            if (ks_offset + i >= S) S_vals[i] = -INFINITY;
            else local_max = fmaxf(local_max, S_vals[i]);
        }
        
        if (local_max > global_max) {
            global_max = local_max;
        }
        
        for (int i = 0; i < 128; ++i) {
            S_vals[i] = fast_exp2f_fn((S_vals[i] - global_max) * 1.44269504089f);
        }
        
        float local_sum = 0.0f;
        for (int i = 0; i < 128; ++i) {
            local_sum += S_vals[i];
        }
        global_sum += local_sum;
        
        if (q_offset + threadIdx.x < S) {
            for (int i = 0; i < 128; i += 2) {
                __nv_bfloat16 p[2];
                p[0] = __float2bfloat16(S_vals[i]);
                p[1] = __float2bfloat16(S_vals[i+1]);
                *(uint32_t*)&smem_P[(q_offset + threadIdx.x - s_offset) * 128 + i] = *(uint32_t*)&p;
            }
        } else {
            for (int i = 0; i < 128; i += 2) {
                smem_P[(q_offset + threadIdx.x - s_offset) * 128 + i] = __float2bfloat16(0.0f);
            }
        }
        
        __syncthreads(); 
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint64_t d_P = make_smem_desc_sm100_fn((char*)smem_P + k * 2, 16, 1024);
                uint64_t d_V0 = make_smem_desc_sm100_fn((char*)smem_V_0 + k * 128, 16384, 1024);
                uint64_t d_V1 = make_smem_desc_sm100_fn((char*)smem_V_1 + k * 128, 16384, 1024);
                umma_f16_cg2_fn(tmem_c_1_0, d_P, d_V0, idesc_PV, 1);
                umma_f16_cg2_fn(tmem_c_1_1, d_V1, idesc_PV, 1);
            }
        }
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
    }

    float O_0[64], O_1[64];
    read_row_64(tmem_c_1_0, threadIdx.x, O_0);
    read_row_64(tmem_c_1_1, threadIdx.x, O_1);

    for (int i = 0; i < 64; i += 2) {
        __nv_bfloat16 o_0[2], o_1[2];
        o_0[0] = __float2bfloat16(O_0[i] / global_sum);
        o_0[1] = __float2bfloat16(O_0[i+1] / global_sum);
        
        o_1[0] = __float2bfloat16(O_1[i] / global_sum);
        o_1[1] = __float2bfloat16(O_1[i+1] / global_sum);
        
        uint32_t row_idx = q_offset + threadIdx.x;
        if (row_idx < S) {
            *(uint32_t*)&O_gmem[row_idx * stride_O + s_block * 128 + i] = *(uint32_t*)&o_0;
            *(uint32_t*)&O_gmem[row_idx * stride_O + s_block * 128 + i + 64] = *(uint32_t*)&o_1;
        }
    }

    uint32_t row_idx = q_offset + threadIdx.x;
    if (row_idx < S) {
        float* LSE_ptr = LSE_gmem + bh_idx * S + row_idx;
        *LSE_ptr = logf(global_sum) + global_max;
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_c_0, 128);
        tmem_dealloc_fn(tmem_c_1_0, 64);
        tmem_dealloc_fn(tmem_c_1_1, 64);
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

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 230000));

    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_kernel, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S, D, 1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda