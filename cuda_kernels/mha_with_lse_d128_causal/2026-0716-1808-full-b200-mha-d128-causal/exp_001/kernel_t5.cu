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

#define CU_CHECK_DRIVER(call) do {                                 \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n",         \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cluster.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__global__ void __launch_bounds__(128, 1) attention_forward_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int S_len, float scale)
{
    int m_block = blockIdx.x;
    int bh_idx = blockIdx.y;
    int abs_m_start = m_block * 128;
    
    if (abs_m_start >= S_len) return;
    
    extern __shared__ __align__(1024) char smem_pool[];
    char* smem_ptr = smem_pool;
    
    // Aligned 16B swizzled blocks (16384 bytes each -> perfectly 1024 aligned)
    __nv_bfloat16* smem_q0 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    __nv_bfloat16* smem_q1 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    __nv_bfloat16* smem_k0 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    __nv_bfloat16* smem_k1 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    __nv_bfloat16* smem_v0 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    __nv_bfloat16* smem_v1 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    
    // Reuse dead K blocks for P to avoid allocating extra dynamic shared memory space
    __nv_bfloat16* smem_p0 = smem_k0;
    __nv_bfloat16* smem_p1 = smem_k1;
    
    uint64_t* smem_mbarrier = (uint64_t*)smem_ptr; smem_ptr += 16;
    
    uint32_t* tmem_S_ptr = (uint32_t*)smem_ptr; smem_ptr += sizeof(uint32_t);
    uint32_t* tmem_O_ptr_0 = (uint32_t*)smem_ptr; smem_ptr += sizeof(uint32_t);
    uint32_t* tmem_O_ptr_1 = (uint32_t*)smem_ptr; smem_ptr += sizeof(uint32_t);
    
    int tid = threadIdx.x;
    
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_S_ptr, 128);
        tmem_alloc_fn(tmem_O_ptr_0, 64);
        tmem_alloc_fn(tmem_O_ptr_1, 64);
        
        init_smem_barrier_fn(smem_mbarrier, 1);
        mbarrier_arrive_and_expect_tx_fn(smem_mbarrier, 32768); 
        
        tma_load_2d_fn(&tma_Q, smem_mbarrier, smem_q0, 0, bh_idx * S_len + abs_m_start);
        tma_load_2d_fn(&tma_Q, smem_mbarrier, smem_q1, 64, bh_idx * S_len + abs_m_start);
    }
    __syncthreads();
    
    float m_val = -INFINITY;
    float l_val = 0;
    
    uint32_t idesc_qkt = 0;
    idesc_qkt |= (1u << 4);
    idesc_qkt |= (1u << 7);
    idesc_qkt |= (1u << 10);
    idesc_qkt |= ((128 / 8) << 17);     
    idesc_qkt |= ((128 / 16) << 24);    
    
    uint32_t idesc_pv = 0;
    idesc_pv |= (1u << 4);
    idesc_pv |= (1u << 7);
    idesc_pv |= (1u << 10);
    idesc_pv |= (1u << 16); // MN-Major V logic mapping
    idesc_pv |= ((64 / 8) << 17);     
    idesc_pv |= ((128 / 16) << 24);    
    
    uint32_t accum = 1;
    uint32_t phase = 0;
    
    mbarrier_wait_fn(smem_mbarrier, phase);
    phase ^= 1;
    fence_proxy_async_fn();
    
    for (int n_block = 0; n_block <= m_block; n_block++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(smem_mbarrier, 65536);
            
            tma_load_2d_fn(&tma_K, smem_mbarrier, smem_k0, 0, bh_idx * S_len + n_block * 128);
            tma_load_2d_fn(&tma_K, smem_mbarrier, smem_k1, 64, bh_idx * S_len + n_block * 128);
            tma_load_2d_fn(&tma_V, smem_mbarrier, smem_v0, 0, bh_idx * S_len + n_block * 128);
            tma_load_2d_fn(&tma_V, smem_mbarrier, smem_v1, 64, bh_idx * S_len + n_block * 128);
        }
        
        mbarrier_wait_fn(smem_mbarrier, phase);
        phase ^= 1;
        fence_proxy_async_fn();
        
        // FMMA replacing scalar math over entire QK^T inner product dimension (swizzled)
        for (uint32_t k_block = 0; k_block < 64; k_block += 16) {
            uint64_t desc_q0 = make_smem_desc_sm100_fn(smem_q0 + k_block, 1, 1024);
            uint64_t desc_q1 = make_smem_desc_sm100_fn(smem_q1 + k_block, 1, 1024);
            uint64_t desc_k0 = make_smem_desc_sm100_fn(smem_k0 + k_block, 1, 1024);
            uint64_t desc_k1 = make_smem_desc_sm100_fn(smem_k1 + k_block, 1, 1024);
            
            umma_f16_cg1_fn(tmem_S_ptr[0], desc_q0, desc_k0, idesc_qkt, accum);
            umma_f16_cg1_fn(tmem_S_ptr[0], desc_q1, desc_k1, idesc_qkt, accum);
        }
        
        umma_commit_1sm_fn(smem_mbarrier);
        mbarrier_wait_fn(smem_mbarrier, phase);
        phase ^= 1;
        fence_proxy_async_fn();
        
        float s_shared[128];
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_ptr[0] + col, &r0, &r1, &r2, &r3);
            s_shared[tid * 128 + col + 0] = __uint_as_float(r0);
            s_shared[tid * 128 + col + 1] = __uint_as_float(r1);
            s_shared[tid * 128 + col + 2] = __uint_as_float(r2);
            s_shared[tid * 128 + col + 3] = __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        
        int row = tid;
        int abs_row = abs_m_start + row;
        
        float row_max = -INFINITY;
        for (int col = 0; col < 128; col++) {
            int abs_col = n_block * 128 + col;
            if (abs_row < S_len && abs_col <= abs_row && abs_col < S_len) {
                row_max = fmaxf(row_max, s_shared[row * 128 + col] * scale);
            }
        }
        
        float new_m = fmaxf(m_val, row_max);
        float o_s = __expf(m_val - new_m);
        
        float row_sum = 0;
        for (int col = 0; col < 128; col++) {
            int abs_col = n_block * 128 + col;
            float p = 0.0f;
            if (abs_row < S_len && abs_col <= abs_row && abs_col < S_len) {
                p = __expf(s_shared[row * 128 + col] * scale - new_m);
                row_sum += p;
            }
            
            int col_in_64 = col % 64;
            int chunk = col_in_64 / 8;
            int swizzled_chunk = chunk ^ (row % 8);
            int final_col = col_in_64 - (chunk * 8) + (swizzled_chunk * 8);
            
            if (col < 64) {
                smem_p0[row * 64 + final_col] = __float2bfloat16(p);
            } else {
                smem_p1[row * 64 + final_col] = __float2bfloat16(p);
            }
        }
        
        l_val = l_val * o_s + row_sum * __expf(row_max - new_m);
        m_val = new_m;
        
        __syncthreads(); 
        fence_proxy_async_fn();
        
        // Accumulate over full K=128 dimension in 16 step strides contracting against swizzled P and V slices
        for (uint32_t k_block = 0; k_block < 64; k_block += 16) {
            uint64_t desc_p0 = make_smem_desc_sm100_fn(smem_p0 + k_block, 1, 1024);
            uint64_t desc_p1 = make_smem_desc_sm100_fn(smem_p1 + k_block, 1, 1024);
            uint64_t desc_v0 = make_smem_desc_sm100_fn(smem_v0 + k_block * 64, 16384, 1024);
            uint64_t desc_v1 = make_smem_desc_sm100_fn(smem_v1 + k_block * 64, 16384, 1024);
            uint64_t desc_v0_p1 = make_smem_desc_sm100_fn(smem_v0 + (64 + k_block) * 64, 16384, 1024);
            uint64_t desc_v1_p1 = make_smem_desc_sm100_fn(smem_v1 + (64 + k_block) * 64, 16384, 1024);
            
            umma_f16_cg1_fn(tmem_O_ptr_0[0], desc_p0, desc_v0, idesc_pv, accum);
            umma_f16_cg1_fn(tmem_O_ptr_1[0], desc_p0, desc_v1, idesc_pv, accum);
            
            umma_f16_cg1_fn(tmem_O_ptr_0[0], desc_p1, desc_v0_p1, idesc_pv, accum);
            umma_f16_cg1_fn(tmem_O_ptr_1[0], desc_p1, desc_v1_p1, idesc_pv, accum);
        }
        
        umma_commit_1sm_fn(smem_mbarrier);
        mbarrier_wait_fn(smem_mbarrier, phase);
        phase ^= 1;
        fence_proxy_async_fn();
        
        __syncthreads();
    }
    
    tmem_load_fence_fn();
    __syncthreads();
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O_ptr_0[0] + col, &r0, &r1, &r2, &r3);
        
        __nv_bfloat16 out[4];
        out[0] = __float2bfloat16(__uint_as_float(r0) / l_val);
        out[1] = __float2bfloat16(__uint_as_float(r1) / l_val);
        out[2] = __float2bfloat16(__uint_as_float(r2) / l_val);
        out[3] = __float2bfloat16(__uint_as_float(r3) / l_val);
        
        int abs_row = abs_m_start + tid;
        if (abs_row < S_len) {
            *reinterpret_cast<uint2*>(&O[bh_idx * S_len * 128 + abs_row * 128 + col]) = *reinterpret_cast<uint2*>(out);
        }
        
        tmem_load_4x_fn(tmem_O_ptr_1[0] + col, &r0, &r1, &r2, &r3);
        
        out[0] = __float2bfloat16(__uint_as_float(r0) / l_val);
        out[1] = __float2bfloat16(__uint_as_float(r1) / l_val);
        out[2] = __float2bfloat16(__uint_as_float(r2) / l_val);
        out[3] = __float2bfloat16(__uint_as_float(r3) / l_val);
        
        if (abs_row < S_len) {
            *reinterpret_cast<uint2*>(&O[bh_idx * S_len * 128 + abs_row * 128 + col + 64]) = *reinterpret_cast<uint2*>(out);
        }
    }
    
    int abs_row = abs_m_start + tid;
    if (abs_row < S_len) {
        LSE[bh_idx * S_len + abs_row] = m_val + __logf(l_val);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, 
         tvm::ffi::TensorView V, tvm::ffi::TensorView O, 
         tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    
    CU_CHECK_DRIVER(create_tma_2d_descriptor_2B(
        &tma_Q, (void*)Q_ptr, D, B * H * S, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK_DRIVER(create_tma_2d_descriptor_2B(
        &tma_K, (void*)K_ptr, D, B * H * S, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK_DRIVER(create_tma_2d_descriptor_2B(
        &tma_V, (void*)V_ptr, D, B * H * S, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    float scale = 1.0f / std::sqrt((float)D);
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    int smem_size = 140000; 
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(
        attention_forward_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        smem_size));
    
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
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_forward_kernel, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S, scale));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda