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

__device__ __forceinline__ void tma_load_2d_cta_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "l"((uint64_t)bar),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_expect_and_load(const CUtensorMap* tma, uint64_t* mbar, void* smem, int c0, int c1) {
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 128 * 128 * 2);
        tma_load_2d_cta_fn(tma, mbar, smem, c0, c1);
    }
}

__device__ __forceinline__ void tma_load_sync(uint64_t* mbar, int& phase) {
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(mbar, phase);
    }
    __syncthreads();
    phase ^= 1;
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint32_t make_idesc(bool a_mn_major, bool b_mn_major, int M, int N) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32 output
    d |= (1u << 7);    // BF16 A
    d |= (1u << 10);   // BF16 B
    d |= ((a_mn_major ? 1u : 0u) << 15);
    d |= ((b_mn_major ? 1u : 0u) << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_K_Major(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t SBO = 128;
    uint32_t LBO = 2048;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_MN_Major(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t LBO = 128;
    uint32_t SBO = 2048;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ void issue_mma_f16(uint32_t tmem_addr, uint64_t desc_A, uint64_t desc_B, uint32_t idesc, int accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_addr), "l"(desc_A), "l"(desc_B), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void mma_128x128x128(uint32_t tmem_addr, void* ptr_A, void* ptr_B, bool a_mn_major, bool b_mn_major, int accum) {
    uint32_t idesc = make_idesc(a_mn_major, b_mn_major, 128, 128);
    for (int k = 0; k < 8; ++k) {
        uint32_t offset_A = a_mn_major ? (k * 4096) : (k * 32);
        uint32_t offset_B = b_mn_major ? (k * 4096) : (k * 32);
        
        uint64_t desc_A = a_mn_major ? make_smem_desc_MN_Major((char*)ptr_A + offset_A) : make_smem_desc_K_Major((char*)ptr_A + offset_A);
        uint64_t desc_B = b_mn_major ? make_smem_desc_MN_Major((char*)ptr_B + offset_B) : make_smem_desc_K_Major((char*)ptr_B + offset_B);
        int acc = (k == 0) ? accum : 1;
        issue_mma_f16(tmem_addr, desc_A, desc_B, idesc, acc);
    }
}

__device__ __forceinline__ void sync_mma(uint64_t* mbar, int& phase) {
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "l"((uint64_t)mbar) : "memory");
        mbarrier_wait_fn(mbar, phase);
    }
    __syncthreads();
    phase ^= 1;
}

__device__ __forceinline__ void compute_PT(uint32_t tmem_S, __nv_bfloat16* smem_PT, const float* smem_LSE, float scale) {
    for (int col = 0; col < 128; col += 8) {
        uint32_t r[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
              "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(tmem_S + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        int smem_idx = threadIdx.x * 128 + col;
        for(int k=0; k<8; ++k) {
            float s = __uint_as_float(r[k]);
            float lse = smem_LSE[col + k];
            float p = expf(s * scale - lse);
            smem_PT[smem_idx + k] = __float2bfloat16(p);
        }
    }
}

__device__ __forceinline__ void compute_dST(uint32_t tmem_dP, const __nv_bfloat16* smem_PT, __nv_bfloat16* smem_dST, const float* smem_D) {
    for (int col = 0; col < 128; col += 8) {
        uint32_t r[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
              "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(tmem_dP + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        int smem_idx = threadIdx.x * 128 + col;
        for(int k=0; k<8; ++k) {
            float dp = __uint_as_float(r[k]);
            float d_val = smem_D[col + k];
            float p = (float)smem_PT[smem_idx + k];
            float ds = p * (dp - d_val);
            smem_dST[smem_idx + k] = __float2bfloat16(ds);
        }
    }
}

__device__ __forceinline__ void add_dQ_global_coalesced(uint32_t tmem_dQ, __nv_bfloat16* gmem_dQ, int offset, int d, __nv_bfloat16* smem_buf) {
    for (int col = 0; col < 128; col += 8) {
        uint32_t r[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
              "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(tmem_dQ + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        int row = threadIdx.x;
        int smem_idx = row * 128 + col;
        for(int k=0; k<8; ++k) {
            smem_buf[smem_idx + k] = __float2bfloat16(__uint_as_float(r[k]));
        }
    }
    __syncthreads();
    
    for (int step = 0; step < 16; ++step) {
        int row = step * 8 + (threadIdx.x / 16);
        int col_vec = threadIdx.x % 16; 
        int col = col_vec * 8;
        size_t idx = (size_t)(offset + row) * d + col;
        
        uint4 curr_data = *reinterpret_cast<uint4*>(&gmem_dQ[idx]);
        __nv_bfloat16* curr_bf16 = reinterpret_cast<__nv_bfloat16*>(&curr_data);
        
        uint4 smem_data = *reinterpret_cast<uint4*>(&smem_buf[row * 128 + col]);
        __nv_bfloat16* smem_bf16 = reinterpret_cast<__nv_bfloat16*>(&smem_data);
        
        uint4 out_data;
        __nv_bfloat16* out_bf16 = reinterpret_cast<__nv_bfloat16*>(&out_data);
        for(int k=0; k<8; ++k) {
            out_bf16[k] = __float2bfloat16((float)curr_bf16[k] + (float)smem_bf16[k]);
        }
        
        *reinterpret_cast<uint4*>(&gmem_dQ[idx]) = out_data;
    }
    __syncthreads();
}

__device__ __forceinline__ void write_tmem_global_coalesced(uint32_t tmem_addr, __nv_bfloat16* gmem_ptr, int offset, int d, __nv_bfloat16* smem_buf) {
    for (int col = 0; col < 128; col += 8) {
        uint32_t r[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
              "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(tmem_addr + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        int row = threadIdx.x;
        int smem_idx = row * 128 + col;
        for(int k=0; k<8; ++k) {
            smem_buf[smem_idx + k] = __float2bfloat16(__uint_as_float(r[k]));
        }
    }
    __syncthreads();
    
    for (int step = 0; step < 16; ++step) {
        int row = step * 8 + (threadIdx.x / 16);
        int col_vec = threadIdx.x % 16; 
        int col = col_vec * 8;
        size_t idx = (size_t)(offset + row) * d + col;
        uint4 data = *reinterpret_cast<uint4*>(&smem_buf[row * 128 + col]);
        *reinterpret_cast<uint4*>(&gmem_ptr[idx]) = data;
    }
    __syncthreads();
}

__global__ void precompute_D_kernel(
    const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int B, int H, int S, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H * S) {
        float sum = 0;
        for (int i = 0; i < d; ++i) {
            sum += (float)O[idx * d + i] * (float)dO[idx * d + i];
        }
        D[idx] = sum;
    }
}

__global__ void fa_bwd_main_kernel(
    const CUtensorMap tma_K, const CUtensorMap tma_V, 
    const CUtensorMap tma_Q, const CUtensorMap tma_dO,
    const float* LSE_ptr, const float* D_ptr,
    __nv_bfloat16* dK_ptr, __nv_bfloat16* dV_ptr, __nv_bfloat16* dQ_ptr,
    int B, int H, int S, int d, float scale) 
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int head_idx = b * H + h;
    int S_blocks = S / 128;
    
    extern __shared__ __align__(16) char smem[];
    
    __nv_bfloat16* smem_K = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_V = smem_K + 128 * 128;
    __nv_bfloat16* smem_Q = smem_V + 128 * 128;
    __nv_bfloat16* smem_dO = smem_Q + 128 * 128;
    __nv_bfloat16* smem_PT = smem_dO + 128 * 128;
    __nv_bfloat16* smem_dST = smem_PT + 128 * 128;
    
    uint64_t* mbar_K = (uint64_t*)(smem_dST + 128 * 128);
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_Q = mbar_V + 1;
    uint64_t* mbar_dO = mbar_Q + 1;
    uint64_t* mbar_mma = mbar_dO + 1;
    
    float* smem_LSE = (float*)(mbar_mma + 1);
    float* smem_D = smem_LSE + 128;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_dO, 1);
        init_smem_barrier_fn(mbar_mma, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    
    uint32_t tmem_S, tmem_dQ, tmem_dV, tmem_dK;
    __shared__ uint32_t smem_addr;
    if (threadIdx.x < 32) { 
        tmem_alloc_fn(&smem_addr, 512);
    }
    __syncthreads();
    
    tmem_S = smem_addr;
    tmem_dQ = smem_addr + 128;
    tmem_dV = smem_addr + 256;
    tmem_dK = smem_addr + 384;
    
    int phase_K = 0, phase_V = 0, phase_Q = 0, phase_dO = 0, phase_mma = 0;
    
    for (int i = 0; i < S_blocks; ++i) {
        int i_offset = head_idx * S + i * 128;
        
        tma_expect_and_load(&tma_K, mbar_K, smem_K, 0, i_offset);
        tma_expect_and_load(&tma_V, mbar_V, smem_V, 0, i_offset);
        tma_load_sync(mbar_K, phase_K);
        tma_load_sync(mbar_V, phase_V);
        
        for (int j = 0; j < S_blocks; ++j) {
            int j_offset = head_idx * S + j * 128;
            
            tma_expect_and_load(&tma_Q, mbar_Q, smem_Q, 0, j_offset);
            tma_expect_and_load(&tma_dO, mbar_dO, smem_dO, 0, j_offset);
            
            int tid = threadIdx.x;
            smem_LSE[tid] = LSE_ptr[j_offset + tid];
            smem_D[tid] = D_ptr[j_offset + tid];
            
            tma_load_sync(mbar_Q, phase_Q);
            tma_load_sync(mbar_dO, phase_dO);
            
            if (threadIdx.x == 0) {
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                mma_128x128x128(tmem_S, smem_K, smem_Q, false, false, 0); 
            }
            sync_mma(mbar_mma, phase_mma);
            
            compute_PT(tmem_S, smem_PT, smem_LSE, scale);
            
            __syncthreads();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            
            if (threadIdx.x == 0) {
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                mma_128x128x128(tmem_S, smem_V, smem_dO, false, false, 0); 
            }
            sync_mma(mbar_mma, phase_mma);
            
            compute_dST(tmem_S, smem_PT, smem_dST, smem_D);
            
            __syncthreads();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            
            if (threadIdx.x == 0) {
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                int acc_KV = (j == 0) ? 0 : 1;
                mma_128x128x128(tmem_dV, smem_PT, smem_dO, false, true, acc_KV); 
                mma_128x128x128(tmem_dK, smem_dST, smem_Q, false, true, acc_KV); 
                mma_128x128x128(tmem_dQ, smem_dST, smem_K, true, true, 0);       
            }
            sync_mma(mbar_mma, phase_mma);
            
            add_dQ_global_coalesced(tmem_dQ, dQ_ptr, j_offset, d, smem_PT);
        }
        
        write_tmem_global_coalesced(tmem_dV, dV_ptr, i_offset, d, smem_PT);
        write_tmem_global_coalesced(tmem_dK, dK_ptr, i_offset, d, smem_PT);
    }
    
    if (threadIdx.x < 32) { 
        tmem_dealloc_fn(tmem_S, 512);
    }
}

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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t bytes = (size_t)B * H * S * d * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, bytes, stream));
    
    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, B * H * S * sizeof(float), stream));
    
    int num_threads = 256;
    int num_blocks = (B * H * S + num_threads - 1) / num_threads;
    precompute_D_kernel<<<num_blocks, num_threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        D_ptr, B, H, S, d);
        
    CUtensorMap tma_K, tma_V, tma_Q, tma_dO;
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), d, B * H * S, d, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K error\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), d, B * H * S, d, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V error\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), d, B * H * S, d, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q error\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), d, B * H * S, d, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA dO error\n"); exit(1); }
    
    dim3 grid(B, H, 1);
    dim3 block(128, 1, 1);
    float scale = 1.0f / sqrtf((float)d);
    
    int smem_size = 197672; 
    CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    fa_bwd_main_kernel<<<grid, block, smem_size, stream>>>(
        tma_K, tma_V, tma_Q, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        D_ptr,
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        (int)B, (int)H, (int)S, (int)d, scale
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda