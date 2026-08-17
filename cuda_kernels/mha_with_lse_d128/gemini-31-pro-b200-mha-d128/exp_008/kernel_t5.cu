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

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ uint16_t float_to_bf16_u16(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    return *reinterpret_cast<uint16_t*>(&b);
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* ptr) {
    uint32_t sbo = 1024;
    uint32_t lbo = 1;
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= base_offset << 49;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* ptr, uint32_t K_dim) {
    uint32_t sbo = 1024;
    uint32_t lbo = (K_dim / 8) * sbo;
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= base_offset << 49;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, bool a_mn_major, bool b_mn_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= ((uint32_t)a_mn_major << 15);
    d |= ((uint32_t)b_mn_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

__global__ void __launch_bounds__(128) flash_attn_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* __restrict__ LSE_data,
    int S)
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_base = blockIdx.x * 128;
    int thread_idx = threadIdx.x;
    
    if (q_base >= S) return;

    // 1024-byte aligned dynamically allocated shared memory
    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* mbar_tma = (uint64_t*)smem;
    uint64_t* mbar_umma = (uint64_t*)(smem + 8);
    
    // Each chunk requires 16384 bytes
    uint8_t* smem_Q0 = smem + 1024;
    uint8_t* smem_Q1 = smem_Q0 + 16384;
    uint8_t* smem_K0 = smem_Q1 + 16384;
    uint8_t* smem_K1 = smem_K0 + 16384;
    uint8_t* smem_V0 = smem_K1 + 16384;
    uint8_t* smem_V1 = smem_V0 + 16384;
    uint8_t* smem_P0 = smem_V1 + 16384;
    uint8_t* smem_P1 = smem_P0 + 16384;
    
    __shared__ uint32_t shared_tmem_addr;
    
    if (thread_idx == 0) {
        init_smem_barrier_fn(mbar_tma, 1);
        init_smem_barrier_fn(mbar_umma, 1);
    }
    __syncthreads();
    
    // Allocate Tensor Memory space for S and O block accumulation (256 columns together)
    if (thread_idx < 32) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&shared_tmem_addr)), "r"(256));
    }
    __syncthreads();
    uint32_t tmem_S = shared_tmem_addr;
    uint32_t tmem_O = shared_tmem_addr + 128;
    
    if (thread_idx == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_tma, 32768);
        tma_load_3d_fn(&tma_Q, mbar_tma, smem_Q0, 0, q_base, b * 48 + h);
        tma_load_3d_fn(&tma_Q, mbar_tma, smem_Q1, 64, q_base, b * 48 + h);
    }
    mbarrier_wait_fn(mbar_tma, 0);
    
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float O_acc[128];
    for(int i = 0; i < 128; ++i) O_acc[i] = 0.0f;
    
    float scale = 0.0883883476f; // 1.0f / sqrt(128.0f)
    int num_k_blocks = (S + 127) / 128;
    
    uint32_t idesc_S = make_instr_desc(128, 128, false, false);
    uint32_t idesc_O = make_instr_desc(128, 64, false, true); // V is MN-major (Transpose B)
    
    for (int k_idx = 0; k_idx < num_k_blocks; ++k_idx) {
        int k_base = k_idx * 128;
        
        if (thread_idx == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma, 65536);
            tma_load_3d_fn(&tma_K, mbar_tma, smem_K0, 0, k_base, b * 48 + h);
            tma_load_3d_fn(&tma_K, mbar_tma, smem_K1, 64, k_base, b * 48 + h);
            tma_load_3d_fn(&tma_V, mbar_tma, smem_V0, 0, k_base, b * 48 + h);
            tma_load_3d_fn(&tma_V, mbar_tma, smem_V1, 64, k_base, b * 48 + h);
        }
        mbarrier_wait_fn(mbar_tma, (k_idx + 1) & 1);
        
        tcgen05_fence_after_fn();
        
        // Compute S = Q @ K^T over D=128 internally breaking into 2 halves (64 dimensions)
        for (int step = 0; step < 4; ++step) {
            uint64_t a_desc = make_smem_desc_k_major(smem_Q0 + step * 32);
            uint64_t b_desc = make_smem_desc_k_major(smem_K0 + step * 32);
            uint32_t accum = (step == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_S, a_desc, b_desc, idesc_S, accum);
        }
        for (int step = 0; step < 4; ++step) {
            uint64_t a_desc = make_smem_desc_k_major(smem_Q1 + step * 32);
            uint64_t b_desc = make_smem_desc_k_major(smem_K1 + step * 32);
            umma_f16_cg1_fn(tmem_S, a_desc, b_desc, idesc_S, 1);
        }
        
        if (thread_idx == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                         :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_umma[0])));
        }
        mbarrier_wait_fn(mbar_umma, (k_idx * 2) & 1);
        
        float m_local = -INFINITY;
        float row_S[128];
        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(tmem_S + c));
            tmem_load_fence_fn();
            
            for(int i = 0; i < 8; ++i) {
                float f = __uint_as_float(r[i]) * scale;
                if (k_base + c + i >= S) f = -INFINITY;
                if (q_base + thread_idx >= S) f = -INFINITY;
                row_S[c+i] = f;
                m_local = fmaxf(m_local, f);
            }
        }
        
        float m_new = fmaxf(m_prev, m_local);
        float exp_m = (m_prev == -INFINITY) ? 0.0f : exp2f((m_prev - m_new) * 1.44269504f);
        
        for(int i = 0; i < 128; ++i) O_acc[i] *= exp_m;
        
        float l_local = 0.0f;
        for (int c = 0; c < 128; ++c) {
            float p = 0.0f;
            if (m_new != -INFINITY) p = exp2f((row_S[c] - m_new) * 1.44269504f);
            row_S[c] = p;
            l_local += p;
        }
        
        l_prev = l_prev * exp_m + l_local;
        m_prev = m_new;
        
        for (int c_chunk = 0; c_chunk < 8; ++c_chunk) {
            uint32_t swizzled_c = ((thread_idx % 8) ^ c_chunk) * 8;
            
            uint16_t p0[8];
            for(int i = 0; i < 8; ++i) p0[i] = float_to_bf16_u16(row_S[c_chunk * 8 + i]);
            *(uint4*)(&smem_P0[thread_idx * 128 + swizzled_c * 2]) = *(uint4*)p0;
            
            uint16_t p1[8];
            for(int i = 0; i < 8; ++i) p1[i] = float_to_bf16_u16(row_S[64 + c_chunk * 8 + i]);
            *(uint4*)(&smem_P1[thread_idx * 128 + swizzled_c * 2]) = *(uint4*)p1;
        }
        
        fence_proxy_async_fn();
        __syncthreads();
        tcgen05_fence_after_fn();
        
        // P @ V Top 64 Elements Column Output
        for (int step = 0; step < 4; ++step) {
            uint64_t a_desc = make_smem_desc_k_major(smem_P0 + step * 32);
            uint64_t b_desc = make_smem_desc_mn_major(smem_V0 + step * 2048, 128);
            uint32_t acc = (step == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_O, a_desc, b_desc, idesc_O, acc);
        }
        for (int step = 0; step < 4; ++step) {
            uint64_t a_desc = make_smem_desc_k_major(smem_P1 + step * 32);
            uint64_t b_desc = make_smem_desc_mn_major(smem_V0 + 8192 + step * 2048, 128);
            umma_f16_cg1_fn(tmem_O, a_desc, b_desc, idesc_O, 1);
        }
        
        // P @ V Bottom 64 Elements Column Output
        for (int step = 0; step < 4; ++step) {
            uint64_t a_desc = make_smem_desc_k_major(smem_P0 + step * 32);
            uint64_t b_desc = make_smem_desc_mn_major(smem_V1 + step * 2048, 128);
            uint32_t acc = (step == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_O + 64, a_desc, b_desc, idesc_O, acc);
        }
        for (int step = 0; step < 4; ++step) {
            uint64_t a_desc = make_smem_desc_k_major(smem_P1 + step * 32);
            uint64_t b_desc = make_smem_desc_mn_major(smem_V1 + 8192 + step * 2048, 128);
            umma_f16_cg1_fn(tmem_O + 64, a_desc, b_desc, idesc_O, 1);
        }
        
        if (thread_idx == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                         :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_umma[0])));
        }
        mbarrier_wait_fn(mbar_umma, (k_idx * 2 + 1) & 1);
        
        for (int c = 0; c < 64; c += 8) {
            uint32_t r[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(tmem_O + c));
            tmem_load_fence_fn();
            for(int i = 0; i < 8; ++i) O_acc[c+i] += __uint_as_float(r[i]);
        }
        for (int c = 0; c < 64; c += 8) {
            uint32_t r[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(tmem_O + 64 + c));
            tmem_load_fence_fn();
            for(int i = 0; i < 8; ++i) O_acc[64+c+i] += __uint_as_float(r[i]);
        }
    }
    
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for(int i = 0; i < 128; ++i) O_acc[i] *= inv_l;
    
    for (int c_chunk = 0; c_chunk < 8; ++c_chunk) {
        uint32_t swizzled_c = ((thread_idx % 8) ^ c_chunk) * 8;
        
        uint16_t o0[8];
        for(int i = 0; i < 8; ++i) o0[i] = float_to_bf16_u16(O_acc[c_chunk * 8 + i]);
        *(uint4*)(&smem_Q0[thread_idx * 128 + swizzled_c * 2]) = *(uint4*)o0;
        
        uint16_t o1[8];
        for(int i = 0; i < 8; ++i) o1[i] = float_to_bf16_u16(O_acc[64 + c_chunk * 8 + i]);
        *(uint4*)(&smem_Q1[thread_idx * 128 + swizzled_c * 2]) = *(uint4*)o1;
    }
    
    fence_proxy_async_fn();
    __syncthreads();
    
    if (thread_idx == 0) {
        tma_store_3d_fn(&tma_O, smem_Q0, 0, q_base, b * 48 + h);
        tma_store_3d_fn(&tma_O, smem_Q1, 64, q_base, b * 48 + h);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    if (q_base + thread_idx < S) {
        LSE_data[b * (48 * S) + h * S + q_base + thread_idx] = (m_prev == -INFINITY) ? -INFINITY : (m_prev + logf(l_prev));
    }
    
    __syncthreads();
    
    if (thread_idx < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_S), "r"(256));
    }
}

CUresult create_tma_3d_descriptor(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2) {
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
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    );
}

CUresult create_tma_3d_descriptor_store(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2) {
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

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    void* Q_data = Q.data_ptr();
    void* K_data = K.data_ptr();
    void* V_data = V.data_ptr();
    void* O_data = O.data_ptr();
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    create_tma_3d_descriptor(&tma_Q, Q_data, 128, S, B * H, 64, 128, 1);
    create_tma_3d_descriptor(&tma_K, K_data, 128, S, B * H, 64, 128, 1);
    create_tma_3d_descriptor(&tma_V, V_data, 128, S, B * H, 64, 128, 1);
    create_tma_3d_descriptor_store(&tma_O, O_data, 128, S, B * H, 64, 128, 1);

    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(flash_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 132096));
    
    flash_attn_kernel<<<grid, block, 132096, stream>>>(tma_Q, tma_K, tma_V, tma_O, LSE_data, S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda