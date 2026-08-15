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
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_name, __FILE__, __LINE__);                     \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

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

__device__ __forceinline__ uint64_t make_smem_desc_k_major_none(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t lbo = 32768; // (128/8)*2048 = 32768
    uint32_t sbo = 2048;  // 8 rows * 256 bytes = 2048
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)0 << 61;   // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_none(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t lbo = 2048;  // 8 rows * 256 bytes = 2048
    uint32_t sbo = 32768; // (128/8)*2048 = 32768
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)0 << 61;   // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ void advance_desc_k_16_none(uint64_t& desc) {
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr += 32; // 16 elements * 2 bytes = 32 bytes (advancing inner dimension K)
    desc = (desc & ~0x3FFFull) | ((addr >> 4) & 0x3FFF);
}

__device__ __forceinline__ void advance_desc_mn_16_none(uint64_t& desc) {
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr += 4096; // 16 rows * 256 bytes = 4096 bytes (advancing outer dimension K)
    desc = (desc & ~0x3FFFull) | ((addr >> 4) & 0x3FFF);
}

__device__ __forceinline__ uint32_t make_idesc_qk() {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (0u << 16);   // b_major = 0 (K-Major)
    d |= ((128 / 8) << 17); // n_dim = 16
    d |= ((128 / 16) << 24); // m_dim = 8
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_pv() {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (1u << 16);   // b_major = 1 (MN-Major)
    d |= ((128 / 8) << 17); // n_dim = 16
    d |= ((128 / 16) << 24); // m_dim = 8
    return d;
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
        :: "r"(r0),"r"(r1),"r"(r2),"r"(r3), "r"(col) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    float2 f2 = make_float2(a, b);
    __nv_bfloat162 bf2 = __float22bfloat162_rn(f2);
    return *reinterpret_cast<uint32_t*>(&bf2);
}

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    int B, int H, int S, int D)
{
    int tid = threadIdx.x;
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int q_start = blockIdx.x * 128;
    int global_row = q_start + tid;

    extern __shared__ __align__(128) char smem[];
    void* smem_q = smem;           
    void* smem_k = smem + 32768;   
    void* smem_v = smem + 65536;   
    void* smem_p = smem + 98304;   
    void* smem_o = smem + 98304;   

    __shared__ __align__(8) uint64_t mbar_q[1];
    __shared__ __align__(8) uint64_t mbar_k[1];
    __shared__ __align__(8) uint64_t mbar_v[1];
    __shared__ __align__(8) uint64_t mbar_umma[1];
    __shared__ __align__(4) uint32_t tmem_alloc_ptr;

    if (tid == 0) {
        init_smem_barrier_fn(mbar_q, 1);
        init_smem_barrier_fn(mbar_k, 1);
        init_smem_barrier_fn(mbar_v, 1);
        init_smem_barrier_fn(mbar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    
    if (tid < 32) {
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(&tmem_alloc_ptr);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(addr), "r"(256));
    }
    __syncthreads();

    uint32_t tmem_base = tmem_alloc_ptr;
    uint32_t p_tmem = tmem_base;
    uint32_t o_tmem = tmem_base + 128;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_q, 32768);
        tma_load_4d_fn(&tma_Q, mbar_q, smem_q, 0, q_start, head, batch);
    }
    mbarrier_wait_fn(mbar_q, 0);

    // OOB zero-fill for Q to ensure clean padding
    int valid_q = S - q_start;
    if (valid_q < 128) {
        int vl = valid_q < 0 ? 0 : valid_q;
        for (int r = vl + (tid / 32); r < 128; r += 4) {
            int lane = tid % 32;
            *(uint64_t*)((char*)smem_q + r * 256 + lane * 8) = 0;
        }
    }
    __syncthreads();

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    int phase_k = 0, phase_v = 0, phase_umma = 0;
    
    uint32_t idesc_QK = make_idesc_qk();
    uint32_t idesc_PV = make_idesc_pv();

    for (int j = 0; j < (S + 127) / 128; ++j) {
        int k_start = j * 128;
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_k, 32768);
            tma_load_4d_fn(&tma_K, mbar_k, smem_k, 0, k_start, head, batch);
            mbarrier_arrive_and_expect_tx_fn(mbar_v, 32768);
            tma_load_4d_fn(&tma_V, mbar_v, smem_v, 0, k_start, head, batch);
        }
        
        mbarrier_wait_fn(mbar_k, phase_k);
        phase_k ^= 1;
        
        // OOB zero-fill for K and V to avoid NaNs on out-of-bounds FMA
        int valid_len = S - j * 128;
        if (valid_len < 128) {
            int vl = valid_len < 0 ? 0 : valid_len;
            for (int r = vl + (tid / 32); r < 128; r += 4) {
                int lane = tid % 32;
                *(uint64_t*)((char*)smem_k + r * 256 + lane * 8) = 0;
                *(uint64_t*)((char*)smem_v + r * 256 + lane * 8) = 0;
            }
        }
        __syncthreads();
        
        uint64_t desc_A_QK = make_smem_desc_k_major_none(smem_q);
        uint64_t desc_B_QK = make_smem_desc_k_major_none(smem_k);
        
        fence_async_shared_fn(); 
        
        if (tid == 0) {
            for (int step = 0; step < 8; ++step) {
                uint32_t accum = (step == 0) ? 0 : 1;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(p_tmem), "l"(desc_A_QK), "l"(desc_B_QK), "r"(idesc_QK), "r"(accum));
                advance_desc_k_16_none(desc_A_QK);
                advance_desc_k_16_none(desc_B_QK);
            }
            uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar_umma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_addr));
        }
        
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        float row_max = -INFINITY;
        int max_vl = valid_len > 128 ? 128 : valid_len;
        float scale_q = 1.0f / sqrtf(128.0f);
        
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(p_tmem + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            float f0 = (c + 0 < max_vl) ? __uint_as_float(r0) * scale_q : -INFINITY;
            float f1 = (c + 1 < max_vl) ? __uint_as_float(r1) * scale_q : -INFINITY;
            float f2 = (c + 2 < max_vl) ? __uint_as_float(r2) * scale_q : -INFINITY;
            float f3 = (c + 3 < max_vl) ? __uint_as_float(r3) * scale_q : -INFINITY;
            row_max = fmaxf(row_max, f0);
            row_max = fmaxf(row_max, f1);
            row_max = fmaxf(row_max, f2);
            row_max = fmaxf(row_max, f3);
        }
        float m_curr = fmaxf(m_prev, row_max);
        float m_diff = m_prev - m_curr; 
        float o_scale = exp2f(m_diff * 1.44269504f);
        
        if (j > 0 && o_scale < 1.0f) {
            for (int c = 0; c < 128; c += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(o_tmem + c, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                float f0 = __uint_as_float(r0) * o_scale;
                float f1 = __uint_as_float(r1) * o_scale;
                float f2 = __uint_as_float(r2) * o_scale;
                float f3 = __uint_as_float(r3) * o_scale;
                tmem_store_4x_fn(o_tmem + c, __float_as_uint(f0), __float_as_uint(f1), __float_as_uint(f2), __float_as_uint(f3));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        float row_sum = 0.0f;
        uint32_t base_p = (uint32_t)__cvta_generic_to_shared(smem_p);
        
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(p_tmem + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            float f0 = (c + 0 < max_vl) ? __uint_as_float(r0) * scale_q - m_curr : -INFINITY;
            float f1 = (c + 1 < max_vl) ? __uint_as_float(r1) * scale_q - m_curr : -INFINITY;
            float f2 = (c + 2 < max_vl) ? __uint_as_float(r2) * scale_q - m_curr : -INFINITY;
            float f3 = (c + 3 < max_vl) ? __uint_as_float(r3) * scale_q - m_curr : -INFINITY;
            
            f0 = exp2f(f0 * 1.44269504f);
            f1 = exp2f(f1 * 1.44269504f);
            f2 = exp2f(f2 * 1.44269504f);
            f3 = exp2f(f3 * 1.44269504f);
            
            row_sum += f0 + f1 + f2 + f3;
            
            uint32_t b01 = pack_bf16_fn(f0, f1);
            uint32_t b23 = pack_bf16_fn(f2, f3);
            
            uint32_t smem_addr = base_p + tid * 256 + c * 2;
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(b01), "r"(b23));
        }
        
        float l_curr = l_prev * o_scale + row_sum;
        m_prev = m_curr;
        l_prev = l_curr;
        
        __syncthreads();
        fence_async_shared_fn();
        
        mbarrier_wait_fn(mbar_v, phase_v);
        phase_v ^= 1;
        
        uint64_t desc_A_PV = make_smem_desc_k_major_none(smem_p);
        uint64_t desc_B_PV = make_smem_desc_mn_major_none(smem_v);
        
        if (tid == 0) {
            for (int step = 0; step < 8; ++step) {
                uint32_t accum = (j == 0 && step == 0) ? 0 : 1;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(o_tmem), "l"(desc_A_PV), "l"(desc_B_PV), "r"(idesc_PV), "r"(accum));
                advance_desc_k_16_none(desc_A_PV);
                advance_desc_mn_16_none(desc_B_PV);
            }
            uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar_umma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_addr));
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
    }

    float out_scale = 1.0f / l_prev;
    uint32_t base_o = (uint32_t)__cvta_generic_to_shared(smem_o);
    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(o_tmem + c, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        float f0 = __uint_as_float(r0) * out_scale;
        float f1 = __uint_as_float(r1) * out_scale;
        float f2 = __uint_as_float(r2) * out_scale;
        float f3 = __uint_as_float(r3) * out_scale;
        
        uint32_t b01 = pack_bf16_fn(f0, f1);
        uint32_t b23 = pack_bf16_fn(f2, f3);
        
        uint32_t smem_addr = base_o + tid * 256 + c * 2;
        asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(b01), "r"(b23));
    }
    
    __syncthreads();

    uint4* smem_flat = (uint4*)smem_o;
    uint4* global_flat = (uint4*)(O + batch * (H * S * D) + head * (S * D) + q_start * D);
    
    for (int i = 0; i < 16; ++i) {
        int idx = i * 128 + tid;
        int row = idx / 16;
        if (q_start + row < S) {
            global_flat[idx] = smem_flat[idx];
        }
    }

    if (global_row < S) {
        float lse = m_prev + logf(l_prev);
        LSE[batch * (H * S) + head * S + global_row] = lse;
    }

    __syncthreads();
    if (tid < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_alloc_ptr), "r"(256));
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    cuuint64_t globalDim[4] = {128, (cuuint64_t)S, (cuuint64_t)H, (cuuint64_t)B};
    cuuint64_t globalStrides[3] = {256, 256 * (cuuint64_t)S, 256 * (cuuint64_t)S * (cuuint64_t)H};
    cuuint32_t boxDim[4] = {128, 128, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        Q_ptr, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        K_ptr, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        V_ptr, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    int64_t blocks_x = (S + 127) / 128;
    int64_t blocks_y = H;
    int64_t blocks_z = B;
    dim3 grid(blocks_x, blocks_y, blocks_z);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 131072));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 131072;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, B, H, S, D));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda