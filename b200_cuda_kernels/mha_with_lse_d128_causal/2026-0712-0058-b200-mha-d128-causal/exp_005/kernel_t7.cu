#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

namespace tvm_ffi_kernel {

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t swizzle_128B_64B(uint32_t row, uint32_t col) {
    uint32_t x_chunk = col / 8;
    uint32_t x_rem = col % 8;
    uint32_t y_chunk = row % 8;
    uint32_t swizzled_x_chunk = x_chunk ^ y_chunk;
    return row * 64 + swizzled_x_chunk * 8 + x_rem;
}

__device__ __forceinline__ void write_swizzled_128B_64B_atomic(__nv_bfloat16* smem, int row, int col, __nv_bfloat16 val) {
    uint32_t idx = swizzle_128B_64B(row, col);
    smem[idx] = val;
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled_128B_64B_atomic(const __nv_bfloat16* smem, int row, int col) {
    uint32_t idx = swizzle_128B_64B(row, col);
    return smem[idx];
}

struct SharedStorage {
    __align__(1024) __nv_bfloat16 Q_T[128 * 128];
    __align__(1024) __nv_bfloat16 K[128 * 128];
    __align__(1024) __nv_bfloat16 V_T[128 * 128];
    __align__(1024) __nv_bfloat16 P_T[64 * 64];
    __align__(1024) __nv_bfloat16 O_half[64 * 128];
    __align__(1024) uint64_t smem_mbar_QK;
    __align__(1024) uint64_t smem_mbar_PV;
    __align__(1024) uint64_t bar_Q;
    __align__(1024) uint64_t bar_K;
    __align__(1024) uint64_t bar_V;
};

__device__ __forceinline__ uint32_t make_instr_desc_fn_64x64() {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (1u << 15);   
    d |= (1u << 16);   
    d |= ((64 / 8) << 17);     
    d |= ((64 / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_64x64_pv() {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (1u << 15);   
    d |= (1u << 16);   
    d |= ((64 / 8) << 17);     
    d |= ((64 / 16) << 24);    
    return d;
}

__device__ __forceinline__ void transpose_64x64_128B_swizzled(__nv_bfloat16* out_B, const __nv_bfloat16* in_A) {
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
        int r = i / 64;
        int c = i % 64;
        __nv_bfloat16 val = in_A[swizzle_128B_64B(r, c)];
        out_B[swizzle_128B_64B(c, r)] = val;
    }
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

__global__ void __launch_bounds__(128) attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_gmem,
    float* __restrict__ LSE_gmem,
    int S_len) 
{
    extern __shared__ __align__(1024) uint8_t smem_dynamic[];
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_dynamic);

    int batch_head_idx = blockIdx.y;
    int block_idx = blockIdx.x;
    int seq_start = block_idx * 64;
    if (seq_start >= S_len) return;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&shared.smem_mbar_QK, 1);
        init_smem_barrier_fn(&shared.smem_mbar_PV, 1);
        init_smem_barrier_fn(&shared.bar_Q, 1);
        init_smem_barrier_fn(&shared.bar_K, 1);
        init_smem_barrier_fn(&shared.bar_V, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    __shared__ uint32_t tmem_S_addr;
    __shared__ uint32_t tmem_O_addr;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S_addr, 64);
        tmem_alloc_fn(&tmem_O_addr, 64);
        
        mbarrier_arrive_and_expect_tx_fn(&shared.bar_Q, 16384);

        tma_load_2d_fn(&tma_Q, &shared.bar_Q, shared.Q_T, 0, batch_head_idx * S_len + seq_start);
        tma_load_2d_fn(&tma_Q, &shared.bar_Q, (char*)shared.Q_T + 8192, 64, batch_head_idx * S_len + seq_start);
    }
    
    int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf(128.0f);
    const float LOG2E = 1.4426950408889634f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    uint32_t phase_PV = 0;
    uint32_t step = 0;

    mbarrier_wait_fn(&shared.bar_Q, 0);
    transpose_64x64_128B_swizzled(shared.Q_T, shared.Q_T);
    transpose_64x64_128B_swizzled((char*)shared.Q_T + 8192, (char*)shared.Q_T + 8192);

    uint32_t qt_base = (uint32_t)__cvta_generic_to_shared(shared.Q_T);
    uint32_t LBO_Q = 8192;
    uint32_t SBO_Q = 1024;

    for (int j = 0; j <= block_idx * 64; j += 64) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&shared.bar_K, 16384);
            mbarrier_arrive_and_expect_tx_fn(&shared.bar_V, 16384);

            tma_load_2d_fn(&tma_K, &shared.bar_K, shared.K, 0, batch_head_idx * S_len + j);
            tma_load_2d_fn(&tma_K, &shared.bar_K, (char*)shared.K + 8192, 64, batch_head_idx * S_len + j);
            
            tma_load_2d_fn(&tma_V, &shared.bar_V, shared.V_T, 0, batch_head_idx * S_len + j);
            tma_load_2d_fn(&tma_V, &shared.bar_V, (char*)shared.V_T + 8192, 64, batch_head_idx * S_len + j);
        }
        
        mbarrier_wait_fn(&shared.bar_K, step % 2);
        mbarrier_wait_fn(&shared.bar_V, step % 2);
        __syncthreads(); 
        
        transpose_64x64_128B_swizzled(shared.V_T, shared.V_T);
        transpose_64x64_128B_swizzled((char*)shared.V_T + 8192, (char*)shared.V_T + 8192);
        __syncthreads();
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            uint32_t kt_base = (uint32_t)__cvta_generic_to_shared(shared.K);
            uint32_t LBO_K = 0;
            uint32_t SBO_K = 1024;
            
            int accum = 0;
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_Q_curr = make_smem_desc_sm100_fn((void*)(qt_base + k * 2048), LBO_Q, SBO_Q);
                uint64_t desc_K_curr = make_smem_desc_sm100_fn((void*)(kt_base + k * 2048), LBO_K, SBO_K);
                
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_S_addr), "l"(desc_Q_curr), "l"(desc_K_curr), "r"(make_instr_desc_fn_64x64()), "r"(accum));
                accum = 1;
            }

            asm volatile(
                "tcgen05.commit.cta_group::1"
                ".mbarrier::arrive::one.shared.b64"
                " [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(&shared.smem_mbar_QK)));
        }
        mbarrier_wait_fn(&shared.smem_mbar_QK, step % 2);
        
        float rowmax = -INFINITY;
        for (int c = 0; c < 128; c += 4) {
            uint32_t addr = (tid << 16) + c;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(addr, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            float s0 = __uint_as_float(r0) * scale;
            float s1 = __uint_as_float(r1) * scale;
            float s2 = __uint_as_float(r2) * scale;
            float s3 = __uint_as_float(r3) * scale;
            
            if (j + c > seq_start + tid) s0 = -INFINITY;
            if (j + c + 1 > seq_start + tid) s1 = -INFINITY;
            if (j + c + 2 > seq_start + tid) s2 = -INFINITY;
            if (j + c + 3 > seq_start + tid) s3 = -INFINITY;
            
            rowmax = fmaxf(rowmax, fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)));
        }
        
        float new_max = fmaxf(running_max, rowmax);
        float alpha = fast_exp2f_fn((running_max - new_max) * LOG2E);
        float beta = fast_exp2f_fn((rowmax - new_max) * LOG2E);
        
        float rowsum = 0;
        for (int c = 0; c < 128; c += 4) {
            uint32_t addr = (tid << 16) + c;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(addr, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float s0 = __uint_as_float(r0) * scale;
            float s1 = __uint_as_float(r1) * scale;
            float s2 = __uint_as_float(r2) * scale;
            float s3 = __uint_as_float(r3) * scale;
            
            float p0 = (j + c > seq_start + tid) ? 0.0f : fast_exp2f_fn((s0 - rowmax) * LOG2E);
            float p1 = (j + c + 1 > seq_start + tid) ? 0.0f : fast_exp2f_fn((s1 - rowmax) * LOG2E);
            float p2 = (j + c + 2 > seq_start + tid) ? 0.0f : fast_exp2f_fn((s2 - rowmax) * LOG2E);
            float p3 = (j + c + 3 > seq_start + tid) ? 0.0f : fast_exp2f_fn((s3 - rowmax) * LOG2E);
            
            rowsum += p0 + p1 + p2 + p3;
            
            write_swizzled_128B_64B_atomic(shared.P_T, tid, c, __float2bfloat16(p0));
            write_swizzled_128B_64B_atomic(shared.P_T, tid, c + 1, __float2bfloat16(p1));
            write_swizzled_128B_64B_atomic(shared.P_T, tid, c + 2, __float2bfloat16(p2));
            write_swizzled_128B_64B_atomic(shared.P_T, tid, c + 3, __float2bfloat16(p3));
        }
        
        running_sum = running_sum * alpha + rowsum * beta;
        running_max = new_max;
        
        named_barrier_sync_fn(1, 128); 
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            uint32_t pt_base = (uint32_t)__cvta_generic_to_shared(shared.P_T);
            uint32_t vt_base = (uint32_t)__cvta_generic_to_shared(shared.V_T);
            
            uint32_t LBO_P = 1024;
            uint32_t SBO_P = 1024;
            
            uint32_t LBO_VT = 1024;
            uint32_t SBO_VT = 1024;
            
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_P_curr = make_smem_desc_sm100_fn((void*)(pt_base + k * 1024), LBO_P, SBO_P);
                uint64_t desc_VT_curr = make_smem_desc_sm100_fn((void*)(vt_base + k * 1024), LBO_VT, SBO_VT);
                
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_O_addr), "l"(desc_P_curr), "l"(desc_VT_curr), "r"(make_instr_desc_fn_64x64_pv()), "r"(1));
            }

            asm volatile(
                "tcgen05.commit.cta_group::1"
                ".mbarrier::arrive::one.shared.b64"
                " [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(&shared.smem_mbar_PV)));
        }
        mbarrier_wait_fn(&shared.smem_mbar_PV, phase_PV);
        phase_PV ^= 1;
        step++;
    }
    
    if (tid < 128) {
        LSE_gmem[batch_head_idx * S_len + seq_start + tid] = running_max + __logf(running_sum);
    }
    
    float inv_sum = 1.0f / running_sum;
    
    for (int c = 0; c < 128; c += 4) {
        uint32_t addr = (tid << 16) + c;
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(addr, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        __nv_bfloat16* ptr = O_gmem + batch_head_idx * S_len * 128 + (seq_start + tid) * 128 + c;
        if (seq_start + tid < S_len && c + 3 < 128) {
            ptr[0] = __float2bfloat16(f0 * inv_sum);
            ptr[1] = __float2bfloat16(f1 * inv_sum);
            ptr[2] = __float2bfloat16(f2 * inv_sum);
            ptr[3] = __float2bfloat16(f3 * inv_sum);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S_addr, 64);
        tmem_dealloc_fn(tmem_O_addr, 64);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_data, 128, B * H * S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        
    create_tma_2d_descriptor_2B(&tma_K, (void*)K_data, 128, B * H * S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        
    create_tma_2d_descriptor_2B(&tma_V, (void*)V_data, 128, B * H * S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    uint32_t num_blocks = (S + 63) / 64;
    uint32_t grid_x = (num_blocks + 1) & ~1; 
    dim3 grid(grid_x, B * H);
    int threads = 128;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = dim3(threads, 1, 1);
    config.dynamicSmemBytes = 128 * 1024;
    config.stream = stream;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_kernel, 
        tma_Q, tma_K, tma_V, O_data, LSE_data, S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel