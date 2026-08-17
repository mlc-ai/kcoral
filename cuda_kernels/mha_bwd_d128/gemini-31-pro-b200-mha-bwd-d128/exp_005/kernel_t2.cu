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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n", err_str,         \
                __FILE__, __LINE__);                               \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int a_maj, int b_maj) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (a_maj << 15);
    d |= (b_maj << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ void issue_mma_8steps(uint32_t tmem_addr, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, bool accumulate, int a_maj, int b_maj) {
    if (threadIdx.x == 0) {
        uint64_t da = desc_a;
        uint64_t db = desc_b;
        uint32_t step_a = (a_maj == 0) ? 2 : 256;
        uint32_t step_b = (b_maj == 0) ? 2 : 256;
        for (int step = 0; step < 8; ++step) {
            uint32_t p = (accumulate || step > 0) ? 1 : 0;
            asm volatile(
                "{\n.reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                :: "r"(tmem_addr), "l"(da), "l"(db), "r"(idesc), "r"(p));
            da += step_a;
            db += step_b;
        }
    }
}

__device__ void wait_mma(uint64_t* mbar_mma, int& phase) {
    if (threadIdx.x == 0) {
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(mbar_mma);
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "r"(smem_addr));
    }
    mbarrier_wait_fn(mbar_mma, phase);
    phase ^= 1;
}

__device__ void write_tmem_to_tma(const CUtensorMap* tma_desc, uint8_t* smem_staging, uint32_t tmem_base, int b, int h, int j) {
    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_base + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * 128 + c;
        ((__nv_bfloat16*)smem_staging)[base + 0] = __float2bfloat16(__uint_as_float(r0));
        ((__nv_bfloat16*)smem_staging)[base + 1] = __float2bfloat16(__uint_as_float(r1));
        ((__nv_bfloat16*)smem_staging)[base + 2] = __float2bfloat16(__uint_as_float(r2));
        ((__nv_bfloat16*)smem_staging)[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    fence_async_shared_fn();
    if (threadIdx.x == 0) {
        tma_store_4d_fn(tma_desc, smem_staging, 0, j * 128, h, b);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();
}

__global__ void compute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        const __nv_bfloat16* o_ptr = O + idx * 128;
        const __nv_bfloat16* do_ptr = dO + idx * 128;
        float sum = 0;
        for (int i = 0; i < 128; ++i) {
            sum += __bfloat162float(o_ptr[i]) * __bfloat162float(do_ptr[i]);
        }
        D[idx] = sum;
    }
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* L_ptr, const float* D_ptr,
    __nv_bfloat16* dQ_ptr,
    int S)
{
    int j = blockIdx.x; 
    int h = blockIdx.y;
    int b = blockIdx.z;
    int H = gridDim.y;

    extern __shared__ __align__(256) uint8_t smem_pool[];
    
    uint8_t* smem_K = smem_pool;
    uint8_t* smem_V = smem_pool + 32768;
    uint8_t* smem_Q = smem_pool + 65536;
    uint8_t* smem_dO = smem_pool + 98304;
    uint8_t* smem_P_T = smem_pool + 131072;
    uint8_t* smem_dS_T = smem_pool + 163840;
    float* smem_LSE = (float*)(smem_pool + 196608);
    float* smem_D = (float*)(smem_pool + 197120);
    uint64_t* mbar_tma = (uint64_t*)(smem_pool + 197632);
    uint64_t* mbar_mma = (uint64_t*)(smem_pool + 197640);
    uint32_t* tmem_addr = (uint32_t*)(smem_pool + 197648);

    if (threadIdx.x < 32) {
        tmem_alloc_fn(tmem_addr, 512);
    }
    
    // .sync.aligned instructions require all threads in the warp
    setmaxnreg_inc_sync_fn<256>();
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_tma, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    __syncthreads();

    uint64_t desc_K_Kmaj = make_smem_desc_sm100_fn(smem_K, 2048, 128);
    uint64_t desc_V_Kmaj = make_smem_desc_sm100_fn(smem_V, 2048, 128);
    uint64_t desc_Q_Kmaj = make_smem_desc_sm100_fn(smem_Q, 2048, 128);
    uint64_t desc_dO_Kmaj = make_smem_desc_sm100_fn(smem_dO, 2048, 128);
    uint64_t desc_PT_Kmaj = make_smem_desc_sm100_fn(smem_P_T, 2048, 128);
    uint64_t desc_dST_Kmaj = make_smem_desc_sm100_fn(smem_dS_T, 2048, 128);

    uint64_t desc_dO_MNmaj = make_smem_desc_sm100_fn(smem_dO, 128, 2048);
    uint64_t desc_Q_MNmaj = make_smem_desc_sm100_fn(smem_Q, 128, 2048);
    uint64_t desc_dS_MNmaj = make_smem_desc_sm100_fn(smem_dS_T, 128, 2048);
    uint64_t desc_K_MNmaj = make_smem_desc_sm100_fn(smem_K, 128, 2048);

    uint32_t idesc_K_K = make_instr_desc_fn(128, 128, 0, 0);
    uint32_t idesc_K_MN = make_instr_desc_fn(128, 128, 0, 1);
    uint32_t idesc_MN_MN = make_instr_desc_fn(128, 128, 1, 1);

    int phase_tma = 0;
    int phase_mma = 0;
    int H_stride = S * 128;
    int B_stride = H * H_stride;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_tma, 32768 * 2);
        tma_load_4d_fn(&tma_K, mbar_tma, smem_K, 0, j * 128, h, b);
        tma_load_4d_fn(&tma_V, mbar_tma, smem_V, 0, j * 128, h, b);
    }
    mbarrier_wait_fn(mbar_tma, phase_tma);
    phase_tma ^= 1;
    __syncthreads();

    for (int i = 0; i < S / 128; ++i) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma, 32768 * 2);
            tma_load_4d_fn(&tma_Q, mbar_tma, smem_Q, 0, i * 128, h, b);
            tma_load_4d_fn(&tma_dO, mbar_tma, smem_dO, 0, i * 128, h, b);
        }
        
        int base_S = b * H * S + h * S + i * 128;
        if (threadIdx.x < 128) {
            smem_LSE[threadIdx.x] = L_ptr[base_S + threadIdx.x];
            smem_D[threadIdx.x] = D_ptr[base_S + threadIdx.x];
        }
        
        mbarrier_wait_fn(mbar_tma, phase_tma);
        phase_tma ^= 1;
        __syncthreads();

        // 1. S^T = K @ Q^T
        issue_mma_8steps(0, desc_K_Kmaj, desc_Q_MNmaj, idesc_K_MN, false, 0, 1);
        wait_mma(mbar_mma, phase_mma);

        // 2. P^T = exp(S^T - LSE)
        uint32_t row = threadIdx.x;
        float lse = smem_LSE[row];
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            f0 = fast_exp2f_fn(f0 * 0.12751532f - lse * 1.44269504f);
            f1 = fast_exp2f_fn(f1 * 0.12751532f - lse * 1.44269504f);
            f2 = fast_exp2f_fn(f2 * 0.12751532f - lse * 1.44269504f);
            f3 = fast_exp2f_fn(f3 * 0.12751532f - lse * 1.44269504f);
            uint32_t base = row * 128 + c;
            ((__nv_bfloat16*)smem_P_T)[base + 0] = __float2bfloat16(f0);
            ((__nv_bfloat16*)smem_P_T)[base + 1] = __float2bfloat16(f1);
            ((__nv_bfloat16*)smem_P_T)[base + 2] = __float2bfloat16(f2);
            ((__nv_bfloat16*)smem_P_T)[base + 3] = __float2bfloat16(f3);
        }
        __syncthreads();
        fence_async_shared_fn();

        // 3. dP^T = V @ dO^T
        issue_mma_8steps(0, desc_V_Kmaj, desc_dO_MNmaj, idesc_K_MN, false, 0, 1);
        wait_mma(mbar_mma, phase_mma);

        // 4. dS^T = P^T * (dP^T - D)
        float d_val = smem_D[row];
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float dp0 = __uint_as_float(r0);
            float dp1 = __uint_as_float(r1);
            float dp2 = __uint_as_float(r2);
            float dp3 = __uint_as_float(r3);
            
            uint32_t base = row * 128 + c;
            float p0 = __bfloat162float(((__nv_bfloat16*)smem_P_T)[base + 0]);
            float p1 = __bfloat162float(((__nv_bfloat16*)smem_P_T)[base + 1]);
            float p2 = __bfloat162float(((__nv_bfloat16*)smem_P_T)[base + 2]);
            float p3 = __bfloat162float(((__nv_bfloat16*)smem_P_T)[base + 3]);
            
            float scale = 0.0883883476f;
            float ds0 = p0 * (dp0 - d_val) * scale;
            float ds1 = p1 * (dp1 - d_val) * scale;
            float ds2 = p2 * (dp2 - d_val) * scale;
            float ds3 = p3 * (dp3 - d_val) * scale;
            
            ((__nv_bfloat16*)smem_dS_T)[base + 0] = __float2bfloat16(ds0);
            ((__nv_bfloat16*)smem_dS_T)[base + 1] = __float2bfloat16(ds1);
            ((__nv_bfloat16*)smem_dS_T)[base + 2] = __float2bfloat16(ds2);
            ((__nv_bfloat16*)smem_dS_T)[base + 3] = __float2bfloat16(ds3);
        }
        __syncthreads();
        fence_async_shared_fn();

        // 5. dV += P^T @ dO
        issue_mma_8steps(128, desc_PT_Kmaj, desc_dO_Kmaj, idesc_K_K, (i > 0), 0, 0);

        // 6. dK += dS^T @ Q
        issue_mma_8steps(256, desc_dST_Kmaj, desc_Q_Kmaj, idesc_K_K, (i > 0), 0, 0);

        // 7. dQ = dS @ K
        issue_mma_8steps(0, desc_dS_MNmaj, desc_K_MNmaj, idesc_MN_MN, false, 1, 1);
        
        wait_mma(mbar_mma, phase_mma);

        // Write dQ to global via atomicAdd
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t base = threadIdx.x * 128 + c;
            ((__nv_bfloat16*)smem_Q)[base + 0] = __float2bfloat16(__uint_as_float(r0));
            ((__nv_bfloat16*)smem_Q)[base + 1] = __float2bfloat16(__uint_as_float(r1));
            ((__nv_bfloat16*)smem_Q)[base + 2] = __float2bfloat16(__uint_as_float(r2));
            ((__nv_bfloat16*)smem_Q)[base + 3] = __float2bfloat16(__uint_as_float(r3));
        }
        __syncthreads();
        
        __nv_bfloat16* g_dQ = dQ_ptr + b * B_stride + h * H_stride + i * 128 * 128;
        for(int step = 0; step < 128; step += 4) {
            int r = step + threadIdx.x / 32;
            int c = (threadIdx.x % 32) * 4;
            __nv_bfloat162* src = (__nv_bfloat162*)(smem_Q + (r * 128 + c) * 2);
            __nv_bfloat162* dst = (__nv_bfloat162*)(g_dQ + r * 128 + c);
            atomicAdd(dst, src[0]);
            atomicAdd(dst + 1, src[1]);
        }
        __syncthreads(); 
    }

    wait_mma(mbar_mma, phase_mma);

    write_tmem_to_tma(&tma_dV, smem_V, 128, b, h, j);
    write_tmem_to_tma(&tma_dK, smem_K, 256, b, h, j);

    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr[0], 512);
    }
    __syncthreads();
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t S, uint64_t H, uint64_t B) {
    cuuint64_t globalDim[4] = {128, S, H, B};
    cuuint64_t globalStrides[3] = {128*2, S*128*2, H*S*128*2};
    cuuint32_t boxDim[4] = {128, 128, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    int64_t numel_D = B * H * S;

    float* d_D = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_D, numel_D * sizeof(float), stream));

    int64_t numel_dQ = B * H * S * d;
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, numel_dQ * 2, stream));

    int64_t threads_D = 256;
    int64_t blocks_D = (numel_D + threads_D - 1) / threads_D;
    compute_D_kernel<<<blocks_D, threads_D, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        d_D,
        numel_D
    );

    CUtensorMap tma_K, tma_V, tma_Q, tma_dO, tma_dK, tma_dV;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_dO, dO.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_dK, dK.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_dV, dV.data_ptr(), S, H, B));

    dim3 blocks(S / 128, H, B);
    dim3 threads(128);
    int smem_size = 197652;
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_bwd_kernel,
        tma_K, tma_V, tma_Q, tma_dO, tma_dK, tma_dV,
        static_cast<const float*>(L.data_ptr()),
        d_D,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<int>(S)
    ));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_D, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

}  // namespace tvm_ffi_mha_bwd