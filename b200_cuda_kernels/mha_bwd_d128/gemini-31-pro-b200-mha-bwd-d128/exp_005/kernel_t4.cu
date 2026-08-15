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

__device__ __forceinline__ uint64_t make_smem_desc_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)0 << 61;   // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int a_maj, int b_maj) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (a_maj << 15);
    d |= (b_maj << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

__device__ void write_tmem_to_tma(const CUtensorMap* tma_desc, uint8_t* smem_staging, uint32_t tmem_base, int b, int h, int row_idx) {
    for (int c = 0; c < 128; c += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" 
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_base + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t p0 = pack_bf16_fn(r0, r1);
        uint32_t p1 = pack_bf16_fn(r2, r3);
        uint32_t p2 = pack_bf16_fn(r4, r5);
        uint32_t p3 = pack_bf16_fn(r6, r7);
        
        int byte_offset = threadIdx.x * 256 + c * 2;
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :: "r"((uint32_t)__cvta_generic_to_shared(smem_staging + byte_offset)),
                        "r"(p0), "r"(p1), "r"(p2), "r"(p3) : "memory");
    }
    __syncthreads();
    fence_async_shared_fn();
    if (threadIdx.x == 0) {
        tma_store_4d_fn(tma_desc, smem_staging, 0, row_idx * 128, h, b);
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
    
    setmaxnreg_inc_sync_fn<256>();
    __syncthreads();

    uint32_t base_tmem = tmem_addr[0];

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_tma, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    __syncthreads();

    // With SWIZZLE_NONE:
    // K-Major LBO=2048, SBO=128
    // MN-Major LBO=128, SBO=2048
    uint64_t desc_K_Kmaj = make_smem_desc_fn(smem_K, 2048, 128);
    uint64_t desc_V_Kmaj = make_smem_desc_fn(smem_V, 2048, 128);
    uint64_t desc_Q_Kmaj = make_smem_desc_fn(smem_Q, 2048, 128);
    uint64_t desc_dO_Kmaj = make_smem_desc_fn(smem_dO, 2048, 128);
    uint64_t desc_PT_Kmaj = make_smem_desc_fn(smem_P_T, 2048, 128);
    uint64_t desc_dST_Kmaj = make_smem_desc_fn(smem_dS_T, 2048, 128);

    uint64_t desc_dO_MNmaj = make_smem_desc_fn(smem_dO, 128, 2048);
    uint64_t desc_Q_MNmaj = make_smem_desc_fn(smem_Q, 128, 2048);
    uint64_t desc_dS_MNmaj = make_smem_desc_fn(smem_dS_T, 128, 2048);
    uint64_t desc_K_MNmaj = make_smem_desc_fn(smem_K, 128, 2048);

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

        // 1. S^T = K @ Q^T (K is row-major, Q^T is col-major -> K-Maj, K-Maj)
        issue_mma_8steps(base_tmem + 0, desc_K_Kmaj, desc_Q_Kmaj, idesc_K_K, false, 0, 0);
        wait_mma(mbar_mma, phase_mma);

        // 2. P^T = exp(S^T - LSE)
        for (int c = 0; c < 128; c += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" 
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(base_tmem + 0 + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float l0 = smem_LSE[c + 0];
            float l1 = smem_LSE[c + 1];
            float l2 = smem_LSE[c + 2];
            float l3 = smem_LSE[c + 3];
            float l4 = smem_LSE[c + 4];
            float l5 = smem_LSE[c + 5];
            float l6 = smem_LSE[c + 6];
            float l7 = smem_LSE[c + 7];

            float f0 = fast_exp2f_fn(__uint_as_float(r0) * 0.12751532f - l0 * 1.44269504f);
            float f1 = fast_exp2f_fn(__uint_as_float(r1) * 0.12751532f - l1 * 1.44269504f);
            float f2 = fast_exp2f_fn(__uint_as_float(r2) * 0.12751532f - l2 * 1.44269504f);
            float f3 = fast_exp2f_fn(__uint_as_float(r3) * 0.12751532f - l3 * 1.44269504f);
            float f4 = fast_exp2f_fn(__uint_as_float(r4) * 0.12751532f - l4 * 1.44269504f);
            float f5 = fast_exp2f_fn(__uint_as_float(r5) * 0.12751532f - l5 * 1.44269504f);
            float f6 = fast_exp2f_fn(__uint_as_float(r6) * 0.12751532f - l6 * 1.44269504f);
            float f7 = fast_exp2f_fn(__uint_as_float(r7) * 0.12751532f - l7 * 1.44269504f);
            
            uint32_t p0 = pack_bf16_fn(*(uint32_t*)&f0, *(uint32_t*)&f1);
            uint32_t p1 = pack_bf16_fn(*(uint32_t*)&f2, *(uint32_t*)&f3);
            uint32_t p2 = pack_bf16_fn(*(uint32_t*)&f4, *(uint32_t*)&f5);
            uint32_t p3 = pack_bf16_fn(*(uint32_t*)&f6, *(uint32_t*)&f7);
            
            int byte_offset = threadIdx.x * 256 + c * 2;
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :: "r"((uint32_t)__cvta_generic_to_shared(smem_P_T + byte_offset)),
                            "r"(p0), "r"(p1), "r"(p2), "r"(p3) : "memory");
        }
        __syncthreads();
        fence_async_shared_fn();

        // 3. dP^T = V @ dO^T
        issue_mma_8steps(base_tmem + 0, desc_V_Kmaj, desc_dO_Kmaj, idesc_K_K, false, 0, 0);
        wait_mma(mbar_mma, phase_mma);

        // 4. dS^T = P^T * (dP^T - D)
        for (int c = 0; c < 128; c += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" 
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(base_tmem + 0 + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float d0 = smem_D[c + 0];
            float d1 = smem_D[c + 1];
            float d2 = smem_D[c + 2];
            float d3 = smem_D[c + 3];
            float d4 = smem_D[c + 4];
            float d5 = smem_D[c + 5];
            float d6 = smem_D[c + 6];
            float d7 = smem_D[c + 7];
            
            int byte_offset = threadIdx.x * 256 + c * 2;
            uint4 p_data = *reinterpret_cast<uint4*>(smem_P_T + byte_offset);
            
            float p0 = __bfloat162float(*(reinterpret_cast<__nv_bfloat16*>(&p_data.x)));
            float p1 = __bfloat162float(*((reinterpret_cast<__nv_bfloat16*>(&p_data.x)) + 1));
            float p2 = __bfloat162float(*(reinterpret_cast<__nv_bfloat16*>(&p_data.y)));
            float p3 = __bfloat162float(*((reinterpret_cast<__nv_bfloat16*>(&p_data.y)) + 1));
            float p4 = __bfloat162float(*(reinterpret_cast<__nv_bfloat16*>(&p_data.z)));
            float p5 = __bfloat162float(*((reinterpret_cast<__nv_bfloat16*>(&p_data.z)) + 1));
            float p6 = __bfloat162float(*(reinterpret_cast<__nv_bfloat16*>(&p_data.w)));
            float p7 = __bfloat162float(*((reinterpret_cast<__nv_bfloat16*>(&p_data.w)) + 1));
            
            float scale = 0.0883883476f;
            float ds0 = p0 * (__uint_as_float(r0) - d0) * scale;
            float ds1 = p1 * (__uint_as_float(r1) - d1) * scale;
            float ds2 = p2 * (__uint_as_float(r2) - d2) * scale;
            float ds3 = p3 * (__uint_as_float(r3) - d3) * scale;
            float ds4 = p4 * (__uint_as_float(r4) - d4) * scale;
            float ds5 = p5 * (__uint_as_float(r5) - d5) * scale;
            float ds6 = p6 * (__uint_as_float(r6) - d6) * scale;
            float ds7 = p7 * (__uint_as_float(r7) - d7) * scale;
            
            uint32_t out0 = pack_bf16_fn(*(uint32_t*)&ds0, *(uint32_t*)&ds1);
            uint32_t out1 = pack_bf16_fn(*(uint32_t*)&ds2, *(uint32_t*)&ds3);
            uint32_t out2 = pack_bf16_fn(*(uint32_t*)&ds4, *(uint32_t*)&ds5);
            uint32_t out3 = pack_bf16_fn(*(uint32_t*)&ds6, *(uint32_t*)&ds7);
            
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :: "r"((uint32_t)__cvta_generic_to_shared(smem_dS_T + byte_offset)),
                            "r"(out0), "r"(out1), "r"(out2), "r"(out3) : "memory");
        }
        __syncthreads();
        fence_async_shared_fn();

        // 5. dV += P^T @ dO
        issue_mma_8steps(base_tmem + 128, desc_PT_Kmaj, desc_dO_MNmaj, idesc_K_MN, (i > 0), 0, 1);

        // 6. dK += dS^T @ Q
        issue_mma_8steps(base_tmem + 256, desc_dST_Kmaj, desc_Q_MNmaj, idesc_K_MN, (i > 0), 0, 1);

        // 7. dQ = dS @ K
        issue_mma_8steps(base_tmem + 0, desc_dS_MNmaj, desc_K_MNmaj, idesc_MN_MN, false, 1, 1);
        
        wait_mma(mbar_mma, phase_mma);

        // Write dQ to global via atomicAdd
        for (int c = 0; c < 128; c += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" 
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(base_tmem + 0 + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t p0 = pack_bf16_fn(r0, r1);
            uint32_t p1 = pack_bf16_fn(r2, r3);
            uint32_t p2 = pack_bf16_fn(r4, r5);
            uint32_t p3 = pack_bf16_fn(r6, r7);
            
            int byte_offset = threadIdx.x * 256 + c * 2;
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :: "r"((uint32_t)__cvta_generic_to_shared(smem_Q + byte_offset)),
                            "r"(p0), "r"(p1), "r"(p2), "r"(p3) : "memory");
        }
        __syncthreads();
        
        __nv_bfloat16* g_dQ = dQ_ptr + b * B_stride + h * H_stride + i * 128 * 128;
        for (int step = 0; step < 16; ++step) {
            int r = step * 8 + threadIdx.x / 16;
            int chunk = threadIdx.x % 16;
            int byte_offset = r * 256 + chunk * 16;
            
            uint4 data = *reinterpret_cast<uint4*>(smem_Q + byte_offset);
            
            int global_col = chunk * 8;
            __nv_bfloat162* g_ptr = (__nv_bfloat162*)(g_dQ + r * 128 + global_col);
            atomicAdd(g_ptr,     *(reinterpret_cast<__nv_bfloat162*>(&data.x)));
            atomicAdd(g_ptr + 1, *(reinterpret_cast<__nv_bfloat162*>(&data.y)));
            atomicAdd(g_ptr + 2, *(reinterpret_cast<__nv_bfloat162*>(&data.z)));
            atomicAdd(g_ptr + 3, *(reinterpret_cast<__nv_bfloat162*>(&data.w)));
        }
        __syncthreads(); 
    }

    wait_mma(mbar_mma, phase_mma);

    write_tmem_to_tma(&tma_dV, smem_V, base_tmem + 128, b, h, j);
    write_tmem_to_tma(&tma_dK, smem_K, base_tmem + 256, b, h, j);

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