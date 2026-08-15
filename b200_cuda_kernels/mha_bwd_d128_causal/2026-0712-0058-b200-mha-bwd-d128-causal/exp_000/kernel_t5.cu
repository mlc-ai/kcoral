#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <mma.h>

#define CUTLASS_CHECK(call) do { \
    if ((call) != cudaSuccess) { \
        fprintf(stderr, "CUTLASS error %d at %s:%d\n", (int)(call), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace tvm_ffi_kernel {
using namespace nvcuda;

// ---------------- Async & TMA Instructions ----------------

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async_fn() { asm volatile("fence.proxy.async;\n" ::: "memory"); }

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(smem)), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

__device__ __forceinline__ void tma_store_commit_fn() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() { asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory"); }

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

// ---------------- WGMMA Descriptors ----------------

uint64_t create_desc_A_A0(void* p_ptr) {
    uint32_t ptr = (uint32_t)__cvta_generic_to_shared(p_ptr);
    uint32_t p0 = __shared_to_register(ptr, 0);
    uint32_t p1 = __shared_to_register(ptr, 4);
    uint32_t p2 = __shared_to_register(ptr, 8);
    uint32_t p3 = __shared_to_register(ptr, 12);
    uint32_t p4 = __shared_to_register(ptr, 16);
    uint32_t p5 = __shared_to_register(ptr, 20);
    uint32_t p6 = __shared_to_register(ptr, 24);
    uint32_t p7 = __shared_to_register(ptr, 28);
    p0 = __shuffle_sync(p0, 1, 0);
    p2 = __shuffle_sync(p2, 1, 0);
    p4 = __shuffle_sync(p4, 1, 0);
    p6 = __shuffle_sync(p6, 1, 0);
    uint32_t lo = (p0 << 16) | p1;
    uint32_t hi = (p4 << 16) | p5;
    uint32_t sbo = (p2 << 16) | p3;
    uint32_t size = (p7 << 16) | p6;
    lo = (lo & 0xFFFF0FFF) | (1 << 12);
    hi = (hi & 0xFFF0FFFE) | (2 << 25);
    return (((uint64_t)hi) << 32) | lo;
}

uint64_t create_desc_A_A1(void* p_ptr) {
    uint32_t ptr = (uint32_t)__cvta_generic_to_shared(p_ptr);
    uint32_t p0 = __shared_to_register(ptr, 0);
    uint32_t p1 = __shared_to_register(ptr, 4);
    uint32_t p2 = __shared_to_register(ptr, 8);
    uint32_t p3 = __shared_to_register(ptr, 12);
    uint32_t p4 = __shared_to_register(ptr, 16);
    uint32_t p5 = __shared_to_register(ptr, 20);
    uint32_t p6 = __shared_to_register(ptr, 24);
    uint32_t p7 = __shared_to_register(ptr, 28);
    p0 = __shuffle_sync(p0, 1, 0);
    p2 = __shuffle_sync(p2, 1, 0);
    p4 = __shuffle_sync(p4, 1, 0);
    p6 = __shuffle_sync(p6, 1, 0);
    uint32_t lo = (p0 << 16) | p1;
    uint32_t hi = (p4 << 16) | p5;
    uint32_t sbo = (p2 << 16) | p3;
    uint32_t size = (p7 << 16) | p6;
    lo = (lo & 0xFFFF0FFF) | (1 << 12);
    hi = (hi & 0xFFF0FFFE) | (2 << 25);
    return (((uint64_t)hi) << 32) | lo;
}

uint64_t create_desc_B_B0(void* p_ptr) {
    uint32_t ptr = (uint32_t)__cvta_generic_to_shared(p_ptr);
    uint32_t p0 = __shared_to_register(ptr, 0);
    uint32_t p1 = __shared_to_register(ptr, 4);
    uint32_t p2 = __shared_to_register(ptr, 8);
    uint32_t p3 = __shared_to_register(ptr, 12);
    uint32_t p4 = __shared_to_register(ptr, 16);
    uint32_t p5 = __shared_to_register(ptr, 20);
    uint32_t p6 = __shared_to_register(ptr, 24);
    uint32_t p7 = __shared_to_register(ptr, 28);
    p0 = __shuffle_sync(p0, 1, 0);
    p2 = __shuffle_sync(p2, 1, 0);
    p4 = __shuffle_sync(p4, 1, 0);
    p6 = __shuffle_sync(p6, 1, 0);
    uint32_t lo = (p0 << 16) | p1;
    uint32_t hi = (p4 << 16) | p5;
    uint32_t sbo = (p2 << 16) | p3;
    uint32_t size = (p7 << 16) | p6;
    lo = (lo & 0xFFFF0FFF) | (1 << 12);
    hi = (hi & 0xFFF0FFFE) | (2 << 25);
    return (((uint64_t)hi) << 32) | lo;
}

uint64_t create_desc_B_B1(void* p_ptr) {
    uint32_t ptr = (uint32_t)__cvta_generic_to_shared(p_ptr);
    uint32_t p0 = __shared_to_register(ptr, 0);
    uint32_t p1 = __shared_to_register(ptr, 4);
    uint32_t p2 = __shared_to_register(ptr, 8);
    uint32_t p3 = __shared_to_register(ptr, 12);
    uint32_t p4 = __shared_to_register(ptr, 16);
    uint32_t p5 = __shared_to_register(ptr, 20);
    uint32_t p6 = __shared_to_register(ptr, 24);
    uint32_t p7 = __shared_to_register(ptr, 28);
    p0 = __shuffle_sync(p0, 1, 0);
    p2 = __shuffle_sync(p2, 1, 0);
    p4 = __shuffle_sync(p4, 1, 0);
    p6 = __shuffle_sync(p6, 1, 0);
    uint32_t lo = (p0 << 16) | p1;
    uint32_t hi = (p4 << 16) | p5;
    uint32_t sbo = (p2 << 16) | p3;
    uint32_t size = (p7 << 16) | p6;
    lo = (lo & 0xFFFF0FFF) | (1 << 12);
    hi = (hi & 0xFFF0FFFE) | (2 << 25);
    return (((uint64_t)hi) << 32) | lo;
}

template<bool trans_a, bool trans_b>
__device__ __forceinline__ uint32_t make_idesc() {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (trans_a ? (1u << 15) : 0);
    d |= (trans_b ? (1u << 16) : 0);
    d |= (8 << 17);     // N >> 3
    d |= (4 << 24);     // M >> 4
    return d;
}

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t r, uint32_t c) {
    uint32_t x = c / 8;
    uint32_t rem = c % 8;
    uint32_t sx = (r % 8) ^ x;
    return r * 64 + sx * 8 + rem;
}

__device__ __forceinline__ __nv_bfloat16 read_smem_64x64(__nv_bfloat16* smem, uint32_t r, uint32_t c) {
    return smem[swizzle_128B(r, c)];
}

// ---------------- Kernels ----------------

__global__ void __launch_bounds__(128) bwd_dq_kernel(
    const __grid_constant__ CUtensorMap tma_Q_0, const __grid_constant__ CUtensorMap tma_Q_1,
    const __grid_constant__ CUtensorMap tma_K_0, const __grid_constant__ CUtensorMap tma_K_1,
    const __grid_constant__ CUtensorMap tma_V_0, const __grid_constant__ CUtensorMap tma_V_1,
    const __grid_constant__ CUtensorMap tma_O_0, const __grid_constant__ CUtensorMap tma_O_1,
    const __grid_constant__ CUtensorMap tma_dO_0, const __grid_constant__ CUtensorMap tma_dO_1,
    const __grid_constant__ CUtensorMap tma_dQ_0, const __grid_constant__ CUtensorMap tma_dQ_1,
    const float* L, __nv_bfloat16* dQ, int S) 
{
    int bh = blockIdx.y;
    int q_blk = blockIdx.x;

    extern __shared__ __align__(128) uint8_t smem[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem;                 // 0 KB
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem + 8192);        // 8 KB
    __nv_bfloat16* smem_O0 = (__nv_bfloat16*)(smem + 16384);       // 16 KB
    __nv_bfloat16* smem_O1 = (__nv_bfloat16*)(smem + 24576);       // 24 KB
    __nv_bfloat16* smem_dO0 = (__nv_bfloat16*)(smem + 32768);      // 32 KB
    __nv_bfloat16* smem_dO1 = (__nv_bfloat16*)(smem + 40960);      // 40 KB
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem + 49152);       // 48 KB
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem + 57344);       // 56 KB
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem + 65536);       // 64 KB
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem + 73728);       // 72 KB
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem + 81920);        // 80 KB
    __nv_bfloat16* smem_dP = (__nv_bfloat16*)(smem + 90112);       // 88 KB
    float* smem_D_o = (float*)(smem + 98304);                      // 96 KB
    float* smem_D = (float*)(smem + 98560);                        // 96.25 KB
    float* smem_L_row = (float*)(smem + 98816);                    // 96.5 KB
    uint64_t* mbar_load = (uint64_t*)(smem + 99072);               // 96.75 KB

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_load[0], 1);
        init_smem_barrier_fn(&mbar_load[1], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    bool is_causal = true;
    float scale = 1.0f / sqrtf(128.0f);

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_load[0], 16384 * 6);
        tma_load_2d_fn(&tma_Q_0, &mbar_load[0], smem_Q0, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_Q_1, &mbar_load[0], smem_Q1, 64, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_O_0, &mbar_load[0], smem_O0, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_O_1, &mbar_load[0], smem_O1, 64, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_dO_0, &mbar_load[0], smem_dO0, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_dO_1, &mbar_load[0], smem_dO1, 64, bh * S + q_blk * 64);
    }

    float my_L = -1e20f;
    if (threadIdx.x < 64) {
        my_L = (q_blk * 64 + threadIdx.x < S) ? L[bh * S + q_blk * 64 + threadIdx.x] : -1e20f;
        smem_L_row[threadIdx.x] = my_L;
    }

    mbarrier_wait_fn(&mbar_load[0], 0);
    fence_proxy_async_fn();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> Q_frag0, Q_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> K_frag0, K_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> V_frag0, V_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> dO_frag0, dO_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> dP_frag0, dP_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> dS_frag0, dS_frag1;

    wmma::accumulator dQ_acc0(64, 64);
    wmma::accumulator dQ_acc1(64, 64);

    constexpr uint32_t idesc_Q_K = make_idesc<false, true>();
    constexpr uint32_t idesc_dS_K = make_idesc<false, false>();

    int tid = threadIdx.x;
    float D_o[64] = {0};
    float D[64] = {0};

    // Pass 1: Compute D_o and D
    for (int k_blk = 0; k_blk <= q_blk; k_blk++) {
        if (is_causal && k_blk > q_blk) break;

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_load[1], 16384 * 4);
            tma_load_2d_fn(&tma_K_0, &mbar_load[1], smem_K0, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_K_1, &mbar_load[1], smem_K1, 64, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V_0, &mbar_load[1], smem_V0, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V_1, &mbar_load[1], smem_V1, 64, bh * S + k_blk * 64);
        }

        mbarrier_wait_fn(&mbar_load[1], k_blk % 2);
        fence_proxy_async_fn();
        __syncthreads();

        wmma::fill_fragment(dP_frag0, 0.0f);
        wmma::load_matrix_sync(Q_frag0, smem_Q0, 128);
        wmma::load_matrix_sync(K_frag0, smem_K0, 128);
        wmma::execute_async(dP_frag0, Q_frag0, K_frag0, dP_frag0, idesc_Q_K);
        wmma::load_matrix_sync(Q_frag1, smem_Q1, 128);
        wmma::load_matrix_sync(K_frag1, smem_K1, 128);
        wmma::execute_async(dP_frag0, Q_frag1, K_frag1, dP_frag0, idesc_Q_K);
        wmma::commit_sync();
        wmma::wait_sync(0);
        wmma::store_matrix_sync(dP_frag0, smem_P, 128);

        wmma::fill_fragment(dP_frag1, 0.0f);
        wmma::load_matrix_sync(dO_frag0, smem_dO0, 128);
        wmma::load_matrix_sync(V_frag0, smem_V0, 128);
        wmma::execute_async(dP_frag1, dO_frag0, V_frag0, dP_frag1, idesc_Q_K);
        wmma::load_matrix_sync(dO_frag1, smem_dO1, 128);
        wmma::load_matrix_sync(V_frag1, smem_V1, 128);
        wmma::execute_async(dP_frag1, dO_frag1, V_frag1, dP_frag1, idesc_Q_K);
        wmma::commit_sync();
        wmma::wait_sync(0);
        wmma::store_matrix_sync(dP_frag1, smem_dP, 128);

        __syncthreads();

        if (tid < 64) {
            for (int col = 0; col < 64; col++) {
                float p_val0 = __bfloat162float(read_smem_64x64(smem_P, tid, col));
                p_val0 *= scale;
                p_val0 = expf(p_val0 - my_L);
                
                float dp_val0 = __bfloat162float(read_smem_64x64(smem_dP, tid, col));
                bool valid = (k_blk * 64 + col <= q_blk * 64 + tid);
                if (valid) {
                    D_o[tid] += p_val0 * dp_val0;
                }
            }
        }
        __syncthreads();
    }

    // Pass 2: Compute dQ
    for (int k_blk = 0; k_blk <= q_blk; k_blk++) {
        if (is_causal && k_blk > q_blk) break;

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_load[1], 16384 * 4);
            tma_load_2d_fn(&tma_K_0, &mbar_load[1], smem_K0, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_K_1, &mbar_load[1], smem_K1, 64, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V_0, &mbar_load[1], smem_V0, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V_1, &mbar_load[1], smem_V1, 64, bh * S + k_blk * 64);
        }

        mbarrier_wait_fn(&mbar_load[1], k_blk % 2);
        fence_proxy_async_fn();
        __syncthreads();

        wmma::fill_fragment(dP_frag0, 0.0f);
        wmma::load_matrix_sync(Q_frag0, smem_Q0, 128);
        wmma::load_matrix_sync(K_frag0, smem_K0, 128);
        wmma::execute_async(dP_frag0, Q_frag0, K_frag0, dP_frag0, idesc_Q_K);
        wmma::load_matrix_sync(Q_frag1, smem_Q1, 128);
        wmma::load_matrix_sync(K_frag1, smem_K1, 128);
        wmma::execute_async(dP_frag0, Q_frag1, K_frag1, dP_frag0, idesc_Q_K);
        wmma::commit_sync();
        wmma::wait_sync(0);
        wmma::store_matrix_sync(dP_frag0, smem_P, 128);

        wmma::fill_fragment(dP_frag1, 0.0f);
        wmma::load_matrix_sync(dO_frag0, smem_dO0, 128);
        wmma::load_matrix_sync(V_frag0, smem_V0, 128);
        wmma::execute_async(dP_frag1, dO_frag0, V_frag0, dP_frag1, idesc_Q_K);
        wmma::load_matrix_sync(dO_frag1, smem_dO1, 128);
        wmma::load_matrix_sync(V_frag1, smem_V1, 128);
        wmma::execute_async(dP_frag1, dO_frag1, V_frag1, dP_frag1, idesc_Q_K);
        wmma::commit_sync();
        wmma::wait_sync(0);
        wmma::store_matrix_sync(dP_frag1, smem_dP, 128);

        __syncthreads();

        if (tid < 64) {
            for (int col = 0; col < 64; col++) {
                float p_val0 = __bfloat162float(read_smem_64x64(smem_P, tid, col));
                p_val0 *= scale;
                p_val0 = expf(p_val0 - my_L);
                
                float dp_val0 = __bfloat162float(read_smem_64x64(smem_dP, tid, col));
                bool valid = (k_blk * 64 + col <= q_blk * 64 + tid);
                
                float s_val = valid ? (p_val0 * (dp_val0 - D_o[tid])) : 0.0f;
                smem_dP[swizzle_128B(tid, col)] = __float2bfloat16(s_val);
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(dS_frag0, smem_dP, 128);
        wmma::load_matrix_sync(K_frag0, smem_K0, 128);
        wmma::execute_async(dQ_acc0, dS_frag0, K_frag0, dQ_acc0, idesc_dS_K);

        wmma::load_matrix_sync(dS_frag1, smem_dP, 128);
        wmma::load_matrix_sync(K_frag1, smem_K1, 128);
        wmma::execute_async(dQ_acc1, dS_frag1, K_frag1, dQ_acc1, idesc_dS_K);
        
        wmma::commit_sync();
        wmma::wait_sync(0);
        __syncthreads();
    }

    wmma::store_matrix_sync(dQ_acc0, smem_Q0, 128);
    wmma::store_matrix_sync(dQ_acc1, smem_Q1, 128);

    __syncthreads();

    if (threadIdx.x == 0) {
        tma_store_fence_fn();
        tma_store_2d_fn(&tma_dQ_0, smem_Q0, 0, bh * S + q_blk * 64);
        tma_store_2d_fn(&tma_dQ_1, smem_Q1, 64, bh * S + q_blk * 64);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
}

__global__ void __launch_bounds__(128) bwd_dkv_kernel(
    const __grid_constant__ CUtensorMap tma_Q_0, const __grid_constant__ CUtensorMap tma_Q_1,
    const __grid_constant__ CUtensorMap tma_K_0, const __grid_constant__ CUtensorMap tma_K_1,
    const __grid_constant__ CUtensorMap tma_V_0, const __grid_constant__ CUtensorMap tma_V_1,
    const __grid_constant__ CUtensorMap tma_O_0, const __grid_constant__ CUtensorMap tma_O_1,
    const __grid_constant__ CUtensorMap tma_dO_0, const __grid_constant__ CUtensorMap tma_dO_1,
    const __grid_constant__ CUtensorMap tma_dK_0, const __grid_constant__ CUtensorMap tma_dK_1,
    const __grid_constant__ CUtensorMap tma_dV_0, const __grid_constant__ CUtensorMap tma_dV_1,
    const float* L, __nv_bfloat16* dK, __nv_bfloat16* dV, int S) 
{
    int bh = blockIdx.y;
    int k_blk = blockIdx.x;

    extern __shared__ __align__(128) uint8_t smem[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem;                  // 0 KB
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem + 8192);         // 8 KB
    __nv_bfloat16* smem_O0 = (__nv_bfloat16*)(smem + 16384);        // 16 KB
    __nv_bfloat16* smem_O1 = (__nv_bfloat16*)(smem + 24576);        // 24 KB
    __nv_bfloat16* smem_dO0 = (__nv_bfloat16*)(smem + 32768);       // 32 KB
    __nv_bfloat16* smem_dO1 = (__nv_bfloat16*)(smem + 40960);       // 40 KB
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem + 49152);        // 48 KB
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem + 57344);        // 56 KB
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem + 65536);        // 64 KB
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem + 73728);        // 72 KB
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem + 81920);         // 80 KB
    __nv_bfloat16* smem_dP = (__nv_bfloat16*)(smem + 90112);        // 88 KB
    __nv_bfloat16* smem_dS = (__nv_bfloat16*)(smem + 98304);        // 96 KB
    __nv_bfloat16* smem_P_T = (__nv_bfloat16*)(smem + 106496);      // 104 KB
    __nv_bfloat16* smem_dS_T = (__nv_bfloat16*)(smem + 114688);     // 112 KB

    __nv_bfloat16* smem_Q_T = smem_dS_T;                            // 112 KB
    __nv_bfloat16* smem_K_T = smem_P_T;                             // 104 KB
    __nv_bfloat16* smem_O_T = smem_V0;                              // 64 KB
    __nv_bfloat16* smem_V_T = smem_V1;                              // 72 KB
    __nv_bfloat16* smem_dO_T = smem_O0;                             // 16 KB

    float* smem_D_o = (float*)(smem + 122880);                      // 120 KB
    float* smem_D = (float*)(smem + 123136);                        // 120.25 KB
    float* smem_L_row = (float*)(smem + 123392);                    // 120.5 KB
    uint64_t* mbar_load = (uint64_t*)(smem + 123648);               // 120.75 KB

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_load[0], 1);
        init_smem_barrier_fn(&mbar_load[1], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    bool is_causal = true;
    float scale = 1.0f / sqrtf(128.0f);

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_load[0], 16384 * 4);
        tma_load_2d_fn(&tma_K_0, &mbar_load[0], smem_K0, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_K_1, &mbar_load[0], smem_K1, 64, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V_0, &mbar_load[0], smem_V0, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V_1, &mbar_load[0], smem_V1, 64, bh * S + k_blk * 64);
    }

    mbarrier_wait_fn(&mbar_load[0], 0);
    fence_proxy_async_fn();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> Q_frag0, Q_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> K_frag0, K_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> V_frag0, V_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> dO_frag0, dO_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> P_frag0, P_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> dP_frag0, dP_frag1;

    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> P_T_frag0, P_T_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> Q_T_frag0, Q_T_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> O_T_frag0, O_T_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> K_T_frag0, K_T_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> dO_T_frag0, dO_T_frag1;
    wmma::fragment<wmma::matrix_a, 64, 64, 64, __nv_bfloat16, wmma::row_major> dS_T_frag0, dS_T_frag1;

    wmma::accumulator dK_acc0(64, 64);
    wmma::accumulator dK_acc1(64, 64);
    wmma::accumulator dV_acc0(64, 64);
    wmma::accumulator dV_acc1(64, 64);

    constexpr uint32_t idesc_Q_K = make_idesc<false, true>();
    constexpr uint32_t idesc_dS_Q = make_idesc<true, true>();
    constexpr uint32_t idesc_P_dO = make_idesc<true, true>();

    int num_q_blocks = (S + 63) / 64;

    int tid = threadIdx.x;
    bool valid_k = (k_blk * 64 + tid < S);

    for (int q_blk = k_blk; q_blk < num_q_blocks; q_blk++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_load[1], 16384 * 6);
            tma_load_2d_fn(&tma_Q_0, &mbar_load[1], smem_Q0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_Q_1, &mbar_load[1], smem_Q1, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O_0, &mbar_load[1], smem_O0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O_1, &mbar_load[1], smem_O1, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO_0, &mbar_load[1], smem_dO0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO_1, &mbar_load[1], smem_dO1, 64, bh * S + q_blk * 64);
        }

        mbarrier_wait_fn(&mbar_load[1], q_blk % 2);
        fence_proxy_async_fn();
        __syncthreads();

        if (tid < 64) {
            smem_L_row[tid] = (q_blk * 64 + tid < S) ? L[bh * S + q_blk * 64 + tid] : -1e20f;
        }

        wmma::fill_fragment(P_frag0, 0.0f);
        wmma::load_matrix_sync(Q_frag0, smem_Q0, 128);
        wmma::load_matrix_sync(K_frag0, smem_K0, 128);
        wmma::execute_async(P_frag0, Q_frag0, K_frag0, P_frag0, idesc_Q_K);
        wmma::load_matrix_sync(Q_frag1, smem_Q1, 128);
        wmma::load_matrix_sync(K_frag1, smem_K1, 128);
        wmma::execute_async(P_frag0, Q_frag1, K_frag1, P_frag0, idesc_Q_K);
        wmma::commit_sync();
        wmma::wait_sync(0);
        wmma::store_matrix_sync(P_frag0, smem_P, 128);

        wmma::fill_fragment(dP_frag0, 0.0f);
        wmma::load_matrix_sync(dO_frag0, smem_dO0, 128);
        wmma::load_matrix_sync(V_frag0, smem_V0, 128);
        wmma::execute_async(dP_frag0, dO_frag0, V_frag0, dP_frag0, idesc_Q_K);
        wmma::load_matrix_sync(dO_frag1, smem_dO1, 128);
        wmma::load_matrix_sync(V_frag1, smem_V1, 128);
        wmma::execute_async(dP_frag0, dO_frag1, V_frag1, dP_frag0, idesc_Q_K);
        wmma::commit_sync();
        wmma::wait_sync(0);
        wmma::store_matrix_sync(dP_frag0, smem_dP, 128);

        __syncthreads();

        if (valid_k) {
            smem_D_o[tid] = 0.0f;
            smem_D[tid] = 0.0f;
        }

        float my_L = smem_L_row[tid];

        for (int col = 0; col < 64; col++) {
            float p_val0 = __bfloat162float(read_smem_64x64(smem_P, tid, col));
            p_val0 *= scale;
            p_val0 = expf(p_val0 - my_L);
            
            float dp_val0 = __bfloat162float(read_smem_64x64(smem_dP, tid, col));
            
            bool valid_q = (q_blk * 64 + tid < S);
            if (valid_k && valid_q && (k_blk * 64 + col <= q_blk * 64 + tid)) {
                smem_D_o[tid] += p_val0 * dp_val0;
                smem_D[tid] += p_val0 * dp_val0;
            }
        }
        __syncthreads();

        float D_o_val = smem_D_o[tid];
        float D_val = smem_D[tid];

        for (int col = 0; col < 64; col++) {
            float p_val0 = __bfloat162float(read_smem_64x64(smem_P, tid, col));
            p_val0 *= scale;
            p_val0 = expf(p_val0 - my_L);
            
            float dp_val0 = __bfloat162float(read_smem_64x64(smem_dP, tid, col));
            
            bool valid_q = (q_blk * 64 + tid < S);
            bool valid = valid_k && valid_q && (k_blk * 64 + col <= q_blk * 64 + tid);
            
            float s_val = valid ? (p_val0 * (dp_val0 - D_o_val)) : 0.0f;
            smem_dS[swizzle_128B(col, tid)] = __float2bfloat16(s_val); 
            smem_P_T[swizzle_128B(col, tid)] = smem_P[swizzle_128B(tid, col)];
        }

        __syncthreads();

        wmma::load_matrix_sync(dS_T_frag0, smem_dS, 128);
        wmma::load_matrix_sync(Q_T_frag0, smem_Q_T, 128);
        wmma::execute_async(dK_acc0, dS_T_frag0, Q_T_frag0, dK_acc0, idesc_dS_Q);

        wmma::load_matrix_sync(dS_T_frag1, smem_dS, 128);
        wmma::load_matrix_sync(Q_T_frag1, smem_Q_T, 128);
        wmma::execute_async(dK_acc1, dS_T_frag1, Q_T_frag1, dK_acc1, idesc_dS_Q);

        wmma::load_matrix_sync(P_T_frag0, smem_P_T, 128);
        wmma::load_matrix_sync(dO_T_frag0, smem_dO_T, 128);
        wmma::execute_async(dV_acc0, P_T_frag0, dO_T_frag0, dV_acc0, idesc_P_dO);

        wmma::load_matrix_sync(P_T_frag1, smem_P_T, 128);
        wmma::load_matrix_sync(dO_T_frag1, smem_dO_T, 128);
        wmma::execute_async(dV_acc1, P_T_frag1, dO_T_frag1, dV_acc1, idesc_P_dO);
        
        wmma::commit_sync();
        wmma::wait_sync(0);
        __syncthreads();
    }

    wmma::store_matrix_sync(dK_acc0, smem_K0, 128);
    wmma::store_matrix_sync(dK_acc1, smem_K1, 128);
    wmma::store_matrix_sync(dV_acc0, smem_V0, 128);
    wmma::store_matrix_sync(dV_acc1, smem_V1, 128);

    __syncthreads();

    if (threadIdx.x == 0) {
        tma_store_fence_fn();
        tma_store_2d_fn(&tma_dK_0, smem_K0, 0, bh * S + k_blk * 64);
        tma_store_2d_fn(&tma_dK_1, smem_K1, 64, bh * S + k_blk * 64);
        tma_store_2d_fn(&tma_dV_0, smem_V0, 0, bh * S + k_blk * 64);
        tma_store_2d_fn(&tma_dV_1, smem_V1, 64, bh * S + k_blk * 64);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, 
         tvm::ffi::TensorView V, tvm::ffi::TensorView O, 
         tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, 
         tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 
    
    CUtensorMap tma_Q_0, tma_Q_1, tma_K_0, tma_K_1, tma_V_0, tma_V_1, tma_O_0, tma_O_1, tma_dO_0, tma_dO_1;
    CUtensorMap tma_dQ_0, tma_dQ_1, tma_dK_0, tma_dK_1, tma_dV_0, tma_dV_1;

    create_tma_2d_descriptor_2B(&tma_Q_0, Q.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_Q_1, Q.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    create_tma_2d_descriptor_2B(&tma_K_0, K.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K_1, K.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    create_tma_2d_descriptor_2B(&tma_V_0, V.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V_1, V.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    create_tma_2d_descriptor_2B(&tma_O_0, O.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O_1, O.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    create_tma_2d_descriptor_2B(&tma_dO_0, dO.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO_1, dO.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    create_tma_2d_descriptor_2B(&tma_dQ_0, dQ.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dQ_1, dQ.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    create_tma_2d_descriptor_2B(&tma_dK_0, dK.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dK_1, dK.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    create_tma_2d_descriptor_2B(&tma_dV_0, dV.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dV_1, dV.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    int num_q_blocks = (S + 63) / 64;
    dim3 grid_dq(num_q_blocks, B * H);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304 + 128));
    bwd_dq_kernel<<<grid_dq, block, 98304 + 128, stream>>>(
        tma_Q_0, tma_Q_1, tma_K_0, tma_K_1, tma_V_0, tma_V_1,
        tma_O_0, tma_O_1, tma_dO_0, tma_dO_1, tma_dQ_0, tma_dQ_1,
        L.data_ptr(), dQ.data_ptr(), S
    );

    dim3 grid_dkv(num_q_blocks, B * H);
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 139264 + 128));
    bwd_dkv_kernel<<<grid_dkv, block, 139264 + 128, stream>>>(
        tma_Q_0, tma_Q_1, tma_K_0, tma_K_1, tma_V_0, tma_V_1,
        tma_O_0, tma_O_1, tma_dO_0, tma_dO_1, tma_dK_0, tma_dK_1, tma_dV_0, tma_dV_1,
        L.data_ptr(), dK.data_ptr(), dV.data_ptr(), S
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel