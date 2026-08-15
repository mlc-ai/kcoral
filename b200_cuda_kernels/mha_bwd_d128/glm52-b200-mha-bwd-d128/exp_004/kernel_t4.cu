#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace flash_attn_bwd {

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _err; \
        cuGetErrorString(_r, &_err); \
        fprintf(stderr, "CU error %s at %s:%d\n", _err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

constexpr int D = 128;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int THREADS = 128;

constexpr uint32_t TMEM_SP = 0;
constexpr uint32_t TMEM_DP = 128;
constexpr uint32_t TMEM_dV = 256;
constexpr uint32_t TMEM_dK = 384;

__device__ __forceinline__ uint32_t make_idesc(uint32_t ta) {
    uint32_t d = 0;
    d |= (1u << 4); d |= (1u << 7); d |= (1u << 10);
    d |= (ta << 15); d |= (0u << 16);
    d |= (16u << 17); d |= (8u << 24);
    return d;
}

__device__ __forceinline__ uint64_t make_sdesc(void* ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)0 << 16;
    d |= (uint64_t)64 << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t desc_off(uint64_t base, uint32_t bytes) {
    uint64_t d = base;
    uint32_t old_addr = (uint32_t)((d & 0x3FFF) << 4);
    uint32_t new_addr = old_addr + bytes;
    d &= ~((uint64_t)0x3FFF);
    d |= (uint64_t)(new_addr & 0x3FFFF) >> 4;
    return d;
}

__device__ __forceinline__ uint32_t k_kmajor_off(int k) {
    return (k / 4) * 16384 + (k % 4) * 32;
}

__device__ __forceinline__ void umma_smem_a(uint32_t td, uint64_t da, uint64_t db, uint32_t id, uint32_t acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(td), "l"(da), "l"(db), "r"(id), "r"(acc));
}

__device__ __forceinline__ void umma_tmem_a(uint32_t td, uint32_t ta, uint64_t db, uint32_t id, uint32_t acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(td), "r"(ta), "l"(db), "r"(id), "r"(acc));
}

__device__ __forceinline__ void umma_commit(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_ld4(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_st4(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tcgen05_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tma_load(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_etx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void fence_mbarrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

__device__ __forceinline__ uint32_t pack_bf16(float f0, float f1) {
    __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void unpack_bf16(uint32_t packed, float* f0, float* f1) {
    __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&packed);
    float2 f = __bfloat1622float2(v);
    *f0 = f.x; *f1 = f.y;
}

__global__ void compute_D_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D_out, int total_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    const __nv_bfloat162* dO_r = reinterpret_cast<const __nv_bfloat162*>(dO + (size_t)row * D);
    const __nv_bfloat162* O_r = reinterpret_cast<const __nv_bfloat162*>(O + (size_t)row * D);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < D / 2; i++) {
        float2 d = __bfloat1622float2(dO_r[i]);
        float2 o = __bfloat1622float2(O_r[i]);
        sum += d.x * o.x + d.y * o.y;
    }
    D_out[row] = sum;
}

__global__ void flash_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vals,
    float* __restrict__ dQ_workspace,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int kv_tile = blockIdx.y;
    int bn_start = kv_tile * BN;
    int b = bh / H, h = bh % H;
    int64_t bh_off = (int64_t)(b * H + h) * S;

    const float* L_bh = L + bh_off;
    const float* D_bh = D_vals + bh_off;
    float* dQ_bh = dQ_workspace + bh_off * D;
    __nv_bfloat16* dK_bh = dK_out + bh_off * D;
    __nv_bfloat16* dV_bh = dV_out + bh_off * D;

    extern __shared__ char smem_raw[];
    uintptr_t smem_aligned = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
    char* base = reinterpret_cast<char*>(smem_aligned);
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(base);
    __nv_bfloat16* V_smem = K_smem + 16384;
    __nv_bfloat16* Q_smem = V_smem + 16384;
    __nv_bfloat16* dO_smem = Q_smem + 16384;

    __shared__ __align__(8) uint64_t bar_tma;
    __shared__ __align__(8) uint64_t bar_mma;
    __shared__ uint32_t tmem_addr_smem;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    if (tid == 0) {
        mbarrier_init(&bar_tma, 1);
        mbarrier_init(&bar_mma, 1);
        fence_mbarrier_init();
    }
    __syncthreads();

    if (warp_id == 0) {
        tmem_alloc(&tmem_addr_smem, 512);
    }
    __syncthreads();
    uint32_t tmem_base = tmem_addr_smem;

    uint64_t k_desc = make_sdesc(K_smem);
    uint64_t v_desc = make_sdesc(V_smem);
    uint64_t q_desc = make_sdesc(Q_smem);
    uint64_t do_desc = make_sdesc(dO_smem);

    uint32_t idesc_nn = make_idesc(0);
    uint32_t idesc_tn = make_idesc(1);

    const float scale = 0.08838834764831845f;
    uint32_t mma_phase = 0;
    uint32_t tma_phase = 0;

    int kv_coord = (int)(bh_off + bn_start);

    // Load K (2 TMA loads of 16384 bytes each)
    if (tid == 0) {
        mbarrier_arrive_etx(&bar_tma, 32768);
        tma_load(&tma_K, &bar_tma, K_smem, 0, kv_coord);
        tma_load(&tma_K, &bar_tma, (char*)K_smem + 16384, 64, kv_coord);
    }
    mbarrier_wait(&bar_tma, tma_phase); tma_phase ^= 1;
    fence_async_shared();

    // Load V
    if (tid == 0) {
        mbarrier_arrive_etx(&bar_tma, 32768);
        tma_load(&tma_V, &bar_tma, V_smem, 0, kv_coord);
        tma_load(&tma_V, &bar_tma, (char*)V_smem + 16384, 64, kv_coord);
    }
    mbarrier_wait(&bar_tma, tma_phase); tma_phase ^= 1;
    fence_async_shared();

    int num_q_tiles = (S + BM - 1) / BM;

    for (int qt = 0; qt < num_q_tiles; qt++) {
        int bm_start = qt * BM;
        int q_coord = (int)(bh_off + bm_start);

        // Load Q and dO (4 TMA loads, 65536 bytes total)
        if (tid == 0) {
            mbarrier_arrive_etx(&bar_tma, 65536);
            tma_load(&tma_Q, &bar_tma, Q_smem, 0, q_coord);
            tma_load(&tma_Q, &bar_tma, (char*)Q_smem + 16384, 64, q_coord);
            tma_load(&tma_dO, &bar_tma, dO_smem, 0, q_coord);
            tma_load(&tma_dO, &bar_tma, (char*)dO_smem + 16384, 64, q_coord);
        }
        mbarrier_wait(&bar_tma, tma_phase); tma_phase ^= 1;
        fence_async_shared();

        float l_reg = (bm_start + tid < S) ? L_bh[bm_start + tid] : 0.0f;
        float d_reg = (bm_start + tid < S) ? D_bh[bm_start + tid] : 0.0f;
        int row = bm_start + tid;
        bool row_valid = (row < S);

        // MMA 1: S = Q @ K^T (ta=0, tb=0)
        if (tid == 0) {
            for (int k = 0; k < 8; k++) {
                umma_smem_a(tmem_base + TMEM_SP,
                    desc_off(q_desc, k_kmajor_off(k)),
                    desc_off(k_desc, k_kmajor_off(k)),
                    idesc_nn, (k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();

        // Softmax: P = exp(S * scale - L), store as BF16 packed to TMEM_SP
        for (int chunk = 0; chunk < 16; chunk++) {
            uint32_t s0,s1,s2,s3,s4,s5,s6,s7;
            tmem_ld4(tmem_base + TMEM_SP + chunk * 8, &s0,&s1,&s2,&s3);
            tmem_ld4(tmem_base + TMEM_SP + chunk * 8 + 4, &s4,&s5,&s6,&s7);
            tmem_wait_ld();

            float pv[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint32_t sv = (i < 4) ? (&s0)[i] : (&s4)[i-4];
                int col = bn_start + chunk * 8 + i;
                bool valid = row_valid && (col < S);
                pv[i] = valid ? __expf(__uint_as_float(sv) * scale - l_reg) : 0.0f;
            }
            tmem_st4(tmem_base + TMEM_SP + chunk * 4,
                pack_bf16(pv[0],pv[1]), pack_bf16(pv[2],pv[3]),
                pack_bf16(pv[4],pv[5]), pack_bf16(pv[6],pv[7]));
        }
        tmem_wait_st();
        __syncthreads();

        // MMA 2: dP = dO @ V^T (ta=0, tb=0)
        if (tid == 0) {
            tcgen05_fence_after();
            for (int k = 0; k < 8; k++) {
                umma_smem_a(tmem_base + TMEM_DP,
                    desc_off(do_desc, k_kmajor_off(k)),
                    desc_off(v_desc, k_kmajor_off(k)),
                    idesc_nn, (k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();

        // MMA 3: dV += P^T @ dO (ta=1, tb=0)
        if (tid == 0) {
            tcgen05_fence_after();
            for (int k = 0; k < 8; k++) {
                uint32_t ta = (tmem_base + TMEM_SP) | ((k * 16) << 16);
                umma_tmem_a(tmem_base + TMEM_dV, ta,
                    desc_off(do_desc, k_kmajor_off(k)),
                    idesc_tn, (qt == 0 && k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();

        // dS = P * (dP - D) * scale, store as BF16 packed to TMEM_DP
        for (int chunk = 0; chunk < 16; chunk++) {
            uint32_t dp0,dp1,dp2,dp3,dp4,dp5,dp6,dp7;
            tmem_ld4(tmem_base + TMEM_DP + chunk * 8, &dp0,&dp1,&dp2,&dp3);
            tmem_ld4(tmem_base + TMEM_DP + chunk * 8 + 4, &dp4,&dp5,&dp6,&dp7);

            uint32_t p0,p1,p2,p3;
            tmem_ld4(tmem_base + TMEM_SP + chunk * 4, &p0,&p1,&p2,&p3);
            tmem_wait_ld();

            float pf[8];
            unpack_bf16(p0, &pf[0], &pf[1]);
            unpack_bf16(p1, &pf[2], &pf[3]);
            unpack_bf16(p2, &pf[4], &pf[5]);
            unpack_bf16(p3, &pf[6], &pf[7]);

            float ds[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint32_t dpv = (i < 4) ? (&dp0)[i] : (&dp4)[i-4];
                int col = bn_start + chunk * 8 + i;
                bool valid = row_valid && (col < S);
                ds[i] = valid ? pf[i] * (__uint_as_float(dpv) - d_reg) * scale : 0.0f;
            }
            tmem_st4(tmem_base + TMEM_DP + chunk * 4,
                pack_bf16(ds[0],ds[1]), pack_bf16(ds[2],ds[3]),
                pack_bf16(ds[4],ds[5]), pack_bf16(ds[6],ds[7]));
        }
        tmem_wait_st();
        __syncthreads();

        // MMA 4: dK += dS^T @ Q (ta=1, tb=0)
        if (tid == 0) {
            tcgen05_fence_after();
            for (int k = 0; k < 8; k++) {
                uint32_t ta = (tmem_base + TMEM_DP) | ((k * 16) << 16);
                umma_tmem_a(tmem_base + TMEM_dK, ta,
                    desc_off(q_desc, k_kmajor_off(k)),
                    idesc_tn, (qt == 0 && k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();

        // MMA 5: dQ = dS @ K (ta=0 A from TMEM, tb=0 B from SMEM)
        if (tid == 0) {
            tcgen05_fence_after();
            for (int k = 0; k < 8; k++) {
                uint32_t ta = tmem_base + TMEM_DP + k * 8;
                umma_tmem_a(tmem_base + TMEM_SP, ta,
                    desc_off(k_desc, k_kmajor_off(k)),
                    idesc_nn, (k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();

        // Read dQ from TMEM_SP, atomic add to global
        for (int chunk = 0; chunk < 32; chunk++) {
            uint32_t r0,r1,r2,r3;
            tmem_ld4(tmem_base + TMEM_SP + chunk * 4, &r0,&r1,&r2,&r3);
            tmem_wait_ld();
            if (row_valid) {
                atomicAdd(&dQ_bh[(int64_t)row * D + chunk * 4 + 0], __uint_as_float(r0));
                atomicAdd(&dQ_bh[(int64_t)row * D + chunk * 4 + 1], __uint_as_float(r1));
                atomicAdd(&dQ_bh[(int64_t)row * D + chunk * 4 + 2], __uint_as_float(r2));
                atomicAdd(&dQ_bh[(int64_t)row * D + chunk * 4 + 3], __uint_as_float(r3));
            }
        }
        __syncthreads();
    }

    // Epilogue: store dK and dV to global
    for (int is_dK = 0; is_dK <= 1; is_dK++) {
        uint32_t tmem_off = is_dK ? (tmem_base + TMEM_dK) : (tmem_base + TMEM_dV);
        __nv_bfloat16* out_ptr = is_dK ? dK_bh : dV_bh;
        int row = bn_start + tid;
        bool row_valid = (row < S);
        for (int chunk = 0; chunk < 32; chunk++) {
            uint32_t r0,r1,r2,r3;
            tmem_ld4(tmem_off + chunk * 4, &r0,&r1,&r2,&r3);
            tmem_wait_ld();
            if (row_valid) {
                out_ptr[(int64_t)row * D + chunk * 4 + 0] = __float2bfloat16(__uint_as_float(r0));
                out_ptr[(int64_t)row * D + chunk * 4 + 1] = __float2bfloat16(__uint_as_float(r1));
                out_ptr[(int64_t)row * D + chunk * 4 + 2] = __float2bfloat16(__uint_as_float(r2));
                out_ptr[(int64_t)row * D + chunk * 4 + 3] = __float2bfloat16(__uint_as_float(r3));
            }
        }
        __syncthreads();
    }

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_base, 512);
    }
}

__global__ void convert_bf16_kernel(const float* src, __nv_bfloat16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

CUresult create_tma_desc(CUtensorMap* d, void* ptr, uint64_t inner, uint64_t outer) {
    cuuint64_t globalDim[2] = {inner, outer};
    cuuint64_t globalStrides[1] = {inner * 2};
    cuuint32_t boxDim[2] = {64, 128};
    cuuint32_t elemStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        ptr, globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = (int)Q.size(0), H = (int)Q.size(1), S = (int)Q.size(2);
    int total_rows = B * H * S;
    int total_elems = total_rows * D;

    float *D_temp, *dQ_ws;
    CUDA_CHECK(cudaMalloc(&D_temp, (size_t)total_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQ_ws, (size_t)total_elems * sizeof(float)));

    {
        int threads = 256, blocks = (total_rows + threads - 1) / threads;
        compute_D_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const __nv_bfloat16*>(O.data_ptr()),
            D_temp, total_rows);
    }

    CUDA_CHECK(cudaMemsetAsync(dQ_ws, 0, (size_t)total_elems * sizeof(float), stream));

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    int64_t total_outer = (int64_t)B * H * S;
    CU_CHECK(create_tma_desc(&tma_Q, Q.data_ptr(), D, total_outer));
    CU_CHECK(create_tma_desc(&tma_K, K.data_ptr(), D, total_outer));
    CU_CHECK(create_tma_desc(&tma_V, V.data_ptr(), D, total_outer));
    CU_CHECK(create_tma_desc(&tma_dO, dO.data_ptr(), D, total_outer));

    {
        dim3 grid(B * H, (S + BN - 1) / BN, 1);
        dim3 block(THREADS, 1, 1);
        int smem_size = 4 * 32768 + 2048;
        CUDA_CHECK(cudaFuncSetAttribute(flash_bwd_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        flash_bwd_kernel<<<grid, block, smem_size, stream>>>(
            tma_Q, tma_K, tma_V, tma_dO,
            static_cast<const float*>(L.data_ptr()), D_temp, dQ_ws,
            static_cast<__nv_bfloat16*>(dK.data_ptr()),
            static_cast<__nv_bfloat16*>(dV.data_ptr()), B, H, S);
    }
    CUDA_CHECK(cudaGetLastError());

    {
        int threads = 256, blocks = (total_elems + threads - 1) / threads;
        convert_bf16_kernel<<<blocks, threads, 0, stream>>>(
            dQ_ws, static_cast<__nv_bfloat16*>(dQ.data_ptr()), total_elems);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_temp));
    CUDA_CHECK(cudaFree(dQ_ws));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_bwd::run);

}  // namespace flash_attn_bwd