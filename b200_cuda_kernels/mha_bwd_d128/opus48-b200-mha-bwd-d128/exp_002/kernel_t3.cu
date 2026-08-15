#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;
typedef __nv_bfloat16 bf16;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {

constexpr int LD128 = 136; // padded leading dim for 128-wide bf16/float tiles
constexpr int LD64  = 72;  // padded leading dim for 64-wide tiles

__device__ __forceinline__ void cpa_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }
template<int N> __device__ __forceinline__ void cpa_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

// Load 64 rows x 128 cols bf16 into padded smem[64][LD128] via cp.async
__device__ __forceinline__ void load_tile_async(const bf16* src, bf16* smem, int base, int S) {
    const int4* s4 = reinterpret_cast<const int4*>(src);
    #pragma unroll
    for (int e = threadIdx.x; e < 64 * 16; e += 256) {
        int row = e >> 4, w = e & 15, g = base + row;
        bf16* dst = smem + row * LD128 + w * 8;
        uint32_t sdst = (uint32_t)__cvta_generic_to_shared(dst);
        if (g < S) {
            const int4* gp = s4 + (size_t)g * 16 + w;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" :: "r"(sdst), "l"(gp) : "memory");
        } else {
            asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};\n" :: "r"(sdst) : "memory");
        }
    }
}

__device__ __forceinline__ void store_tile(bf16* dst, const float* sOut, int base, int S) {
    int4* d4 = reinterpret_cast<int4*>(dst);
    #pragma unroll
    for (int e = threadIdx.x; e < 64 * 16; e += 256) {
        int row = e >> 4, w = e & 15, g = base + row;
        const float* p = sOut + row * LD128 + w * 8;
        float4 f0 = *reinterpret_cast<const float4*>(p);
        float4 f1 = *reinterpret_cast<const float4*>(p + 4);
        __nv_bfloat162 h0 = __floats2bfloat162_rn(f0.x, f0.y);
        __nv_bfloat162 h1 = __floats2bfloat162_rn(f0.z, f0.w);
        __nv_bfloat162 h2 = __floats2bfloat162_rn(f1.x, f1.y);
        __nv_bfloat162 h3 = __floats2bfloat162_rn(f1.z, f1.w);
        int4 out;
        out.x = *reinterpret_cast<int*>(&h0);
        out.y = *reinterpret_cast<int*>(&h1);
        out.z = *reinterpret_cast<int*>(&h2);
        out.w = *reinterpret_cast<int*>(&h3);
        if (g < S) d4[(size_t)g * 16 + w] = out;
    }
}

// ---------------- Preprocess: Delta = rowsum(O * dO) ----------------
__global__ void compute_delta(const bf16* __restrict__ O,
                              const bf16* __restrict__ dO,
                              float* __restrict__ Delta, int total_rows) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (row >= total_rows) return;
    const int4* o4 = reinterpret_cast<const int4*>(O + (size_t)row * 128);
    const int4* g4 = reinterpret_cast<const int4*>(dO + (size_t)row * 128);
    float s = 0.f;
    if (lane < 16) {
        int4 ov = o4[lane];
        int4 gv = g4[lane];
        const bf16* ob = reinterpret_cast<const bf16*>(&ov);
        const bf16* gb = reinterpret_cast<const bf16*>(&gv);
        #pragma unroll
        for (int k = 0; k < 8; k++) s += __bfloat162float(ob[k]) * __bfloat162float(gb[k]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    if (lane == 0) Delta[row] = s;
}

// C[64][64] = A[64][128] @ B[64][128]^T (contract 128). 8 warps, A-fragment reuse.
__device__ __forceinline__ void gemm_qkt(const bf16* A, const bf16* B, float* C, int warp) {
    int rt = warp >> 1, cbase = (warp & 1) * 2;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fill_fragment(acc[0], 0.f);
    wmma::fill_fragment(acc[1], 0.f);
    #pragma unroll
    for (int ks = 0; ks < 8; ks++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
        wmma::load_matrix_sync(a, A + rt * 16 * LD128 + ks * 16, LD128);
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
            wmma::load_matrix_sync(b, B + (cbase + c) * 16 * LD128 + ks * 16, LD128);
            wmma::mma_sync(acc[c], a, b, acc[c]);
        }
    }
    #pragma unroll
    for (int c = 0; c < 2; c++)
        wmma::store_matrix_sync(C + rt * 16 * LD64 + (cbase + c) * 16, acc[c], LD64, wmma::mem_row_major);
}

// ---------------- dK / dV kernel ----------------
__global__ void __launch_bounds__(256) bwd_dkv_kernel(
        const bf16* __restrict__ Q, const bf16* __restrict__ K,
        const bf16* __restrict__ V, const bf16* __restrict__ dO,
        const float* __restrict__ L, const float* __restrict__ Delta,
        bf16* __restrict__ dK, bf16* __restrict__ dV,
        int S, int H, float scale) {
    int b = blockIdx.z, h = blockIdx.y, kv_block = blockIdx.x;
    int j0 = kv_block * 64;
    int tid = threadIdx.x, warp = tid >> 5;

    size_t bh = (size_t)b * H + h;
    const bf16* Qbh  = Q  + bh * S * 128;
    const bf16* Kbh  = K  + bh * S * 128;
    const bf16* Vbh  = V  + bh * S * 128;
    const bf16* dObh = dO + bh * S * 128;
    const float* Lbh = L + bh * S;
    const float* Dbh = Delta + bh * S;
    bf16* dKbh = dK + bh * S * 128;
    bf16* dVbh = dV + bh * S * 128;

    extern __shared__ char smem_raw[];
    char* p = smem_raw;
    bf16* sK   = (bf16*)p; p += 64 * LD128 * 2;
    bf16* sV   = (bf16*)p; p += 64 * LD128 * 2;
    bf16* sQ2  = (bf16*)p; p += 2 * 64 * LD128 * 2;
    bf16* sdO2 = (bf16*)p; p += 2 * 64 * LD128 * 2;
    bf16* sPbf = (bf16*)p; p += 64 * LD64 * 2;
    bf16* sdS  = (bf16*)p; p += 64 * LD64 * 2;
    float* sS  = (float*)p; p += 64 * LD64 * 4;
    float* sPf = (float*)p; p += 64 * LD64 * 4;
    float* sL  = (float*)p; p += 64 * 4;
    float* sD  = (float*)p; p += 64 * 4;
    float* sOut = (float*)sK;
    bf16* sQ[2]  = { sQ2, sQ2 + 64 * LD128 };
    bf16* sdO[2] = { sdO2, sdO2 + 64 * LD128 };

    load_tile_async(Kbh, sK, j0, S);
    load_tile_async(Vbh, sV, j0, S);
    cpa_commit();
    load_tile_async(Qbh, sQ[0], 0, S);
    load_tile_async(dObh, sdO[0], 0, S);
    cpa_commit();
    cpa_wait<1>();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_acc[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_acc[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) { wmma::fill_fragment(dV_acc[j], 0.f); wmma::fill_fragment(dK_acc[j], 0.f); }
    int rt = warp >> 1;
    int cbase = (warp & 1) * 4;

    int num_q = (S + 63) / 64;
    for (int qb = 0; qb < num_q; qb++) {
        int cur = qb & 1, nxt = (qb + 1) & 1;
        int i0 = qb * 64;
        bool has_next = (qb + 1 < num_q);
        if (has_next) {
            load_tile_async(Qbh, sQ[nxt], (qb + 1) * 64, S);
            load_tile_async(dObh, sdO[nxt], (qb + 1) * 64, S);
            cpa_commit();
            cpa_wait<1>();
        } else {
            cpa_wait<0>();
        }
        for (int i = tid; i < 64; i += 256) {
            int g = i0 + i;
            sL[i] = (g < S) ? Lbh[g] : INFINITY;
            sD[i] = (g < S) ? Dbh[g] : 0.f;
        }
        __syncthreads();

        gemm_qkt(sQ[cur], sK, sS, warp);   // S = Q @ K^T
        __syncthreads();

        #pragma unroll
        for (int u = tid; u < 64 * 16; u += 256) {
            int i = u >> 4, jg = u & 15, col = jg * 4;
            float4 sv = *reinterpret_cast<float4*>(sS + i * LD64 + col);
            float lv = sL[i];
            float p0 = (j0 + col + 0 < S) ? __expf(scale * sv.x - lv) : 0.f;
            float p1 = (j0 + col + 1 < S) ? __expf(scale * sv.y - lv) : 0.f;
            float p2 = (j0 + col + 2 < S) ? __expf(scale * sv.z - lv) : 0.f;
            float p3 = (j0 + col + 3 < S) ? __expf(scale * sv.w - lv) : 0.f;
            float4 pv; pv.x = p0; pv.y = p1; pv.z = p2; pv.w = p3;
            *reinterpret_cast<float4*>(sPf + i * LD64 + col) = pv;
            __nv_bfloat162* pb = reinterpret_cast<__nv_bfloat162*>(sPbf + i * LD64 + col);
            pb[0] = __floats2bfloat162_rn(p0, p1);
            pb[1] = __floats2bfloat162_rn(p2, p3);
        }
        __syncthreads();

        // dV += P^T @ dO
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> a;
            wmma::load_matrix_sync(a, sPbf + ks * 16 * LD64 + rt * 16, LD64);
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> bfr;
                wmma::load_matrix_sync(bfr, sdO[cur] + ks * 16 * LD128 + (cbase + c) * 16, LD128);
                wmma::mma_sync(dV_acc[c], a, bfr, dV_acc[c]);
            }
        }
        // dP = dO @ V^T into sS
        gemm_qkt(sdO[cur], sV, sS, warp);
        __syncthreads();

        #pragma unroll
        for (int u = tid; u < 64 * 16; u += 256) {
            int i = u >> 4, jg = u & 15, col = jg * 4;
            float4 pf = *reinterpret_cast<float4*>(sPf + i * LD64 + col);
            float4 dp = *reinterpret_cast<float4*>(sS + i * LD64 + col);
            float dv = sD[i];
            float d0 = scale * pf.x * (dp.x - dv);
            float d1 = scale * pf.y * (dp.y - dv);
            float d2 = scale * pf.z * (dp.z - dv);
            float d3 = scale * pf.w * (dp.w - dv);
            __nv_bfloat162* ds = reinterpret_cast<__nv_bfloat162*>(sdS + i * LD64 + col);
            ds[0] = __floats2bfloat162_rn(d0, d1);
            ds[1] = __floats2bfloat162_rn(d2, d3);
        }
        __syncthreads();

        // dK += dS'^T @ Q
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> a;
            wmma::load_matrix_sync(a, sdS + ks * 16 * LD64 + rt * 16, LD64);
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> bfr;
                wmma::load_matrix_sync(bfr, sQ[cur] + ks * 16 * LD128 + (cbase + c) * 16, LD128);
                wmma::mma_sync(dK_acc[c], a, bfr, dK_acc[c]);
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int c = 0; c < 4; c++)
        wmma::store_matrix_sync(sOut + rt * 16 * LD128 + (cbase + c) * 16, dV_acc[c], LD128, wmma::mem_row_major);
    __syncthreads();
    store_tile(dVbh, sOut, j0, S);
    __syncthreads();
    #pragma unroll
    for (int c = 0; c < 4; c++)
        wmma::store_matrix_sync(sOut + rt * 16 * LD128 + (cbase + c) * 16, dK_acc[c], LD128, wmma::mem_row_major);
    __syncthreads();
    store_tile(dKbh, sOut, j0, S);
}

// ---------------- dQ kernel ----------------
__global__ void __launch_bounds__(256) bwd_dq_kernel(
        const bf16* __restrict__ Q, const bf16* __restrict__ K,
        const bf16* __restrict__ V, const bf16* __restrict__ dO,
        const float* __restrict__ L, const float* __restrict__ Delta,
        bf16* __restrict__ dQ,
        int S, int H, float scale) {
    int b = blockIdx.z, h = blockIdx.y, q_block = blockIdx.x;
    int i0 = q_block * 64;
    int tid = threadIdx.x, warp = tid >> 5;

    size_t bh = (size_t)b * H + h;
    const bf16* Qbh  = Q  + bh * S * 128;
    const bf16* Kbh  = K  + bh * S * 128;
    const bf16* Vbh  = V  + bh * S * 128;
    const bf16* dObh = dO + bh * S * 128;
    const float* Lbh = L + bh * S;
    const float* Dbh = Delta + bh * S;
    bf16* dQbh = dQ + bh * S * 128;

    extern __shared__ char smem_raw[];
    char* p = smem_raw;
    bf16* sQ   = (bf16*)p; p += 64 * LD128 * 2;
    bf16* sdO  = (bf16*)p; p += 64 * LD128 * 2;
    bf16* sK2  = (bf16*)p; p += 2 * 64 * LD128 * 2;
    bf16* sV2  = (bf16*)p; p += 2 * 64 * LD128 * 2;
    float* sS  = (float*)p; p += 64 * LD64 * 4;
    float* sPf = (float*)p; p += 64 * LD64 * 4;
    bf16* sdS  = (bf16*)p; p += 64 * LD64 * 2;
    float* sL  = (float*)p; p += 64 * 4;
    float* sD  = (float*)p; p += 64 * 4;
    float* sOut = (float*)sK2;
    bf16* sK[2] = { sK2, sK2 + 64 * LD128 };
    bf16* sV[2] = { sV2, sV2 + 64 * LD128 };

    load_tile_async(Qbh, sQ, i0, S);
    load_tile_async(dObh, sdO, i0, S);
    cpa_commit();
    for (int i = tid; i < 64; i += 256) {
        int g = i0 + i;
        sL[i] = (g < S) ? Lbh[g] : INFINITY;
        sD[i] = (g < S) ? Dbh[g] : 0.f;
    }
    load_tile_async(Kbh, sK[0], 0, S);
    load_tile_async(Vbh, sV[0], 0, S);
    cpa_commit();
    cpa_wait<1>();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_acc[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) wmma::fill_fragment(dQ_acc[j], 0.f);
    int rt = warp >> 1;
    int cbase = (warp & 1) * 4;

    int num_kv = (S + 63) / 64;
    for (int kv = 0; kv < num_kv; kv++) {
        int cur = kv & 1, nxt = (kv + 1) & 1;
        int j0 = kv * 64;
        bool has_next = (kv + 1 < num_kv);
        if (has_next) {
            load_tile_async(Kbh, sK[nxt], (kv + 1) * 64, S);
            load_tile_async(Vbh, sV[nxt], (kv + 1) * 64, S);
            cpa_commit();
            cpa_wait<1>();
        } else {
            cpa_wait<0>();
        }
        __syncthreads();

        gemm_qkt(sQ, sK[cur], sS, warp);   // S = Q @ K^T
        __syncthreads();

        #pragma unroll
        for (int u = tid; u < 64 * 16; u += 256) {
            int i = u >> 4, jg = u & 15, col = jg * 4;
            float4 sv = *reinterpret_cast<float4*>(sS + i * LD64 + col);
            float lv = sL[i];
            float4 pv;
            pv.x = (j0 + col + 0 < S) ? __expf(scale * sv.x - lv) : 0.f;
            pv.y = (j0 + col + 1 < S) ? __expf(scale * sv.y - lv) : 0.f;
            pv.z = (j0 + col + 2 < S) ? __expf(scale * sv.z - lv) : 0.f;
            pv.w = (j0 + col + 3 < S) ? __expf(scale * sv.w - lv) : 0.f;
            *reinterpret_cast<float4*>(sPf + i * LD64 + col) = pv;
        }
        __syncthreads();

        gemm_qkt(sdO, sV[cur], sS, warp);  // dP = dO @ V^T
        __syncthreads();

        #pragma unroll
        for (int u = tid; u < 64 * 16; u += 256) {
            int i = u >> 4, jg = u & 15, col = jg * 4;
            float4 pf = *reinterpret_cast<float4*>(sPf + i * LD64 + col);
            float4 dp = *reinterpret_cast<float4*>(sS + i * LD64 + col);
            float dv = sD[i];
            float d0 = scale * pf.x * (dp.x - dv);
            float d1 = scale * pf.y * (dp.y - dv);
            float d2 = scale * pf.z * (dp.z - dv);
            float d3 = scale * pf.w * (dp.w - dv);
            __nv_bfloat162* ds = reinterpret_cast<__nv_bfloat162*>(sdS + i * LD64 + col);
            ds[0] = __floats2bfloat162_rn(d0, d1);
            ds[1] = __floats2bfloat162_rn(d2, d3);
        }
        __syncthreads();

        // dQ += dS' @ K
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
            wmma::load_matrix_sync(a, sdS + rt * 16 * LD64 + ks * 16, LD64);
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> bfr;
                wmma::load_matrix_sync(bfr, sK[cur] + ks * 16 * LD128 + (cbase + c) * 16, LD128);
                wmma::mma_sync(dQ_acc[c], a, bfr, dQ_acc[c]);
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int c = 0; c < 4; c++)
        wmma::store_matrix_sync(sOut + rt * 16 * LD128 + (cbase + c) * 16, dQ_acc[c], LD128, wmma::mem_row_major);
    __syncthreads();
    store_tile(dQbh, sOut, i0, S);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const bf16* Qp  = static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp  = static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp  = static_cast<const bf16*>(V.data_ptr());
    const bf16* Op  = static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp = static_cast<const bf16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    bf16* dQp = static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp = static_cast<bf16*>(dK.data_ptr());
    bf16* dVp = static_cast<bf16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float scale = 1.0f / sqrtf(128.0f);

    float* Delta = nullptr;
    size_t total = (size_t)B * H * S;
    CUDA_CHECK(cudaMallocAsync(&Delta, total * sizeof(float), stream));

    int dgrid = (int)((total + 3) / 4);
    compute_delta<<<dgrid, 128, 0, stream>>>(Op, dOp, Delta, (int)total);
    CUDA_CHECK(cudaGetLastError());

    const int DKV_SMEM = 160256;
    const int DQ_SMEM  = 151040;
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dkv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, DKV_SMEM));
        CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dq_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, DQ_SMEM));
        attr_set = true;
    }

    int nblk = (S + 63) / 64;
    dim3 grid(nblk, H, B);

    bwd_dkv_kernel<<<grid, 256, DKV_SMEM, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dKp, dVp, S, H, scale);
    CUDA_CHECK(cudaGetLastError());

    bwd_dq_kernel<<<grid, 256, DQ_SMEM, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dQp, S, H, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Delta, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd