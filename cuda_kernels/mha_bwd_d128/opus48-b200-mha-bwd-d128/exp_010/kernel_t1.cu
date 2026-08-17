#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {

using bf16 = __nv_bfloat16;

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int DIM = 128;
static constexpr int NTHREAD = 256;

__device__ __forceinline__ uint32_t ldu32(const bf16* p){
    return *reinterpret_cast<const uint32_t*>(p);
}

__device__ __forceinline__ void mma16816(
    float &d0,float &d1,float &d2,float &d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
    uint32_t b0,uint32_t b1){
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

// ---------- Delta = rowsum(O * dO) ----------
__global__ void compute_delta_kernel(const bf16* __restrict__ O, const bf16* __restrict__ dO,
                                     float* __restrict__ Delta, long long total_rows, int d) {
    long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp >= total_rows) return;
    const bf16* o = O + (size_t)warp * d;
    const bf16* g = dO + (size_t)warp * d;
    float acc = 0.f;
    for (int c = lane; c < d; c += 32) acc += __bfloat162float(o[c]) * __bfloat162float(g[c]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) Delta[warp] = acc;
}

// ---------- dK, dV kernel ----------
// grid: (num_kv_blocks, B*H), block: 256 threads (8 warps 4x2)
__global__ void bwd_dkdv_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                                const bf16* __restrict__ V, const bf16* __restrict__ dO,
                                const float* __restrict__ L, const float* __restrict__ Delta,
                                bf16* __restrict__ dK, bf16* __restrict__ dV,
                                int S, float scale) {
    int kv_blk = blockIdx.x;
    int bh = blockIdx.y;
    int kv0 = kv_blk * BN;

    const bf16* Kbh  = K  + (size_t)bh * S * DIM;
    const bf16* Vbh  = V  + (size_t)bh * S * DIM;
    const bf16* Qbh  = Q  + (size_t)bh * S * DIM;
    const bf16* dObh = dO + (size_t)bh * S * DIM;
    const float* Lbh = L  + (size_t)bh * S;
    const float* Dbh = Delta + (size_t)bh * S;
    bf16* dKbh = dK + (size_t)bh * S * DIM;
    bf16* dVbh = dV + (size_t)bh * S * DIM;

    extern __shared__ char smem[];
    bf16* sK  = reinterpret_cast<bf16*>(smem);
    bf16* sV  = sK  + BN*DIM;
    bf16* sQ  = sV  + BN*DIM;
    bf16* sdO = sQ  + BM*DIM;
    bf16* sQt = sdO + BM*DIM;   // [DIM][BM]
    bf16* sdOt= sQt + DIM*BM;   // [DIM][BM]
    bf16* sPt = sdOt+ DIM*BM;   // [BN][BM]
    bf16* sdSt= sPt + BN*BM;    // [BN][BM]
    float* sL     = reinterpret_cast<float*>(sdSt + BN*BM);
    float* sDelta = sL + BM;

    int tid = threadIdx.x;
    int lane = tid & 31, gid = lane >> 2, tg = lane & 3;
    int warp = tid >> 5, wm = warp & 3, wn = warp >> 2;

    bf16 z = __float2bfloat16(0.f);

    // load K,V
    for (int idx = tid; idx < BN*DIM; idx += NTHREAD) {
        int j = idx >> 7, c = idx & 127;
        int grow = kv0 + j;
        bf16 kv=z, vv=z;
        if (grow < S) { kv = Kbh[(size_t)grow*DIM + c]; vv = Vbh[(size_t)grow*DIM + c]; }
        sK[idx] = kv; sV[idx] = vv;
    }

    float dV[8][4], dKa[8][4];
    #pragma unroll
    for (int s=0;s<8;s++)
        #pragma unroll
        for (int r=0;r<4;r++){ dV[s][r]=0.f; dKa[s][r]=0.f; }

    int num_q = (S + BM - 1) / BM;
    for (int qb = 0; qb < num_q; ++qb) {
        int q0 = qb * BM;
        __syncthreads();
        for (int idx = tid; idx < BM*DIM; idx += NTHREAD) {
            int i = idx >> 7, c = idx & 127;
            int grow = q0 + i;
            bf16 qv=z, gv=z;
            if (grow < S) { qv = Qbh[(size_t)grow*DIM + c]; gv = dObh[(size_t)grow*DIM + c]; }
            sQ[idx] = qv; sdO[idx] = gv;
            sQt[c*BM + i] = qv; sdOt[c*BM + i] = gv;
        }
        for (int idx = tid; idx < BM; idx += NTHREAD) {
            int grow = q0 + idx;
            sL[idx]     = (grow < S) ? Lbh[grow] : 0.f;
            sDelta[idx] = (grow < S) ? Dbh[grow] : 0.f;
        }
        __syncthreads();

        // Phase A: S^T = K@Q^T, dP^T = V@dO^T -> P^T, dS^T
        int m0 = wm*16;
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int n0 = wn*32 + s*8;   // i base
            float sS0=0,sS1=0,sS2=0,sS3=0, sD0=0,sD1=0,sD2=0,sD3=0;
            #pragma unroll
            for (int kk=0; kk<8; kk++){
                int c0 = kk*16;
                uint32_t aK0=ldu32(&sK[(m0+gid)*DIM + c0 + tg*2]);
                uint32_t aK1=ldu32(&sK[(m0+8+gid)*DIM + c0 + tg*2]);
                uint32_t aK2=ldu32(&sK[(m0+gid)*DIM + c0 + tg*2+8]);
                uint32_t aK3=ldu32(&sK[(m0+8+gid)*DIM + c0 + tg*2+8]);
                uint32_t aV0=ldu32(&sV[(m0+gid)*DIM + c0 + tg*2]);
                uint32_t aV1=ldu32(&sV[(m0+8+gid)*DIM + c0 + tg*2]);
                uint32_t aV2=ldu32(&sV[(m0+gid)*DIM + c0 + tg*2+8]);
                uint32_t aV3=ldu32(&sV[(m0+8+gid)*DIM + c0 + tg*2+8]);
                uint32_t bQ0=ldu32(&sQ[(n0+gid)*DIM + c0 + tg*2]);
                uint32_t bQ1=ldu32(&sQ[(n0+gid)*DIM + c0 + tg*2+8]);
                uint32_t bO0=ldu32(&sdO[(n0+gid)*DIM + c0 + tg*2]);
                uint32_t bO1=ldu32(&sdO[(n0+gid)*DIM + c0 + tg*2+8]);
                mma16816(sS0,sS1,sS2,sS3, aK0,aK1,aK2,aK3, bQ0,bQ1);
                mma16816(sD0,sD1,sD2,sD3, aV0,aV1,aV2,aV3, bO0,bO1);
            }
            int iA = n0 + tg*2, iB = iA + 1;
            int jA = m0 + gid,  jB = m0 + 8 + gid;
            float LiA=sL[iA], LiB=sL[iB], DiA=sDelta[iA], DiB=sDelta[iB];
            float p0=__expf(scale*sS0 - LiA);
            float p1=__expf(scale*sS1 - LiB);
            float p2=__expf(scale*sS2 - LiA);
            float p3=__expf(scale*sS3 - LiB);
            sPt[jA*BM + iA]=__float2bfloat16(p0);
            sPt[jA*BM + iB]=__float2bfloat16(p1);
            sPt[jB*BM + iA]=__float2bfloat16(p2);
            sPt[jB*BM + iB]=__float2bfloat16(p3);
            sdSt[jA*BM + iA]=__float2bfloat16(p0*(sD0-DiA));
            sdSt[jA*BM + iB]=__float2bfloat16(p1*(sD1-DiB));
            sdSt[jB*BM + iA]=__float2bfloat16(p2*(sD2-DiA));
            sdSt[jB*BM + iB]=__float2bfloat16(p3*(sD3-DiB));
        }
        __syncthreads();

        // Phase B: dV += P^T @ dO ; dK += dS^T @ Q  (contract over i=BM)
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            int n0 = wn*64 + s*8;   // c base
            #pragma unroll
            for (int kk=0; kk<4; kk++){
                int i0 = kk*16;
                uint32_t aP0=ldu32(&sPt[(m0+gid)*BM + i0 + tg*2]);
                uint32_t aP1=ldu32(&sPt[(m0+8+gid)*BM + i0 + tg*2]);
                uint32_t aP2=ldu32(&sPt[(m0+gid)*BM + i0 + tg*2+8]);
                uint32_t aP3=ldu32(&sPt[(m0+8+gid)*BM + i0 + tg*2+8]);
                uint32_t aS0=ldu32(&sdSt[(m0+gid)*BM + i0 + tg*2]);
                uint32_t aS1=ldu32(&sdSt[(m0+8+gid)*BM + i0 + tg*2]);
                uint32_t aS2=ldu32(&sdSt[(m0+gid)*BM + i0 + tg*2+8]);
                uint32_t aS3=ldu32(&sdSt[(m0+8+gid)*BM + i0 + tg*2+8]);
                uint32_t bO0=ldu32(&sdOt[(n0+gid)*BM + i0 + tg*2]);
                uint32_t bO1=ldu32(&sdOt[(n0+gid)*BM + i0 + tg*2+8]);
                uint32_t bQ0=ldu32(&sQt[(n0+gid)*BM + i0 + tg*2]);
                uint32_t bQ1=ldu32(&sQt[(n0+gid)*BM + i0 + tg*2+8]);
                mma16816(dV[s][0],dV[s][1],dV[s][2],dV[s][3], aP0,aP1,aP2,aP3, bO0,bO1);
                mma16816(dKa[s][0],dKa[s][1],dKa[s][2],dKa[s][3], aS0,aS1,aS2,aS3, bQ0,bQ1);
            }
        }
    }

    // write out
    int m0 = wm*16;
    #pragma unroll
    for (int s=0;s<8;s++){
        int n0 = wn*64 + s*8;
        int cA = n0 + tg*2, cB = cA + 1;
        int jA = kv0 + m0 + gid, jB = kv0 + m0 + 8 + gid;
        if (jA < S){
            dVbh[(size_t)jA*DIM + cA] = __float2bfloat16(dV[s][0]);
            dVbh[(size_t)jA*DIM + cB] = __float2bfloat16(dV[s][1]);
            dKbh[(size_t)jA*DIM + cA] = __float2bfloat16(scale*dKa[s][0]);
            dKbh[(size_t)jA*DIM + cB] = __float2bfloat16(scale*dKa[s][1]);
        }
        if (jB < S){
            dVbh[(size_t)jB*DIM + cA] = __float2bfloat16(dV[s][2]);
            dVbh[(size_t)jB*DIM + cB] = __float2bfloat16(dV[s][3]);
            dKbh[(size_t)jB*DIM + cA] = __float2bfloat16(scale*dKa[s][2]);
            dKbh[(size_t)jB*DIM + cB] = __float2bfloat16(scale*dKa[s][3]);
        }
    }
}

// ---------- dQ kernel ----------
// grid: (num_q_blocks, B*H), block: 256 threads
__global__ void bwd_dq_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                              const bf16* __restrict__ V, const bf16* __restrict__ dO,
                              const float* __restrict__ L, const float* __restrict__ Delta,
                              bf16* __restrict__ dQ,
                              int S, float scale) {
    int q_blk = blockIdx.x;
    int bh = blockIdx.y;
    int q0 = q_blk * BM;

    const bf16* Kbh  = K  + (size_t)bh * S * DIM;
    const bf16* Vbh  = V  + (size_t)bh * S * DIM;
    const bf16* Qbh  = Q  + (size_t)bh * S * DIM;
    const bf16* dObh = dO + (size_t)bh * S * DIM;
    const float* Lbh = L  + (size_t)bh * S;
    const float* Dbh = Delta + (size_t)bh * S;
    bf16* dQbh = dQ + (size_t)bh * S * DIM;

    extern __shared__ char smem[];
    bf16* sQ  = reinterpret_cast<bf16*>(smem);
    bf16* sdO = sQ  + BM*DIM;
    bf16* sK  = sdO + BM*DIM;
    bf16* sV  = sK  + BN*DIM;
    bf16* sKt = sV  + BN*DIM;   // [DIM][BN]
    bf16* sdS = sKt + DIM*BN;   // [BM][BN]
    float* sL     = reinterpret_cast<float*>(sdS + BM*BN);
    float* sDelta = sL + BM;

    int tid = threadIdx.x;
    int lane = tid & 31, gid = lane >> 2, tg = lane & 3;
    int warp = tid >> 5, wm = warp & 3, wn = warp >> 2;

    bf16 z = __float2bfloat16(0.f);

    // load Q, dO (persistent)
    for (int idx = tid; idx < BM*DIM; idx += NTHREAD) {
        int i = idx >> 7, c = idx & 127;
        int grow = q0 + i;
        bf16 qv=z, gv=z;
        if (grow < S) { qv = Qbh[(size_t)grow*DIM + c]; gv = dObh[(size_t)grow*DIM + c]; }
        sQ[idx] = qv; sdO[idx] = gv;
    }
    for (int idx = tid; idx < BM; idx += NTHREAD) {
        int grow = q0 + idx;
        sL[idx]     = (grow < S) ? Lbh[grow] : 0.f;
        sDelta[idx] = (grow < S) ? Dbh[grow] : 0.f;
    }

    float dQa[8][4];
    #pragma unroll
    for (int s=0;s<8;s++)
        #pragma unroll
        for (int r=0;r<4;r++) dQa[s][r]=0.f;

    int m0 = wm*16;
    int num_kv = (S + BN - 1) / BN;
    for (int kvb = 0; kvb < num_kv; ++kvb) {
        int kv0 = kvb * BN;
        __syncthreads();
        for (int idx = tid; idx < BN*DIM; idx += NTHREAD) {
            int j = idx >> 7, c = idx & 127;
            int grow = kv0 + j;
            bf16 kv=z, vv=z;
            if (grow < S) { kv = Kbh[(size_t)grow*DIM + c]; vv = Vbh[(size_t)grow*DIM + c]; }
            sK[idx] = kv; sV[idx] = vv;
            sKt[c*BN + j] = kv;
        }
        __syncthreads();

        // Phase A: S = Q@K^T, dP = dO@V^T -> P, dS  (store sdS[i][j])
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int n0 = wn*32 + s*8;   // j base
            float sS0=0,sS1=0,sS2=0,sS3=0, sD0=0,sD1=0,sD2=0,sD3=0;
            #pragma unroll
            for (int kk=0; kk<8; kk++){
                int c0 = kk*16;
                uint32_t aQ0=ldu32(&sQ[(m0+gid)*DIM + c0 + tg*2]);
                uint32_t aQ1=ldu32(&sQ[(m0+8+gid)*DIM + c0 + tg*2]);
                uint32_t aQ2=ldu32(&sQ[(m0+gid)*DIM + c0 + tg*2+8]);
                uint32_t aQ3=ldu32(&sQ[(m0+8+gid)*DIM + c0 + tg*2+8]);
                uint32_t aO0=ldu32(&sdO[(m0+gid)*DIM + c0 + tg*2]);
                uint32_t aO1=ldu32(&sdO[(m0+8+gid)*DIM + c0 + tg*2]);
                uint32_t aO2=ldu32(&sdO[(m0+gid)*DIM + c0 + tg*2+8]);
                uint32_t aO3=ldu32(&sdO[(m0+8+gid)*DIM + c0 + tg*2+8]);
                uint32_t bK0=ldu32(&sK[(n0+gid)*DIM + c0 + tg*2]);
                uint32_t bK1=ldu32(&sK[(n0+gid)*DIM + c0 + tg*2+8]);
                uint32_t bV0=ldu32(&sV[(n0+gid)*DIM + c0 + tg*2]);
                uint32_t bV1=ldu32(&sV[(n0+gid)*DIM + c0 + tg*2+8]);
                mma16816(sS0,sS1,sS2,sS3, aQ0,aQ1,aQ2,aQ3, bK0,bK1);
                mma16816(sD0,sD1,sD2,sD3, aO0,aO1,aO2,aO3, bV0,bV1);
            }
            int jA = n0 + tg*2, jB = jA + 1;
            int iA = m0 + gid,  iB = m0 + 8 + gid;
            float LiA=sL[iA], LiB=sL[iB], DiA=sDelta[iA], DiB=sDelta[iB];
            float p0=__expf(scale*sS0 - LiA);
            float p1=__expf(scale*sS1 - LiA);
            float p2=__expf(scale*sS2 - LiB);
            float p3=__expf(scale*sS3 - LiB);
            sdS[iA*BN + jA]=__float2bfloat16(p0*(sD0-DiA));
            sdS[iA*BN + jB]=__float2bfloat16(p1*(sD1-DiA));
            sdS[iB*BN + jA]=__float2bfloat16(p2*(sD2-DiB));
            sdS[iB*BN + jB]=__float2bfloat16(p3*(sD3-DiB));
        }
        __syncthreads();

        // Phase B: dQ += dS @ K  (contract over j=BN)
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            int n0 = wn*64 + s*8;   // c base
            #pragma unroll
            for (int kk=0; kk<4; kk++){
                int j0 = kk*16;
                uint32_t aS0=ldu32(&sdS[(m0+gid)*BN + j0 + tg*2]);
                uint32_t aS1=ldu32(&sdS[(m0+8+gid)*BN + j0 + tg*2]);
                uint32_t aS2=ldu32(&sdS[(m0+gid)*BN + j0 + tg*2+8]);
                uint32_t aS3=ldu32(&sdS[(m0+8+gid)*BN + j0 + tg*2+8]);
                uint32_t bK0=ldu32(&sKt[(n0+gid)*BN + j0 + tg*2]);
                uint32_t bK1=ldu32(&sKt[(n0+gid)*BN + j0 + tg*2+8]);
                mma16816(dQa[s][0],dQa[s][1],dQa[s][2],dQa[s][3], aS0,aS1,aS2,aS3, bK0,bK1);
            }
        }
    }

    // write dQ
    #pragma unroll
    for (int s=0;s<8;s++){
        int n0 = wn*64 + s*8;
        int cA = n0 + tg*2, cB = cA + 1;
        int iA = q0 + m0 + gid, iB = q0 + m0 + 8 + gid;
        if (iA < S){
            dQbh[(size_t)iA*DIM + cA] = __float2bfloat16(scale*dQa[s][0]);
            dQbh[(size_t)iA*DIM + cB] = __float2bfloat16(scale*dQa[s][1]);
        }
        if (iB < S){
            dQbh[(size_t)iB*DIM + cA] = __float2bfloat16(scale*dQa[s][2]);
            dQbh[(size_t)iB*DIM + cB] = __float2bfloat16(scale*dQa[s][3]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), d = Q.size(3);
    int64_t BH = B * H;
    float scale = 1.0f / sqrtf((float)d);

    const bf16* Qp  = static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp  = static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp  = static_cast<const bf16*>(V.data_ptr());
    const bf16* Op  = static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp = static_cast<const bf16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    bf16* dQp = static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp = static_cast<bf16*>(dK.data_ptr());
    bf16* dVp = static_cast<bf16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Delta = nullptr;
    size_t deltaSz = (size_t)BH * S * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&Delta, deltaSz, stream));

    long long total_rows = (long long)BH * S;
    int dblocks = (int)((total_rows + 7) / 8);
    compute_delta_kernel<<<dblocks, 256, 0, stream>>>(Op, dOp, Delta, total_rows, (int)d);
    CUDA_CHECK(cudaGetLastError());

    size_t dkdv_smem = ((size_t)6*BM*DIM + 2*BN*BM)*sizeof(bf16) + 2*BM*sizeof(float);
    size_t dq_smem   = ((size_t)5*BM*DIM + BM*BN)*sizeof(bf16) + 2*BM*sizeof(float);

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dkdv_smem));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dq_smem));
        attr_set = true;
    }

    int num_kv = (int)((S + BN - 1) / BN);
    int num_q  = (int)((S + BM - 1) / BM);

    dim3 grid_kv(num_kv, (unsigned)BH);
    bwd_dkdv_kernel<<<grid_kv, NTHREAD, dkdv_smem, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dKp, dVp, (int)S, scale);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_q(num_q, (unsigned)BH);
    bwd_dq_kernel<<<grid_q, NTHREAD, dq_smem, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dQp, (int)S, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Delta, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd