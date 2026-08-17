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
static constexpr int NT = 512;
static constexpr int LDK = DIM + 8;
static constexpr int LDT = 72;

__device__ __forceinline__ void cpasync16(void* dst, const void* src){
    uint32_t s = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ void ldm4(const bf16* tile, int row0, int col0, int lda,
    uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
    int lane = threadIdx.x & 31;
    const bf16* p = tile + (row0 + (lane&15))*lda + col0 + ((lane>>4)<<3);
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm4t(const bf16* tile, int row0, int col0, int lda,
    uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
    int lane = threadIdx.x & 31;
    const bf16* p = tile + (row0 + (lane&15))*lda + col0 + ((lane>>4)<<3);
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma16(float&d0,float&d1,float&d2,float&d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint32_t b0,uint32_t b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

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

__global__ __launch_bounds__(512,1) void bwd_dkdv_kernel(
        const bf16* __restrict__ Q, const bf16* __restrict__ K,
        const bf16* __restrict__ V, const bf16* __restrict__ dO,
        const float* __restrict__ L, const float* __restrict__ Delta,
        bf16* __restrict__ dK, bf16* __restrict__ dV,
        int S, float scale) {
    int kv0 = blockIdx.x * BN;
    int bh = blockIdx.y;

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
    bf16* sV  = sK  + BN*LDK;
    bf16* sQ0 = sV  + BN*LDK;
    bf16* sQ1 = sQ0 + BM*LDK;
    bf16* sO0 = sQ1 + BM*LDK;
    bf16* sO1 = sO0 + BM*LDK;
    bf16* sPt = sO1 + BM*LDK;
    bf16* sdSt= sPt + BN*LDT;
    float* sL     = reinterpret_cast<float*>(sdSt + BN*LDT);
    float* sDelta = sL + BM;
    bf16* sQb[2] = {sQ0, sQ1};
    bf16* sOb[2] = {sO0, sO1};

    int tid = threadIdx.x;
    int lane = tid & 31, g = lane >> 2, t = lane & 3;
    int warp = tid >> 5, wm = warp & 3, wn = warp >> 2;

    for (int idx = tid; idx < BN*(DIM/8); idx += NT) {
        int j = idx / (DIM/8), cc = (idx % (DIM/8))*8;
        int grow = kv0 + j;
        if (grow < S) { cpasync16(&sK[j*LDK+cc], &Kbh[(size_t)grow*DIM+cc]);
                        cpasync16(&sV[j*LDK+cc], &Vbh[(size_t)grow*DIM+cc]); }
        else { *(int4*)&sK[j*LDK+cc]=make_int4(0,0,0,0); *(int4*)&sV[j*LDK+cc]=make_int4(0,0,0,0); }
    }
    cp_commit(); cp_wait<0>(); __syncthreads();

    float accV[4][4], accK[4][4];
    #pragma unroll
    for (int s=0;s<4;s++)
        #pragma unroll
        for (int r=0;r<4;r++){ accV[s][r]=0.f; accK[s][r]=0.f; }

    int num_q = (S + BM - 1) / BM;
    {
        bf16* bq=sQb[0]; bf16* bo=sOb[0];
        for (int idx = tid; idx < BM*(DIM/8); idx += NT) {
            int i = idx / (DIM/8), cc = (idx % (DIM/8))*8;
            if (i < S) { cpasync16(&bq[i*LDK+cc], &Qbh[(size_t)i*DIM+cc]);
                         cpasync16(&bo[i*LDK+cc], &dObh[(size_t)i*DIM+cc]); }
            else { *(int4*)&bq[i*LDK+cc]=make_int4(0,0,0,0); *(int4*)&bo[i*LDK+cc]=make_int4(0,0,0,0); }
        }
        cp_commit();
    }

    int jb = wm*16;      // kv rows for this warp
    int ib = wn*16;      // q cols for this warp (phase A)
    for (int qb = 0; qb < num_q; ++qb) {
        int cur = qb & 1;
        int q0 = qb*BM;
        if (qb+1 < num_q) {
            int q0n = (qb+1)*BM;
            bf16* bq=sQb[(qb+1)&1]; bf16* bo=sOb[(qb+1)&1];
            for (int idx = tid; idx < BM*(DIM/8); idx += NT) {
                int i = idx / (DIM/8), cc = (idx % (DIM/8))*8;
                int grow = q0n + i;
                if (grow < S) { cpasync16(&bq[i*LDK+cc], &Qbh[(size_t)grow*DIM+cc]);
                                cpasync16(&bo[i*LDK+cc], &dObh[(size_t)grow*DIM+cc]); }
                else { *(int4*)&bq[i*LDK+cc]=make_int4(0,0,0,0); *(int4*)&bo[i*LDK+cc]=make_int4(0,0,0,0); }
            }
            cp_commit(); cp_wait<1>();
        } else { cp_wait<0>(); }

        for (int idx = tid; idx < BM; idx += NT) {
            int grow = q0 + idx;
            sL[idx]     = (grow < S) ? Lbh[grow] : 0.f;
            sDelta[idx] = (grow < S) ? Dbh[grow] : 0.f;
        }
        __syncthreads();

        bf16* sQ = sQb[cur]; bf16* sO = sOb[cur];

        // Phase A: S^T=K@Q^T, dP^T=V@dO^T (this warp: 16x16)
        float As[2][4], Ap[2][4];
        #pragma unroll
        for (int h=0;h<2;h++)
            #pragma unroll
            for (int r=0;r<4;r++){As[h][r]=0.f;Ap[h][r]=0.f;}
        #pragma unroll
        for (int kc=0; kc<8; kc++){
            int c0 = kc*16;
            uint32_t aK0,aK1,aK2,aK3, aV0,aV1,aV2,aV3, bQ0,bQ1,bQ2,bQ3, bO0,bO1,bO2,bO3;
            ldm4(sK, jb, c0, LDK, aK0,aK1,aK2,aK3);
            ldm4(sV, jb, c0, LDK, aV0,aV1,aV2,aV3);
            ldm4(sQ, ib, c0, LDK, bQ0,bQ1,bQ2,bQ3);
            ldm4(sO, ib, c0, LDK, bO0,bO1,bO2,bO3);
            mma16(As[0][0],As[0][1],As[0][2],As[0][3], aK0,aK1,aK2,aK3, bQ0,bQ2);
            mma16(As[1][0],As[1][1],As[1][2],As[1][3], aK0,aK1,aK2,aK3, bQ1,bQ3);
            mma16(Ap[0][0],Ap[0][1],Ap[0][2],Ap[0][3], aV0,aV1,aV2,aV3, bO0,bO2);
            mma16(Ap[1][0],Ap[1][1],Ap[1][2],Ap[1][3], aV0,aV1,aV2,aV3, bO1,bO3);
        }
        #pragma unroll
        for (int h=0;h<2;h++){
            int iA = ib + h*8 + t*2, iB = iA+1;
            int jA = jb + g,  jB = jb + g + 8;
            float LiA=sL[iA], LiB=sL[iB], DiA=sDelta[iA], DiB=sDelta[iB];
            float p0=__expf(scale*As[h][0]-LiA);
            float p1=__expf(scale*As[h][1]-LiB);
            float p2=__expf(scale*As[h][2]-LiA);
            float p3=__expf(scale*As[h][3]-LiB);
            sPt[jA*LDT+iA]=__float2bfloat16(p0);
            sPt[jA*LDT+iB]=__float2bfloat16(p1);
            sPt[jB*LDT+iA]=__float2bfloat16(p2);
            sPt[jB*LDT+iB]=__float2bfloat16(p3);
            sdSt[jA*LDT+iA]=__float2bfloat16(p0*(Ap[h][0]-DiA));
            sdSt[jA*LDT+iB]=__float2bfloat16(p1*(Ap[h][1]-DiB));
            sdSt[jB*LDT+iA]=__float2bfloat16(p2*(Ap[h][2]-DiA));
            sdSt[jB*LDT+iB]=__float2bfloat16(p3*(Ap[h][3]-DiB));
        }
        __syncthreads();

        // Phase B: dV += P^T@dO ; dK += dS^T@Q  (contract i=BM); cols wn*32..+32
        #pragma unroll
        for (int kc=0; kc<4; kc++){
            int k0 = kc*16;
            uint32_t aP0,aP1,aP2,aP3, aS0,aS1,aS2,aS3;
            ldm4(sPt, jb, k0, LDT, aP0,aP1,aP2,aP3);
            ldm4(sdSt, jb, k0, LDT, aS0,aS1,aS2,aS3);
            #pragma unroll
            for (int ct=0; ct<2; ct++){
                int c0 = wn*32 + ct*16;
                uint32_t bo0,bo1,bo2,bo3, bq0,bq1,bq2,bq3;
                ldm4t(sO, k0, c0, LDK, bo0,bo1,bo2,bo3);
                ldm4t(sQ, k0, c0, LDK, bq0,bq1,bq2,bq3);
                mma16(accV[ct*2+0][0],accV[ct*2+0][1],accV[ct*2+0][2],accV[ct*2+0][3], aP0,aP1,aP2,aP3, bo0,bo1);
                mma16(accV[ct*2+1][0],accV[ct*2+1][1],accV[ct*2+1][2],accV[ct*2+1][3], aP0,aP1,aP2,aP3, bo2,bo3);
                mma16(accK[ct*2+0][0],accK[ct*2+0][1],accK[ct*2+0][2],accK[ct*2+0][3], aS0,aS1,aS2,aS3, bq0,bq1);
                mma16(accK[ct*2+1][0],accK[ct*2+1][1],accK[ct*2+1][2],accK[ct*2+1][3], aS0,aS1,aS2,aS3, bq2,bq3);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int t8=0;t8<4;t8++){
        int cbase = wn*32 + t8*8;
        int cA = cbase + t*2, cB = cA+1;
        int j0 = kv0 + wm*16 + g, j1 = j0 + 8;
        if (j0 < S){
            dVbh[(size_t)j0*DIM+cA] = __float2bfloat16(accV[t8][0]);
            dVbh[(size_t)j0*DIM+cB] = __float2bfloat16(accV[t8][1]);
            dKbh[(size_t)j0*DIM+cA] = __float2bfloat16(scale*accK[t8][0]);
            dKbh[(size_t)j0*DIM+cB] = __float2bfloat16(scale*accK[t8][1]);
        }
        if (j1 < S){
            dVbh[(size_t)j1*DIM+cA] = __float2bfloat16(accV[t8][2]);
            dVbh[(size_t)j1*DIM+cB] = __float2bfloat16(accV[t8][3]);
            dKbh[(size_t)j1*DIM+cA] = __float2bfloat16(scale*accK[t8][2]);
            dKbh[(size_t)j1*DIM+cB] = __float2bfloat16(scale*accK[t8][3]);
        }
    }
}

__global__ __launch_bounds__(512,1) void bwd_dq_kernel(
        const bf16* __restrict__ Q, const bf16* __restrict__ K,
        const bf16* __restrict__ V, const bf16* __restrict__ dO,
        const float* __restrict__ L, const float* __restrict__ Delta,
        bf16* __restrict__ dQ,
        int S, float scale) {
    int q0 = blockIdx.x * BM;
    int bh = blockIdx.y;

    const bf16* Kbh  = K  + (size_t)bh * S * DIM;
    const bf16* Vbh  = V  + (size_t)bh * S * DIM;
    const bf16* Qbh  = Q  + (size_t)bh * S * DIM;
    const bf16* dObh = dO + (size_t)bh * S * DIM;
    const float* Lbh = L  + (size_t)bh * S;
    const float* Dbh = Delta + (size_t)bh * S;
    bf16* dQbh = dQ + (size_t)bh * S * DIM;

    extern __shared__ char smem[];
    bf16* sQ  = reinterpret_cast<bf16*>(smem);
    bf16* sO  = sQ  + BM*LDK;
    bf16* sK0 = sO  + BM*LDK;
    bf16* sK1 = sK0 + BN*LDK;
    bf16* sV0 = sK1 + BN*LDK;
    bf16* sV1 = sV0 + BN*LDK;
    bf16* sdS = sV1 + BN*LDK;
    float* sL     = reinterpret_cast<float*>(sdS + BM*LDT);
    float* sDelta = sL + BM;
    bf16* sKb[2] = {sK0, sK1};
    bf16* sVb[2] = {sV0, sV1};

    int tid = threadIdx.x;
    int lane = tid & 31, g = lane >> 2, t = lane & 3;
    int warp = tid >> 5, wm = warp & 3, wn = warp >> 2;

    for (int idx = tid; idx < BM*(DIM/8); idx += NT) {
        int i = idx / (DIM/8), cc = (idx % (DIM/8))*8;
        int grow = q0 + i;
        if (grow < S) { cpasync16(&sQ[i*LDK+cc], &Qbh[(size_t)grow*DIM+cc]);
                        cpasync16(&sO[i*LDK+cc], &dObh[(size_t)grow*DIM+cc]); }
        else { *(int4*)&sQ[i*LDK+cc]=make_int4(0,0,0,0); *(int4*)&sO[i*LDK+cc]=make_int4(0,0,0,0); }
    }
    cp_commit(); cp_wait<0>(); __syncthreads();
    for (int idx = tid; idx < BM; idx += NT) {
        int grow = q0 + idx;
        sL[idx]     = (grow < S) ? Lbh[grow] : 0.f;
        sDelta[idx] = (grow < S) ? Dbh[grow] : 0.f;
    }
    __syncthreads();

    float accQ[4][4];
    #pragma unroll
    for (int s=0;s<4;s++)
        #pragma unroll
        for (int r=0;r<4;r++) accQ[s][r]=0.f;

    int ibM = wm*16;    // q rows for this warp
    int jbA = wn*16;    // kv cols for this warp (phase A)
    int num_kv = (S + BN - 1) / BN;
    {
        bf16* bk=sKb[0]; bf16* bv=sVb[0];
        for (int idx = tid; idx < BN*(DIM/8); idx += NT) {
            int j = idx / (DIM/8), cc = (idx % (DIM/8))*8;
            if (j < S) { cpasync16(&bk[j*LDK+cc], &Kbh[(size_t)j*DIM+cc]);
                         cpasync16(&bv[j*LDK+cc], &Vbh[(size_t)j*DIM+cc]); }
            else { *(int4*)&bk[j*LDK+cc]=make_int4(0,0,0,0); *(int4*)&bv[j*LDK+cc]=make_int4(0,0,0,0); }
        }
        cp_commit();
    }

    for (int kvb = 0; kvb < num_kv; ++kvb) {
        int cur = kvb & 1;
        if (kvb+1 < num_kv) {
            int kv0n = (kvb+1)*BN;
            bf16* bk=sKb[(kvb+1)&1]; bf16* bv=sVb[(kvb+1)&1];
            for (int idx = tid; idx < BN*(DIM/8); idx += NT) {
                int j = idx / (DIM/8), cc = (idx % (DIM/8))*8;
                int grow = kv0n + j;
                if (grow < S) { cpasync16(&bk[j*LDK+cc], &Kbh[(size_t)grow*DIM+cc]);
                                cpasync16(&bv[j*LDK+cc], &Vbh[(size_t)grow*DIM+cc]); }
                else { *(int4*)&bk[j*LDK+cc]=make_int4(0,0,0,0); *(int4*)&bv[j*LDK+cc]=make_int4(0,0,0,0); }
            }
            cp_commit(); cp_wait<1>();
        } else { cp_wait<0>(); }
        __syncthreads();

        bf16* sK = sKb[cur]; bf16* sV = sVb[cur];

        // Phase A: S=Q@K^T, dP=dO@V^T (this warp: 16x16) -> dS[i][j]
        float As[2][4], Ap[2][4];
        #pragma unroll
        for (int h=0;h<2;h++)
            #pragma unroll
            for (int r=0;r<4;r++){As[h][r]=0.f;Ap[h][r]=0.f;}
        #pragma unroll
        for (int kc=0; kc<8; kc++){
            int c0 = kc*16;
            uint32_t aQ0,aQ1,aQ2,aQ3, aO0,aO1,aO2,aO3, bK0,bK1,bK2,bK3, bV0,bV1,bV2,bV3;
            ldm4(sQ, ibM, c0, LDK, aQ0,aQ1,aQ2,aQ3);
            ldm4(sO, ibM, c0, LDK, aO0,aO1,aO2,aO3);
            ldm4(sK, jbA, c0, LDK, bK0,bK1,bK2,bK3);
            ldm4(sV, jbA, c0, LDK, bV0,bV1,bV2,bV3);
            mma16(As[0][0],As[0][1],As[0][2],As[0][3], aQ0,aQ1,aQ2,aQ3, bK0,bK2);
            mma16(As[1][0],As[1][1],As[1][2],As[1][3], aQ0,aQ1,aQ2,aQ3, bK1,bK3);
            mma16(Ap[0][0],Ap[0][1],Ap[0][2],Ap[0][3], aO0,aO1,aO2,aO3, bV0,bV2);
            mma16(Ap[1][0],Ap[1][1],Ap[1][2],Ap[1][3], aO0,aO1,aO2,aO3, bV1,bV3);
        }
        #pragma unroll
        for (int h=0;h<2;h++){
            int jA = jbA + h*8 + t*2, jB = jA+1;
            int iA = ibM + g, iB = ibM + g + 8;
            float LiA=sL[iA], LiB=sL[iB], DiA=sDelta[iA], DiB=sDelta[iB];
            float p0=__expf(scale*As[h][0]-LiA);
            float p1=__expf(scale*As[h][1]-LiA);
            float p2=__expf(scale*As[h][2]-LiB);
            float p3=__expf(scale*As[h][3]-LiB);
            sdS[iA*LDT+jA]=__float2bfloat16(p0*(Ap[h][0]-DiA));
            sdS[iA*LDT+jB]=__float2bfloat16(p1*(Ap[h][1]-DiA));
            sdS[iB*LDT+jA]=__float2bfloat16(p2*(Ap[h][2]-DiB));
            sdS[iB*LDT+jB]=__float2bfloat16(p3*(Ap[h][3]-DiB));
        }
        __syncthreads();

        // Phase B: dQ += dS@K (contract j=BN); cols wn*32..+32
        #pragma unroll
        for (int kc=0; kc<4; kc++){
            int j0 = kc*16;
            uint32_t aS0,aS1,aS2,aS3;
            ldm4(sdS, ibM, j0, LDT, aS0,aS1,aS2,aS3);
            #pragma unroll
            for (int ct=0; ct<2; ct++){
                int c0 = wn*32 + ct*16;
                uint32_t bk0,bk1,bk2,bk3;
                ldm4t(sK, j0, c0, LDK, bk0,bk1,bk2,bk3);
                mma16(accQ[ct*2+0][0],accQ[ct*2+0][1],accQ[ct*2+0][2],accQ[ct*2+0][3], aS0,aS1,aS2,aS3, bk0,bk1);
                mma16(accQ[ct*2+1][0],accQ[ct*2+1][1],accQ[ct*2+1][2],accQ[ct*2+1][3], aS0,aS1,aS2,aS3, bk2,bk3);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int t8=0;t8<4;t8++){
        int cbase = wn*32 + t8*8;
        int cA = cbase + t*2, cB = cA+1;
        int i0 = q0 + wm*16 + g, i1 = i0 + 8;
        if (i0 < S){
            dQbh[(size_t)i0*DIM+cA] = __float2bfloat16(scale*accQ[t8][0]);
            dQbh[(size_t)i0*DIM+cB] = __float2bfloat16(scale*accQ[t8][1]);
        }
        if (i1 < S){
            dQbh[(size_t)i1*DIM+cA] = __float2bfloat16(scale*accQ[t8][2]);
            dQbh[(size_t)i1*DIM+cB] = __float2bfloat16(scale*accQ[t8][3]);
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
    CUDA_CHECK(cudaMallocAsync(&Delta, (size_t)BH*S*sizeof(float), stream));

    long long total_rows = (long long)BH * S;
    int dblocks = (int)((total_rows + 7) / 8);
    compute_delta_kernel<<<dblocks, 256, 0, stream>>>(Op, dOp, Delta, total_rows, (int)d);
    CUDA_CHECK(cudaGetLastError());

    size_t dkdv_smem = ((size_t)(2*BN*LDK + 4*BM*LDK + 2*BN*LDT))*sizeof(bf16) + 2*BM*sizeof(float);
    size_t dq_smem   = ((size_t)(2*BM*LDK + 4*BN*LDK + BM*LDT))*sizeof(bf16) + 2*BM*sizeof(float);

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
    bwd_dkdv_kernel<<<grid_kv, NT, dkdv_smem, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dKp, dVp, (int)S, scale);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_q(num_q, (unsigned)BH);
    bwd_dq_kernel<<<grid_q, NT, dq_smem, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dQp, (int)S, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Delta, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd