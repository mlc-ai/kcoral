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

static constexpr int BM = 64;   // query tile
static constexpr int BN = 64;   // key tile
static constexpr int DIM = 128; // head dim
// dynamic shared bytes: 4 bf16 tiles (64x128) + 2 float tiles (64x64) + 2 float vecs(64)
static constexpr size_t SMEM_BYTES = (size_t)(4*BM*DIM*2 + 2*BM*BN*4 + 2*BM*4);

// -------- Delta = rowsum(O * dO) --------
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

// -------- dK, dV kernel --------
// grid: (num_kv_blocks, B*H). block: (16,16)=256 threads
__global__ void bwd_dkdv_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                                const bf16* __restrict__ V, const bf16* __restrict__ dO,
                                const float* __restrict__ L, const float* __restrict__ Delta,
                                bf16* __restrict__ dK, bf16* __restrict__ dV,
                                int S, int d, float scale) {
    int kv_blk = blockIdx.x;
    int bh = blockIdx.y;
    int kv0 = kv_blk * BN;

    const bf16* Kbh  = K  + (size_t)bh * S * d;
    const bf16* Vbh  = V  + (size_t)bh * S * d;
    const bf16* Qbh  = Q  + (size_t)bh * S * d;
    const bf16* dObh = dO + (size_t)bh * S * d;
    const float* Lbh = L  + (size_t)bh * S;
    const float* Dbh = Delta + (size_t)bh * S;
    bf16* dKbh = dK + (size_t)bh * S * d;
    bf16* dVbh = dV + (size_t)bh * S * d;

    extern __shared__ char smem[];
    bf16* sK  = reinterpret_cast<bf16*>(smem);
    bf16* sV  = sK + BN*DIM;
    bf16* sQ  = sV + BN*DIM;
    bf16* sdO = sQ + BM*DIM;
    float* sP    = reinterpret_cast<float*>(sdO + BM*DIM);
    float* sdS   = sP + BN*BM;
    float* sL    = sdS + BN*BM;
    float* sDelta= sL + BM;

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * 16 + tx;

    // load K, V for this kv tile
    for (int idx = tid; idx < BN*DIM; idx += 256) {
        int j = idx >> 7, c = idx & 127;
        int grow = kv0 + j;
        bf16 kv = __float2bfloat16(0.f), vv = __float2bfloat16(0.f);
        if (grow < S) { kv = Kbh[(size_t)grow*d + c]; vv = Vbh[(size_t)grow*d + c]; }
        sK[idx] = kv; sV[idx] = vv;
    }

    float dV_acc[4][8], dK_acc[4][8];
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (int cc = 0; cc < 8; cc++) { dV_acc[r][cc] = 0.f; dK_acc[r][cc] = 0.f; }

    int num_q = (S + BM - 1) / BM;
    for (int qb = 0; qb < num_q; ++qb) {
        int q0 = qb * BM;
        __syncthreads();
        for (int idx = tid; idx < BM*DIM; idx += 256) {
            int i = idx >> 7, c = idx & 127;
            int grow = q0 + i;
            bf16 qv = __float2bfloat16(0.f), gv = __float2bfloat16(0.f);
            if (grow < S) { qv = Qbh[(size_t)grow*d + c]; gv = dObh[(size_t)grow*d + c]; }
            sQ[idx] = qv; sdO[idx] = gv;
        }
        for (int idx = tid; idx < BM; idx += 256) {
            int grow = q0 + idx;
            sL[idx]     = (grow < S) ? Lbh[grow] : 0.f;
            sDelta[idx] = (grow < S) ? Dbh[grow] : 0.f;
        }
        __syncthreads();

        // GEMM1&2: S[j][i]=K.Q ; dP[j][i]=V.dO -> P, dS   (rows j=ty*4, cols i=tx*4)
        {
            float accS[4][4], accP[4][4];
            #pragma unroll
            for (int r=0;r<4;r++)
                #pragma unroll
                for (int cc=0;cc<4;cc++){accS[r][cc]=0.f;accP[r][cc]=0.f;}
            int jb = ty*4, ib = tx*4;
            #pragma unroll 4
            for (int c = 0; c < DIM; c++) {
                float kf[4], vf[4], qf[4], gf[4];
                #pragma unroll
                for (int r=0;r<4;r++){ kf[r]=__bfloat162float(sK[(jb+r)*DIM+c]); vf[r]=__bfloat162float(sV[(jb+r)*DIM+c]); }
                #pragma unroll
                for (int cc=0;cc<4;cc++){ qf[cc]=__bfloat162float(sQ[(ib+cc)*DIM+c]); gf[cc]=__bfloat162float(sdO[(ib+cc)*DIM+c]); }
                #pragma unroll
                for (int r=0;r<4;r++)
                    #pragma unroll
                    for (int cc=0;cc<4;cc++){ accS[r][cc]+=kf[r]*qf[cc]; accP[r][cc]+=vf[r]*gf[cc]; }
            }
            #pragma unroll
            for (int r=0;r<4;r++)
                #pragma unroll
                for (int cc=0;cc<4;cc++){
                    int i = ib+cc;
                    float arg = scale*accS[r][cc] - sL[i];
                    float p = __expf(fminf(arg, 30.0f));
                    float ds = p*(accP[r][cc] - sDelta[i]);
                    sP[(jb+r)*BM + i] = p;
                    sdS[(jb+r)*BM + i] = ds;
                }
        }
        __syncthreads();

        // GEMM3&4: dV[j][c]+=P.dO ; dK[j][c]+=dS.Q   (rows j=ty*4, cols c=tx*8)
        {
            int jb = ty*4, cb = tx*8;
            #pragma unroll 4
            for (int i = 0; i < BM; i++) {
                float pf[4], dsf[4];
                #pragma unroll
                for (int r=0;r<4;r++){ pf[r]=sP[(jb+r)*BM+i]; dsf[r]=sdS[(jb+r)*BM+i]; }
                float gf[8], qf[8];
                #pragma unroll
                for (int cc=0;cc<8;cc++){ gf[cc]=__bfloat162float(sdO[i*DIM+cb+cc]); qf[cc]=__bfloat162float(sQ[i*DIM+cb+cc]); }
                #pragma unroll
                for (int r=0;r<4;r++)
                    #pragma unroll
                    for (int cc=0;cc<8;cc++){ dV_acc[r][cc]+=pf[r]*gf[cc]; dK_acc[r][cc]+=dsf[r]*qf[cc]; }
            }
        }
    }

    // write
    int jb = ty*4, cb = tx*8;
    #pragma unroll
    for (int r=0;r<4;r++){
        int grow = kv0 + jb + r;
        if (grow < S){
            #pragma unroll
            for (int cc=0;cc<8;cc++){
                dVbh[(size_t)grow*d + cb+cc] = __float2bfloat16(dV_acc[r][cc]);
                dKbh[(size_t)grow*d + cb+cc] = __float2bfloat16(scale*dK_acc[r][cc]);
            }
        }
    }
}

// -------- dQ kernel --------
// grid: (num_q_blocks, B*H). block: (16,16)=256 threads
__global__ void bwd_dq_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                              const bf16* __restrict__ V, const bf16* __restrict__ dO,
                              const float* __restrict__ L, const float* __restrict__ Delta,
                              bf16* __restrict__ dQ,
                              int S, int d, float scale) {
    int q_blk = blockIdx.x;
    int bh = blockIdx.y;
    int q0 = q_blk * BM;

    const bf16* Kbh  = K  + (size_t)bh * S * d;
    const bf16* Vbh  = V  + (size_t)bh * S * d;
    const bf16* Qbh  = Q  + (size_t)bh * S * d;
    const bf16* dObh = dO + (size_t)bh * S * d;
    const float* Lbh = L  + (size_t)bh * S;
    const float* Dbh = Delta + (size_t)bh * S;
    bf16* dQbh = dQ + (size_t)bh * S * d;

    extern __shared__ char smem[];
    bf16* sK  = reinterpret_cast<bf16*>(smem);
    bf16* sV  = sK + BN*DIM;
    bf16* sQ  = sV + BN*DIM;
    bf16* sdO = sQ + BM*DIM;
    float* sP    = reinterpret_cast<float*>(sdO + BM*DIM);
    float* sdS   = sP + BN*BM;
    float* sL    = sdS + BN*BM;
    float* sDelta= sL + BM;

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * 16 + tx;

    // load Q, dO for this query tile (persistent)
    for (int idx = tid; idx < BM*DIM; idx += 256) {
        int i = idx >> 7, c = idx & 127;
        int grow = q0 + i;
        bf16 qv = __float2bfloat16(0.f), gv = __float2bfloat16(0.f);
        if (grow < S) { qv = Qbh[(size_t)grow*d + c]; gv = dObh[(size_t)grow*d + c]; }
        sQ[idx] = qv; sdO[idx] = gv;
    }
    for (int idx = tid; idx < BM; idx += 256) {
        int grow = q0 + idx;
        sL[idx]     = (grow < S) ? Lbh[grow] : 0.f;
        sDelta[idx] = (grow < S) ? Dbh[grow] : 0.f;
    }

    float dQ_acc[4][8];
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (int cc = 0; cc < 8; cc++) dQ_acc[r][cc] = 0.f;

    int num_kv = (S + BN - 1) / BN;
    for (int kvb = 0; kvb < num_kv; ++kvb) {
        int kv0 = kvb * BN;
        __syncthreads();
        for (int idx = tid; idx < BN*DIM; idx += 256) {
            int j = idx >> 7, c = idx & 127;
            int grow = kv0 + j;
            bf16 kv = __float2bfloat16(0.f), vv = __float2bfloat16(0.f);
            if (grow < S) { kv = Kbh[(size_t)grow*d + c]; vv = Vbh[(size_t)grow*d + c]; }
            sK[idx] = kv; sV[idx] = vv;
        }
        __syncthreads();

        // GEMM1&2: S[i][j]=Q.K ; dP[i][j]=dO.V -> P, dS  (rows i=ty*4, cols j=tx*4)
        {
            float accS[4][4], accP[4][4];
            #pragma unroll
            for (int r=0;r<4;r++)
                #pragma unroll
                for (int cc=0;cc<4;cc++){accS[r][cc]=0.f;accP[r][cc]=0.f;}
            int ib = ty*4, jb = tx*4;
            #pragma unroll 4
            for (int c = 0; c < DIM; c++) {
                float qf[4], gf[4], kf[4], vf[4];
                #pragma unroll
                for (int r=0;r<4;r++){ qf[r]=__bfloat162float(sQ[(ib+r)*DIM+c]); gf[r]=__bfloat162float(sdO[(ib+r)*DIM+c]); }
                #pragma unroll
                for (int cc=0;cc<4;cc++){ kf[cc]=__bfloat162float(sK[(jb+cc)*DIM+c]); vf[cc]=__bfloat162float(sV[(jb+cc)*DIM+c]); }
                #pragma unroll
                for (int r=0;r<4;r++)
                    #pragma unroll
                    for (int cc=0;cc<4;cc++){ accS[r][cc]+=qf[r]*kf[cc]; accP[r][cc]+=gf[r]*vf[cc]; }
            }
            #pragma unroll
            for (int r=0;r<4;r++)
                #pragma unroll
                for (int cc=0;cc<4;cc++){
                    int i = ib+r;
                    float arg = scale*accS[r][cc] - sL[i];
                    float p = __expf(fminf(arg, 30.0f));
                    float ds = p*(accP[r][cc] - sDelta[i]);
                    sP[i*BN + (jb+cc)] = p;
                    sdS[i*BN + (jb+cc)] = ds;
                }
        }
        __syncthreads();

        // GEMM3: dQ[i][c] += dS.K   (rows i=ty*4, cols c=tx*8)
        {
            int ib = ty*4, cb = tx*8;
            #pragma unroll 4
            for (int j = 0; j < BN; j++) {
                float dsf[4];
                #pragma unroll
                for (int r=0;r<4;r++) dsf[r]=sdS[(ib+r)*BN + j];
                float kf[8];
                #pragma unroll
                for (int cc=0;cc<8;cc++) kf[cc]=__bfloat162float(sK[j*DIM + cb+cc]);
                #pragma unroll
                for (int r=0;r<4;r++)
                    #pragma unroll
                    for (int cc=0;cc<8;cc++) dQ_acc[r][cc]+=dsf[r]*kf[cc];
            }
        }
    }

    int ib = ty*4, cb = tx*8;
    #pragma unroll
    for (int r=0;r<4;r++){
        int grow = q0 + ib + r;
        if (grow < S){
            #pragma unroll
            for (int cc=0;cc<8;cc++){
                dQbh[(size_t)grow*d + cb+cc] = __float2bfloat16(scale*dQ_acc[r][cc]);
            }
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

    // Delta buffer
    float* Delta = nullptr;
    size_t deltaSz = (size_t)BH * S * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&Delta, deltaSz, stream));

    long long total_rows = (long long)BH * S;
    int dblocks = (int)((total_rows + 7) / 8);
    compute_delta_kernel<<<dblocks, 256, 0, stream>>>(Op, dOp, Delta, total_rows, (int)d);
    CUDA_CHECK(cudaGetLastError());

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        attr_set = true;
    }

    dim3 block(16, 16);
    int num_kv = (int)((S + BN - 1) / BN);
    int num_q  = (int)((S + BM - 1) / BM);

    dim3 grid_kv(num_kv, (unsigned)BH);
    bwd_dkdv_kernel<<<grid_kv, block, SMEM_BYTES, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dKp, dVp, (int)S, (int)d, scale);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_q(num_q, (unsigned)BH);
    bwd_dq_kernel<<<grid_q, block, SMEM_BYTES, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dQp, (int)S, (int)d, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Delta, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd