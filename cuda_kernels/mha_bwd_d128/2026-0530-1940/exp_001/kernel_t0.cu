#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
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
namespace wmma = nvcuda::wmma;

using FragA    = wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major>;
using FragB    = wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major>;
using FragBcol = wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major>;
using FragC    = wmma::fragment<wmma::accumulator,16,16,16,float>;

// ---------------- Delta kernel: D = rowsum(dO * O) ----------------
__global__ void delta_kernel(const bf16* __restrict__ O, const bf16* __restrict__ dO,
                             float* __restrict__ Delta, long total) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int4* o4 = reinterpret_cast<const int4*>(O + idx*128);
    const int4* g4 = reinterpret_cast<const int4*>(dO + idx*128);
    float acc = 0.f;
    #pragma unroll
    for (int j=0;j<16;j++){
        int4 ov=o4[j], gv=g4[j];
        const bf16* op=reinterpret_cast<const bf16*>(&ov);
        const bf16* gp=reinterpret_cast<const bf16*>(&gv);
        #pragma unroll
        for(int k=0;k<8;k++) acc += __bfloat162float(op[k])*__bfloat162float(gp[k]);
    }
    Delta[idx]=acc;
}

__device__ __forceinline__ void load_tile(bf16 dst[64][128], const bf16* src,
                                           int row0, int S, int tid){
    #pragma unroll
    for(int idx=tid; idx<64*16; idx+=128){
        int r=idx>>4;
        int c8=(idx&15)<<3;
        int row=row0+r;
        int4 v=make_int4(0,0,0,0);
        if(row<S) v=*reinterpret_cast<const int4*>(src + (size_t)row*128 + c8);
        *reinterpret_cast<int4*>(&dst[r][c8])=v;
    }
}

// ---------------- dK / dV kernel ----------------
__global__ __launch_bounds__(128) void dkdv_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ Lg, const float* __restrict__ Dg,
    bf16* __restrict__ dKo, bf16* __restrict__ dVo, int S, float scale)
{
    extern __shared__ char smem[];
    bf16  (*sK)[128]  = reinterpret_cast<bf16(*)[128]>(smem + 0);
    bf16  (*sV)[128]  = reinterpret_cast<bf16(*)[128]>(smem + 16384);
    bf16  (*sQ)[128]  = reinterpret_cast<bf16(*)[128]>(smem + 32768);
    bf16  (*sdO)[128] = reinterpret_cast<bf16(*)[128]>(smem + 49152);
    float (*sSt)[64]  = reinterpret_cast<float(*)[64]>(smem + 65536);
    float (*sdPt)[64] = reinterpret_cast<float(*)[64]>(smem + 81920);
    bf16  (*sPt)[64]  = reinterpret_cast<bf16(*)[64]>(smem + 98304);
    bf16  (*sdSt)[64] = reinterpret_cast<bf16(*)[64]>(smem + 106496);
    float* sL = reinterpret_cast<float*>(smem + 114688);
    float* sD = reinterpret_cast<float*>(smem + 114944);
    float (*sOut)[128]= reinterpret_cast<float(*)[128]>(smem + 65536);

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int bh = blockIdx.y;
    int kv0 = blockIdx.x * 64;

    const bf16* Kb = K  + (size_t)bh*S*128;
    const bf16* Vb = V  + (size_t)bh*S*128;
    const bf16* Qb = Q  + (size_t)bh*S*128;
    const bf16* dOb= dO + (size_t)bh*S*128;
    const float* Lb = Lg + (size_t)bh*S;
    const float* Db = Dg + (size_t)bh*S;
    bf16* dKb = dKo + (size_t)bh*S*128;
    bf16* dVb = dVo + (size_t)bh*S*128;

    load_tile(sK, Kb, kv0, S, tid);
    load_tile(sV, Vb, kv0, S, tid);

    FragC accV[8], accK[8];
    #pragma unroll
    for(int i=0;i<8;i++){ wmma::fill_fragment(accV[i],0.0f); wmma::fill_fragment(accK[i],0.0f); }

    __syncthreads();

    int nQ = (S + 63)/64;
    for(int qb=0; qb<nQ; qb++){
        int q0 = qb*64;
        load_tile(sQ, Qb, q0, S, tid);
        load_tile(sdO, dOb, q0, S, tid);
        for(int i=tid;i<64;i+=128){
            int row=q0+i;
            sL[i] = (row<S)? Lb[row] : 0.f;
            sD[i] = (row<S)? Db[row] : 0.f;
        }
        __syncthreads();

        // S^T = K @ Q^T
        {
            FragC acc[4];
            #pragma unroll
            for(int j=0;j<4;j++) wmma::fill_fragment(acc[j],0.0f);
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                FragA a; wmma::load_matrix_sync(a, &sK[16*warp][kt*16], 128);
                #pragma unroll
                for(int j=0;j<4;j++){
                    FragBcol b; wmma::load_matrix_sync(b, &sQ[16*j][kt*16], 128);
                    wmma::mma_sync(acc[j], a, b, acc[j]);
                }
            }
            #pragma unroll
            for(int j=0;j<4;j++) wmma::store_matrix_sync(&sSt[16*warp][16*j], acc[j], 64, wmma::mem_row_major);
        }
        // dP^T = V @ dO^T
        {
            FragC acc[4];
            #pragma unroll
            for(int j=0;j<4;j++) wmma::fill_fragment(acc[j],0.0f);
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                FragA a; wmma::load_matrix_sync(a, &sV[16*warp][kt*16], 128);
                #pragma unroll
                for(int j=0;j<4;j++){
                    FragBcol b; wmma::load_matrix_sync(b, &sdO[16*j][kt*16], 128);
                    wmma::mma_sync(acc[j], a, b, acc[j]);
                }
            }
            #pragma unroll
            for(int j=0;j<4;j++) wmma::store_matrix_sync(&sdPt[16*warp][16*j], acc[j], 64, wmma::mem_row_major);
        }
        __syncthreads();

        // elementwise: m = KV row, n = query col
        for(int idx=tid; idx<64*64; idx+=128){
            int m=idx>>6, n=idx&63;
            float s=sSt[m][n];
            float dp=sdPt[m][n];
            float p=__expf(scale*s - sL[n]);
            float ds=p*(dp - sD[n]);
            sPt[m][n]=__float2bfloat16(p);
            sdSt[m][n]=__float2bfloat16(ds);
        }
        __syncthreads();

        // dV += P^T @ dO ; dK += dS^T @ Q
        #pragma unroll
        for(int kt=0;kt<4;kt++){
            FragA aP; wmma::load_matrix_sync(aP, &sPt[16*warp][kt*16], 64);
            FragA aS; wmma::load_matrix_sync(aS, &sdSt[16*warp][kt*16], 64);
            #pragma unroll
            for(int nt=0;nt<8;nt++){
                FragB bO; wmma::load_matrix_sync(bO, &sdO[kt*16][nt*16], 128);
                wmma::mma_sync(accV[nt], aP, bO, accV[nt]);
                FragB bQ; wmma::load_matrix_sync(bQ, &sQ[kt*16][nt*16], 128);
                wmma::mma_sync(accK[nt], aS, bQ, accK[nt]);
            }
        }
        __syncthreads();
    }

    // write dV
    #pragma unroll
    for(int nt=0;nt<8;nt++) wmma::store_matrix_sync(&sOut[16*warp][nt*16], accV[nt], 128, wmma::mem_row_major);
    __syncthreads();
    for(int idx=tid; idx<64*128; idx+=128){
        int m=idx>>7, e=idx&127;
        int row=kv0+m;
        if(row<S) dVb[(size_t)row*128+e]=__float2bfloat16(sOut[m][e]);
    }
    __syncthreads();
    // write dK (scaled)
    #pragma unroll
    for(int nt=0;nt<8;nt++){
        #pragma unroll
        for(int i=0;i<accK[nt].num_elements;i++) accK[nt].x[i]*=scale;
        wmma::store_matrix_sync(&sOut[16*warp][nt*16], accK[nt], 128, wmma::mem_row_major);
    }
    __syncthreads();
    for(int idx=tid; idx<64*128; idx+=128){
        int m=idx>>7, e=idx&127;
        int row=kv0+m;
        if(row<S) dKb[(size_t)row*128+e]=__float2bfloat16(sOut[m][e]);
    }
}

// ---------------- dQ kernel ----------------
__global__ __launch_bounds__(128) void dq_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ Lg, const float* __restrict__ Dg,
    bf16* __restrict__ dQo, int S, float scale)
{
    extern __shared__ char smem[];
    bf16  (*sK)[128]  = reinterpret_cast<bf16(*)[128]>(smem + 0);
    bf16  (*sV)[128]  = reinterpret_cast<bf16(*)[128]>(smem + 16384);
    bf16  (*sQ)[128]  = reinterpret_cast<bf16(*)[128]>(smem + 32768);
    bf16  (*sdO)[128] = reinterpret_cast<bf16(*)[128]>(smem + 49152);
    float (*sS)[64]   = reinterpret_cast<float(*)[64]>(smem + 65536);
    float (*sdP)[64]  = reinterpret_cast<float(*)[64]>(smem + 81920);
    bf16  (*sdS)[64]  = reinterpret_cast<bf16(*)[64]>(smem + 98304);
    float* sL = reinterpret_cast<float*>(smem + 106496);
    float* sD = reinterpret_cast<float*>(smem + 106752);
    float (*sOut)[128]= reinterpret_cast<float(*)[128]>(smem + 65536);

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int bh = blockIdx.y;
    int q0 = blockIdx.x * 64;

    const bf16* Kb = K  + (size_t)bh*S*128;
    const bf16* Vb = V  + (size_t)bh*S*128;
    const bf16* Qb = Q  + (size_t)bh*S*128;
    const bf16* dOb= dO + (size_t)bh*S*128;
    const float* Lb = Lg + (size_t)bh*S;
    const float* Db = Dg + (size_t)bh*S;
    bf16* dQb = dQo + (size_t)bh*S*128;

    load_tile(sQ, Qb, q0, S, tid);
    load_tile(sdO, dOb, q0, S, tid);
    for(int i=tid;i<64;i+=128){
        int row=q0+i;
        sL[i] = (row<S)? Lb[row] : 0.f;
        sD[i] = (row<S)? Db[row] : 0.f;
    }

    FragC accQ[8];
    #pragma unroll
    for(int i=0;i<8;i++) wmma::fill_fragment(accQ[i],0.0f);

    __syncthreads();

    int nK = (S + 63)/64;
    for(int kb=0; kb<nK; kb++){
        int kv0 = kb*64;
        load_tile(sK, Kb, kv0, S, tid);
        load_tile(sV, Vb, kv0, S, tid);
        __syncthreads();

        // S = Q @ K^T  (rows=query of warp, cols=key)
        {
            FragC acc[4];
            #pragma unroll
            for(int j=0;j<4;j++) wmma::fill_fragment(acc[j],0.0f);
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                FragA a; wmma::load_matrix_sync(a, &sQ[16*warp][kt*16], 128);
                #pragma unroll
                for(int j=0;j<4;j++){
                    FragBcol b; wmma::load_matrix_sync(b, &sK[16*j][kt*16], 128);
                    wmma::mma_sync(acc[j], a, b, acc[j]);
                }
            }
            #pragma unroll
            for(int j=0;j<4;j++) wmma::store_matrix_sync(&sS[16*warp][16*j], acc[j], 64, wmma::mem_row_major);
        }
        // dP = dO @ V^T
        {
            FragC acc[4];
            #pragma unroll
            for(int j=0;j<4;j++) wmma::fill_fragment(acc[j],0.0f);
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                FragA a; wmma::load_matrix_sync(a, &sdO[16*warp][kt*16], 128);
                #pragma unroll
                for(int j=0;j<4;j++){
                    FragBcol b; wmma::load_matrix_sync(b, &sV[16*j][kt*16], 128);
                    wmma::mma_sync(acc[j], a, b, acc[j]);
                }
            }
            #pragma unroll
            for(int j=0;j<4;j++) wmma::store_matrix_sync(&sdP[16*warp][16*j], acc[j], 64, wmma::mem_row_major);
        }
        __syncthreads();

        // elementwise: n = query row, m = key col
        for(int idx=tid; idx<64*64; idx+=128){
            int n=idx>>6, m=idx&63;
            float s=sS[n][m];
            float dp=sdP[n][m];
            float p=__expf(scale*s - sL[n]);
            float ds=p*(dp - sD[n]);
            sdS[n][m]=__float2bfloat16(ds);
        }
        __syncthreads();

        // dQ += dS @ K
        #pragma unroll
        for(int kt=0;kt<4;kt++){
            FragA aS; wmma::load_matrix_sync(aS, &sdS[16*warp][kt*16], 64);
            #pragma unroll
            for(int nt=0;nt<8;nt++){
                FragB bK; wmma::load_matrix_sync(bK, &sK[kt*16][nt*16], 128);
                wmma::mma_sync(accQ[nt], aS, bK, accQ[nt]);
            }
        }
        __syncthreads();
    }

    // write dQ (scaled)
    #pragma unroll
    for(int nt=0;nt<8;nt++){
        #pragma unroll
        for(int i=0;i<accQ[nt].num_elements;i++) accQ[nt].x[i]*=scale;
        wmma::store_matrix_sync(&sOut[16*warp][nt*16], accQ[nt], 128, wmma::mem_row_major);
    }
    __syncthreads();
    for(int idx=tid; idx<64*128; idx+=128){
        int m=idx>>7, e=idx&127;
        int row=q0+m;
        if(row<S) dQb[(size_t)row*128+e]=__float2bfloat16(sOut[m][e]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    long BH = (long)B*H;
    float scale = 1.0f / sqrtf(128.0f);

    const bf16* Qp = static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp = static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp = static_cast<const bf16*>(V.data_ptr());
    const bf16* Op = static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp= static_cast<const bf16*>(dO.data_ptr());
    const float* Lp= static_cast<const float*>(L.data_ptr());
    bf16* dQp = static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp = static_cast<bf16*>(dK.data_ptr());
    bf16* dVp = static_cast<bf16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Delta = nullptr;
    CUDA_CHECK(cudaMalloc(&Delta, sizeof(float)*BH*S));

    long total = BH*S;
    int dthreads = 256;
    long dblocks = (total + dthreads - 1)/dthreads;
    delta_kernel<<<dblocks, dthreads, 0, stream>>>(Op, dOp, Delta, total);

    int nblk = (S + 63)/64;
    dim3 grid(nblk, (unsigned)BH);

    size_t smem_dkdv = 115200;
    size_t smem_dq   = 107008;

    static bool attr_set = false;
    CUDA_CHECK(cudaFuncSetAttribute((const void*)dkdv_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dkdv));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)dq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));
    (void)attr_set;

    dkdv_kernel<<<grid, 128, smem_dkdv, stream>>>(Qp, Kp, Vp, dOp, Lp, Delta, dKp, dVp, S, scale);
    dq_kernel<<<grid, 128, smem_dq, stream>>>(Qp, Kp, Vp, dOp, Lp, Delta, dQp, S, scale);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd