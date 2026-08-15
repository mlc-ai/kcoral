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

// D = rowsum(dO * O)
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

// Load 64x128 bf16 tile (row0..row0+63) into contiguous dst, zero-padded.
__device__ __forceinline__ void load_tile(bf16* dst, const bf16* src, int row0, int S, int tid){
    #pragma unroll
    for(int idx=tid; idx<64*16; idx+=128){
        int r=idx>>4;
        int c8=(idx&15)<<3;
        int row=row0+r;
        int4 v=make_int4(0,0,0,0);
        if(row<S) v=*reinterpret_cast<const int4*>(src + (size_t)row*128 + c8);
        *reinterpret_cast<int4*>(dst + r*128 + c8)=v;
    }
}

// Out[ar][bc] = sum_e A[ar][e]*B[bc][e]  (A@B^T), 64x64 out, contract 128.
// A,B contiguous [64][128]; Out contiguous [64][64]. warp handles rows [16w,16w+16).
__device__ __forceinline__ void qkt(const bf16* A, const bf16* B, float* Out, int warp){
    FragC acc[4];
    #pragma unroll
    for(int j=0;j<4;j++) wmma::fill_fragment(acc[j],0.f);
    #pragma unroll
    for(int kt=0;kt<8;kt++){
        FragA a; wmma::load_matrix_sync(a, A + (16*warp)*128 + kt*16, 128);
        #pragma unroll
        for(int j=0;j<4;j++){
            FragBcol b; wmma::load_matrix_sync(b, B + (16*j)*128 + kt*16, 128);
            wmma::mma_sync(acc[j],a,b,acc[j]);
        }
    }
    #pragma unroll
    for(int j=0;j<4;j++)
        wmma::store_matrix_sync(Out + (16*warp)*64 + 16*j, acc[j], 64, wmma::mem_row_major);
}

// acc[nt] += A@B : out[ar][e]=sum_k A[ar][k]*B[k][e], contract 64, out cols 128.
// A contiguous [64][64]; B contiguous [64][128]; acc[8] resident.
__device__ __forceinline__ void pv(const bf16* A, const bf16* B, FragC* acc, int warp){
    #pragma unroll
    for(int kt=0;kt<4;kt++){
        FragA a; wmma::load_matrix_sync(a, A + (16*warp)*64 + kt*16, 64);
        #pragma unroll
        for(int nt=0;nt<8;nt++){
            FragB b; wmma::load_matrix_sync(b, B + (kt*16)*128 + nt*16, 128);
            wmma::mma_sync(acc[nt],a,b,acc[nt]);
        }
    }
}

__device__ __forceinline__ void write_out(FragC* acc, float* sOut, bf16* gbase,
                                          int base0, int S, int tid, int warp, float sc){
    #pragma unroll
    for(int nt=0;nt<8;nt++){
        if(sc!=1.0f){
            #pragma unroll
            for(int i=0;i<acc[nt].num_elements;i++) acc[nt].x[i]*=sc;
        }
        wmma::store_matrix_sync(sOut + (16*warp)*128 + 16*nt, acc[nt], 128, wmma::mem_row_major);
    }
    __syncthreads();
    for(int idx=tid; idx<64*128; idx+=128){
        int m=idx>>7, e=idx&127;
        int row=base0+m;
        if(row<S) gbase[(size_t)row*128+e]=__float2bfloat16(sOut[m*128+e]);
    }
}

// ---------------- dV kernel ----------------
__global__ __launch_bounds__(128,2) void dv_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ dO, const float* __restrict__ Lg,
    bf16* __restrict__ dVo, int S, float scale)
{
    extern __shared__ char smem[];
    bf16*  sK    = (bf16*)(smem + 0);       // [64][128]
    bf16*  sQ    = (bf16*)(smem + 16384);
    bf16*  sdO   = (bf16*)(smem + 32768);
    float* sScore= (float*)(smem + 49152);  // [64][64]
    bf16*  sPt   = (bf16*)(smem + 65536);   // [64][64]
    float* sL    = (float*)(smem + 73728);  // [64]
    float* sOut  = (float*)(smem + 49152);  // [64][128] (overlaps score region)

    int tid=threadIdx.x, warp=tid>>5, bh=blockIdx.y, kv0=blockIdx.x*64;
    const bf16* Kb=K+(size_t)bh*S*128;
    const bf16* Qb=Q+(size_t)bh*S*128;
    const bf16* dOb=dO+(size_t)bh*S*128;
    const float* Lb=Lg+(size_t)bh*S;
    bf16* dVb=dVo+(size_t)bh*S*128;

    load_tile(sK, Kb, kv0, S, tid);
    FragC accV[8];
    #pragma unroll
    for(int i=0;i<8;i++) wmma::fill_fragment(accV[i],0.f);

    int nQ=(S+63)/64;
    for(int qb=0; qb<nQ; qb++){
        int q0=qb*64;
        load_tile(sQ, Qb, q0, S, tid);
        load_tile(sdO, dOb, q0, S, tid);
        for(int i=tid;i<64;i+=128){ int r=q0+i; sL[i]=(r<S)?Lb[r]:0.f; }
        __syncthreads();
        qkt(sK, sQ, sScore, warp);          // S^T = K@Q^T
        __syncthreads();
        for(int idx=tid; idx<4096; idx+=128){
            int m=idx>>6, n=idx&63;
            sPt[m*64+n]=__float2bfloat16(__expf(scale*sScore[m*64+n]-sL[n]));
        }
        __syncthreads();
        pv(sPt, sdO, accV, warp);           // dV += P^T@dO
        __syncthreads();
    }
    write_out(accV, sOut, dVb, kv0, S, tid, warp, 1.0f);
}

// ---------------- dK kernel ----------------
__global__ __launch_bounds__(128,2) void dk_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ Lg, const float* __restrict__ Dg,
    bf16* __restrict__ dKo, int S, float scale)
{
    extern __shared__ char smem[];
    bf16*  sK    = (bf16*)(smem + 0);
    bf16*  sQ    = (bf16*)(smem + 16384);
    bf16*  sV    = (bf16*)(smem + 32768);
    bf16*  sdO   = (bf16*)(smem + 49152);
    float* sScore= (float*)(smem + 65536);  // [64][64]
    bf16*  sPt   = (bf16*)(smem + 81920);   // [64][64]
    bf16*  sdSt  = (bf16*)(smem + 90112);   // [64][64]
    float* sL    = (float*)(smem + 98304);
    float* sD    = (float*)(smem + 98560);
    float* sOut  = (float*)(smem + 65536);  // [64][128]

    int tid=threadIdx.x, warp=tid>>5, bh=blockIdx.y, kv0=blockIdx.x*64;
    const bf16* Kb=K+(size_t)bh*S*128;
    const bf16* Qb=Q+(size_t)bh*S*128;
    const bf16* Vb=V+(size_t)bh*S*128;
    const bf16* dOb=dO+(size_t)bh*S*128;
    const float* Lb=Lg+(size_t)bh*S;
    const float* Db=Dg+(size_t)bh*S;
    bf16* dKb=dKo+(size_t)bh*S*128;

    load_tile(sK, Kb, kv0, S, tid);
    load_tile(sV, Vb, kv0, S, tid);
    FragC accK[8];
    #pragma unroll
    for(int i=0;i<8;i++) wmma::fill_fragment(accK[i],0.f);

    int nQ=(S+63)/64;
    for(int qb=0; qb<nQ; qb++){
        int q0=qb*64;
        load_tile(sQ, Qb, q0, S, tid);
        load_tile(sdO, dOb, q0, S, tid);
        for(int i=tid;i<64;i+=128){ int r=q0+i; sL[i]=(r<S)?Lb[r]:0.f; sD[i]=(r<S)?Db[r]:0.f; }
        __syncthreads();
        qkt(sK, sQ, sScore, warp);          // S^T
        __syncthreads();
        for(int idx=tid; idx<4096; idx+=128){
            int m=idx>>6, n=idx&63;
            sPt[m*64+n]=__float2bfloat16(__expf(scale*sScore[m*64+n]-sL[n]));
        }
        __syncthreads();
        qkt(sV, sdO, sScore, warp);          // dP^T (reuse sScore)
        __syncthreads();
        for(int idx=tid; idx<4096; idx+=128){
            int m=idx>>6, n=idx&63;
            float p=__bfloat162float(sPt[m*64+n]);
            sdSt[m*64+n]=__float2bfloat16(p*(sScore[m*64+n]-sD[n]));
        }
        __syncthreads();
        pv(sdSt, sQ, accK, warp);            // dK += dS^T@Q
        __syncthreads();
    }
    write_out(accK, sOut, dKb, kv0, S, tid, warp, scale);
}

// ---------------- dQ kernel ----------------
__global__ __launch_bounds__(128,2) void dq_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ Lg, const float* __restrict__ Dg,
    bf16* __restrict__ dQo, int S, float scale)
{
    extern __shared__ char smem[];
    bf16*  sQ    = (bf16*)(smem + 0);
    bf16*  sdO   = (bf16*)(smem + 16384);
    bf16*  sK    = (bf16*)(smem + 32768);
    bf16*  sV    = (bf16*)(smem + 49152);
    float* sScore= (float*)(smem + 65536);
    bf16*  sPt   = (bf16*)(smem + 81920);
    bf16*  sdS   = (bf16*)(smem + 90112);
    float* sL    = (float*)(smem + 98304);
    float* sD    = (float*)(smem + 98560);
    float* sOut  = (float*)(smem + 65536);

    int tid=threadIdx.x, warp=tid>>5, bh=blockIdx.y, q0=blockIdx.x*64;
    const bf16* Kb=K+(size_t)bh*S*128;
    const bf16* Qb=Q+(size_t)bh*S*128;
    const bf16* Vb=V+(size_t)bh*S*128;
    const bf16* dOb=dO+(size_t)bh*S*128;
    const float* Lb=Lg+(size_t)bh*S;
    const float* Db=Dg+(size_t)bh*S;
    bf16* dQb=dQo+(size_t)bh*S*128;

    load_tile(sQ, Qb, q0, S, tid);
    load_tile(sdO, dOb, q0, S, tid);
    for(int i=tid;i<64;i+=128){ int r=q0+i; sL[i]=(r<S)?Lb[r]:0.f; sD[i]=(r<S)?Db[r]:0.f; }

    FragC accQ[8];
    #pragma unroll
    for(int i=0;i<8;i++) wmma::fill_fragment(accQ[i],0.f);

    int nK=(S+63)/64;
    for(int kb=0; kb<nK; kb++){
        int kv0=kb*64;
        load_tile(sK, Kb, kv0, S, tid);
        load_tile(sV, Vb, kv0, S, tid);
        __syncthreads();
        qkt(sQ, sK, sScore, warp);          // S[n][m] = Q@K^T
        __syncthreads();
        for(int idx=tid; idx<4096; idx+=128){
            int n=idx>>6, m=idx&63;
            sPt[n*64+m]=__float2bfloat16(__expf(scale*sScore[n*64+m]-sL[n]));
        }
        __syncthreads();
        qkt(sdO, sV, sScore, warp);          // dP[n][m] = dO@V^T
        __syncthreads();
        for(int idx=tid; idx<4096; idx+=128){
            int n=idx>>6, m=idx&63;
            float p=__bfloat162float(sPt[n*64+m]);
            sdS[n*64+m]=__float2bfloat16(p*(sScore[n*64+m]-sD[n]));
        }
        __syncthreads();
        pv(sdS, sK, accQ, warp);             // dQ += dS@K
        __syncthreads();
    }
    write_out(accQ, sOut, dQb, q0, S, tid, warp, scale);
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

    size_t smem_dv = 81920;
    size_t smem_dk = 98816;
    size_t smem_dq = 98816;

    CUDA_CHECK(cudaFuncSetAttribute((const void*)dv_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dv));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)dk_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dk));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)dq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));

    dv_kernel<<<grid, 128, smem_dv, stream>>>(Qp, Kp, dOp, Lp, dVp, S, scale);
    dk_kernel<<<grid, 128, smem_dk, stream>>>(Qp, Kp, Vp, dOp, Lp, Delta, dKp, S, scale);
    dq_kernel<<<grid, 128, smem_dq, stream>>>(Qp, Kp, Vp, dOp, Lp, Delta, dQp, S, scale);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd