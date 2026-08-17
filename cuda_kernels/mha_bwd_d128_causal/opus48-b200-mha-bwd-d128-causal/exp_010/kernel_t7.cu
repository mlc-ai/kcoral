#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {

typedef wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> FragA;
typedef wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> FragBrow;
typedef wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> FragBcol;
typedef wmma::fragment<wmma::accumulator, 16,16,16, float> FragC;

__device__ __forceinline__ half b2h(__nv_bfloat16 x){ return __float2half(__bfloat162float(x)); }

__global__ void compute_D_kernel(const __nv_bfloat16* __restrict__ O,
                                 const __nv_bfloat16* __restrict__ dO,
                                 float* __restrict__ D, int64_t total_rows){
    int64_t row = blockIdx.x;
    if(row >= total_rows) return;
    int t = threadIdx.x;
    float v = __bfloat162float(O[row*128 + t]) * __bfloat162float(dO[row*128 + t]);
    for(int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
    __shared__ float wr[4];
    if((t&31)==0) wr[t>>5]=v;
    __syncthreads();
    if(t==0) D[row]=wr[0]+wr[1]+wr[2]+wr[3];
}

// ---- dK/dV kernel: key-block BN=64 (parallel), query inner BM=128, 4 warps ----
__launch_bounds__(128,1)
__global__ void bwd_dkdv_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S, float scale)
{
    int bh = blockIdx.y;
    int kb = blockIdx.x;
    int n0 = kb*64;
    if(n0 >= S) return;
    int64_t base  = (int64_t)bh * S * 128;
    int64_t lbase = (int64_t)bh * S;

    extern __shared__ char smem[];
    half*  Ksh  = (half*)smem;          // [64,128]
    half*  Vsh  = Ksh  + 64*128;        // [64,128]
    half*  Qsh  = Vsh  + 64*128;        // [128,128]
    half*  dOsh = Qsh  + 128*128;       // [128,128]
    float* Sbuf = (float*)(dOsh + 128*128); // [64,128]
    half*  PTsh = (half*)(Sbuf + 64*128);   // [64,128]
    half*  dSTsh= PTsh + 64*128;            // [64,128]
    float* Lsh  = (float*)(dSTsh + 64*128); // [128]
    float* Drow = Lsh + 128;                // [128]

    int tid = threadIdx.x;
    int warp = tid >> 5;   // 0..3 -> key band
    int wk = warp;         // owns keys [wk*16, +16)

    for(int i=tid;i<64*128;i+=128){
        int r=i>>7, c=i&127; int gr=n0+r;
        if(gr<S){ Ksh[i]=b2h(K[base+(int64_t)gr*128+c]); Vsh[i]=b2h(V[base+(int64_t)gr*128+c]); }
        else    { Ksh[i]=__float2half(0.f);              Vsh[i]=__float2half(0.f); }
    }

    FragC dV_frag[8], dK_frag[8];
    #pragma unroll
    for(int t=0;t<8;t++){ wmma::fill_fragment(dV_frag[t],0.f); wmma::fill_fragment(dK_frag[t],0.f); }

    int num_qb = (S+127)/128;
    int qb0 = kb/2;
    for(int qb=qb0; qb<num_qb; qb++){
        int m0=qb*128;
        __syncthreads();
        for(int i=tid;i<128*128;i+=128){
            int r=i>>7,c=i&127; int gr=m0+r;
            if(gr<S){ Qsh[i]=b2h(Q[base+(int64_t)gr*128+c]); dOsh[i]=b2h(dO[base+(int64_t)gr*128+c]); }
            else    { Qsh[i]=__float2half(0.f);              dOsh[i]=__float2half(0.f); }
        }
        for(int i=tid;i<128;i+=128){
            int gr=m0+i;
            Lsh[i]= gr<S? L[lbase+gr]:0.f;
            Drow[i]= gr<S? D[lbase+gr]:0.f;
        }
        __syncthreads();

        // S^T = K @ Q^T   [64 key, 128 query]
        #pragma unroll
        for(int qt=0;qt<8;qt++){
            FragC acc; wmma::fill_fragment(acc,0.f);
            #pragma unroll
            for(int kk=0;kk<8;kk++){
                FragA aK; wmma::load_matrix_sync(aK, &Ksh[wk*16*128 + kk*16], 128);
                FragBcol bQ; wmma::load_matrix_sync(bQ, &Qsh[qt*16*128 + kk*16], 128);
                wmma::mma_sync(acc,aK,bQ,acc);
            }
            wmma::store_matrix_sync(&Sbuf[wk*16*128 + qt*16], acc, 128, wmma::mem_row_major);
        }
        __syncthreads();

        // P^T = exp(scale*S^T - L[q]) masked
        for(int i=tid;i<64*128;i+=128){
            int key=i>>7, q=i&127;
            int gk=n0+key, gq=m0+q;
            bool valid = (gk<S)&&(gq<S)&&(gk<=gq);
            float p = valid ? __expf(scale*Sbuf[i] - Lsh[q]) : 0.f;
            PTsh[i]=__float2half(p);
        }
        __syncthreads();

        // dP^T = V @ dO^T   [64 key, 128 query]  (reuse Sbuf)
        #pragma unroll
        for(int qt=0;qt<8;qt++){
            FragC acc; wmma::fill_fragment(acc,0.f);
            #pragma unroll
            for(int kk=0;kk<8;kk++){
                FragA aV; wmma::load_matrix_sync(aV, &Vsh[wk*16*128 + kk*16], 128);
                FragBcol bO; wmma::load_matrix_sync(bO, &dOsh[qt*16*128 + kk*16], 128);
                wmma::mma_sync(acc,aV,bO,acc);
            }
            wmma::store_matrix_sync(&Sbuf[wk*16*128 + qt*16], acc, 128, wmma::mem_row_major);
        }
        __syncthreads();

        // dS^T = scale*P^T*(dP^T - D[q])
        for(int i=tid;i<64*128;i+=128){
            int q=i&127;
            float p=__half2float(PTsh[i]);
            float ds=scale*p*(Sbuf[i] - Drow[q]);
            dSTsh[i]=__float2half(ds);
        }
        __syncthreads();

        // dV += P^T @ dO ; dK += dS^T @ Q   (contract query=128)
        #pragma unroll
        for(int kk=0;kk<8;kk++){
            FragA aP; wmma::load_matrix_sync(aP, &PTsh[wk*16*128 + kk*16], 128);
            FragA aD; wmma::load_matrix_sync(aD, &dSTsh[wk*16*128 + kk*16], 128);
            #pragma unroll
            for(int dt=0;dt<8;dt++){
                FragBrow bO; wmma::load_matrix_sync(bO, &dOsh[kk*16*128 + dt*16], 128);
                FragBrow bQ; wmma::load_matrix_sync(bQ, &Qsh[kk*16*128 + dt*16], 128);
                wmma::mma_sync(dV_frag[dt], aP, bO, dV_frag[dt]);
                wmma::mma_sync(dK_frag[dt], aD, bQ, dK_frag[dt]);
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for(int dt=0;dt<8;dt++) wmma::store_matrix_sync(&Sbuf[wk*16*128 + dt*16], dV_frag[dt], 128, wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<64*128;i+=128){
        int key=i>>7, dd=i&127; int gk=n0+key;
        if(gk<S) dV[base+(int64_t)gk*128 + dd]=__float2bfloat16(Sbuf[i]);
    }
    __syncthreads();
    #pragma unroll
    for(int dt=0;dt<8;dt++) wmma::store_matrix_sync(&Sbuf[wk*16*128 + dt*16], dK_frag[dt], 128, wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<64*128;i+=128){
        int key=i>>7, dd=i&127; int gk=n0+key;
        if(gk<S) dK[base+(int64_t)gk*128 + dd]=__float2bfloat16(Sbuf[i]);
    }
}

// ---- dQ kernel: query-block BM=64 (parallel), key inner BN=128, 4 warps ----
__launch_bounds__(128,1)
__global__ void bwd_dq_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dQ,
    int S, float scale)
{
    int bh=blockIdx.y;
    int qb=blockIdx.x;
    int m0=qb*64;
    if(m0>=S) return;
    int64_t base=(int64_t)bh*S*128;
    int64_t lbase=(int64_t)bh*S;

    extern __shared__ char smem[];
    half*  Ksh  = (half*)smem;          // [128,128]
    half*  Vsh  = Ksh  + 128*128;       // [128,128]
    half*  Qsh  = Vsh  + 128*128;       // [64,128]
    half*  dOsh = Qsh  + 64*128;        // [64,128]
    float* Sbuf = (float*)(dOsh + 64*128); // [64,128]
    half*  Psh  = (half*)(Sbuf + 64*128);  // [64,128]
    half*  dSsh = Psh  + 64*128;           // [64,128]
    float* Lsh  = (float*)(dSsh + 64*128); // [64]
    float* Drow = Lsh + 64;                // [64]

    int tid=threadIdx.x;
    int warp=tid>>5;   // owns queries [warp*16,+16)
    int wq=warp;

    for(int i=tid;i<64*128;i+=128){
        int r=i>>7,c=i&127; int gr=m0+r;
        if(gr<S){ Qsh[i]=b2h(Q[base+(int64_t)gr*128+c]); dOsh[i]=b2h(dO[base+(int64_t)gr*128+c]); }
        else    { Qsh[i]=__float2half(0.f);              dOsh[i]=__float2half(0.f); }
    }
    for(int i=tid;i<64;i+=128){
        int gr=m0+i;
        Lsh[i]= gr<S? L[lbase+gr]:0.f;
        Drow[i]= gr<S? D[lbase+gr]:0.f;
    }

    FragC dQ_frag[8];
    #pragma unroll
    for(int t=0;t<8;t++) wmma::fill_fragment(dQ_frag[t],0.f);

    int kb_max = (m0+63)/128;
    for(int kb=0; kb<=kb_max; kb++){
        int n0=kb*128;
        __syncthreads();
        for(int i=tid;i<128*128;i+=128){
            int r=i>>7,c=i&127; int gr=n0+r;
            if(gr<S){ Ksh[i]=b2h(K[base+(int64_t)gr*128+c]); Vsh[i]=b2h(V[base+(int64_t)gr*128+c]); }
            else    { Ksh[i]=__float2half(0.f);              Vsh[i]=__float2half(0.f); }
        }
        __syncthreads();

        // S = Q @ K^T   [64 query, 128 key]
        #pragma unroll
        for(int kt=0;kt<8;kt++){
            FragC acc; wmma::fill_fragment(acc,0.f);
            #pragma unroll
            for(int kk=0;kk<8;kk++){
                FragA aQ; wmma::load_matrix_sync(aQ, &Qsh[wq*16*128 + kk*16], 128);
                FragBcol bK; wmma::load_matrix_sync(bK, &Ksh[kt*16*128 + kk*16], 128);
                wmma::mma_sync(acc,aQ,bK,acc);
            }
            wmma::store_matrix_sync(&Sbuf[wq*16*128 + kt*16], acc, 128, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(scale*S - L[q]) masked
        for(int i=tid;i<64*128;i+=128){
            int q=i>>7, key=i&127;
            int gq=m0+q, gk=n0+key;
            bool valid=(gk<S)&&(gq<S)&&(gk<=gq);
            float p=valid? __expf(scale*Sbuf[i] - Lsh[q]):0.f;
            Psh[i]=__float2half(p);
        }
        __syncthreads();

        // dP = dO @ V^T   [64 query, 128 key]
        #pragma unroll
        for(int kt=0;kt<8;kt++){
            FragC acc; wmma::fill_fragment(acc,0.f);
            #pragma unroll
            for(int kk=0;kk<8;kk++){
                FragA aO; wmma::load_matrix_sync(aO, &dOsh[wq*16*128 + kk*16], 128);
                FragBcol bV; wmma::load_matrix_sync(bV, &Vsh[kt*16*128 + kk*16], 128);
                wmma::mma_sync(acc,aO,bV,acc);
            }
            wmma::store_matrix_sync(&Sbuf[wq*16*128 + kt*16], acc, 128, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = scale*P*(dP - D[q])
        for(int i=tid;i<64*128;i+=128){
            int q=i>>7;
            float p=__half2float(Psh[i]);
            float ds=scale*p*(Sbuf[i] - Drow[q]);
            dSsh[i]=__float2half(ds);
        }
        __syncthreads();

        // dQ += dS @ K   (contract key=128)
        #pragma unroll
        for(int kk=0;kk<8;kk++){
            FragA aD; wmma::load_matrix_sync(aD, &dSsh[wq*16*128 + kk*16], 128);
            #pragma unroll
            for(int dt=0;dt<8;dt++){
                FragBrow bK; wmma::load_matrix_sync(bK, &Ksh[kk*16*128 + dt*16], 128);
                wmma::mma_sync(dQ_frag[dt], aD, bK, dQ_frag[dt]);
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for(int dt=0;dt<8;dt++) wmma::store_matrix_sync(&Sbuf[wq*16*128 + dt*16], dQ_frag[dt], 128, wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<64*128;i+=128){
        int q=i>>7, dd=i&127; int gq=m0+q;
        if(gq<S) dQ[base+(int64_t)gq*128 + dd]=__float2bfloat16(Sbuf[i]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int d = (int)Q.size(3);
    int BH = B*H;
    float scale = 1.0f / sqrtf((float)d);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp= static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    int64_t total_rows = (int64_t)BH * S;
    float* Dp = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&Dp, total_rows*sizeof(float), stream));

    compute_D_kernel<<<(unsigned int)total_rows, 128, 0, stream>>>(Op, dOp, Dp, total_rows);
    CUDA_CHECK(cudaGetLastError());

    // dkdv smem: K,V[64,128] + Q,dO[128,128] halves ; Sbuf[64,128] f32 ; PT,dST[64,128] half ; L,D[128]
    size_t smem_dkdv = (size_t)(2*64*128 + 2*128*128)*2 + (size_t)64*128*4 + (size_t)2*64*128*2 + (size_t)2*128*4;
    // dq smem: K,V[128,128] + Q,dO[64,128] halves ; Sbuf[64,128] f32 ; P,dS[64,128] half ; L,D[64]
    size_t smem_dq   = (size_t)(2*128*128 + 2*64*128)*2 + (size_t)64*128*4 + (size_t)2*64*128*2 + (size_t)2*64*4;

    static bool attr_set = false;
    if(!attr_set){
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dkdv));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));
        attr_set = true;
    }

    dim3 grid_dkdv((S+63)/64, BH);
    dim3 grid_dq((S+63)/64, BH);

    bwd_dkdv_kernel<<<grid_dkdv, 128, smem_dkdv, stream>>>(Qp,Kp,Vp,dOp,Lp,Dp,dKp,dVp,S,scale);
    CUDA_CHECK(cudaGetLastError());

    bwd_dq_kernel<<<grid_dq, 128, smem_dq, stream>>>(Qp,Kp,Vp,dOp,Lp,Dp,dQp,S,scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Dp, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd