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

typedef wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> FragA;
typedef wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> FragBrow;
typedef wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::col_major> FragBcol;
typedef wmma::fragment<wmma::accumulator, 16,16,16, float> FragC;

__device__ __forceinline__ void split_bf16(float x, __nv_bfloat16 &hi, __nv_bfloat16 &lo){
    hi = __float2bfloat16(x);
    float h = __bfloat162float(hi);
    lo = __float2bfloat16(x - h);
}

// load a [64,128] bf16 tile (rows row0..row0+63) into smem, vectorized uint4
__device__ __forceinline__ void load_tile(__nv_bfloat16* dst, const __nv_bfloat16* base,
                                           int row0, int S, int tid){
    #pragma unroll
    for(int j=tid;j<1024;j+=256){
        int gr=j>>4, cb=j&15;
        int row=row0+gr;
        uint4 v;
        if(row<S){ v = *reinterpret_cast<const uint4*>(base + (int64_t)row*128 + cb*8); }
        else     { v = make_uint4(0,0,0,0); }
        reinterpret_cast<uint4*>(dst)[gr*16 + cb] = v;
    }
}

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

// ------------- dK/dV kernel (parallel over KV blocks), 8 warps -------------
__launch_bounds__(256,2)
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
    __nv_bfloat16* Ksh  = (__nv_bfloat16*)smem;
    __nv_bfloat16* Vsh  = Ksh  + 64*128;
    __nv_bfloat16* Qsh  = Vsh  + 64*128;
    __nv_bfloat16* dOsh = Qsh  + 64*128;
    float* Sbuf = (float*)(dOsh + 64*128);
    __nv_bfloat16* Ph = (__nv_bfloat16*)(Sbuf + 64*64);
    __nv_bfloat16* Pl = Ph + 64*64;
    __nv_bfloat16* Dh = Pl + 64*64;
    __nv_bfloat16* Dl = Dh + 64*64;
    float* Lsh = (float*)(Dl + 64*64);
    float* Drow = Lsh + 64;
    float* Escr = (float*)Ksh;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int wq = warp >> 1;   // key band 0..3 (16 keys)
    int wh = warp & 1;    // d-half / q-half

    load_tile(Ksh, K+base, n0, S, tid);
    load_tile(Vsh, V+base, n0, S, tid);

    FragC dV_frag[4], dK_frag[4];
    #pragma unroll
    for(int t=0;t<4;t++){ wmma::fill_fragment(dV_frag[t],0.f); wmma::fill_fragment(dK_frag[t],0.f); }

    int num_qb = (S+63)/64;
    for(int qb=kb; qb<num_qb; qb++){
        int m0=qb*64;
        __syncthreads();
        load_tile(Qsh,  Q+base,  m0, S, tid);
        load_tile(dOsh, dO+base, m0, S, tid);
        for(int i=tid;i<64;i+=256){
            int gr=m0+i;
            Lsh[i]= gr<S? L[lbase+gr]:0.f;
            Drow[i]= gr<S? D[lbase+gr]:0.f;
        }
        __syncthreads();

        // S^T = K@Q^T
        {
            FragC accS0,accS1; wmma::fill_fragment(accS0,0.f); wmma::fill_fragment(accS1,0.f);
            #pragma unroll
            for(int kk=0;kk<8;kk++){
                FragA aK; wmma::load_matrix_sync(aK, &Ksh[wq*16*128 + kk*16], 128);
                FragBcol bQ0,bQ1;
                wmma::load_matrix_sync(bQ0, &Qsh[(wh*2+0)*16*128 + kk*16], 128);
                wmma::load_matrix_sync(bQ1, &Qsh[(wh*2+1)*16*128 + kk*16], 128);
                wmma::mma_sync(accS0,aK,bQ0,accS0);
                wmma::mma_sync(accS1,aK,bQ1,accS1);
            }
            wmma::store_matrix_sync(&Sbuf[wq*16*64 + (wh*2+0)*16], accS0, 64, wmma::mem_row_major);
            wmma::store_matrix_sync(&Sbuf[wq*16*64 + (wh*2+1)*16], accS1, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // P^T = exp(scale*S^T - L[q]) masked ; split hi/lo
        for(int i=tid;i<64*64;i+=256){
            int key=i>>6, q=i&63;
            int gk=n0+key, gq=m0+q;
            bool valid = (gk<S)&&(gq<S)&&(gk<=gq);
            float p = valid ? __expf(scale*Sbuf[i] - Lsh[q]) : 0.f;
            __nv_bfloat16 hi,lo; split_bf16(p,hi,lo);
            Ph[i]=hi; Pl[i]=lo;
        }
        __syncthreads();

        // dP^T = V@dO^T  (reuse Sbuf)
        {
            FragC accP0,accP1; wmma::fill_fragment(accP0,0.f); wmma::fill_fragment(accP1,0.f);
            #pragma unroll
            for(int kk=0;kk<8;kk++){
                FragA aV; wmma::load_matrix_sync(aV, &Vsh[wq*16*128 + kk*16], 128);
                FragBcol bO0,bO1;
                wmma::load_matrix_sync(bO0, &dOsh[(wh*2+0)*16*128 + kk*16], 128);
                wmma::load_matrix_sync(bO1, &dOsh[(wh*2+1)*16*128 + kk*16], 128);
                wmma::mma_sync(accP0,aV,bO0,accP0);
                wmma::mma_sync(accP1,aV,bO1,accP1);
            }
            wmma::store_matrix_sync(&Sbuf[wq*16*64 + (wh*2+0)*16], accP0, 64, wmma::mem_row_major);
            wmma::store_matrix_sync(&Sbuf[wq*16*64 + (wh*2+1)*16], accP1, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // dS^T = scale*P^T*(dP^T - D[q]) ; split hi/lo
        for(int i=tid;i<64*64;i+=256){
            int q=i&63;
            float p=__bfloat162float(Ph[i]) + __bfloat162float(Pl[i]);
            float ds=scale*p*(Sbuf[i] - Drow[q]);
            __nv_bfloat16 hi,lo; split_bf16(ds,hi,lo);
            Dh[i]=hi; Dl[i]=lo;
        }
        __syncthreads();

        // dV += P^T@dO ; dK += dS^T@Q
        #pragma unroll
        for(int kk=0;kk<4;kk++){
            FragA aPh,aPl,aDh,aDl;
            wmma::load_matrix_sync(aPh, &Ph[wq*16*64 + kk*16], 64);
            wmma::load_matrix_sync(aPl, &Pl[wq*16*64 + kk*16], 64);
            wmma::load_matrix_sync(aDh, &Dh[wq*16*64 + kk*16], 64);
            wmma::load_matrix_sync(aDl, &Dl[wq*16*64 + kk*16], 64);
            #pragma unroll
            for(int dt=0;dt<4;dt++){
                int dcol=wh*64 + dt*16;
                FragBrow bO,bQ;
                wmma::load_matrix_sync(bO, &dOsh[kk*16*128 + dcol], 128);
                wmma::load_matrix_sync(bQ, &Qsh[kk*16*128 + dcol], 128);
                wmma::mma_sync(dV_frag[dt], aPh, bO, dV_frag[dt]);
                wmma::mma_sync(dV_frag[dt], aPl, bO, dV_frag[dt]);
                wmma::mma_sync(dK_frag[dt], aDh, bQ, dK_frag[dt]);
                wmma::mma_sync(dK_frag[dt], aDl, bQ, dK_frag[dt]);
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for(int dt=0;dt<4;dt++){
        wmma::store_matrix_sync(&Escr[warp*256], dV_frag[dt], 16, wmma::mem_row_major);
        __syncwarp();
        for(int e=lane;e<256;e+=32){
            int kr=e>>4, dc=e&15;
            int gk=n0 + wq*16 + kr;
            int gd=wh*64 + dt*16 + dc;
            if(gk<S) dV[base+(int64_t)gk*128 + gd] = __float2bfloat16(Escr[warp*256 + e]);
        }
        __syncwarp();
        wmma::store_matrix_sync(&Escr[warp*256], dK_frag[dt], 16, wmma::mem_row_major);
        __syncwarp();
        for(int e=lane;e<256;e+=32){
            int kr=e>>4, dc=e&15;
            int gk=n0 + wq*16 + kr;
            int gd=wh*64 + dt*16 + dc;
            if(gk<S) dK[base+(int64_t)gk*128 + gd] = __float2bfloat16(Escr[warp*256 + e]);
        }
        __syncwarp();
    }
}

// ------------- dQ kernel (parallel over Q blocks), 8 warps -------------
__launch_bounds__(256,2)
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
    __nv_bfloat16* Ksh  = (__nv_bfloat16*)smem;
    __nv_bfloat16* Vsh  = Ksh  + 64*128;
    __nv_bfloat16* Qsh  = Vsh  + 64*128;
    __nv_bfloat16* dOsh = Qsh  + 64*128;
    float* Sbuf = (float*)(dOsh + 64*128);
    __nv_bfloat16* Ph = (__nv_bfloat16*)(Sbuf + 64*64);
    __nv_bfloat16* Pl = Ph + 64*64;
    __nv_bfloat16* Dh = Pl + 64*64;
    __nv_bfloat16* Dl = Dh + 64*64;
    float* Lsh = (float*)(Dl + 64*64);
    float* Drow = Lsh + 64;
    float* Escr = (float*)Ksh;

    int tid=threadIdx.x;
    int warp=tid>>5;
    int lane=tid&31;
    int wq=warp>>1;
    int wh=warp&1;

    load_tile(Qsh,  Q+base,  m0, S, tid);
    load_tile(dOsh, dO+base, m0, S, tid);
    for(int i=tid;i<64;i+=256){
        int gr=m0+i;
        Lsh[i]= gr<S? L[lbase+gr]:0.f;
        Drow[i]= gr<S? D[lbase+gr]:0.f;
    }

    FragC dQ_frag[4];
    #pragma unroll
    for(int t=0;t<4;t++) wmma::fill_fragment(dQ_frag[t],0.f);

    for(int kb=0; kb<=qb; kb++){
        int n0=kb*64;
        __syncthreads();
        load_tile(Ksh, K+base, n0, S, tid);
        load_tile(Vsh, V+base, n0, S, tid);
        __syncthreads();

        // S = Q@K^T
        {
            FragC accS0,accS1; wmma::fill_fragment(accS0,0.f); wmma::fill_fragment(accS1,0.f);
            #pragma unroll
            for(int kk=0;kk<8;kk++){
                FragA aQ; wmma::load_matrix_sync(aQ, &Qsh[wq*16*128 + kk*16], 128);
                FragBcol bK0,bK1;
                wmma::load_matrix_sync(bK0, &Ksh[(wh*2+0)*16*128 + kk*16], 128);
                wmma::load_matrix_sync(bK1, &Ksh[(wh*2+1)*16*128 + kk*16], 128);
                wmma::mma_sync(accS0,aQ,bK0,accS0);
                wmma::mma_sync(accS1,aQ,bK1,accS1);
            }
            wmma::store_matrix_sync(&Sbuf[wq*16*64 + (wh*2+0)*16], accS0, 64, wmma::mem_row_major);
            wmma::store_matrix_sync(&Sbuf[wq*16*64 + (wh*2+1)*16], accS1, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(scale*S - L[q]) masked
        for(int i=tid;i<64*64;i+=256){
            int q=i>>6, key=i&63;
            int gq=m0+q, gk=n0+key;
            bool valid=(gk<S)&&(gq<S)&&(gk<=gq);
            float p=valid? __expf(scale*Sbuf[i] - Lsh[q]):0.f;
            __nv_bfloat16 hi,lo; split_bf16(p,hi,lo);
            Ph[i]=hi; Pl[i]=lo;
        }
        __syncthreads();

        // dP = dO@V^T
        {
            FragC accP0,accP1; wmma::fill_fragment(accP0,0.f); wmma::fill_fragment(accP1,0.f);
            #pragma unroll
            for(int kk=0;kk<8;kk++){
                FragA aO; wmma::load_matrix_sync(aO, &dOsh[wq*16*128 + kk*16], 128);
                FragBcol bV0,bV1;
                wmma::load_matrix_sync(bV0, &Vsh[(wh*2+0)*16*128 + kk*16], 128);
                wmma::load_matrix_sync(bV1, &Vsh[(wh*2+1)*16*128 + kk*16], 128);
                wmma::mma_sync(accP0,aO,bV0,accP0);
                wmma::mma_sync(accP1,aO,bV1,accP1);
            }
            wmma::store_matrix_sync(&Sbuf[wq*16*64 + (wh*2+0)*16], accP0, 64, wmma::mem_row_major);
            wmma::store_matrix_sync(&Sbuf[wq*16*64 + (wh*2+1)*16], accP1, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = scale*P*(dP - D[q])
        for(int i=tid;i<64*64;i+=256){
            int q=i>>6;
            float p=__bfloat162float(Ph[i]) + __bfloat162float(Pl[i]);
            float ds=scale*p*(Sbuf[i] - Drow[q]);
            __nv_bfloat16 hi,lo; split_bf16(ds,hi,lo);
            Dh[i]=hi; Dl[i]=lo;
        }
        __syncthreads();

        // dQ += dS@K
        #pragma unroll
        for(int kk=0;kk<4;kk++){
            FragA aDh,aDl;
            wmma::load_matrix_sync(aDh, &Dh[wq*16*64 + kk*16], 64);
            wmma::load_matrix_sync(aDl, &Dl[wq*16*64 + kk*16], 64);
            #pragma unroll
            for(int dt=0;dt<4;dt++){
                int dcol=wh*64 + dt*16;
                FragBrow bK;
                wmma::load_matrix_sync(bK, &Ksh[kk*16*128 + dcol], 128);
                wmma::mma_sync(dQ_frag[dt], aDh, bK, dQ_frag[dt]);
                wmma::mma_sync(dQ_frag[dt], aDl, bK, dQ_frag[dt]);
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for(int dt=0;dt<4;dt++){
        wmma::store_matrix_sync(&Escr[warp*256], dQ_frag[dt], 16, wmma::mem_row_major);
        __syncwarp();
        for(int e=lane;e<256;e+=32){
            int qr=e>>4, dc=e&15;
            int gq=m0 + wq*16 + qr;
            int gd=wh*64 + dt*16 + dc;
            if(gq<S) dQ[base+(int64_t)gq*128 + gd]=__float2bfloat16(Escr[warp*256 + e]);
        }
        __syncwarp();
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

    // smem: 4*64*128 bf16 + 64*64 float + 4*(64*64) bf16 + 2*64 float
    size_t smem = (size_t)4*64*128*2 + (size_t)64*64*4 + (size_t)4*64*64*2 + (size_t)2*64*4;

    static bool attr_set = false;
    if(!attr_set){
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr_set = true;
    }

    int num_blocks = (S+63)/64;
    dim3 grid(num_blocks, BH);

    bwd_dkdv_kernel<<<grid, 256, smem, stream>>>(Qp,Kp,Vp,dOp,Lp,Dp,dKp,dVp,S,scale);
    CUDA_CHECK(cudaGetLastError());

    bwd_dq_kernel<<<grid, 256, smem, stream>>>(Qp,Kp,Vp,dOp,Lp,Dp,dQp,S,scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Dp, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd