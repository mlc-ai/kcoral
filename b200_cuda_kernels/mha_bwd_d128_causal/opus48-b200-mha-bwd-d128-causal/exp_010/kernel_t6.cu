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

__global__ void convert_dq_kernel(const float* __restrict__ dQf,
                                  __nv_bfloat16* __restrict__ dQ, int64_t n){
    int64_t i = (int64_t)blockIdx.x*256 + threadIdx.x;
    if(i<n) dQ[i]=__float2bfloat16(dQf[i]);
}

// ------------- Merged backward kernel (parallel over KV blocks), 8 warps -------------
__launch_bounds__(256,1)
__global__ void merged_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    float* __restrict__ dQf,
    int S, float scale)
{
    int bh = blockIdx.y;
    int kb = blockIdx.x;
    int n0 = kb*64;
    if(n0 >= S) return;
    int64_t base  = (int64_t)bh * S * 128;
    int64_t lbase = (int64_t)bh * S;

    extern __shared__ char smem[];
    half*  Ksh  = (half*)smem;
    half*  Vsh  = Ksh  + 64*128;
    half*  Qsh  = Vsh  + 64*128;
    half*  dOsh = Qsh  + 64*128;
    float* Sbuf = (float*)(dOsh + 64*128);
    half*  PTsh = (half*)(Sbuf + 64*64);
    half*  dSTsh= PTsh + 64*64;
    half*  dSsh = dSTsh + 64*64;
    float* Lsh  = (float*)(dSsh + 64*64);
    float* Drow = Lsh + 64;
    float* dQstg= Drow + 64;         // 64*128 floats
    float* Escr = Sbuf;              // reuse for dV/dK epilogue

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int wq = warp >> 1;   // row band 0..3 (16 rows)
    int wh = warp & 1;    // d-half / q-half

    // load K,V (fixed for this block)
    for(int i=tid;i<64*128;i+=256){
        int r=i>>7, c=i&127; int gr=n0+r;
        if(gr<S){ Ksh[i]=b2h(K[base+(int64_t)gr*128+c]); Vsh[i]=b2h(V[base+(int64_t)gr*128+c]); }
        else    { Ksh[i]=__float2half(0.f);              Vsh[i]=__float2half(0.f); }
    }

    FragC dV_frag[4], dK_frag[4];
    #pragma unroll
    for(int t=0;t<4;t++){ wmma::fill_fragment(dV_frag[t],0.f); wmma::fill_fragment(dK_frag[t],0.f); }

    int num_qb = (S+63)/64;
    for(int qb=kb; qb<num_qb; qb++){
        int m0=qb*64;
        __syncthreads();
        for(int i=tid;i<64*128;i+=256){
            int r=i>>7,c=i&127; int gr=m0+r;
            if(gr<S){ Qsh[i]=b2h(Q[base+(int64_t)gr*128+c]); dOsh[i]=b2h(dO[base+(int64_t)gr*128+c]); }
            else    { Qsh[i]=__float2half(0.f);              dOsh[i]=__float2half(0.f); }
        }
        for(int i=tid;i<64;i+=256){
            int gr=m0+i;
            Lsh[i]= gr<S? L[lbase+gr]:0.f;
            Drow[i]= gr<S? D[lbase+gr]:0.f;
        }
        __syncthreads();

        // S^T = K@Q^T  (A=K row, B=Q col)
        #pragma unroll
        for(int c=0;c<2;c++){
            int jt=wh*2+c;
            FragC accS; wmma::fill_fragment(accS,0.f);
            #pragma unroll
            for(int kk=0;kk<8;kk++){
                FragA aK; wmma::load_matrix_sync(aK, &Ksh[wq*16*128 + kk*16], 128);
                FragBcol bQ; wmma::load_matrix_sync(bQ, &Qsh[jt*16*128 + kk*16], 128);
                wmma::mma_sync(accS,aK,bQ,accS);
            }
            wmma::store_matrix_sync(&Sbuf[wq*16*64 + jt*16], accS, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // P^T = exp(scale*S^T - L[q]) masked
        for(int i=tid;i<64*64;i+=256){
            int key=i>>6, q=i&63;
            int gk=n0+key, gq=m0+q;
            bool valid = (gk<S)&&(gq<S)&&(gk<=gq);
            float p = valid ? __expf(scale*Sbuf[i] - Lsh[q]) : 0.f;
            PTsh[i]=__float2half(p);
        }
        __syncthreads();

        // dP^T = V@dO^T (reuse Sbuf)
        #pragma unroll
        for(int c=0;c<2;c++){
            int jt=wh*2+c;
            FragC accP; wmma::fill_fragment(accP,0.f);
            #pragma unroll
            for(int kk=0;kk<8;kk++){
                FragA aV; wmma::load_matrix_sync(aV, &Vsh[wq*16*128 + kk*16], 128);
                FragBcol bO; wmma::load_matrix_sync(bO, &dOsh[jt*16*128 + kk*16], 128);
                wmma::mma_sync(accP,aV,bO,accP);
            }
            wmma::store_matrix_sync(&Sbuf[wq*16*64 + jt*16], accP, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // dS^T = scale*P^T*(dP^T - D[q]) ; store both transposed and non-transposed
        for(int i=tid;i<64*64;i+=256){
            int key=i>>6, q=i&63;
            float p=__half2float(PTsh[i]);
            float ds=scale*p*(Sbuf[i] - Drow[q]);
            half h=__float2half(ds);
            dSTsh[i]=h;                 // [key,q]
            dSsh[q*64+key]=h;           // [q,key]
        }
        __syncthreads();

        // dV += P^T@dO ; dK += dS^T@Q
        #pragma unroll
        for(int kk=0;kk<4;kk++){
            FragA aP; wmma::load_matrix_sync(aP, &PTsh[wq*16*64 + kk*16], 64);
            FragA aD; wmma::load_matrix_sync(aD, &dSTsh[wq*16*64 + kk*16], 64);
            #pragma unroll
            for(int dt=0;dt<4;dt++){
                int dcol=wh*64 + dt*16;
                FragBrow bO; wmma::load_matrix_sync(bO, &dOsh[kk*16*128 + dcol], 128);
                FragBrow bQ; wmma::load_matrix_sync(bQ, &Qsh[kk*16*128 + dcol], 128);
                wmma::mma_sync(dV_frag[dt], aP, bO, dV_frag[dt]);
                wmma::mma_sync(dK_frag[dt], aD, bQ, dK_frag[dt]);
            }
        }
        // dQ = dS@K -> dQstg  (single accumulator per dt to keep regs low)
        #pragma unroll
        for(int dt=0;dt<4;dt++){
            int dcol=wh*64 + dt*16;
            FragC acc; wmma::fill_fragment(acc,0.f);
            #pragma unroll
            for(int kk=0;kk<4;kk++){
                FragA aS; wmma::load_matrix_sync(aS, &dSsh[wq*16*64 + kk*16], 64);
                FragBrow bK; wmma::load_matrix_sync(bK, &Ksh[kk*16*128 + dcol], 128);
                wmma::mma_sync(acc, aS, bK, acc);
            }
            wmma::store_matrix_sync(&dQstg[(wq*16)*128 + dcol], acc, 128, wmma::mem_row_major);
        }
        __syncthreads();

        // atomicAdd dQ contribution
        for(int i=tid;i<64*128;i+=256){
            int q=i>>7, dd=i&127; int gq=m0+q;
            if(gq<S) atomicAdd(&dQf[base + (int64_t)gq*128 + dd], dQstg[i]);
        }
    }

    __syncthreads();
    // epilogue dV, dK (this CTA fully owns key-block -> direct write)
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
    int64_t total_elem = total_rows * 128;
    float* Dp = nullptr;
    float* dQf = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&Dp, total_rows*sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&dQf, total_elem*sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQf, 0, total_elem*sizeof(float), stream));

    compute_D_kernel<<<(unsigned int)total_rows, 128, 0, stream>>>(Op, dOp, Dp, total_rows);
    CUDA_CHECK(cudaGetLastError());

    size_t smem = (size_t)4*64*128*2 + (size_t)64*64*4 + (size_t)3*64*64*2
                  + (size_t)2*64*4 + (size_t)64*128*4;

    static bool attr_set = false;
    if(!attr_set){
        CUDA_CHECK(cudaFuncSetAttribute(merged_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr_set = true;
    }

    int num_blocks = (S+63)/64;
    dim3 grid(num_blocks, BH);

    merged_bwd_kernel<<<grid, 256, smem, stream>>>(Qp,Kp,Vp,dOp,Lp,Dp,dKp,dVp,dQf,S,scale);
    CUDA_CHECK(cudaGetLastError());

    int64_t cblocks = (total_elem + 255)/256;
    convert_dq_kernel<<<(unsigned int)cblocks, 256, 0, stream>>>(dQf, dQp, total_elem);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Dp, stream));
    CUDA_CHECK(cudaFreeAsync(dQf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd