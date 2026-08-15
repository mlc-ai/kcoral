#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

namespace mha_d128 {

constexpr int D  = 128;
constexpr int BM = 128;   // query rows per block
constexpr int BN = 64;    // KV rows per tile
constexpr int NTHREADS = 256;
constexpr int UPR = D/8;  // int4 per row = 16

__device__ __forceinline__ void cp_async_16(void* dst, const void* src){
    unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" :: "r"(s), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }
template<int N> __device__ __forceinline__ void cp_async_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ __forceinline__ int imin(int a,int b){ return a<b?a:b; }

__device__ __forceinline__ void prefetch_kv(
    const __nv_bfloat16* Kbase, const __nv_bfloat16* Vbase,
    __nv_bfloat16* Kdst, __nv_bfloat16* Vdst,
    int kv_start, int valid, int tid) {
    const int total = BN*UPR;
    int4 z; z.x=z.y=z.z=z.w=0;
    for (int u=tid; u<total; u+=NTHREADS){
        int r=u/UPR, c=u%UPR;
        if (r<valid){
            const int4* ks = reinterpret_cast<const int4*>(Kbase+(int64_t)(kv_start+r)*D)+c;
            const int4* vs = reinterpret_cast<const int4*>(Vbase+(int64_t)(kv_start+r)*D)+c;
            cp_async_16(reinterpret_cast<int4*>(Kdst)+u, ks);
            cp_async_16(reinterpret_cast<int4*>(Vdst)+u, vs);
        } else {
            reinterpret_cast<int4*>(Kdst)[u]=z;
            reinterpret_cast<int4*>(Vdst)[u]=z;
        }
    }
}

__global__ __launch_bounds__(NTHREADS, 1) void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, float scale)
{
    extern __shared__ char smem_raw[];
    float*         Ost = reinterpret_cast<float*>(smem_raw);           // [D][BM]
    float*         Sst = Ost + D*BM;                                   // [BN][BM]
    __nv_bfloat16* Qs  = reinterpret_cast<__nv_bfloat16*>(Sst + BN*BM);// [BM][D]
    __nv_bfloat16* Ks  = Qs + BM*D;                                    // [2][BN][D]
    __nv_bfloat16* Vs  = Ks + 2*BN*D;                                  // [2][BN][D]
    __nv_bfloat16* Pst = Vs + 2*BN*D;                                  // [BN][BM]
    float*         ms  = reinterpret_cast<float*>(Pst + BN*BM);        // [BM]
    float*         ls  = ms + BM;                                      // [BM]

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;         // 0..7
    const int wrow = warp*16;          // this warp's query band base
    const int bh = blockIdx.y;
    const int q_start = blockIdx.x * BM;

    const __nv_bfloat16* Qbase = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* Kbase = K + (int64_t)bh * S * D;
    const __nv_bfloat16* Vbase = V + (int64_t)bh * S * D;
    __nv_bfloat16* Obase = O + (int64_t)bh * S * D;
    float* LSEbase = LSE + (int64_t)bh * S;

    for (int i=tid;i<D*BM;i+=NTHREADS) Ost[i]=0.0f;
    for (int i=tid;i<BM;i+=NTHREADS){ ms[i]=-1e30f; ls[i]=0.0f; }

    // load Q
    {
        int4 z; z.x=z.y=z.z=z.w=0;
        for (int u=tid;u<BM*UPR;u+=NTHREADS){
            int r=u/UPR, c=u%UPR;
            int gq=q_start+r;
            reinterpret_cast<int4*>(Qs)[u] = (gq<S)
                ? reinterpret_cast<const int4*>(Qbase+(int64_t)gq*D)[c] : z;
        }
    }

    int num_kv = (S + BN - 1)/BN;
    // prefetch tile 0
    prefetch_kv(Kbase,Vbase, Ks, Vs, 0, imin(BN,S), tid);
    cp_async_commit();
    __syncthreads();  // Q ready

    for (int kt=0; kt<num_kv; ++kt){
        int cur = kt & 1;
        int valid = imin(BN, S - kt*BN);

        if (kt+1<num_kv){
            int nxt=(kt+1)&1;
            int vN=imin(BN, S-(kt+1)*BN);
            prefetch_kv(Kbase,Vbase, Ks+nxt*BN*D, Vs+nxt*BN*D, (kt+1)*BN, vN, tid);
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        __nv_bfloat16* Kc = Ks + cur*BN*D;
        __nv_bfloat16* Vc = Vs + cur*BN*D;

        // S = Q @ K^T  -> store transposed into Sst[key][query]
        #pragma unroll
        for (int n=0;n<BN/16;n++){
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::fill_fragment(c, 0.0f);
            #pragma unroll
            for (int k=0;k<D/16;k++){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
                wmma::load_matrix_sync(a, Qs + wrow*D + k*16, D);
                wmma::load_matrix_sync(b, Kc + (n*16)*D + k*16, D);
                wmma::mma_sync(c,a,b,c);
            }
            wmma::store_matrix_sync(Sst + (n*16)*BM + wrow, c, BM, wmma::mem_col_major);
        }
        __syncthreads();

        // softmax (conflict-free: thread i reads column i of transposed buffers)
        if (tid < BM){
            int i=tid;
            float mprev = ms[i];
            float tmax=-1e30f;
            for (int j=0;j<valid;j++) tmax=fmaxf(tmax, Sst[j*BM+i]*scale);
            float mnew=fmaxf(mprev,tmax);
            float alpha = (mnew>mprev) ? __expf(mprev-mnew) : 1.0f;
            float rs=0.0f;
            #pragma unroll
            for (int j=0;j<BN;j++){
                float p=0.0f;
                if (j<valid) p=__expf(Sst[j*BM+i]*scale - mnew);
                rs+=p;
                Pst[j*BM+i]=__float2bfloat16(p);
            }
            float lnew = ls[i]*alpha + rs;
            if (alpha!=1.0f){
                #pragma unroll
                for (int d=0;d<D;d++) Ost[d*BM+i]*=alpha;
            }
            ms[i]=mnew; ls[i]=lnew;
        }
        __syncthreads();

        // O += P @ V  (all transposed: Ost[d][query], Pst[key][query], Vc[key][d])
        #pragma unroll
        for (int nd=0;nd<D/16;nd++){
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::load_matrix_sync(c, Ost + (nd*16)*BM + wrow, BM, wmma::mem_col_major);
            #pragma unroll
            for (int kk=0;kk<BN/16;kk++){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
                wmma::load_matrix_sync(a, Pst + (kk*16)*BM + wrow, BM);
                wmma::load_matrix_sync(b, Vc + (kk*16)*D + nd*16, D);
                wmma::mma_sync(c,a,b,c);
            }
            wmma::store_matrix_sync(Ost + (nd*16)*BM + wrow, c, BM, wmma::mem_col_major);
        }
        __syncthreads();
    }

    // finalize
    if (tid < BM){
        int i=tid; int gq=q_start+i;
        if (gq<S){
            float l=ls[i];
            float inv = (l>0.0f) ? 1.0f/l : 0.0f;
            __nv_bfloat16* out = Obase + (int64_t)gq*D;
            #pragma unroll
            for (int d=0;d<D;d++) out[d]=__float2bfloat16(Ost[d*BM+i]*inv);
            LSEbase[gq]=ms[i]+logf(l);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int Bd = (int)Q.size(0);
    int Hd = (int)Q.size(1);
    int Sd = (int)Q.size(2);
    int Dd = (int)Q.size(3);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp = static_cast<float*>(LSE.data_ptr());

    float scale = 1.0f / sqrtf((float)Dd);

    int num_q_tiles = (Sd + BM - 1)/BM;
    dim3 grid(num_q_tiles, Bd*Hd);
    dim3 block(NTHREADS);

    size_t smem = (size_t)D*BM*4 + (size_t)BN*BM*4 + (size_t)BM*D*2
                + (size_t)2*BN*D*2 + (size_t)2*BN*D*2 + (size_t)BN*BM*2
                + (size_t)BM*4 + (size_t)BM*4;

    static bool attr_set=false;
    if(!attr_set){
        CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr_set=true;
    }

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, Lp, Sd, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}  // namespace mha_d128