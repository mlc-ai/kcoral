#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <type_traits>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {

constexpr int D  = 128;
constexpr int BR = 64;   // Q tile
constexpr int BC = 64;   // KV tile

// ---------------- cooperative block GEMM using WMMA ----------------
// Computes C[MT][NT] = A[MT x KT] * B[KT x NT] (optionally accumulate into C)
// AL/BL are wmma::row_major or wmma::col_major.
// lda/ldb/ldc are physical leading dimensions (stride between rows for row_major,
// stride between columns for col_major).
template<int MT,int NT,int KT, typename AL, typename BL>
__device__ __forceinline__ void block_gemm(
    const bf16* A, int lda,
    const bf16* B, int ldb,
    float* C, int ldc,
    bool addToC, int warp_id, int num_warps)
{
    constexpr int MTIL = MT/16;
    constexpr int NTIL = NT/16;
    constexpr int NSUB = MTIL*NTIL;
    for(int s=warp_id; s<NSUB; s+=num_warps){
        int mt=s/NTIL, nt=s%NTIL;
        int m0=mt*16, n0=nt*16;
        wmma::fragment<wmma::accumulator,16,16,16,float> cf;
        if(addToC) wmma::load_matrix_sync(cf, C+m0*ldc+n0, ldc, wmma::mem_row_major);
        else       wmma::fill_fragment(cf, 0.0f);
        #pragma unroll
        for(int k0=0;k0<KT;k0+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,bf16,AL> af;
            wmma::fragment<wmma::matrix_b,16,16,16,bf16,BL> bfr;
            const bf16* ap;
            if constexpr (std::is_same<AL,wmma::row_major>::value) ap = A + m0*lda + k0;
            else                                                   ap = A + m0 + k0*lda;
            const bf16* bp;
            if constexpr (std::is_same<BL,wmma::row_major>::value) bp = B + k0*ldb + n0;
            else                                                   bp = B + k0 + n0*ldb;
            wmma::load_matrix_sync(af, ap, lda);
            wmma::load_matrix_sync(bfr, bp, ldb);
            wmma::mma_sync(cf, af, bfr, cf);
        }
        wmma::store_matrix_sync(C+m0*ldc+n0, cf, ldc, wmma::mem_row_major);
    }
}

// vectorized tile load [ROWS x D] from global (row-major) into shared, masking rows >= S
template<int ROWS>
__device__ __forceinline__ void load_tile(bf16* smem, const bf16* g, int row0, int S,
                                          int tid, int nthreads){
    constexpr int VEC = 8;
    constexpr int per_row = D/VEC;
    constexpr int total = ROWS*per_row;
    for(int idx=tid; idx<total; idx+=nthreads){
        int r = idx/per_row, cv = idx%per_row;
        int gr = row0 + r;
        int4 val;
        if(gr < S) val = *reinterpret_cast<const int4*>(g + (long)gr*D + cv*VEC);
        else       val = make_int4(0,0,0,0);
        *reinterpret_cast<int4*>(smem + r*D + cv*VEC) = val;
    }
}

// ---------------- compute D_i = sum_k O_ik * dO_ik ----------------
__global__ void compute_D_kernel(const bf16* O, const bf16* dO, float* Dg, int total){
    int warps_per_block = blockDim.x/32;
    int row = blockIdx.x*warps_per_block + (threadIdx.x/32);
    if(row >= total) return;
    int lane = threadIdx.x%32;
    const bf16* Op  = O  + (long)row*D;
    const bf16* dOp = dO + (long)row*D;
    float acc = 0.f;
    #pragma unroll
    for(int k=lane; k<D; k+=32){
        acc += __bfloat162float(Op[k]) * __bfloat162float(dOp[k]);
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if(lane==0) Dg[row] = acc;
}

// ---------------- Kernel 1: dK, dV ----------------
__global__ void bwd_kv_kernel(
    const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
    const float* Lg, const float* Dg,
    bf16* dK, bf16* dV,
    int S, float scale)
{
    int bh = blockIdx.x;
    int kv_tile = blockIdx.y;
    int kvrow0 = kv_tile*BC;
    if(kvrow0 >= S) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int warp_id = tid/32;
    int num_warps = nthreads/32;

    long base = (long)bh * S * D;
    const bf16* Qb  = Q  + base;
    const bf16* Kb  = K  + base;
    const bf16* Vb  = V  + base;
    const bf16* dOb = dO + base;
    const float* Lb = Lg + (long)bh*S;
    const float* Db = Dg + (long)bh*S;
    bf16* dKg = dK + base;
    bf16* dVg = dV + base;

    extern __shared__ char smem[];
    bf16* Qs   = (bf16*)smem;
    bf16* Ks   = Qs + BR*D;
    bf16* Vs   = Ks + BC*D;
    bf16* dOs  = Vs + BC*D;
    bf16* Pbf  = dOs + BR*D;
    bf16* dSbf = Pbf + BR*BC;
    float* buf0  = (float*)(dSbf + BR*BC);
    float* buf1  = buf0 + BR*BC;
    float* dVacc = buf1 + BR*BC;
    float* dKacc = dVacc + BC*D;
    float* Ls    = dKacc + BC*D;
    float* Ds    = Ls + BR;

    // load K,V tile
    load_tile<BC>(Ks, Kb, kvrow0, S, tid, nthreads);
    load_tile<BC>(Vs, Vb, kvrow0, S, tid, nthreads);
    // zero accumulators
    for(int idx=tid; idx<BC*D; idx+=nthreads){ dVacc[idx]=0.f; dKacc[idx]=0.f; }
    __syncthreads();

    int num_q = (S+BR-1)/BR;
    for(int qt=0; qt<num_q; ++qt){
        int qrow0 = qt*BR;
        load_tile<BR>(Qs,  Qb,  qrow0, S, tid, nthreads);
        load_tile<BR>(dOs, dOb, qrow0, S, tid, nthreads);
        for(int i=tid;i<BR;i+=nthreads){
            int gi=qrow0+i;
            Ls[i] = (gi<S)?Lb[gi]:0.f;
            Ds[i] = (gi<S)?Db[gi]:0.f;
        }
        __syncthreads();

        // S = Q @ K^T -> buf0
        block_gemm<BR,BC,D, wmma::row_major, wmma::col_major>(Qs, D, Ks, D, buf0, BC, false, warp_id, num_warps);
        __syncthreads();

        // P = exp(scale*S - L)
        for(int idx=tid; idx<BR*BC; idx+=nthreads){
            int i=idx/BC, j=idx%BC;
            int gi=qrow0+i, gj=kvrow0+j;
            float p;
            if(gi<S && gj<S) p = __expf(scale*buf0[idx] - Ls[i]);
            else p = 0.f;
            buf0[idx] = p;
            Pbf[idx]  = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T -> buf1
        block_gemm<BR,BC,D, wmma::row_major, wmma::col_major>(dOs, D, Vs, D, buf1, BC, false, warp_id, num_warps);
        __syncthreads();

        // dS = P*(dP - D)
        for(int idx=tid; idx<BR*BC; idx+=nthreads){
            int i=idx/BC, j=idx%BC;
            int gi=qrow0+i, gj=kvrow0+j;
            float ds;
            if(gi<S && gj<S) ds = buf0[idx]*(buf1[idx] - Ds[i]);
            else ds = 0.f;
            dSbf[idx] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dV += P^T @ dO
        block_gemm<BC,D,BR, wmma::col_major, wmma::row_major>(Pbf, BC, dOs, D, dVacc, D, true, warp_id, num_warps);
        __syncthreads();
        // dK += dS^T @ Q
        block_gemm<BC,D,BR, wmma::col_major, wmma::row_major>(dSbf, BC, Qs, D, dKacc, D, true, warp_id, num_warps);
        __syncthreads();
    }

    // write outputs
    for(int idx=tid; idx<BC*D; idx+=nthreads){
        int j=idx/D, k=idx%D;
        int gj=kvrow0+j;
        if(gj<S){
            dVg[(long)gj*D + k] = __float2bfloat16(dVacc[idx]);
            dKg[(long)gj*D + k] = __float2bfloat16(dKacc[idx]*scale);
        }
    }
}

// ---------------- Kernel 2: dQ ----------------
__global__ void bwd_q_kernel(
    const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
    const float* Lg, const float* Dg,
    bf16* dQ,
    int S, float scale)
{
    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int qrow0 = q_tile*BR;
    if(qrow0 >= S) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int warp_id = tid/32;
    int num_warps = nthreads/32;

    long base = (long)bh * S * D;
    const bf16* Qb  = Q  + base;
    const bf16* Kb  = K  + base;
    const bf16* Vb  = V  + base;
    const bf16* dOb = dO + base;
    const float* Lb = Lg + (long)bh*S;
    const float* Db = Dg + (long)bh*S;
    bf16* dQg = dQ + base;

    extern __shared__ char smem[];
    bf16* Qs   = (bf16*)smem;
    bf16* dOs  = Qs + BR*D;
    bf16* Ks   = dOs + BR*D;
    bf16* Vs   = Ks + BC*D;
    bf16* dSbf = Vs + BC*D;
    float* buf0  = (float*)(dSbf + BR*BC);
    float* buf1  = buf0 + BR*BC;
    float* dQacc = buf1 + BR*BC;
    float* Ls    = dQacc + BR*D;
    float* Ds    = Ls + BR;

    load_tile<BR>(Qs,  Qb,  qrow0, S, tid, nthreads);
    load_tile<BR>(dOs, dOb, qrow0, S, tid, nthreads);
    for(int i=tid;i<BR;i+=nthreads){
        int gi=qrow0+i;
        Ls[i] = (gi<S)?Lb[gi]:0.f;
        Ds[i] = (gi<S)?Db[gi]:0.f;
    }
    for(int idx=tid; idx<BR*D; idx+=nthreads) dQacc[idx]=0.f;
    __syncthreads();

    int num_kv = (S+BC-1)/BC;
    for(int kt=0; kt<num_kv; ++kt){
        int kvrow0 = kt*BC;
        load_tile<BC>(Ks, Kb, kvrow0, S, tid, nthreads);
        load_tile<BC>(Vs, Vb, kvrow0, S, tid, nthreads);
        __syncthreads();

        // S = Q @ K^T -> buf0
        block_gemm<BR,BC,D, wmma::row_major, wmma::col_major>(Qs, D, Ks, D, buf0, BC, false, warp_id, num_warps);
        __syncthreads();

        // P
        for(int idx=tid; idx<BR*BC; idx+=nthreads){
            int i=idx/BC, j=idx%BC;
            int gi=qrow0+i, gj=kvrow0+j;
            float p;
            if(gi<S && gj<S) p = __expf(scale*buf0[idx] - Ls[i]);
            else p = 0.f;
            buf0[idx] = p;
        }
        __syncthreads();

        // dP = dO @ V^T -> buf1
        block_gemm<BR,BC,D, wmma::row_major, wmma::col_major>(dOs, D, Vs, D, buf1, BC, false, warp_id, num_warps);
        __syncthreads();

        // dS = P*(dP - D)
        for(int idx=tid; idx<BR*BC; idx+=nthreads){
            int i=idx/BC, j=idx%BC;
            int gi=qrow0+i, gj=kvrow0+j;
            float ds;
            if(gi<S && gj<S) ds = buf0[idx]*(buf1[idx] - Ds[i]);
            else ds = 0.f;
            dSbf[idx] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dQ += dS @ K
        block_gemm<BR,D,BC, wmma::row_major, wmma::row_major>(dSbf, BC, Ks, D, dQacc, D, true, warp_id, num_warps);
        __syncthreads();
    }

    for(int idx=tid; idx<BR*D; idx+=nthreads){
        int i=idx/D, k=idx%D;
        int gi=qrow0+i;
        if(gi<S) dQg[(long)gi*D + k] = __float2bfloat16(dQacc[idx]*scale);
    }
}

// persistent scratch for D
static float* g_D = nullptr;
static size_t  g_Dsz = 0;

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int d = (int)Q.size(3);
    (void)d;
    int BH = B*H;
    float scale = 1.0f / sqrtf((float)D);

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

    // D scratch
    size_t need = (size_t)BH*S*sizeof(float);
    if(need > g_Dsz){
        if(g_D) cudaFree(g_D);
        CUDA_CHECK(cudaMalloc(&g_D, need));
        g_Dsz = need;
    }

    // compute D
    int total_rows = BH*S;
    int dblock = 128;
    int drows_per_block = dblock/32;
    int dgrid = (total_rows + drows_per_block - 1)/drows_per_block;
    compute_D_kernel<<<dgrid, dblock, 0, stream>>>(Op, dOp, g_D, total_rows);
    CUDA_CHECK(cudaGetLastError());

    // shared sizes
    size_t smem1 = (size_t)((BR*D + BC*D + BC*D + BR*D + BR*BC + BR*BC)) * sizeof(bf16)
                 + (size_t)((BR*BC + BR*BC + BC*D + BC*D + BR + BR)) * sizeof(float);
    size_t smem2 = (size_t)((BR*D + BR*D + BC*D + BC*D + BR*BC)) * sizeof(bf16)
                 + (size_t)((BR*BC + BR*BC + BR*D + BR + BR)) * sizeof(float);

    static bool attr_set = false;
    if(!attr_set){
        CUDA_CHECK(cudaFuncSetAttribute(bwd_kv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_q_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2));
        attr_set = true;
    }

    int block = 256;
    int num_kv = (S+BC-1)/BC;
    int num_q  = (S+BR-1)/BR;

    dim3 grid1(BH, num_kv);
    bwd_kv_kernel<<<grid1, block, smem1, stream>>>(Qp,Kp,Vp,dOp,Lp,g_D,dKp,dVp,S,scale);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid2(BH, num_q);
    bwd_q_kernel<<<grid2, block, smem2, stream>>>(Qp,Kp,Vp,dOp,Lp,g_D,dQp,S,scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd