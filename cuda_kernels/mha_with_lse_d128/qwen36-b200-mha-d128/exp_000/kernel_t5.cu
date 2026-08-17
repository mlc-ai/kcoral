#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <float.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_impl {

template<int BM, int BN, int D>
__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q, 
    const __nv_bfloat16* __restrict__ K, 
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O, 
    float* __restrict__ LSE,
    int B, int H, int S,
    int stride_tensor, int stride_lse)
{
    constexpr int NT = 256;
    constexpr int ELEMS = (BM * D) / NT;
    
    uint32_t bh_id = blockIdx.x;
    uint32_t bm_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;
    
    int b = bh_id / H;
    int h = bh_id % H;
    
    const __nv_bfloat16* Q_off = Q + b * stride_tensor + h * (stride_tensor / D);
    const __nv_bfloat16* K_off = K + b * stride_tensor + h * (stride_tensor / D);
    const __nv_bfloat16* V_off = V + b * stride_tensor + h * (stride_tensor / D);
    __nv_bfloat16* O_off = O + b * stride_tensor + h * (stride_tensor / D);
    float* LSE_off = LSE + b * stride_lse + h * (stride_lse / D);
    
    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + BM * D;
    __nv_bfloat16* sV = sK + BN * D;
    
    int r_arr[ELEMS], c_arr[ELEMS];
    for(int e=0; e<ELEMS; ++e) {
        int idx = tid * ELEMS + e;
        r_arr[e] = idx / D;
        c_arr[e] = idx % D;
    }
    
    float o_reg[ELEMS] = {};
    float m_reg[ELEMS] = {-FLT_MAX};
    float d_reg[ELEMS] = {0.0f};
    float scores[BN] = {};
    
    float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    int m_start = bm_idx * BM;
    bool m_valid = m_start < S;
    
    if(m_valid) {
        // Cooperative Q load
        for(int i=tid; i<BM*D; i+=NT) {
            int r = i / D;
            int c = i % D;
            sQ[i] = Q_off[(m_start+r)*D + c];
        }
        __syncthreads();
        
        const int nblocks = (S + BN - 1) / BN;
        for(int nb=0; nb<nblocks; ++nb) {
            int n_st = nb * BN;
            
            // Cooperative K & V load
            for(int i=tid; i<BN*D; i+=NT) {
                int n = i / D;
                int c = i % D;
                bool v = (n_st+n < S);
                sK[i] = v ? K_off[(n_st+n)*D + c] : __float2bfloat16(0.0f);
                sV[i] = v ? V_off[(n_st+n)*D + c] : __float2bfloat16(0.0f);
            }
            __syncthreads();
            
            #pragma unroll
            for(int e=0; e<ELEMS; ++e) {
                int r = r_arr[e];
                int c = c_arr[e];
                
                float rmax = -FLT_MAX;
                const __nv_bfloat16* q_r = sQ + r*D;
                #pragma unroll
                for(int n=0; n<BN; ++n) {
                    float p = 0.0f;
                    const __nv_bfloat16* k_r = sK + n*D;
                    #pragma unroll
                    for(int di=0; di<D; di+=4) {
                        p += __bfloat162float(q_r[di])   * __bfloat162float(k_r[di]);
                        p += __bfloat162float(q_r[di+1]) * __bfloat162float(k_r[di+1]);
                        p += __bfloat162float(q_r[di+2]) * __bfloat162float(k_r[di+2]);
                        p += __bfloat162float(q_r[di+3]) * __bfloat162float(k_r[di+3]);
                    }
                    float sc = (n_st+n < S) ? (p * inv_sqrt_d) : (-FLT_MAX);
                    scores[n] = sc;
                    if(sc > rmax) rmax = sc;
                }
                
                float pm = m_reg[e];
                float pd = d_reg[e];
                float po = o_reg[e];
                float nm = max(pm, rmax);
                float alpha = expf(pm - nm);
                float ns = 0.0f;
                float no = po * alpha;
                
                #pragma unroll
                for(int n=0; n<BN; ++n) {
                    float pn = expf(scores[n] - nm);
                    ns += pn;
                    no += pn * __bfloat162float(sV[n*D + c]);
                }
                m_reg[e] = nm;
                d_reg[e] = pd * alpha + ns;
                o_reg[e] = no;
            }
            __syncthreads();
        }
        
        // Epilogue: Normalize and store
        #pragma unroll
        for(int e=0; e<ELEMS; ++e) {
            int r = r_arr[e];
            int c = c_arr[e];
            int gr = m_start + r;
            if(gr >= S) continue;
            
            float ld = d_reg[e];
            if(ld <= 0.0f) {
                O_off[gr*D + c] = __float2bfloat16(0.0f);
                if(c == 0) LSE_off[gr] = -FLT_MAX;
            } else {
                float inv = 1.0f / ld;
                O_off[gr*D + c] = __float2bfloat16(o_reg[e] * inv);
                if(c == 0) LSE_off[gr] = m_reg[e] + logf(ld);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    constexpr int BM = 64, BN = 64;
    
    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(256);
    size_t smem = (BM + 2 * BN) * D * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int stride_tensor = static_cast<int>(S * D);
    int stride_lse = static_cast<int>(S);
    
    mha_kernel<BM, BN, D><<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S,
        stride_tensor, stride_lse
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);

} // namespace mha_impl