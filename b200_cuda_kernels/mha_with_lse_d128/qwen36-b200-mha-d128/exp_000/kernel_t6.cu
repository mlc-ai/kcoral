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
    if (_e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d\n", __FILE__, __LINE__); exit(1); } } while(0)

namespace mha_impl {

template<int BM, int BN, int TD>
__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q, 
    const __nv_bfloat16* __restrict__ K, 
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O, 
    float* __restrict__ LSE,
    int B, int H, int S,
    int str_h, int str_s) 
{
    constexpr int NT = 256;
    constexpr int ELEMS = (BM * TD) / NT; 
    
    int bh_id = blockIdx.x;
    int tile_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    int b = bh_id / H;
    int h = bh_id % H;
    
    int base_offset = b * str_h + h * str_h;
    const __nv_bfloat16* q_ptr = Q + base_offset;
    const __nv_bfloat16* k_ptr = K + base_offset;
    const __nv_bfloat16* v_ptr = V + base_offset;
    __nv_bfloat16* o_ptr = O + base_offset;
    float* lse_ptr = LSE + b * H * S + h * S;
    
    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + BM * TD;
    __nv_bfloat16* sV = sK + BN * TD;
    
    int r_arr[ELEMS];
    int c_arr[ELEMS];
    for(int e=0; e<ELEMS; ++e) {
        int idx = tid * ELEMS + e;
        r_arr[e] = idx / TD;
        c_arr[e] = idx % TD;
    }
    
    float o_reg[ELEMS] = {};
    float m_reg[ELEMS] = {-FLT_MAX};
    float d_reg[ELEMS] = {0.0f};
    float scores[BN] = {};
    
    int m_start = tile_idx * BM;
    bool valid_m = m_start < S;
    
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(TD));
    int n_blocks = (S + BN - 1) / BN;
    
    if(valid_m) {
        // Cooperative Q load
        for(int i=tid; i<BM*TD; i+=NT) {
            int r = i / TD;
            int c = i % TD;
            sQ[i] = q_ptr[(m_start+r)*str_s + c];
        }
        __syncthreads();
        
        for(int nb=0; nb<n_blocks; ++nb) {
            int n_st = nb * BN;
            
            // Cooperative K, V load
            for(int i=tid; i<BN*TD; i+=NT) {
                int n = i / TD;
                int c = i % TD;
                bool v = (n_st+n < S);
                sK[i] = v ? k_ptr[(n_st+n)*str_s + c] : __float2bfloat16(0.0f);
                sV[i] = v ? v_ptr[(n_st+n)*str_s + c] : __float2bfloat16(0.0f);
            }
            __syncthreads();
            
            #pragma unroll
            for(int e=0; e<ELEMS; ++e) {
                int r = r_arr[e];
                int c = c_arr[e];
                
                float rmax = -FLT_MAX;
                const __nv_bfloat16* qr = sQ + r*TD;
                for(int n=0; n<BN; ++n) {
                    float s = 0.0f;
                    const __nv_bfloat16* kr = sK + n*TD;
                    #pragma unroll
                    for(int di=0; di<TD; di+=4) {
                        s += __bfloat162float(qr[di])   * __bfloat162float(kr[di]);
                        s += __bfloat162float(qr[di+1]) * __bfloat162float(kr[di+1]);
                        s += __bfloat162float(qr[di+2]) * __bfloat162float(kr[di+2]);
                        s += __bfloat162float(qr[di+3]) * __bfloat162float(kr[di+3]);
                    }
                    float sc = (n_st+n < S) ? (s * inv_sqrt_d) : (-FLT_MAX);
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
                
                for(int n=0; n<BN; ++n) {
                    float pn = expf(scores[n] - nm);
                    ns += pn;
                    no += pn * __bfloat162float(sV[n*TD + c]);
                }
                m_reg[e] = nm;
                d_reg[e] = pd * alpha + ns;
                o_reg[e] = no;
            }
            __syncthreads();
        }
        
        #pragma unroll
        for(int e=0; e<ELEMS; ++e) {
            int r = r_arr[e];
            int c = c_arr[e];
            int gr = m_start + r;
            if(gr >= S) continue;
            
            float ld = d_reg[e];
            if(ld <= 0.0f) {
                o_ptr[gr*str_s + c] = __float2bfloat16(0.0f);
                if(c == 0) lse_ptr[gr] = -FLT_MAX;
            } else {
                float inv = 1.0f / ld;
                o_ptr[gr*str_s + c] = __float2bfloat16(o_reg[e] * inv);
                if(c == 0) lse_ptr[gr] = m_reg[e] + logf(ld);
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
    
    constexpr int BM = 64, BN = 64, TD = 128;
    
    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(256);
    size_t smem_bytes = (BM + 2 * BN) * TD * sizeof(__nv_bfloat16);
    
    int str_s = static_cast<int>(D);
    int str_h = static_cast<int>(S * D);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<BM, BN, TD><<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S,
        str_h, str_s
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);

} // namespace mha_impl