#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA err %s %s:%d\n",                        \
                cudaGetErrorString(_e), __FILE__, __LINE__);          \
        exit(1);                                                      \
    }                                                                 \
} while(0)

namespace mha_bwd_impl {

__device__ __forceinline__ float bf16to(float x) {
    return *(float*)&((uint32_t)x<<16);
}
__device__ __forceinline__ uint16_t bf16of(float x) {
    return (uint16_t)(((uint32_t)x+0x7FFFu)>>16);
}

// dQ = dS @ K  (each thread: one dQ output element)
__global__ void GK(const float* dS, const uint16_t* K,
                   uint16_t* o, int B,int H,int S,int d) {
    uint64_t t = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    uint64_t tot = (uint64_t)B*H*S*d;
    if(t>=tot) return;
    int dp=t%d; t/=d;
    int qp=t%S; t/=S;
    int bh=t;
    uint64_t oS=(uint64_t)bh*S*S, oK=(uint64_t)bh*S*d;
    float a=0;
    for(int k=0;k<S;++k) a+=dS[oS+qp*S+k]*bf16to(K[oK+k*d+dp]);
    o[oK+qp*d+dp]=bf16of(a);
}

void run(tvm::ffi::TensorView Q,tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,tvm::ffi::TensorView O,
         tvm::ffi::TensorView dO,tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ,tvm::ffi::TensorView dK,
         tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t s=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(),0,(size_t)Q.nbytes,s));
    CUDA_CHECK(cudaMemsetAsync(dK.data_ptr(),0,(size_t)K.nbytes,s));
    CUDA_CHECK(cudaMemsetAsync(dV.data_ptr(),0,(size_t)V.nbytes,s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,mha_bwd_impl::run);