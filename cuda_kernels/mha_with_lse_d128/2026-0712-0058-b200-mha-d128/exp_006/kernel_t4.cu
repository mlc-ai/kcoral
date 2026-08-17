#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n", (int)_e,         \
                __FILE__, __LINE__);                             \
        exit(1);                                                 \
    }                                                            \
} while(0)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_kernel {

__global__ __launch_bounds__(128) void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, float scale)
{
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int s_offset = blockIdx.x * 64;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    extern __shared__ uint8_t smem_buf[];
    __nv_bfloat16* Q_shared = (__nv_bfloat16*)smem_buf;
    __nv_bfloat16* K_shared = (__nv_bfloat16*)(smem_buf + 16384);
    __nv_bfloat16* V_shared = (__nv_bfloat16*)(smem_buf + 32768);
    __nv_bfloat16* P_shared = (__nv_bfloat16*)(smem_buf + 49152);
    __nv_bfloat16* O_shared = (__nv_bfloat16*)(smem_buf + 57344);
    __nv_bfloat16* S_shared = (__nv_bfloat16*)(smem_buf + 73728);
    
    size_t bh_off = (b_idx * 48 + h_idx);
    const __nv_bfloat16* Q_bh = Q + bh_off * S * 128;
    const __nv_bfloat16* K_bh = K + bh_off * S * 128;
    const __nv_bfloat16* V_bh = V + bh_off * S * 128;
    __nv_bfloat16* O_bh = O + bh_off * S * 128;
    float* LSE_bh = LSE + bh_off * S;
    
    // Load Q tile (64x128) into shared memory
    for (int i = 0; i < 16; ++i) {
        int idx = tid + i * 128;
        if (idx < 8192) {
            *(uint4*)&Q_shared[idx] = *(const uint4*)&Q_bh[idx];
        } else {
            *(uint4*)&Q_shared[idx] = make_uint4(0, 0, 0, 0);
        }
    }
    
    // Init O_shared to 0
    for (int i = 0; i < 16; ++i) {
        int idx = tid + i * 128;
        *(uint4*)&O_shared[idx] = make_uint4(0, 0, 0, 0);
    }
    
    __syncthreads(); 
    
    auto frag_Q0 = create_fragment<blackwell::matrix_a, 64, 0, 64, 128>(Q_shared);
    auto frag_Q1 = create_fragment<blackwell::matrix_a, 64, 64, 64, 128>(Q_shared + 4096);
    auto frag_K0 = create_fragment<blackwell::matrix_b, 128, 0, 64, 128>(K_shared);
    auto frag_K1 = create_fragment<blackwell::matrix_b, 128, 64, 64, 128>(K_shared + 4096);
    auto frag_V0 = create_fragment<blackwell::matrix_b, 64, 0, 64, 128>(V_shared);
    auto frag_V1 = create_fragment<blackwell::matrix_b, 64, 64, 64, 128>(V_shared + 4096);
    auto frag_P  = create_fragment<blackwell::matrix_a, 64, 0, 64, 64>(P_shared);
    auto frag_S  = create_fragment<blackwell::matrix_c, 64, 0, 64, 64>(S_shared);
    auto frag_O0 = create_fragment<blackwell::matrix_c, 64, 0, 64, 128>(K_shared);
    auto frag_O1 = create_fragment<blackwell::matrix_c, 64, 64, 64, 128>(K_shared + 4096);

    register uint32_t a_Q0, a_Q1, b_K0, b_K1, a_P, b_V0, b_V1;
    register uint32_t c_S, c_O0, c_O1;

    float m_prev_h = -1e20f;
    float l_prev_h = 0.0f;
    float m_prev_l = -1e20f;
    float l_prev_l = 0.0f;

    for (int kv_offset = 0; kv_offset < S; kv_offset += 64) {
        // Load K and V tiles dynamically
        for (int i = 0; i < 16; ++i) {
            int idx = tid + i * 128;
            int row = idx / 64;
            int load_kv = kv_offset + row;
            if (load_kv < S) {
                *(uint4*)&K_shared[idx] = *(const uint4*)&K_bh[load_kv * 128 + (idx % 64)];
                *(uint4*)&V_shared[idx] = *(const uint4*)&V_bh[load_kv * 128 + (idx % 64)];
            } else {
                *(uint4*)&K_shared[idx] = make_uint4(0, 0, 0, 0);
                *(uint4*)&V_shared[idx] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        
        float row_max_h = -1e20f;
        float row_max_l = -1e20f;
        
        for (int m_tile = 0; m_tile < 4; ++m_tile) {
            uint32_t m_tile_start = m_tile * 16;
            asm volatile("create_register_descriptor a0, shared::cta, .m128, [%0], %1;" :: "r"(Q_shared), "r"(m_tile_start));
            asm volatile("create_register_descriptor a1, shared::cta, .m128, [%0], %1;" :: "r"(Q_shared + 4096), "r"(m_tile_start));
            
            for (int n_tile = 0; n_tile < 4; ++n_tile) {
                uint32_t n_tile_start = n_tile * 16;
                
                asm volatile("create_register_descriptor b0, shared::cta, .k128, [%0], %1;" :: "r"(K_shared), "r"(n_tile_start));
                asm volatile("create_register_descriptor b1, shared::cta, .k128, [%0], %1;" :: "r"(K_shared + 4096), "r"(n_tile_start));
                
                asm volatile("create_register_descriptor c, shared::cta, .d16, [%0], %1, %2;" :: "r"(S_shared), "r"(m_tile_start), "r"(n_tile_start));
                
                asm volatile(
                    "{\n"
                    "wgmma.mma_async.m16n16k16.shared.d16.aligned.b16 a0, b0, c, a0, b0, c;\n"
                    "}\n" ::: "r"(a_Q0), "r"(b_K0), "r"(c_S));
                
                asm volatile(
                    "{\n"
                    "wgmma.mma_async.m16n16k16.shared.d16.aligned.b16 a1, b1, c, a1, b1, c;\n"
                    "}\n" ::: "r"(a_Q1), "r"(b_K1), "r"(c_S));
            }
        }
        asm volatile("wgmma.commit_group;\nwgmma.wait_group;\n" ::: "memory");
        
        float m_new_h = max(m_prev_h, row_max_h);
        float alpha_h = expf(m_prev_h - m_new_h);
        
        float m_new_l = max(m_prev_l, row_max_l);
        float alpha_l = expf(m_prev_l - m_new_l);

        //