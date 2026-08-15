#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

using namespace nvcuda;

// Hardware Instruction Wrappers
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;" ::: "memory");
}

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

CUresult create_tma_4d_descriptor_none(CUtensorMap* d, void* globalAddress, uint64_t d_dim, uint64_t S_dim, uint64_t H_dim, uint64_t B_dim) {
    cuuint64_t globalDim[4] = {d_dim, S_dim, H_dim, B_dim};
    cuuint64_t globalStrides[3] = {
        d_dim * 2, 
        S_dim * d_dim * 2, 
        H_dim * S_dim * d_dim * 2
    }; 
    cuuint32_t boxDim[4] = {128, 64, 1, 1}; 
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ void sdpa_bwd_kernel_tma_wmma(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int seq_len
) {
    setmaxnreg_inc_sync_fn(); // Allocate heavy 256 regs for accumulating fragments safely

    extern __shared__ uint4 smem[];
    uint8_t* smem_base = (uint8_t*)smem;
    uint64_t* mbar = (uint64_t*)smem_base;
    uint8_t* tma_base = smem_base + 16;
    
    // Shared Memory Layout Mapping (~128KB total, comfortably inside Hopper's 227KB limit)
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)tma_base;                // 16KB
    __nv_bfloat16* dO_smem = Q_smem + 64 * 128;                      // 16KB
    __nv_bfloat16* K_smem = dO_smem + 64 * 128;                      // 16KB
    __nv_bfloat16* V_smem = K_smem + 64 * 128;                       // 16KB
    __nv_bfloat16* O_smem = V_smem + 64 * 128;                       // 16KB
    
    float* S_smem = (float*)(O_smem + 64 * 128);                     // 16KB
    float* dP_smem = S_smem + 64 * 64;                               // 16KB
    __nv_bfloat16* P_smem = (__nv_bfloat16*)(dP_smem + 64 * 64);     // 8KB
    __nv_bfloat16* dS_smem = P_smem + 64 * 64;                       // 8KB
    float* D_smem = (float*)(dS_smem + 64 * 64);                     // 256B
    float* L_smem = D_smem + 64;                                     // 256B
    
    int block_i = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int Wy = warp_id / 2;
    int Wx = warp_id % 2;
    
    int global_q_row = block_i * 64;
    int batch_offset = (b * gridDim.y + h) * seq_len * 128;
    int l_offset = (b * gridDim.y + h) * seq_len;
    
    if (elect_one_sync_fn() && warp_id == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(128));
    }
    __syncthreads();
    
    int phase = 0;
    
    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" 
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(16384 * 3));
        tma_load_4d_fn(&tma_Q, mbar, Q_smem, 0, global_q_row, h, b);
        tma_load_4d_fn(&tma_dO, mbar, dO_smem, 0, global_q_row, h, b);
        tma_load_4d_fn(&tma_O, mbar, O_smem, 0, global_q_row, h, b);
    } else {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
    }
    
    if (tid < 64) {
        L_smem[tid] = (global_q_row + tid < seq_len) ? L[l_offset + global_q_row + tid] : 0.0f;
    }
    
    asm volatile(
        "{\n.reg .pred P;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(phase));
    phase ^= 1;
    
    if (tid < 64) {
        float d_val = 0.0f;
        for (int k = 0; k < 128; k++) {
            float do_val = __bfloat162float(dO_smem[tid * 128 + k]);
            float o_val  = __bfloat162float(O_smem[tid * 128 + k]);
            d_val += do_val * o_val;
        }
        D_smem[tid] = d_val;
    }
    __syncthreads();
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_frag[2][4];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(dQ_frag[i][j], 0.0f);
        }
    }
    
    for (int block_j = 0; block_j <= block_i; block_j++) {
        int global_k_row = block_j * 64;
        
        if (tid == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" 
                :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(16384 * 2));
            tma_load_4d_fn(&tma_K, mbar, K_smem, 0, global_k_row, h, b);
            tma_load_4d_fn(&tma_V, mbar, V_smem, 0, global_k_row, h, b);
        } else {
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        }
        
        asm volatile(
            "{\n.reg .pred P;\nWAIT_%=:\n"
            "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
            "@!P bra WAIT_%=;\n}\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(phase));
        phase ^= 1;
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_frag[2][2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dP_frag[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                wmma::fill_fragment(S_frag[i][j], 0.0f);
                wmma::fill_fragment(dP_frag[i][j], 0.0f);
            }
        }
        
        // S = Q K^T AND dP = dO V^T
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag[2];
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> v_frag[2];
            
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(q_frag[i], &Q_smem[(Wy * 32 + i * 16) * 128 + k], 128);
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(k_frag[j], &K_smem[(Wx * 32 + j * 16) * 128 + k], 128);
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(do_frag[i], &dO_smem[(Wy * 32 + i * 16) * 128 + k], 128);
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(v_frag[j], &V_smem[(Wx * 32 + j * 16) * 128 + k], 128);
            
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(S_frag[i][j], q_frag[i], k_frag[j], S_frag[i][j]);
                    wmma::mma_sync(dP_frag[i][j], do_frag[i], v_frag[j], dP_frag[i][j]);
                }
            }
        }
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                wmma::store_matrix_sync(&S_smem[(Wy * 32 + i * 16) * 64 + (Wx * 32 + j * 16)], S_frag[i][j], 64, wmma::mem_row_major);
                wmma::store_matrix_sync(&dP_smem[(Wy * 32 + i * 16) * 64 + (Wx * 32 + j * 16)], dP_frag[i][j], 64, wmma::mem_row_major);
            }
        }
        __syncthreads();
        
        // Logical Core - Attention Math (Causal Masking & Derivatives)
        for (int idx = tid; idx < 4096; idx += 128) {
            int r = idx / 64;
            int c = idx % 64;
            int glob_r = global_q_row + r;
            int glob_c = global_k_row + c;
            
            float p = 0.0f;
            float ds = 0.0f;
            
            if (glob_r < seq_len && glob_c < seq_len && glob_c <= glob_r) {
                float s = S_smem[idx];
                float dp = dP_smem[idx];
                float scaled_s = s * 0.088388347648f; 
                p = expf(scaled_s - L_smem[r]);
                ds = p * (dp - D_smem[r]) * 0.088388347648f;
            }
            
            P_smem[idx] = __float2bfloat16(p);
            dS_smem[idx] = __float2bfloat16(ds);
        }
        __syncthreads();
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_frag[2][4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_frag[2][4];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                wmma::fill_fragment(dK_frag[i][j], 0.0f);
                wmma::fill_fragment(dV_frag[i][j], 0.0f);
            }
        }
        
        // dQ += dS K, dK += dS^T Q, dV += P^T dO
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ds_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> k_frag_b[4];
            
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ds_t_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag_b[4];
            
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> p_t_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag_b[4];
            
            for (int i = 0; i < 2; i++) {
                wmma::load_matrix_sync(ds_frag[i], &dS_smem[(Wy * 32 + i * 16) * 64 + k], 64);
                wmma::load_matrix_sync(ds_t_frag[i], &dS_smem[k * 64 + (Wy * 32 + i * 16)], 64);
                wmma::load_matrix_sync(p_t_frag[i], &P_smem[k * 64 + (Wy * 32 + i * 16)], 64);
            }
            for (int j = 0; j < 4; j++) {
                wmma::load_matrix_sync(k_frag_b[j], &K_smem[k * 128 + (Wx * 64 + j * 16)], 128);
                wmma::load_matrix_sync(q_frag_b[j], &Q_smem[k * 128 + (Wx * 64 + j * 16)], 128);
                wmma::load_matrix_sync(do_frag_b[j], &dO_smem[k * 128 + (Wx * 64 + j * 16)], 128);
            }
            
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 4; j++) {
                    wmma::mma_sync(dQ_frag[i][j], ds_frag[i], k_frag_b[j], dQ_frag[i][j]);
                    wmma::mma_sync(dK_frag[i][j], ds_t_frag[i], q_frag_b[j], dK_frag[i][j]);
                    wmma::mma_sync(dV_frag[i][j], p_t_frag[i], do_frag_b[j], dV_frag[i][j]);
                }
            }
        }
        
        // Memory Aliasing - Extrema optimization. K and S buffers have safely concluded consumption.
        float* dK_smem_f = (float*)K_smem; // Reuse K_smem and V_smem (32KB available)
        float* dV_smem_f = S_smem;         // Reuse S_smem and dP_smem (32KB available)
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                wmma::store_matrix_sync(&dK_smem_f[(Wy * 32 + i * 16) * 128 + (Wx * 64 + j * 16)], dK_frag[i][j], 128, wmma::mem_row_major);
                wmma::store_matrix_sync(&dV_smem_f[(Wy * 32 + i * 16) * 128 + (Wx * 64 + j * 16)], dV_frag[i][j], 128, wmma::mem_row_major);
            }
        }
        __syncthreads();
        
        for (int idx = tid; idx < 4096; idx += 128) {
            int r = idx / 64;
            int c = (idx % 64) * 2;
            if (global_k_row + r < seq_len) {
                float dk0 = dK_smem_f[r * 128 + c];
                float dk1 = dK_smem_f[r * 128 + c + 1];
                float dv0 = dV_smem_f[r * 128 + c];
                float dv1 = dV_smem_f[r * 128 + c + 1];
                
                __nv_bfloat162 dk_bf = __floats2bfloat162_rn(dk0, dk1);
                __nv_bfloat162 dv_bf = __floats2bfloat162_rn(dv0, dv1);
                
                __nv_bfloat162* global_dK_ptr = (__nv_bfloat162*)&dK[batch_offset + (global_k_row + r) * 128 + c];
                __nv_bfloat162* global_dV_ptr = (__nv_bfloat162*)&dV[batch_offset + (global_k_row + r) * 128 + c];
                
                atomicAdd(global_dK_ptr, dk_bf);
                atomicAdd(global_dV_ptr, dv_bf);
            }
        }
        __syncthreads(); // Mandatory for iteration overlay integrity
    } // End block_j loop
    
    float* dQ_smem_f = (float*)K_smem; // Safely reuse
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&dQ_smem_f[(Wy * 32 + i * 16) * 128 + (Wx * 64 + j * 16)], dQ_frag[i][j], 128, wmma::mem_row_major);
        }
    }
    __syncthreads();
    
    for (int idx = tid; idx < 4096; idx += 128) {
        int r = idx / 64;
        int c = (idx % 64) * 2;
        if (global_q_row + r < seq_len) {
            float dq0 = dQ_smem_f[r * 128 + c];
            float dq1 = dQ_smem_f[r * 128 + c + 1];
            __nv_bfloat162 dq_bf = __floats2bfloat162_rn(dq0, dq1);
            
            uint32_t* global_dQ_ptr = (uint32_t*)&dQ[batch_offset + (global_q_row + r) * 128 + c];
            *global_dQ_ptr = *(uint32_t*)&dq_bf;
        }
    }
}


namespace tvm_ffi_sdpa_causal {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    if (Q.size(2) == 0) return;
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = 128;
    
    CUDA_CHECK(cudaMemsetAsync(dK.data_ptr(), 0, B * H * S * d * sizeof(uint16_t), stream));
    CUDA_CHECK(cudaMemsetAsync(dV.data_ptr(), 0, B * H * S * d * sizeof(uint16_t), stream));

    CUtensorMap tma_Q, tma_dO, tma_O, tma_K, tma_V;
    create_tma_4d_descriptor_none(&tma_Q, Q.data_ptr(), 128, S, H, B);
    create_tma_4d_descriptor_none(&tma_dO, dO.data_ptr(), 128, S, H, B);
    create_tma_4d_descriptor_none(&tma_O, O.data_ptr(), 128, S, H, B);
    create_tma_4d_descriptor_none(&tma_K, K.data_ptr(), 128, S, H, B);
    create_tma_4d_descriptor_none(&tma_V, V.data_ptr(), 128, S, H, B);

    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);

    int smem_size = 132000;
    CUDA_CHECK(cudaFuncSetAttribute(sdpa_bwd_kernel_tma_wmma, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    sdpa_bwd_kernel_tma_wmma<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_sdpa_causal::run);

}