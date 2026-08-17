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

__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    // Elevate maximum register limit dynamically to prevent heavy accumulator fragments from spilling
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;" ::: "memory");
}

__device__ __forceinline__ void cp_async_bulk_1d(void* smem, const void* gmem, int bytes, uint64_t* mbar) {
    uint32_t smem_ptr = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t mbar_ptr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(mbar_ptr) : "memory"
    );
}

// -------------------------------------------------------------
// Kernel 1: Computes dQ (Outer Loop: Q, Inner Loop: KV)
// -------------------------------------------------------------
__global__ __launch_bounds__(128) void sdpa_bwd_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int seq_len
) {
    setmaxnreg_inc_sync_fn();

    extern __shared__ uint4 smem[];
    uint8_t* smem_base = (uint8_t*)smem;
    uint64_t* mbar = (uint64_t*)smem_base;
    
    // Extrema-optimized dynamic overlap structure ensuring 100% capacity within limits
    __nv_bfloat16* Q_smem  = (__nv_bfloat16*)(smem_base + 16);             
    __nv_bfloat16* dO_smem = Q_smem + 64 * 128;               
    __nv_bfloat16* O_smem  = dO_smem + 64 * 128;              
    __nv_bfloat16* K_smem  = O_smem + 64 * 128;               
    __nv_bfloat16* V_smem  = K_smem + 2 * 64 * 128;               
    float* S_smem_f        = (float*)(V_smem + 2 * 64 * 128);             
    float* dP_smem_f       = S_smem_f + 64 * 64;             
    __nv_bfloat16* dS_smem_bf16 = (__nv_bfloat16*)(dP_smem_f + 64 * 64);                             
    float* L_smem          = (float*)(dS_smem_bf16 + 64 * 64);                             
    float* D_smem          = L_smem + 64;                             

    int block_i = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int Wy = tid / 32;

    int global_q_row = block_i * 64;
    int batch_offset = (b * gridDim.y + h) * seq_len * 128;
    int l_offset = (b * gridDim.y + h) * seq_len;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(128));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[1])), "r"(128));
    }
    __syncthreads();
    
    int phase = 0;
    int valid_q = min(64, seq_len - global_q_row);
    if (valid_q < 0) valid_q = 0;

    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" 
            :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(valid_q * 256 * 3));
    } else {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])));
    }

    if (tid < valid_q) {
        cp_async_bulk_1d(&Q_smem[tid * 128], &Q[batch_offset + (global_q_row + tid) * 128], 256, &mbar[0]);
        cp_async_bulk_1d(&dO_smem[tid * 128], &dO[batch_offset + (global_q_row + tid) * 128], 256, &mbar[0]);
        cp_async_bulk_1d(&O_smem[tid * 128], &O[batch_offset + (global_q_row + tid) * 128], 256, &mbar[0]);
    }

    asm volatile(
        "{\n.reg .pred P;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(phase));
    phase ^= 1;

    for (int idx = tid; idx < 64 * 16; idx += 128) {
        int r = idx / 16;
        if (r >= valid_q) {
            ((uint4*)Q_smem)[idx] = {0,0,0,0};
            ((uint4*)dO_smem)[idx] = {0,0,0,0};
            ((uint4*)O_smem)[idx] = {0,0,0,0};
        }
    }
    
    if (tid < 64) {
        L_smem[tid] = (tid < valid_q) ? L[l_offset + global_q_row + tid] : 0.0f;
    }
    __syncthreads();

    if (tid < 64) {
        float d_val = 0.0f;
        for (int k = 0; k < 128; k++) {
            d_val += __bfloat162float(dO_smem[tid * 128 + k]) * __bfloat162float(O_smem[tid * 128 + k]);
        }
        D_smem[tid] = d_val;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_frag[8];
    for(int i = 0; i < 8; i++) wmma::fill_fragment(dQ_frag[i], 0.0f);

    int valid_k = min(64, seq_len);
    if (valid_k < 0) valid_k = 0;

    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" 
            :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(valid_k * 256 * 2));
    } else {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])));
    }

    if (tid < valid_k) {
        cp_async_bulk_1d(&K_smem[tid * 128], &K[batch_offset + tid * 128], 256, &mbar[0]);
        cp_async_bulk_1d(&V_smem[tid * 128], &V[batch_offset + tid * 128], 256, &mbar[0]);
    }
    
    int phase_k[2] = {phase, 0};

    for (int block_j = 0; block_j <= block_i; block_j++) {
        int buf_idx = block_j % 2;
        int next_buf_idx = (block_j + 1) % 2;
        
        uint32_t cur_mbar = (uint32_t)__cvta_generic_to_shared(&mbar[buf_idx]);
        int cur_phase = phase_k[buf_idx];
        asm volatile(
            "{\n.reg .pred P;\nWAIT_%=:\n"
            "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
            "@!P bra WAIT_%=;\n}\n"
            :: "r"(cur_mbar), "r"(cur_phase));
        phase_k[buf_idx] ^= 1;
        
        int cur_k_offset = buf_idx * 64 * 128;
        int cur_global_k = block_j * 64;
        int cur_valid_k = min(64, seq_len - cur_global_k);
        if (cur_valid_k < 0) cur_valid_k = 0;
        
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int r = idx / 16;
            if (r >= cur_valid_k) {
                ((uint4*)&K_smem[cur_k_offset])[idx] = {0,0,0,0};
                ((uint4*)&V_smem[cur_k_offset])[idx] = {0,0,0,0};
            }
        }
        __syncthreads();

        if (block_j + 1 <= block_i) {
            int next_global_k = (block_j + 1) * 64;
            int next_valid_k = min(64, seq_len - next_global_k);
            if (next_valid_k < 0) next_valid_k = 0;
            
            uint32_t next_mbar = (uint32_t)__cvta_generic_to_shared(&mbar[next_buf_idx]);
            if (tid == 0) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" 
                    :: "r"(next_mbar), "r"(next_valid_k * 256 * 2));
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(next_mbar));
            }
            if (tid < next_valid_k) {
                cp_async_bulk_1d(&K_smem[next_buf_idx * 64 * 128 + tid * 128], &K[batch_offset + (next_global_k + tid) * 128], 256, &mbar[next_buf_idx]);
                cp_async_bulk_1d(&V_smem[next_buf_idx * 64 * 128 + tid * 128], &V[batch_offset + (next_global_k + tid) * 128], 256, &mbar[next_buf_idx]);
            }
        }

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_frag[4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dP_frag[4];
        for(int c = 0; c < 4; c++) { 
            wmma::fill_fragment(S_frag[c], 0.0f); 
            wmma::fill_fragment(dP_frag[c], 0.0f); 
        }

        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag;
            wmma::load_matrix_sync(q_frag, &Q_smem[Wy * 16 * 128 + k], 128);
            wmma::load_matrix_sync(do_frag, &dO_smem[Wy * 16 * 128 + k], 128);

            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> v_frag;
                wmma::load_matrix_sync(k_frag, &K_smem[cur_k_offset + c * 16 * 128 + k], 128);
                wmma::load_matrix_sync(v_frag, &V_smem[cur_k_offset + c * 16 * 128 + k], 128);
                
                wmma::mma_sync(S_frag[c], q_frag, k_frag, S_frag[c]);
                wmma::mma_sync(dP_frag[c], do_frag, v_frag, dP_frag[c]);
            }
        }

        for (int c = 0; c < 4; c++) {
            for (int i = 0; i < 8; i++) {
                int r = (lane / 4) + 8 * (i & 1);
                int col = (lane % 4) * 2 + 8 * (i / 2);
                int local_r = Wy * 16 + r;
                int local_c = c * 16 + col;
                int glob_r = global_q_row + local_r;
                int glob_c = cur_global_k + local_c;
                float ds = 0.0f;
                if (glob_r < seq_len && glob_c < seq_len && glob_c <= glob_r) {
                    float s = S_frag[c].x[i] * 0.088388347648f;
                    float p = expf(s - L_smem[local_r]);
                    ds = p * (dP_frag[c].x[i] - D_smem[local_r]) * 0.088388347648f;
                }
                dS_smem_bf16[local_r * 64 + local_c] = __float2bfloat16(ds);
            }
        }
        __syncthreads();

        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ds_frag;
            wmma::load_matrix_sync(ds_frag, &dS_smem_bf16[Wy * 16 * 64 + k], 64);
            for (int c = 0; c < 8; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> k_frag;
                wmma::load_matrix_sync(k_frag, &K_smem[cur_k_offset + k * 128 + c * 16], 128);
                wmma::mma_sync(dQ_frag[c], ds_frag, k_frag, dQ_frag[c]);
            }
        }
        __syncthreads(); 
    }

    float* dQ_smem_f = (float*)Q_smem;
    for (int c = 0; c < 8; c++) {
        wmma::store_matrix_sync(&dQ_smem_f[Wy * 16 * 128 + c * 16], dQ_frag[c], 128, wmma::mem_row_major);
    }
    __syncthreads();

    for(int idx = tid; idx < 4096; idx += 128) {
        int r = idx / 64;
        int c = (idx % 64) * 2;
        if (global_q_row + r < seq_len) {
            float dq0 = dQ_smem_f[r * 128 + c];
            float dq1 = dQ_smem_f[r * 128 + c + 1];
            __nv_bfloat162 dq_bf = __floats2bfloat162_rn(dq0, dq1);
            *(uint32_t*)&dQ[batch_offset + (global_q_row + r) * 128 + c] = *(uint32_t*)&dq_bf;
        }
    }
}

// -------------------------------------------------------------
// Kernel 2: Computes dK, dV (Outer Loop: KV, Inner Loop: Q)
// -------------------------------------------------------------
__global__ __launch_bounds__(128) void sdpa_bwd_dK_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int seq_len
) {
    setmaxnreg_inc_sync_fn();

    extern __shared__ uint4 smem[];
    uint8_t* smem_base = (uint8_t*)smem;
    uint64_t* mbar = (uint64_t*)smem_base;
    
    __nv_bfloat16* K_smem  = (__nv_bfloat16*)(smem_base + 16);             
    __nv_bfloat16* V_smem  = K_smem + 64 * 128;               
    __nv_bfloat16* Q_smem  = V_smem + 64 * 128;               
    __nv_bfloat16* dO_smem = Q_smem + 2 * 64 * 128;               
    __nv_bfloat16* O_smem  = dO_smem + 2 * 64 * 128;              
    float* S_smem_f        = (float*)(O_smem + 2 * 64 * 128);               
    float* dP_smem_f       = S_smem_f + 64 * 64;               
    __nv_bfloat16* dS_smem_bf16 = (__nv_bfloat16*)(dP_smem_f + 64 * 64);             
    __nv_bfloat16* P_smem_bf16  = dS_smem_bf16 + 64 * 64;                             
    float* L_smem          = (float*)(P_smem_bf16 + 64 * 64);
    float* D_smem          = L_smem + 128;                             

    int block_j = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int Wy = tid / 32;

    int global_k_row = block_j * 64;
    int batch_offset = (b * gridDim.y + h) * seq_len * 128;
    int l_offset = (b * gridDim.y + h) * seq_len;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(128));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[1])), "r"(128));
    }
    __syncthreads();
    
    int valid_k = min(64, seq_len - global_k_row);
    if (valid_k < 0) valid_k = 0;

    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" 
            :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(valid_k * 256 * 2));
    } else {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])));
    }

    if (tid < valid_k) {
        cp_async_bulk_1d(&K_smem[tid * 128], &K[batch_offset + (global_k_row + tid) * 128], 256, &mbar[0]);
        cp_async_bulk_1d(&V_smem[tid * 128], &V[batch_offset + (global_k_row + tid) * 128], 256, &mbar[0]);
    }

    int phase = 0;
    asm volatile(
        "{\n.reg .pred P;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(phase));
    phase ^= 1;

    for (int idx = tid; idx < 64 * 16; idx += 128) {
        int r = idx / 16;
        if (r >= valid_k) {
            ((uint4*)K_smem)[idx] = {0,0,0,0};
            ((uint4*)V_smem)[idx] = {0,0,0,0};
        }
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_frag[8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_frag[8];
    for(int c = 0; c < 8; c++) { 
        wmma::fill_fragment(dK_frag[c], 0.0f); 
        wmma::fill_fragment(dV_frag[c], 0.0f); 
    }

    int start_i = block_j; 
    int end_i = (seq_len + 63) / 64;

    int phase_q[2] = {phase, 0};
    
    if (start_i < end_i) {
        int global_q_row = start_i * 64;
        int valid_q = min(64, seq_len - global_q_row);
        if (valid_q < 0) valid_q = 0;
        
        if (tid == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" 
                :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(valid_q * 256 * 3));
        } else {
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])));
        }

        if (tid < valid_q) {
            cp_async_bulk_1d(&Q_smem[tid * 128], &Q[batch_offset + (global_q_row + tid) * 128], 256, &mbar[0]);
            cp_async_bulk_1d(&dO_smem[tid * 128], &dO[batch_offset + (global_q_row + tid) * 128], 256, &mbar[0]);
            cp_async_bulk_1d(&O_smem[tid * 128], &O[batch_offset + (global_q_row + tid) * 128], 256, &mbar[0]);
        }
    }

    for (int block_i = start_i; block_i < end_i; block_i++) {
        int buf_idx = (block_i - start_i) % 2;
        int next_buf_idx = (block_i - start_i + 1) % 2;
        
        uint32_t cur_mbar = (uint32_t)__cvta_generic_to_shared(&mbar[buf_idx]);
        int cur_phase = phase_q[buf_idx];
        asm volatile(
            "{\n.reg .pred P;\nWAIT_%=:\n"
            "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
            "@!P bra WAIT_%=;\n}\n"
            :: "r"(cur_mbar), "r"(cur_phase));
        phase_q[buf_idx] ^= 1;
        
        int cur_q_offset = buf_idx * 64 * 128;
        int cur_l_offset = buf_idx * 64;
        
        int cur_global_q = block_i * 64;
        int cur_valid_q = min(64, seq_len - cur_global_q);
        if (cur_valid_q < 0) cur_valid_q = 0;
        
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int r = idx / 16;
            if (r >= cur_valid_q) {
                ((uint4*)&Q_smem[cur_q_offset])[idx] = {0,0,0,0};
                ((uint4*)&dO_smem[cur_q_offset])[idx] = {0,0,0,0};
                ((uint4*)&O_smem[cur_q_offset])[idx] = {0,0,0,0};
            }
        }
        if (tid < 64) {
            L_smem[cur_l_offset + tid] = (tid < cur_valid_q) ? L[l_offset + cur_global_q + tid] : 0.0f;
        }
        __syncthreads();
        
        if (tid < 64) {
            float d_val = 0.0f;
            for (int k = 0; k < 128; k++) {
                d_val += __bfloat162float(dO_smem[cur_q_offset + tid * 128 + k]) * __bfloat162float(O_smem[cur_q_offset + tid * 128 + k]);
            }
            D_smem[cur_l_offset + tid] = d_val;
        }
        __syncthreads();
        
        if (block_i + 1 < end_i) {
            int next_global_q = (block_i + 1) * 64;
            int next_valid_q = min(64, seq_len - next_global_q);
            if (next_valid_q < 0) next_valid_q = 0;
            
            uint32_t next_mbar = (uint32_t)__cvta_generic_to_shared(&mbar[next_buf_idx]);
            if (tid == 0) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" 
                    :: "r"(next_mbar), "r"(next_valid_q * 256 * 3));
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(next_mbar));
            }
            if (tid < next_valid_q) {
                cp_async_bulk_1d(&Q_smem[next_buf_idx * 64 * 128 + tid * 128], &Q[batch_offset + (next_global_q + tid) * 128], 256, &mbar[next_buf_idx]);
                cp_async_bulk_1d(&dO_smem[next_buf_idx * 64 * 128 + tid * 128], &dO[batch_offset + (next_global_q + tid) * 128], 256, &mbar[next_buf_idx]);
                cp_async_bulk_1d(&O_smem[next_buf_idx * 64 * 128 + tid * 128], &O[batch_offset + (next_global_q + tid) * 128], 256, &mbar[next_buf_idx]);
            }
        }

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_frag[4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dP_frag[4];
        for(int c = 0; c < 4; c++) { 
            wmma::fill_fragment(S_frag[c], 0.0f); 
            wmma::fill_fragment(dP_frag[c], 0.0f); 
        }

        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> v_frag;
            wmma::load_matrix_sync(k_frag, &K_smem[Wy * 16 * 128 + k], 128);
            wmma::load_matrix_sync(v_frag, &V_smem[Wy * 16 * 128 + k], 128);

            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag;
                wmma::load_matrix_sync(q_frag, &Q_smem[cur_q_offset + c * 16 * 128 + k], 128);
                wmma::load_matrix_sync(do_frag, &dO_smem[cur_q_offset + c * 16 * 128 + k], 128);
                
                wmma::mma_sync(S_frag[c], q_frag, k_frag, S_frag[c]);
                wmma::mma_sync(dP_frag[c], do_frag, v_frag, dP_frag[c]);
            }
        }

        for (int c = 0; c < 4; c++) {
            for (int i = 0; i < 8; i++) {
                int r = (lane / 4) + 8 * (i & 1);
                int col = (lane % 4) * 2 + 8 * (i / 2);
                int local_q_r = c * 16 + r;    
                int local_k_r = Wy * 16 + col;  
                int glob_r = cur_global_q + local_q_r;
                int glob_c = global_k_row + local_k_r;
                
                float ds = 0.0f;
                float p = 0.0f;
                if (glob_r < seq_len && glob_c < seq_len && glob_c <= glob_r) {
                    float s = S_frag[c].x[i] * 0.088388347648f;
                    p = expf(s - L_smem[cur_l_offset + local_q_r]);
                    ds = p * (dP_frag[c].x[i] - D_smem[cur_l_offset + local_q_r]) * 0.088388347648f;
                }
                dS_smem_bf16[local_q_r * 64 + local_k_r] = __float2bfloat16(ds);
                P_smem_bf16[local_q_r * 64 + local_k_r] = __float2bfloat16(p);
            }
        }
        __syncthreads();

        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ds_t_frag;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> p_t_frag;
            wmma::load_matrix_sync(ds_t_frag, &dS_smem_bf16[k * 64 + Wy * 16], 64);
            wmma::load_matrix_sync(p_t_frag, &P_smem_bf16[k * 64 + Wy * 16], 64);

            for (int c = 0; c < 8; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag;
                wmma::load_matrix_sync(q_frag, &Q_smem[cur_q_offset + k * 128 + c * 16], 128);
                wmma::load_matrix_sync(do_frag, &dO_smem[cur_q_offset + k * 128 + c * 16], 128);
                
                wmma::mma_sync(dK_frag[c], ds_t_frag, q_frag, dK_frag[c]);
                wmma::mma_sync(dV_frag[c], p_t_frag, do_frag, dV_frag[c]);
            }
        }
        __syncthreads();
    }

    float* dK_smem_f = (float*)Q_smem;
    float* dV_smem_f = (float*)dO_smem;
    
    for (int c = 0; c < 8; c++) {
        wmma::store_matrix_sync(&dK_smem_f[Wy * 16 * 128 + c * 16], dK_frag[c], 128, wmma::mem_row_major);
        wmma::store_matrix_sync(&dV_smem_f[Wy * 16 * 128 + c * 16], dV_frag[c], 128, wmma::mem_row_major);
    }
    __syncthreads();

    for(int idx = tid; idx < 4096; idx += 128) {
        int r = idx / 64;
        int c = (idx % 64) * 2;
        if (global_k_row + r < seq_len) {
            float dk0 = dK_smem_f[r * 128 + c];
            float dk1 = dK_smem_f[r * 128 + c + 1];
            float dv0 = dV_smem_f[r * 128 + c];
            float dv1 = dV_smem_f[r * 128 + c + 1];
            
            __nv_bfloat162 dk_bf = __floats2bfloat162_rn(dk0, dk1);
            __nv_bfloat162 dv_bf = __floats2bfloat162_rn(dv0, dv1);
            *(uint32_t*)&dK[batch_offset + (global_k_row + r) * 128 + c] = *(uint32_t*)&dk_bf;
            *(uint32_t*)&dV[batch_offset + (global_k_row + r) * 128 + c] = *(uint32_t*)&dv_bf;
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
    
    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);

    int smem_dQ = 160000;
    int smem_dK = 190000;
    
    CUDA_CHECK(cudaFuncSetAttribute(sdpa_bwd_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dQ));
    sdpa_bwd_dQ_kernel<<<grid, block, smem_dQ, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S
    );

    CUDA_CHECK(cudaFuncSetAttribute(sdpa_bwd_dK_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dK));
    sdpa_bwd_dK_dV_kernel<<<grid, block, smem_dK, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S
    );
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_sdpa_causal::run);

}