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

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

using namespace nvcuda;

__device__ __forceinline__ void fence_proxy_async_fn() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n" ".reg .pred P;\n" "WAIT_%=:\n" "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n" "@!P bra WAIT_%=;\n" "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}
template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ uint64_t swizzled_128B_base(uint32_t addr) {
    return (((uint64_t)(addr >> 7)) << 36) | (((uint64_t)(addr & 128)) << 32);
}

__device__ __forceinline__ uint64_t swizzled_128B_col_major_desc(void* ptr, int lbo, int sbo) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t lbo_swizzled = ((addr >> 7) & 7) * 16 + lbo;
    return swizzled_128B_base(addr) | ((uint64_t)((lbo_swizzled & 0x3FFFF) >> 4) << 16) | ((uint64_t)((sbo & 0x3FFFF) >> 4) << 32);
}

__device__ __forceinline__ uint64_t swizzled_128B_row_major_desc(void* ptr, int lbo, int sbo) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t lbo_swizzled = ((addr >> 7) & 7) * 16 + lbo;
    return swizzled_128B_base(addr) | ((uint64_t)((lbo_swizzled & 0x3FFFF) >> 4) << 16) | ((uint64_t)((sbo & 0x3FFFF) >> 4) << 32);
}

__device__ __forceinline__ void load_A_col_major_8x16(wmma::fragment<wmma::matrix_a, 8, 8, 16, __nv_bfloat16, wmma::col_major>& A_frag, void* p0) {
    uint64_t desc0 = swizzled_128B_col_major_desc(p0, 16, 128);
    wmma::load_matrix_sync(A_frag, p0, desc0);
}

__device__ __forceinline__ void load_B_col_major_8x16(wmma::fragment<wmma::matrix_b, 8, 8, 16, __nv_bfloat16, wmma::col_major>& B_frag, void* p0, void* p1) {
    uint64_t desc0 = swizzled_128B_col_major_desc(p0, 16, 128);
    uint64_t desc1 = swizzled_128B_col_major_desc(p1, 16, 128);
    wmma::load_matrix_sync(B_frag, p0, desc0);
    wmma::load_matrix_sync(B_frag, p1, desc1, 16, 8);
}

__device__ __forceinline__ void load_B_row_major_8x16(wmma::fragment<wmma::matrix_b, 8, 8, 16, __nv_bfloat16, wmma::row_major>& B_frag, void* p0, void* p1) {
    uint64_t desc0 = swizzled_128B_row_major_desc(p0, 16, 128);
    uint64_t desc1 = swizzled_128B_row_major_desc(p1, 16, 128);
    wmma::load_matrix_sync(B_frag, p0, desc0);
    wmma::load_matrix_sync(B_frag, p1, desc1, 16, 8);
}

__device__ __forceinline__ void load_gmem_to_smem(const __nv_bfloat16* gmem, __nv_bfloat16* smem_0, __nv_bfloat16* smem_1, int num_rows, int row_step) {
    int tid = threadIdx.x;
    if (tid < 128) {
        int row = tid * 2;
        const int4* gmem_v = (const int4*)(gmem + row * row_step);
        int4 v0 = gmem_v[0];
        int4 v1 = gmem_v[1];
        int4 v2 = gmem_v[2];
        int4 v3 = gmem_v[3];
        
        const int4* gmem_v1 = (const int4*)(gmem + (row + 1) * row_step);
        int4 v4 = gmem_v1[0];
        int4 v5 = gmem_v1[1];
        int4 v6 = gmem_v1[2];
        int4 v7 = gmem_v1[3];
        
        __nv_bfloat16* smem_all = (tid < 64) ? smem_0 : smem_1;
        int x = (tid / 64) % 8;
        int y_swizzled = (row % 8) ^ x;
        int offset = row * 64 + y_swizzled * 8 + (tid % 8);
        *reinterpret_cast<int4*>(&smem_all[offset]) = v0;
        *reinterpret_cast<int4*>(&smem_all[offset + 16]) = v1;
        *reinterpret_cast<int4*>(&smem_all[offset + 32]) = v2;
        *reinterpret_cast<int4*>(&smem_all[offset + 48]) = v3;
        
        int x1 = (tid / 64) % 8;
        int y_swizzled1 = ((row + 1) % 8) ^ x1;
        int offset1 = (row + 1) * 64 + y_swizzled1 * 8 + (tid % 8);
        *reinterpret_cast<int4*>(&smem_all[offset1]) = v4;
        *reinterpret_cast<int4*>(&smem_all[offset1 + 16]) = v5;
        *reinterpret_cast<int4*>(&smem_all[offset1 + 32]) = v6;
        *reinterpret_cast<int4*>(&smem_all[offset1 + 48]) = v7;
    }
}

__device__ __forceinline__ void store_smem_to_gmem(__nv_bfloat16* gmem, const __nv_bfloat16* smem_0, const __nv_bfloat16* smem_1, int num_rows, int row_step) {
    int tid = threadIdx.x;
    if (tid < 128) {
        int row = tid * 2;
        __nv_bfloat16* smem_all = (tid < 64) ? ( __nv_bfloat16* )smem_0 : ( __nv_bfloat16* )smem_1;
        const int4* smem_v = (const int4*)(smem_all + row * 64);
        int4 v0 = smem_v[0];
        int4 v1 = smem_v[1];
        int4 v2 = smem_v[2];
        int4 v3 = smem_v[3];
        int4 v4 = smem_v[4];
        int4 v5 = smem_v[5];
        int4 v6 = smem_v[6];
        int4 v7 = smem_v[7];
        
        int4* gmem_v = (int4*)(gmem + row * row_step);
        gmem_v[0] = v0;
        gmem_v[1] = v1;
        gmem_v[2] = v2;
        gmem_v[3] = v3;
        gmem_v[4] = v4;
        gmem_v[5] = v5;
        gmem_v[6] = v6;
        gmem_v[7] = v7;
        
        const int4* smem_v1 = (const int4*)(smem_all + (row + 1) * 64);
        int4 v8 = smem_v1[0];
        int4 v9 = smem_v1[1];
        int4 v10 = smem_v1[2];
        int4 v11 = smem_v1[3];
        int4 v12 = smem_v1[4];
        int4 v13 = smem_v1[5];
        int4 v14 = smem_v1[6];
        int4 v15 = smem_v1[7];
        
        int4* gmem_v1 = (int4*)(gmem + (row + 1) * row_step);
        gmem_v1[0] = v8;
        gmem_v1[1] = v9;
        gmem_v1[2] = v10;
        gmem_v1[3] = v11;
        gmem_v1[4] = v12;
        gmem_v1[5] = v13;
        gmem_v1[6] = v14;
        gmem_v1[7] = v15;
    }
}

__device__ __forceinline__ void store_and_extract_S_floats(
    const wmma::fragment<wmma::accumulator, 8, 8, 16, float>& C_frag,
    __nv_bfloat16* smem, int wg, float (&out_P)[8][8], float (&out_D)[8][8]) 
{
    for (int c = 0; c < 8; ++c) {
        for (int r = 0; r < 8; ++r) {
            float p_val = wmma::extract(C_frag, r, c);
            out_P[r][c] = p_val;
            int swizzled_x = (c / 8) ^ ((wg * 8 + r) % 8);
            int swizzled_idx = (wg * 8 + r) * 64 + swizzled_x * 8 + (c % 8);
            smem[swizzled_idx] = __float2bfloat16(p_val);
        }
    }
}

// ---- DQ Kernel ----
__global__ void bwd_dq_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O, __nv_bfloat16* __restrict__ dQ,
    const float* __restrict__ L, int S, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();

    extern __shared__ __align__(128) char smem[];
    __nv_bfloat16* smem_Q_0 = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_Q_1 = smem_Q_0 + 64 * 64;
    __nv_bfloat16* smem_K_0 = smem_Q_1 + 64 * 64;
    __nv_bfloat16* smem_K_1 = smem_K_0 + 64 * 64;
    __nv_bfloat16* smem_V_0 = smem_K_1 + 64 * 64;
    __nv_bfloat16* smem_V_1 = smem_V_0 + 64 * 64;
    __nv_bfloat16* smem_dO_0 = smem_V_1 + 64 * 64;
    __nv_bfloat16* smem_dO_1 = smem_dO_0 + 64 * 64;
    __nv_bfloat16* smem_dS = smem_dO_1 + 64 * 64;
    __nv_bfloat16* smem_D = smem_dS + 64 * 64;
    float* smem_L = reinterpret_cast<float*>(smem_D + 64 * 64);
    
    uint64_t* bar_load = reinterpret_cast<uint64_t*>(smem_L + 64);
    uint64_t* bar_compute = reinterpret_cast<uint64_t*>(smem_L + 64 + 1);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_load, 1);
        init_smem_barrier_fn(bar_compute, 128);
    }
    __syncthreads();

    int num_heads = 48;
    int head_id = (blockIdx.y / (S + 63)/64) % num_heads;
    int batch_id = blockIdx.y / ((S + 63)/64 * num_heads);
    const float* L_ptr = L + batch_id * num_heads * S + head_id * S;
    int query_blk = blockIdx.x;
    int query_blk_start = query_blk * 64;
    if (query_blk_start >= S) return;

    const __nv_bfloat16* Q_ptr = Q + batch_id * num_heads * S * 128 + head_id * S * 128 + query_blk_start * 128;
    const __nv_bfloat16* K_ptr = K + batch_id * num_heads * S * 128 + head_id * S * 128;
    const __nv_bfloat16* V_ptr = V + batch_id * num_heads * S * 128 + head_id * S * 128;
    const __nv_bfloat16* dO_ptr = dO + batch_id * num_heads * S * 128 + head_id * S * 128 + query_blk_start * 128;
    __nv_bfloat16* dQ_ptr = dQ + batch_id * num_heads * S * 128 + head_id * S * 128 + query_blk_start * 128;

    int wg = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (threadIdx.x < 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_load, 8192 * 8);
        }
        load_gmem_to_smem(Q_ptr, smem_Q_0, smem_Q_1, 64, 128);
        load_gmem_to_smem(K_ptr, smem_K_0, smem_K_1, 64, 128);
        load_gmem_to_smem(V_ptr, smem_V_0, smem_V_1, 64, 128);
        load_gmem_to_smem(dO_ptr, smem_dO_0, smem_dO_1, 64, 128);
        if (threadIdx.x < 64) {
            smem_L[threadIdx.x] = L_ptr[query_blk_start + threadIdx.x];
        }
    }
    fence_proxy_async_fn();
    mbarrier_wait_fn(bar_load, 0);

    float sum_P_D[8] = {0};

    for (int col_tile = 0; col_tile < 8; ++col_tile) {
        wmma::fragment<wmma::accumulator, 8, 8, 16, float> C_frag_S_wg;
        wmma::fill_fragment(C_frag_S_wg);
        
        wmma::fragment<wmma::matrix_a, 8, 8, 16, __nv_bfloat16, wmma::col_major> A_frag_Q_wg[4];
        wmma::fragment<wmma::matrix_b, 8, 8, 16, __nv_bfloat16, wmma::col_major> B_frag_K_wg[4];
        for(int i=0; i<4; ++i) {
            load_A_col_major_8x16(A_frag_Q_wg[i], smem_Q_0 + wg * 8 * 64 + i * 16);
            load_B_col_major_8x16(B_frag_K_wg[i], smem_K_0 + col_tile * 8 * 64 + i * 16, smem_K_1 + col_tile * 8 * 64 + i * 16);
            wmma::mma_sync(C_frag_S_wg, A_frag_Q_wg[i], B_frag_K_wg[i]);
        }

        wmma::fragment<wmma::accumulator, 8, 8, 16, float> C_frag_D_wg;
        wmma::fill_fragment(C_frag_D_wg);
        wmma::fragment<wmma::matrix_a, 8, 8, 16, __nv_bfloat16, wmma::col_major> A_frag_dO_wg[4];
        wmma::fragment<wmma::matrix_b, 8, 8, 16, __nv_bfloat16, wmma::col_major> B_frag_V_wg[4];
        for(int i=0; i<4; ++i) {
            load_A_col_major_8x16(A_frag_dO_wg[i], smem_dO_0 + wg * 8 * 64 + i * 16);
            load_B_col_major_8x16(B_frag_V_wg[i], smem_V_0 + col_tile * 8 * 64 + i * 16, smem_V_1 + col_tile * 8 * 64 + i * 16);
            wmma::mma_sync(C_frag_D_wg, A_frag_dO_wg[i], B_frag_V_wg[i]);
        }

        float P_vals[8][8], D_vals[8][8];
        store_and_extract_S_floats(C_frag_S_wg, smem_dS, wg, P_vals, D_vals);
        store_and_extract_S_floats(C_frag_D_wg, smem_D, wg, P_vals, D_vals);
        
        float pd_local[8] = {0};
        for(int c=0; c<8; ++c) {
            for(int r=0; r<8; ++r) {
                P_vals[r][c] = expf(P_vals[r][c] - smem_L[wg * 8 + r]);
                pd_local[r] += P_vals[r][c] * D_vals[r][c];
            }
        }
        
        for(int r=0; r<8; ++r) {
            pd_local[r] += __shfl_down_sync(0xFFFFFFFF, pd_local[r], 2);
            pd_local[r] += __shfl_down_sync(0xFFFFFFFF, pd_local[r], 1);
            if (lane_id < 4) sum_P_D[r] += pd_local[r];
        }
        
        for(int c=0; c<8; ++c) {
            for(int r=0; r<8; ++r) {
                float ds_val = P_vals[r][c] * (D_vals[r][c] - sum_P_D[r]) * scale;
                int swizzled_x = (c / 8) ^ ((wg * 8 + r) % 8);
                int swizzled_idx = (wg * 8 + r) * 64 + swizzled_x * 8 + (c % 8);
                smem_dS[swizzled_idx] = __float2bfloat16(ds_val);
            }
        }
    }
    __syncthreads();
    
    float sum_PD[8] = {0}; // Unused variable, but kept to preserve exact original kernel logic structure context mapping
    (void)sum_PD;

    wmma::fragment<wmma::accumulator, 8, 8, 16, float> C_frag_dQ_wg[2][8];
    for (int i=0; i<2; ++i) {
        for (int j=0; j<8; ++j) {
            wmma::fill_fragment(C_frag_dQ_wg[i][j]);
        }
    }
    fence_proxy_async_fn();

    wmma::fragment<wmma::matrix_a, 8, 8, 16, __nv_bfloat16, wmma::col_major> A_frag_dS_wg[4];
    wmma::fragment<wmma::matrix_b, 8, 8, 16, __nv_bfloat16, wmma::row_major> B_frag_dK_wg[4];

    for (int d_half = 0; d_half < 2; ++d_half) {
        for (int col_tile = 0; col_tile < 8; ++col_tile) {
            for (int k = 0; k < 4; ++k) {
                load_A_col_major_8x16(A_frag_dS_wg[k], (d_half == 0) ? (smem_dS + k * 16 * 64 + col_tile * 8) : (smem_dS + k * 16 * 64 + col_tile * 8 + 8*64));
                load_B_row_major_8x16(B_frag_dK_wg[k], 
                    (d_half == 0) ? (smem_K_0 + k * 16 * 64 + wg * 8) : (smem_K_1 + k * 16 * 64 + wg * 8),
                    (d_half == 0) ? (smem_K_0 + k * 16 * 64 + wg * 8 + 8) : (smem_K_1 + k * 16 * 64 + wg * 8 + 8));
                wmma::mma_sync(C_frag_dQ_wg[d_half][col_tile], A_frag_dS_wg[k], B_frag_dK_wg[k]);
            }
        }
    }

    for (int d_half = 0; d_half < 2; ++d_half) {
        for (int col_tile = 0; col_tile < 8; ++col_tile) {
            float P_vals[8][8], D_vals[8][8];
            store_and_extract_S_floats(C_frag_dQ_wg[d_half][col_tile], (d_half == 0) ? smem_Q_0 : smem_Q_1, wg, P_vals, D_vals);
        }
    }
    
    __syncthreads();
    store_smem_to_gmem(dQ_ptr, smem_Q_0, smem_Q_1, 64, 128);
}

// ---- DK/DV Kernel ----
__global__ void bwd_dkv_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O, __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV,
    const float* __restrict__ L, int S, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();

    extern __shared__ __align__(128) char smem[];
    __nv_bfloat16* smem_Q_0 = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_Q_1 = smem_Q_0 + 64 * 64;
    __nv_bfloat16* smem_K_0 = smem_Q_1 + 64 * 64;
    __nv_bfloat16* smem_K_1 = smem_K_0 + 64 * 64;
    __nv_bfloat16* smem_V_0 = smem_K_1 + 64 * 64;
    __nv_bfloat16* smem_V_1 = smem_V_0 + 64 * 64;
    __nv_bfloat16* smem_dO_0 = smem_V_1 + 64 * 64;
    __nv_bfloat16* smem_dO_1 = smem_dO_0 + 64 * 64;
    __nv_bfloat16* smem_dS = smem_dO_1 + 64 * 64;
    __nv_bfloat16* smem_D = smem_dS + 64 * 64;
    __nv_bfloat16* smem_P_T = smem_D + 64 * 64;
    float* smem_L = reinterpret_cast<float*>(smem_P_T + 64 * 64);
    
    uint64_t* bar_load = reinterpret_cast<uint64_t*>(smem_L + 64);
    uint64_t* bar_compute = reinterpret_cast<uint64_t*>(smem_L + 64 + 1);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_load, 1);
        init_smem_barrier_fn(bar_compute, 128);
    }
    __syncthreads();

    int num_heads = 48;
    int head_id = (blockIdx.y / (S + 63)/64) % num_heads;
    int batch_id = blockIdx.y / ((S + 63)/64 * num_heads);
    const float* L_ptr = L + batch_id * num_heads * S + head_id * S;
    
    int key_blk = blockIdx.x;
    int key_blk_start = key_blk * 64;
    if (key_blk_start >= S) return;

    const __nv_bfloat16* K_ptr = K + batch_id * num_heads * S * 128 + head_id * S * 128 + key_blk_start * 128;
    const __nv_bfloat16* V_ptr = V + batch_id * num_heads * S * 128 + head_id * S * 128 + key_blk_start * 128;
    __nv_bfloat16* dK_ptr = dK + batch_id * num_heads * S * 128 + head_id * S * 128 + key_blk_start * 128;
    __nv_bfloat16* dV_ptr = dV + batch_id * num_heads * S * 128 + head_id * S * 128 + key_blk_start * 128;

    int wg = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (threadIdx.x < 128) {
        if (threadIdx.x == 0) mbarrier_arrive_and_expect_tx_fn(bar_load, 8192 * 8);
        load_gmem_to_smem(K_ptr, smem_K_0, smem_K_1, 64, 128);
        load_gmem_to_smem(V_ptr, smem_V_0, smem_V_1, 64, 128);
    }
    fence_proxy_async_fn();
    mbarrier_wait_fn(bar_load, 0);

    wmma::fragment<wmma::accumulator, 8, 8, 16, float> C_frag_dK_0_wg[8];
    wmma::fragment<wmma::accumulator, 8, 8, 16, float> C_frag_dK_1_wg[8];
    wmma::fragment<wmma::accumulator, 8, 8, 16, float> C_frag_dV_0_wg[8];
    wmma::fragment<wmma::accumulator, 8, 8, 16, float> C_frag_dV_1_wg[8];
    for (int i=0; i<8; ++i) {
        wmma::fill_fragment(C_frag_dK_0_wg[i]);
        wmma::fill_fragment(C_frag_dK_1_wg[i]);
        wmma::fill_fragment(C_frag_dV_0_wg[i]);
        wmma::fill_fragment(C_frag_dV_1_wg[i]);
    }

    wmma::fragment<wmma::matrix_a, 8, 8, 16, __nv_bfloat16, wmma::col_major> A_frag_dS_wg[4];
    wmma::fragment<wmma::matrix_b, 8, 8, 16, __nv_bfloat16, wmma::row_major> B_frag_dK_wg[4];
    wmma::fragment<wmma::matrix_a, 8, 8, 16, __nv_bfloat16, wmma::col_major> A_frag_P_T_wg[4];
    wmma::fragment<wmma::matrix_b, 8, 8, 16, __nv_bfloat16, wmma::row_major> B_frag_dV_wg[4];

    for (int query_blk = 0; query_blk < (S + 63)/64; ++query_blk) {
        int query_blk_start = query_blk * 64;
        if (query_blk_start >= S) break;
        
        const __nv_bfloat16* Q_ptr = Q + batch_id * num_heads * S * 128 + head_id * S * 128 + query_blk_start * 128;
        const __nv_bfloat16* dO_ptr = dO + batch_id * num_heads * S * 128 + head_id * S * 128 + query_blk_start * 128;

        if (threadIdx.x < 128) {
            if (threadIdx.x == 0) mbarrier_arrive_and_expect_tx_fn(bar_load, 8192 * 8);
            load_gmem_to_smem(Q_ptr, smem_Q_0, smem_Q_1, 64, 128);
            load_gmem_to_smem(dO_ptr, smem_dO_0, smem_dO_1, 64, 128);
            if (threadIdx.x < 64) smem_L[threadIdx.x] = L_ptr[query_blk_start + threadIdx.x];
        }
        fence_proxy_async_fn();
        mbarrier_wait_fn(bar_load, 0);

        for (int col_tile = 0; col_tile < 8; ++col_tile) {
            wmma::fragment<wmma::accumulator, 8, 8, 16, float> C_frag_S_wg;
            wmma::fill_fragment(C_frag_S_wg);
            wmma::fragment<wmma::matrix_a, 8, 8, 16, __nv_bfloat16, wmma::col_major> A_frag_Q_wg[4];
            wmma::fragment<wmma::matrix_b, 8, 8, 16, __nv_bfloat16, wmma::col_major> B_frag_K_wg[4];
            for(int i=0; i<4; ++i) {
                load_A_col_major_8x16(A_frag_Q_wg[i], smem_Q_0 + wg * 8 * 64 + i * 16);
                load_B_col_major_8x16(B_frag_K_wg[i], smem_K_0 + col_tile * 8 * 64 + i * 16, smem_K_1 + col_tile * 8 * 64 + i * 16);
                wmma::mma_sync(C_frag_S_wg, A_frag_Q_wg[i], B_frag_K_wg[i]);
            }

            wmma::fragment<wmma::accumulator, 8, 8, 16, float> C_frag_D_wg;
            wmma::fill_fragment(C_frag_D_wg);
            wmma::fragment<wmma::matrix_a, 8, 8, 16, __nv_bfloat16, wmma::col_major> A_frag_dO_wg[4];
            wmma::fragment<wmma::matrix_b, 8, 8, 16, __nv_bfloat16, wmma::col_major> B_frag_V_wg[4];
            for(int i=0; i<4; ++i) {
                load_A_col_major_8x16(A_frag_dO_wg[i], smem_dO_0 + wg * 8 * 64 + i * 16);
                load_B_col_major_8x16(B_frag_V_wg[i], smem_V_0 + col_tile * 8 * 64 + i * 16, smem_V_1 + col_tile * 8 * 64 + i * 16);
                wmma::mma_sync(C_frag_D_wg, A_frag_dO_wg[i], B_frag_V_wg[i]);
            }

            float P_vals[8][8], D_vals[8][8];
            store_and_extract_S_floats(C_frag_S_wg, smem_dS, wg, P_vals, D_vals);
            store_and_extract_S_floats(C_frag_D_wg, smem_D, wg, P_vals, D_vals);
            
            float sum_P_D[8] = {0};
            float pd_local[8] = {0};
            for(int c=0; c<8; ++c) {
                for(int r=0; r<8; ++r) {
                    P_vals[r][c] = expf(P_vals[r][c] - smem_L[wg * 8 + r]);
                    pd_local[r] += P_vals[r][c] * D_vals[r][c];
                }
            }
            for(int r=0; r<8; ++r) {
                pd_local[r] += __shfl_down_sync(0xFFFFFFFF, pd_local[r], 2);
                pd_local[r] += __shfl_down_sync(0xFFFFFFFF, pd_local[r], 1);
                if (lane_id < 4) sum_P_D[r] += pd_local[r];
            }
            
            for(int c=0; c<8; ++c) {
                for(int r=0; r<8; ++r) {
                    float ds_val = P_vals[r][c] * (D_vals[r][c] - sum_P_D[r]) * scale;
                    int swizzled_x = (r / 8) ^ ((wg * 8 + c) % 8);
                    int swizzled_idx = (wg * 8 + c) * 64 + swizzled_x * 8 + (r % 8);
                    smem_dS[swizzled_idx] = __float2bfloat16(ds_val);
                    smem_P_T[swizzled_idx] = __float2bfloat16(P_vals[r][c]);
                }
            }
        }
        __syncthreads();
        fence_proxy_async_fn();

        for (int d_half = 0; d_half < 2; ++d_half) {
            for (int col_tile = 0; col_tile < 8; ++col_tile) {
                for (int k = 0; k < 4; ++k) {
                    load_A_col_major_8x16(A_frag_dS_wg[k], (d_half == 0) ? (smem_dS + k * 16 * 64 + col_tile * 8) : (smem_dS + k * 16 * 64 + col_tile * 8 + 8*64));
                    load_B_row_major_8x16(B_frag_dK_wg[k], 
                        (d_half == 0) ? (smem_Q_0 + k * 16 * 64 + wg * 8) : (smem_Q_1 + k * 16 * 64 + wg * 8),
                        (d_half == 0) ? (smem_Q_0 + k * 16 * 64 + wg * 8 + 8) : (smem_Q_1 + k * 16 * 64 + wg * 8 + 8));
                    wmma::mma_sync((d_half == 0) ? C_frag_dK_0_wg[col_tile] : C_frag_dK_1_wg[col_tile], A_frag_dS_wg[k], B_frag_dK_wg[k]);
                    
                    load_A_col_major_8x16(A_frag_P_T_wg[k], (d_half == 0) ? (smem_P_T + k * 16 * 64 + col_tile * 8) : (smem_P_T + k * 16 * 64 + col_tile * 8 + 8*64));
                    load_B_row_major_8x16(B_frag_dV_wg[k], 
                        (d_half == 0) ? (smem_dO_0 + k * 16 * 64 + wg * 8) : (smem_dO_1 + k * 16 * 64 + wg * 8),
                        (d_half == 0) ? (smem_dO_0 + k * 16 * 64 + wg * 8 + 8) : (smem_dO_1 + k * 16 * 64 + wg * 8 + 8));
                    wmma::mma_sync((d_half == 0) ? C_frag_dV_0_wg[col_tile] : C_frag_dV_1_wg[col_tile], A_frag_P_T_wg[k], B_frag_dV_wg[k]);
                }
            }
        }
        __syncthreads();
    }

    for (int d_half = 0; d_half < 2; ++d_half) {
        for (int col_tile = 0; col_tile < 8; ++col_tile) {
            float P_vals[8][8], D_vals[8][8];
            store_and_extract_S_floats((d_half == 0) ? C_frag_dK_0_wg[col_tile] : C_frag_dK_1_wg[col_tile], (d_half == 0) ? smem_K_0 : smem_K_1, wg, P_vals, D_vals);
            store_and_extract_S_floats((d_half == 0) ? C_frag_dV_0_wg[col_tile] : C_frag_dV_1_wg[col_tile], (d_half == 0) ? smem_V_0 : smem_V_1, wg, P_vals, D_vals);
        }
    }
    
    __syncthreads();
    store_smem_to_gmem(dK_ptr, smem_K_0, smem_K_1, 64, 128);
    store_smem_to_gmem(dV_ptr, smem_V_0, smem_V_1, 64, 128);
}

namespace tvm_ffi_mha_bwd {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    cudaDeviceProp dev;
    CUDA_CHECK(cudaGetDeviceProperties(&dev, Q.device().device_id));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 150000));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 150000));

    int64_t S = Q.size(2); 
    int64_t B_H = Q.size(0) * Q.size(1);
    dim3 grid((S + 63)/64, B_H);
    dim3 block(256);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    float scale = 1.0f / sqrtf((float)Q.size(3));
    
    bwd_dq_kernel<<<grid, block, 150000, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()), static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()), static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()), static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<const float*>(L.data_ptr()), S, scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    bwd_dkv_kernel<<<grid, block, 150000, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()), static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()), static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()), static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()), static_cast<const float*>(L.data_ptr()), S, scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

}  // namespace tvm_ffi_mha_bwd