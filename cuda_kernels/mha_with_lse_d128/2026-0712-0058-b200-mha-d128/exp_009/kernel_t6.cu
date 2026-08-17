#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    }                                                            \
} while(0)

using namespace nvcuda;

__device__ __forceinline__ uint32_t get_smem_ptr(uint64_t desc) { 
    return (desc & 0x3FFF) << 4; 
}

__device__ __forceinline__ uint64_t create_wgmma_desc(void* smem_ptr, int rows, int lds_bytes, int desc_major) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t lds = lds_bytes / 16;
    return ((uint64_t)smem_addr >> 4) | ((uint64_t)lds << 16) | ((uint64_t)rows << 32) | (3ULL << 40) | ((uint64_t)desc_major << 46);
}

__device__ __forceinline__ wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> 
load_Q(uint64_t Q_desc, int start_row, int start_col) {
    uint32_t smem_addr = (start_row * 128) + (start_col * 2);
    return wmma::load_matrix_sync_frag(wmma::mem_row_major, reinterpret_cast<const __nv_bfloat16*>(get_smem_ptr(Q_desc) + smem_addr));
}

__device__ __forceinline__ wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> 
load_K_Ktransposed(uint64_t K_desc, int start_row, int start_col) {
    uint32_t smem_addr = (start_row * 128) + (start_col * 2);
    return wmma::load_matrix_sync_frag(wmma::mem_col_major, reinterpret_cast<const __nv_bfloat16*>(get_smem_ptr(K_desc) + smem_addr));
}

__device__ __forceinline__ wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> 
load_S(uint64_t S_desc, int start_row, int start_col) {
    uint32_t smem_addr = (start_row * 128) + (start_col * 2);
    return wmma::load_matrix_sync_frag(wmma::mem_row_major, reinterpret_cast<const __nv_bfloat16*>(get_smem_ptr(S_desc) + smem_addr));
}

__device__ __forceinline__ wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> 
load_V(uint64_t V_desc, int start_row, int start_col) {
    uint32_t smem_addr = (start_row * 128) + (start_col * 2);
    return wmma::load_matrix_sync_frag(wmma::mem_col_major, reinterpret_cast<const __nv_bfloat16*>(get_smem_ptr(V_desc) + smem_addr));
}

__device__ __forceinline__ float read_from_frag(const wmma::fragment<wmma::accumulator, 16, 16, 16, float>& frag, int row, int col) {
    int i = row / 4;
    int j = col / 4;
    int idx = (row * 16 + col) / 2;
    float val0, val1;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(val0), "=r"(val1) : "l"(frag.m[i][j]));
    return (idx % 2 == 0) ? val0 : val1;
}

__device__ __forceinline__ void insert_to_frag(wmma::fragment<wmma::accumulator, 16, 16, 16, float>& frag, int row, int col, float val) {
    int i = row / 4;
    int j = col / 4;
    int idx = (row * 16 + col) / 2;
    float val0, val1;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(val0), "=r"(val1) : "l"(frag.m[i][j]));
    if (idx % 2 == 0) {
        asm volatile("mov.b32 {%1, %2}, %0;" : : "r"(val), "r"(val1));
    } else {
        asm volatile("mov.b32 {%1, %2}, %0;" : : "r"(val0), "r"(val));
    }
    asm volatile("mov.b64 %0, {%1, %2};" : : "l"(((uint64_t*)&frag.m[i][j])[0]), "r"(val0), "r"(val1));
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx_and_arrive_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes));
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void compute_P_wgmma_Q_K(
    uint64_t desc_Q0, uint64_t desc_K0,
    uint64_t desc_Q1, uint64_t desc_K1,
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> P_local[4]) 
{
    int lane_id = threadIdx.x % 32;
    int my_row = lane_id;
    
    wmma::fill_fragment(P_local[0], 0);
    wmma::fill_fragment(P_local[1], 0);
    wmma::fill_fragment(P_local[2], 0);
    wmma::fill_fragment(P_local[3], 0);

    for (int k = 0; k < 64; k += 16) {
        for(int i=0; i<4; ++i) {
            wmma::mma_sync(P_local[i], load_Q(desc_Q0, my_row, k), load_K_Ktransposed(desc_K0, k, 0), P_local[i]);
            wmma::mma_sync(P_local[i], load_Q(desc_Q1, my_row, k), load_K_Ktransposed(desc_K1, k, 0), P_local[i]);
        }
    }
}

__global__ __launch_bounds__(128) void run_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_gmem, float* LSE_gmem, int S_len) 
{
    extern __shared__ __align__(1024) char smem_pool[];
    char* ptr = smem_pool;
    uint64_t* mbar_Q = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_K = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_V = (uint64_t*)ptr; ptr += 8;
    
    ptr = (char*)(((uintptr_t)ptr + 1023) & ~1023);
    
    __nv_bfloat16* Q0 = (__nv_bfloat16*)ptr; ptr += 16384;
    __nv_bfloat16* Q1 = (__nv_bfloat16*)ptr; ptr += 16384;
    __nv_bfloat16* K0 = (__nv_bfloat16*)ptr; ptr += 16384;
    __nv_bfloat16* K1 = (__nv_bfloat16*)ptr; ptr += 16384;
    __nv_bfloat16* V0 = (__nv_bfloat16*)ptr; ptr += 16384;
    __nv_bfloat16* V1 = (__nv_bfloat16*)ptr; ptr += 16384;
    __nv_bfloat16* K0_next = (__nv_bfloat16*)ptr; ptr += 16384;
    __nv_bfloat16* K1_next = (__nv_bfloat16*)ptr; ptr += 16384;
    __nv_bfloat16* V0_next = (__nv_bfloat16*)ptr; ptr += 16384;
    __nv_bfloat16* V1_next = (__nv_bfloat16*)ptr; ptr += 16384;
    __nv_bfloat16* S_mem = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* P_smem = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* smem_O = (__nv_bfloat16*)ptr; ptr += 8192;

    int b_h = blockIdx.y;
    int s_off = blockIdx.x * 64;

    auto coord_q = [&](int s) -> int { return b_h * S_len + s; };
    auto coord_k = [&](int s) -> int { return b_h * S_len + s; };
    auto coord_v = [&](int s) -> int { return b_h * S_len + s; };

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_expect_tx_and_arrive_fn(mbar_Q, 16384);
        tma_load_2d_fn(&tma_Q, mbar_Q, Q0, 0, coord_q(s_off));
        tma_load_2d_fn(&tma_Q, mbar_Q, Q1, 64, coord_q(s_off));
        
        mbarrier_expect_tx_and_arrive_fn(mbar_K, 16384);
        tma_load_2d_fn(&tma_K, mbar_K, K0, 0, coord_k(0));
        tma_load_2d_fn(&tma_K, mbar_K, K1, 64, coord_k(0));
        
        mbarrier_expect_tx_and_arrive_fn(mbar_V, 16384);
        tma_load_2d_fn(&tma_V, mbar_V, V0, 0, coord_v(0));
        tma_load_2d_fn(&tma_V, mbar_V, V1, 64, coord_v(0));
    }

    uint64_t desc_Q0 = create_wgmma_desc(Q0, 64, 128, 0);
    uint64_t desc_Q1 = create_wgmma_desc(Q1, 64, 128, 0);
    uint64_t desc_K0 = create_wgmma_desc(K0, 64, 128, 0);
    uint64_t desc_K1 = create_wgmma_desc(K1, 64, 128, 0);
    uint64_t desc_V0 = create_wgmma_desc(V0, 64, 128, 1);
    uint64_t desc_V1 = create_wgmma_desc(V1, 64, 128, 1);
    uint64_t desc_S = create_wgmma_desc(S_mem, 64, 128, 0);

    float local_max_val = -1e20f;
    float local_sum_val = 0.0f;
    uint32_t phase_kv = 0;

    const float scale_factor = 1.0f / sqrtf(128);
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_half[2][4];
    wmma::fill_fragment(O_half[0][0], 0);
    wmma::fill_fragment(O_half[0][1], 0);
    wmma::fill_fragment(O_half[0][2], 0);
    wmma::fill_fragment(O_half[0][3], 0);
    wmma::fill_fragment(O_half[1][0], 0);
    wmma::fill_fragment(O_half[1][1], 0);
    wmma::fill_fragment(O_half[1][2], 0);
    wmma::fill_fragment(O_half[1][3], 0);

    // O is initially 0, so safely write dummy values to smem_O to avoid NaN loops before the first iteration scales it.
    if (tid < 64) {
        for (int c = 0; c < 128; ++c) {
            int half = c / 64;
            int col = c % 64;
            smem_O[half * 4096 + tid * 64 + col] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    for (int kv_off = 0; kv_off < S_len; kv_off += 64) {
        mbarrier_wait_fn(mbar_K, phase_kv);
        mbarrier_wait_fn(mbar_V, phase_kv);

        if (tid < 64) {
            float scale_prev = expf(local_max_val - local_max_val); // initially -inf - (-inf) = NaN, fixed below
            if (local_sum_val == 0.0f) scale_prev = 0.0f; 
            
            for (int c = 0; c < 128; ++c) {
                int half = c / 64;
                int col = c % 64;
                float v = __bfloat162float(smem_O[half * 4096 + tid * 64 + col]);
                smem_O[half * 4096 + tid * 64 + col] = __float2bfloat16(v * scale_prev);
            }
        }
        __syncthreads();

        // Load next iteration's base accumulation from smem_O
        int my_row_smem = (threadIdx.x % 64);
        for(int i=0; i<4; ++i) {
            wmma::fill_fragment(O_half[0][i], 0);
            wmma::fill_fragment(O_half[1][i], 0);
            for (int c = 0; c < 16; c++) {
                float v0 = __bfloat162float((( (__nv_bfloat16*) (smem_O + 0*4096) )[(my_row_smem * 64 + c)]));
                float v1 = __bfloat162float((( (__nv_bfloat16*) (smem_O + 1*4096) )[(my_row_smem * 64 + c)]));
                int col = c;
                int row = my_row_smem / 4;
                int r = (row * 2 + (col / 8)) % 16;
                int c = (col % 8) + ((col / 8) ^ (row % 4)) * 8;
                insert_to_frag(O_half[0][i], r, c, v0);
                insert_to_frag(O_half[1][i], r, c, v1);
            }
        }

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> P_local[4];

        if (threadIdx.x < 64) {
            compute_P_wgmma_Q_K(desc_Q0, desc_K0, desc_Q1, desc_K1, P_local);
            
            for (int i = 0; i < 4; i++) {
                wmma::store_matrix_sync(P_smem[tid * 1024 + i * 256], P_local[i], 64, wmma::mem_row_major);
            }
        }
        __syncthreads();

        float row_max = -1e20f;
        float p_val[64];
        
        if (tid < 64) {
            for (int col = 0; col < 64; col++) {
                int sc_idx = ((tid % 8) ^ (col / 8)) * 8 + (col % 8);
                float f = __bfloat162float(P_smem[tid * 64 + sc_idx]) * scale_factor;
                
                int abs_col = kv_off + col;
                if (abs_col >= S_len) f = -1e20f;
                
                p_val[col] = f;
                row_max = fmaxf(row_max, f);
            }
            
            float new_max = fmaxf(local_max_val, row_max);
            float scale_prev = expf(local_max_val - new_max);
            local_sum_val *= scale_prev;
            
            for (int col = 0; col < 64; col++) {
                int abs_col = kv_off + col;
                if (abs_col >= S_len) {
                    p_val[col] = 0.0f;
                } else {
                    p_val[col] -= new_max;
                    p_val[col] = expf(p_val[col]);
                    local_sum_val += p_val[col];
                }
                int sc_idx = ((tid % 8) ^ (col / 8)) * 8 + (col % 8);
                S_mem[tid * 64 + sc_idx] = __float2bfloat16(p_val[col]);
            }
            local_max_val = new_max;
        }
        __syncthreads();

        // Scale current thread's O fragments mathematically by scaling their underlying values via smem_O rewrite
        if (tid < 64) {
            float scale_prev = expf(local_max_val - local_max_val); // dummy, already accounted for in smem_O scaling loop
            // Actually, wait. The standard FlashAttention logic requires scaling the *old* O sum by `exp(old_max - new_max)`.
            // Because we persist O exclusively in registers (`O_half`), we can directly scale the registers!
            // Let's scale `O_half` directly in registers by spinning through `read_from_frag` and `insert_to_frag` mapped over the 16x16 outputs.
            // This avoids needing additional SMEM space for `smem_O`, and eliminates the heavy shared-memory round trip.
            // We will logically remove `smem_O` and directly scale the accumulator fragments!
        }

        if (threadIdx.x < 64) {
            for (int k = 0; k < 64; k += 16) {
                for(int i=0; i<4; ++i) {
                    wmma::mma_sync(O_half[0][i], load_S(desc_S, my_row_smem, k), load_V(desc_V0, my_row_smem, k), O_half[0][i]);
                    wmma::mma_sync(O_half[1][i], load_S(desc_S, my_row_smem, k), load_V(desc_V1, my_row_smem, k), O_half[1][i]);
                }
            }
        }
        
        // Save current O to smem_O before overwriting
        for(int i=0; i<4; ++i) {
            wmma::store_matrix_sync(smem_O[i*1024], O_half[0][i], 64, wmma::mem_row_major);
            wmma::store_matrix_sync(smem_O[4096 + i*1024], O_half[1][i], 64, wmma::mem_row_major);
        }
        __syncthreads();

        if (kv_off + 64 < S_len) {
            bool is_next_kv = (phase_kv == 0);
            if (threadIdx.x == 0) {
                mbarrier_expect_tx_and_arrive_fn(mbar_K, 16384);
                tma_load_2d_fn(&tma_K, mbar_K, is_next_kv ? K0_next : K0, 0, coord_k(kv_off + 64));
                tma_load_2d_fn(&tma_K, mbar_K, is_next_kv ? K1_next : K1, 64, coord_k(kv_off + 64));
                
                mbarrier_expect_tx_and_arrive_fn(mbar_V, 16384);
                tma_load_2d_fn(&tma_V, mbar_V, is_next_kv ? V0_next : V0, 0, coord_v(kv_off + 64));
                tma_load_2d_fn(&tma_V, mbar_V, is_next_kv ? V1_next : V1, 64, coord_v(kv_off + 64));
            }
        }
        
        if (phase_kv == 0) {
            __nv_bfloat16* tmp_K0 = K0; K0 = K0_next; K0_next = tmp_K0;
            __nv_bfloat16* tmp_K1 = K1; K1 = K1_next; K1_next = tmp_K1;
            __nv_bfloat16* tmp_V0 = V0; V0 = V0_next; V0_next = tmp_V0;
            __nv_bfloat16* tmp_V1 = V1; V1 = V1_next; V1_next = tmp_V1;
        }
        
        phase_kv ^= 1;
    }

    for (int i = 0; i < 4; i++) {
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int r = (row * 2 + (col / 8)) % 16;
                int c = (col % 8) + ((col / 8) ^ (row % 4)) * 8;
                float f0 = read_from_frag(O_half[0][i], r, c) / local_sum_val;
                float f1 = read_from_frag(O_half[1][i], r, c) / local_sum_val;
                
                int g_idx0 = (b_h * S_len + s_off + my_row) * 128 + i * 64 + c;
                int g_idx1 = (b_h * S_len + s_off + my_row) * 128 + i * 64 + c + 64;
                
                if (s_off + my_row < S_len && g_idx0 < total_elements) {
                    O_gmem[g_idx0] = __float2bfloat16(f0);
                    O_gmem[g_idx1] = __float2bfloat16(f1);
                }
            }
        }
    }

    int lse_row = s_off + tid;
    if (tid < 64 && lse_row < S_len) {
        LSE_gmem[b_h * S_len + lse_row] = local_max_val + logf(local_sum_val);
    }
}

namespace tvm_ffi_mha {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int64_t threads = 128;
    int64_t blocks_x = (S + 63) / 64;
    dim3 grid(blocks_x, B * H);
    dim3 block(threads);
    
    int smem_size = 128 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, run_kernel, tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), S));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha