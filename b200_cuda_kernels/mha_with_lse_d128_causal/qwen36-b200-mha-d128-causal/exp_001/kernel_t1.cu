#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

// ---- Inline Helpers from Reference Material ----

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm ("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xFFFFFFFF;\nselp.b32 %0, 1, 0, p;\n}\n" : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_cluster_fn(uint64_t* bar, uint32_t tx, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;" :: "r"(remote_a), "r"(tx));
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n" :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
                 :: "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(smem)), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void tma_store_wait_fn() { asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cg2_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                 :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ void tcgen05_fence_before_fn() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tmem_load_fence_fn() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transpose_a, bool transpose_b) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= ((transpose_a ? 1u : 0u) << 15);
    d |= ((transpose_b ? 1u : 0u) << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__host__ __forceinline__ CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, CUtensorMapDataType dataType, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

namespace attention_impl {

__global__ void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE,
    int B, int H, int S, int D,
    float inv_sqrt_d)
{
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int NUM_THREADS = 64;
    
    extern __shared__ char smem_base[];

    int off = 0;
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_base + off); off += (BM * BK + 127) & ~127;
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(smem_base + off); off += (BN * BK + 127) & ~127;
    __nv_bfloat16* smem_V = reinterpret_cast<__nv_bfloat16*>(smem_base + off); off += (BN * D + 127) & ~127;
    float* smem_P = reinterpret_cast<float*>(smem_base + off); off += (BM * BN) * sizeof(float);
    uint64_t* bar_Q = reinterpret_cast<uint64_t*>(smem_base + off); off += 8;
    uint64_t* bar_K = reinterpret_cast<uint64_t*>(smem_base + off); off += 8;
    uint64_t* bar_V = reinterpret_cast<uint64_t*>(smem_base + off); off += 8;
    uint32_t* tmem_ptrs = reinterpret_cast<uint32_t*>(smem_base + off); off += 4;
    
    int tid = threadIdx.x;
    int q_start = blockIdx.z * BM;
    if (q_start >= S) return;
    int q_end = min(q_start + BM, S);
    int eff_bm = q_end - q_start;

    int b = blockIdx.x / gridDim.y;
    int h = blockIdx.x % gridDim.y;
    
    float row_max[BM] = {-1e20f};
    float row_sum[BM] = {0.0f};
    if (tid < BM) {
        row_max[tid] = -1e20f;
        row_sum[tid] = 0.0f;
    }

    uint32_t tmem_s_addr;
    if (tid == 0) {
        tmem_alloc_fn(tmem_ptrs, 64);
        tmem_s_addr = tmem_ptrs[0];
    }
    __syncthreads();

    init_smem_barrier_fn(bar_Q, 1);
    init_smem_barrier_fn(bar_K, 1);
    init_smem_barrier_fn(bar_V, 1);
    fence_smem_barrier_init_fn();

    int num_k_steps = (S + BN - 1) / BN;
    bool first_iter = true;
    
    uint32_t idesc_qk = make_instr_desc_fn(BM, BN, false, false);
    
    for (int ks = 0; ks < num_k_steps; ++ks) {
        int k_start = ks * BN;
        int k_end = min(k_start + BN, S);
        int eff_bn = k_end - k_start;
        
        if (tid == 0) {
            tma_load_2d_fn(&tma_Q, bar_Q, smem_Q, q_start, 0);
            tma_load_2d_fn(&tma_K, bar_K, smem_K, k_start, 0);
            tma_load_2d_fn(&tma_V, bar_V, smem_V, k_start, 0);
        }
        mbarrier_wait_fn(bar_Q, 0);
        mbarrier_wait_fn(bar_K, 0);
        mbarrier_wait_fn(bar_V, 0);

        uint64_t desc_q = make_smem_desc_sm100_fn(smem_Q, 1, BK * 2);
        uint64_t desc_k = make_smem_desc_sm100_fn(smem_K, 1, BN * 2);
        
        uint32_t d_addr = tmem_s_addr;
        if (tid == 0) {
            umma_f16_cg2_fn(d_addr, desc_q, desc_k, idesc_qk, first_iter ? 0 : 1);
            tcgen05_fence_before_fn();
            umma_commit_2sm_fn(bar_Q);
            mbarrier_wait_fn(bar_Q, 0);
            first_iter = false;
        }
        __syncthreads();

        for (int r = tid; r < eff_bm; r += NUM_THREADS) {
            int qr = q_start + r;
            float cur_row_max = -1e20f;
            float s_vals[64];
            
            for (int c = 0; c < eff_bn; ++c) {
                uint32_t val_int;
                uint32_t addr = (r << 16) | c;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];" : "=r"(val_int) : "r"(addr));
                tmem_load_fence_fn();
                float s = __uint_as_float(val_int);
                
                int kr = k_start + c;
                if (kr > qr) s = -1e20f;
                else s *= inv_sqrt_d;
                
                s_vals[c] = s;
                if (s > cur_row_max) cur_row_max = s;
            }
            
            float old_max = row_max[r];
            float tmp_max = -1e20f;
            for(int c=0; c<eff_bn; ++c) {
                float v = s_vals[c];
                if(v > tmp_max) tmp_max = v;
            }
            row_max[r] = fmaxf(old_max, tmp_max);
            float scale_old = expf(old_max - row_max[r]);
            float scale_new = expf(tmp_max - row_max[r]);
            float p_sum = 0.0f;
            
            for(int c=0; c<eff_bn; ++c) {
                float p = expf(s_vals[c] - row_max[r]);
                p_sum += p;
                smem_P[r * BN + c] = p;
            }
            row_sum[r] = row_sum[r] * scale_old + p_sum * scale_new;
        }
        __syncthreads();
    }
    
    for (int r = tid; r < eff_bm; ++r += NUM_THREADS) {
        float lse = row_max[r] + logf(row_sum[r]);
        LSE[((int64_t)b * H + h) * S + (q_start + r)] = lse;
    }
    
    if (tid == 0) tmem_dealloc_fn(tmem_s_addr, 64);
}

void run(tvm::ffi::TensorView Q_in, tvm::ffi::TensorView K_in, tvm::ffi::TensorView V_in, 
         tvm::ffi::TensorView O_out, tvm::ffi::TensorView LSE_out) {
    CUDA_CHECK(cudaSetDevice(Q_in.device().device_id));
    
    int64_t B = Q_in.size(0), H = Q_in.size(1), S = Q_in.size(2), D = Q_in.size(3);
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q_in.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K_in.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V_in.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O_out.data_ptr());
    float* LSE_data = static_cast<float*>(LSE_out.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q_in.device().device_type, Q_in.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    constexpr int BM = 64, BN = 64, BK = 16;
    
    create_tma_2d_descriptor_2B(&tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, (void*)Q_data, D, S, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, (void*)K_data, D, S, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, (void*)V_data, D, S, D, BN, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, (void*)O_data, D, S, D, BM, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    dim3 grid((B * H + 1) / 2, 2, (S + BM - 1) / BM);
    dim3 block(64);
    size_t smem_size = (BM*BK*2 + BN*BK*2 + BN*D*2 + BM*BN*4 + 8*3 + 4);
    smem_size = (smem_size + 127) & ~127;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    float inv_sqrt_d = rsqrtf((float)D);
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_kernel, 
        tma_Q, tma_K, tma_V, tma_O, 
        Q_data, K_data, V_data, O_data, LSE_data, 
        B, H, S, D, inv_sqrt_d));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_impl::run);

}  // namespace attention_impl