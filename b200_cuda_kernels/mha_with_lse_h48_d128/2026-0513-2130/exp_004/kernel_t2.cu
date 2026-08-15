#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace tvm_ffi_cuda {

__device__ __forceinline__ void setmaxnreg_inc_sync_fn_248() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 248;" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_mma_cg1_f16(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.b64"
        " [%0];"
        :: "r"(a)); 
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_none_k_major_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // NONE
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_none_mn_major_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_custom_fn(uint32_t M, uint32_t N, bool trans_a, bool trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    if (trans_a) d |= (1u << 15);
    if (trans_b) d |= (1u << 16);
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint64_t update_desc_addr_none(uint64_t desc, uint32_t new_addr_bytes) {
    uint32_t addr_16b = (new_addr_bytes >> 4) & 0x3FFF;
    desc = (desc & ~0x3FFF) | addr_16b;
    return desc;
}

extern __shared__ __align__(1024) uint8_t smem[];

__global__ void __launch_bounds__(128, 1) mha_fwd_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    int S, int H, int B
) {
    setmaxnreg_inc_sync_fn_248();

    int m_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;

    uint32_t smem_Q_addr = 0;
    uint32_t smem_K_addr = 32768;
    uint32_t smem_V_addr = 65536;
    uint32_t smem_P_addr = 98304;

    __nv_bfloat16* smem_q = (__nv_bfloat16*)(smem + smem_Q_addr);
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem + smem_K_addr);
    __nv_bfloat16* smem_v = (__nv_bfloat16*)(smem + smem_V_addr);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem + smem_P_addr);

    uint64_t* mbar_Q = (uint64_t*)(smem + 128 * 1024);
    uint64_t* mbar_K = (uint64_t*)(smem + 128 * 1024 + 8);
    uint64_t* mbar_V = (uint64_t*)(smem + 128 * 1024 + 16);
    uint64_t* mbar_MMA = (uint64_t*)(smem + 128 * 1024 + 24);

    __shared__ uint32_t tmem_base;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_MMA, 1);
    }
    
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&tmem_base, 256);
    }
    
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t S_tmem = tmem_base;
    uint32_t O_tmem = tmem_base + 128;

    uint32_t cvt_smem_Q = (uint32_t)__cvta_generic_to_shared(smem_q);
    uint32_t cvt_smem_K = (uint32_t)__cvta_generic_to_shared(smem_k);
    uint32_t cvt_smem_V = (uint32_t)__cvta_generic_to_shared(smem_v);
    uint32_t cvt_smem_P = (uint32_t)__cvta_generic_to_shared(smem_P);

    uint64_t desc_Q_base = make_smem_desc_sm100_none_k_major_fn(smem_q, 16, 2048);
    uint64_t desc_K_base = make_smem_desc_sm100_none_k_major_fn(smem_k, 16, 2048);
    uint64_t desc_V_base = make_smem_desc_sm100_none_mn_major_fn(smem_v, 128, 256);
    uint64_t desc_P_base = make_smem_desc_sm100_none_k_major_fn(smem_P, 16, 2048);

    uint32_t idesc_QK = make_instr_desc_custom_fn(128, 128, false, false);
    uint32_t idesc_PV = make_instr_desc_custom_fn(128, 128, false, true);

    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_MMA = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_q, 0, m_block * 128, b * H + h);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q);
    phase_Q ^= 1;
    fence_proxy_async_fn();
    tcgen05_fence_after_fn();

    float O_reg[128];
    for (int i = 0; i < 128; i++) O_reg[i] = 0.0f;
    float m_val = -1e20f;
    float l_val = 0.0f;
    float scale = 0.0883883476f; // 1 / sqrt(128)

    int num_n_blocks = (S + 127) / 128;
    for (int n_block = 0; n_block < num_n_blocks; n_block++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 32768);
            tma_load_3d_fn(&tma_K, mbar_K, smem_k, 0, n_block * 128, b * H + h);
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 32768);
            tma_load_3d_fn(&tma_V, mbar_V, smem_v, 0, n_block * 128, b * H + h);
        }
        
        mbarrier_wait_fn(mbar_K, phase_K);
        phase_K ^= 1;
        fence_proxy_async_fn();
        tcgen05_fence_after_fn();

        if (threadIdx.x == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint64_t dQ = update_desc_addr_none(desc_Q_base, cvt_smem_Q + k * 2);
                uint64_t dK = update_desc_addr_none(desc_K_base, cvt_smem_K + k * 2);
                tcgen05_mma_cg1_f16(S_tmem, dQ, dK, idesc_QK, (k == 0 ? 0 : 1));
            }
            tcgen05_commit_cg1_fn(mbar_MMA);
        }
        mbarrier_wait_fn(mbar_MMA, phase_MMA);
        phase_MMA ^= 1;

        float row_max = -1e20f;
        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(S_tmem + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; i++) {
                float val = __uint_as_float(r[i]) * scale;
                uint32_t k_idx = n_block * 128 + c + i;
                if (k_idx >= S) val = -1e20f;
                row_max = fmaxf(row_max, val);
            }
        }

        float new_m = fmaxf(m_val, row_max);
        float rescale = fast_exp2f_fn((m_val - new_m) * 1.44269504f);
        m_val = new_m;
        for (int i = 0; i < 128; i++) O_reg[i] *= rescale;
        l_val *= rescale;

        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(S_tmem + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            uint32_t p_packed[4];
            for (int i = 0; i < 8; i += 2) {
                float v0 = __uint_as_float(r[i]) * scale;
                float v1 = __uint_as_float(r[i+1]) * scale;
                uint32_t k_idx0 = n_block * 128 + c + i;
                uint32_t k_idx1 = n_block * 128 + c + i + 1;
                if (k_idx0 >= S) v0 = -1e20f;
                if (k_idx1 >= S) v1 = -1e20f;
                float p0 = fast_exp2f_fn((v0 - m_val) * 1.44269504f);
                float p1 = fast_exp2f_fn((v1 - m_val) * 1.44269504f);
                l_val += p0 + p1;
                p_packed[i/2] = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            }
            uint32_t y = threadIdx.x;
            uint32_t smem_addr = smem_P_addr + y * 256 + c * 2;
            st_shared_128_fn(smem_addr, p_packed[0], p_packed[1], p_packed[2], p_packed[3]);
        }
        
        __syncthreads();
        fence_async_shared_fn();

        mbarrier_wait_fn(mbar_V, phase_V);
        phase_V ^= 1;
        fence_proxy_async_fn();
        tcgen05_fence_after_fn();

        if (threadIdx.x == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint64_t dP = update_desc_addr_none(desc_P_base, cvt_smem_P + k * 2);
                uint64_t dV = update_desc_addr_none(desc_V_base, cvt_smem_V + k * 256);
                tcgen05_mma_cg1_f16(O_tmem, dP, dV, idesc_PV, (k == 0 ? 0 : 1));
            }
            tcgen05_commit_cg1_fn(mbar_MMA);
        }
        mbarrier_wait_fn(mbar_MMA, phase_MMA);
        phase_MMA ^= 1;

        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(O_tmem + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; i++) {
                O_reg[c+i] += __uint_as_float(r[i]);
            }
        }
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }

    float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
    for (int i = 0; i < 128; i++) {
        O_reg[i] *= inv_l;
    }

    for (int c = 0; c < 128; c += 8) {
        uint32_t p0 = pack_bf16_fn(__float_as_uint(O_reg[c]), __float_as_uint(O_reg[c+1]));
        uint32_t p1 = pack_bf16_fn(__float_as_uint(O_reg[c+2]), __float_as_uint(O_reg[c+3]));
        uint32_t p2 = pack_bf16_fn(__float_as_uint(O_reg[c+4]), __float_as_uint(O_reg[c+5]));
        uint32_t p3 = pack_bf16_fn(__float_as_uint(O_reg[c+6]), __float_as_uint(O_reg[c+7]));
        uint32_t smem_addr = smem_Q_addr + threadIdx.x * 256 + c * 2;
        st_shared_128_fn(smem_addr, p0, p1, p2, p3);
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (int step = 0; step < 128 / 4; step++) {
        uint32_t r = step * 4 + warp_id;
        if (r >= 128) continue;
        uint32_t global_row = m_block * 128 + r;
        if (global_row < S) {
            uint32_t col_start = lane_id * 4;
            uint2 data = *reinterpret_cast<uint2*>(smem_q + r * 128 + col_start);
            *reinterpret_cast<uint2*>(O + b * H * S * 128 + h * S * 128 + global_row * 128 + col_start) = data;
        }
    }

    uint32_t global_row = m_block * 128 + threadIdx.x;
    if (global_row < S) {
        LSE[b * H * S + h * S + global_row] = m_val + logf(l_val);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint32_t smem_dim0, uint32_t smem_dim1, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2}; 
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2}; 
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, 1}; 
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    void* o_ptr = O.data_ptr();
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    CUresult res;
    res = create_tma_3d_descriptor_2B(&tma_Q, q_ptr, 128, S, B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    if(res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    res = create_tma_3d_descriptor_2B(&tma_K, k_ptr, 128, S, B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    if(res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    res = create_tma_3d_descriptor_2B(&tma_V, v_ptr, 128, S, B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    if(res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }

    CUDA_CHECK(cudaFuncSetAttribute(
        mha_fwd_sm100_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        128 * 1024 + 1024));

    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 128 * 1024 + 1024;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, mha_fwd_sm100_kernel, tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(o_ptr), lse_ptr, S, H, B);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_cuda::run);

}