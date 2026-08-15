#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1); } } while(0)

namespace attn_kernel {

constexpr int D = 128;
constexpr int BM = 128;
constexpr int BN = 64;
constexpr int NUM_THREADS = 128;

__device__ __forceinline__ void cp_async_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void st_shared_128(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16(float x, float y) {
    __nv_bfloat16 bx = __float2bfloat16_rn(x);
    __nv_bfloat16 by = __float2bfloat16_rn(y);
    uint16_t ux = *reinterpret_cast<uint16_t*>(&bx);
    uint16_t uy = *reinterpret_cast<uint16_t*>(&by);
    return (uint32_t)ux | ((uint32_t)uy << 16);
}

__device__ __forceinline__ float fast_expf(float x) {
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f)); return y;
}
__device__ __forceinline__ float fast_logf(float x) {
    float y; asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y * 0.6931471805599453f;
}

__device__ __forceinline__ float warp_max4(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return v;
}
__device__ __forceinline__ float warp_sum4(float v) {
    v = v + __shfl_xor_sync(0xffffffff, v, 1);
    v = v + __shfl_xor_sync(0xffffffff, v, 2);
    return v;
}

// TMEM operations
__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc_cg1(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}
__device__ __forceinline__ void tmem_store_4x(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        :: "r"(r0),"r"(r1),"r"(r2),"r"(r3), "r"(col));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// UMMA
__device__ __forceinline__ void umma_cg1_smem_a(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void umma_cg1_tmem_a(uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_d), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

// Mbarrier
__device__ __forceinline__ void init_barrier(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}
__device__ __forceinline__ void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

// SMEM descriptor (no swizzle)
__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    return d; // swizzle=0 (no swizzle)
}

// Instruction descriptor
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N, bool trans_a, bool trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);    // dtype=FP32
    d |= (1u << 7);    // atype=BF16
    d |= (1u << 10);   // btype=BF16
    d |= ((uint32_t)trans_a << 15);
    d |= ((uint32_t)trans_b << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void load_kv_tile(
    const __nv_bfloat16* K_bh, const __nv_bfloat16* V_bh,
    uint32_t K_base, uint32_t V_base, int kb_start, int S, int tid) {
    // 64*128 bf16 = 1024 x 16B chunks, 128 threads -> 8 each
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int chunk = i * 128 + tid;
        int key = chunk / 16;
        int d_bf16 = (chunk % 16) * 8;
        int gkey = kb_start + key;
        uint32_t k_addr = K_base + (uint32_t)(key * 128 + d_bf16) * 2;
        uint32_t v_addr = V_base + (uint32_t)(key * 128 + d_bf16) * 2;
        if (gkey < S) {
            cp_async_16B(k_addr, &K_bh[(size_t)gkey * 128 + d_bf16]);
            cp_async_16B(v_addr, &V_bh[(size_t)gkey * 128 + d_bf16]);
        } else {
            st_shared_128(k_addr, 0, 0, 0, 0);
            st_shared_128(v_addr, 0, 0, 0, 0);
        }
    }
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void attnKernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int bh = blockIdx.x;
    int q_block = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int q_start = q_block * BM;
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    size_t bh_off = (size_t)(b * H + h) * (size_t)S * (size_t)D;
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    __nv_bfloat16* O_bh = O + bh_off;
    float* LSE_bh = LSE + (size_t)(b * H + h) * (size_t)S;

    extern __shared__ char smem[];
    // Layout: mbar(8) + tmem_addrs(8) + Q(32KB) + K[2](32KB) + V[2](32KB)
    uint64_t* mbar = (uint64_t*)smem;
    uint32_t* tmem_addrs = (uint32_t*)(smem + 8);
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)(smem + 16);
    __nv_bfloat16* K_smem_0 = Q_smem + BM * D;
    __nv_bfloat16* V_smem_0 = K_smem_0 + BN * D;
    __nv_bfloat16* K_smem_1 = V_smem_0 + BN * D;
    __nv_bfloat16* V_smem_1 = K_smem_1 + BN * D;

    uint32_t Q_base = (uint32_t)__cvta_generic_to_shared(Q_smem);
    uint32_t K_base[2] = {
        (uint32_t)__cvta_generic_to_shared(K_smem_0),
        (uint32_t)__cvta_generic_to_shared(K_smem_1)
    };
    uint32_t V_base[2] = {
        (uint32_t)__cvta_generic_to_shared(V_smem_0),
        (uint32_t)__cvta_generic_to_shared(V_smem_1)
    };

    // Init mbarrier
    if (tid == 0) {
        init_barrier(mbar, 1);
        fence_barrier_init();
    }
    __syncthreads();

    // Allocate TMEM: S/P (64 cols) + O (128 cols)
    if (warp == 0) {
        tmem_alloc_cg1(&tmem_addrs[0], 64);
        tmem_alloc_cg1(&tmem_addrs[1], 128);
    }
    __syncthreads();
    uint32_t tmem_S = tmem_addrs[0];
    uint32_t tmem_O = tmem_addrs[1];

    const float scale = 0.08838834764831845f;

    // Load Q to SMEM (128*128*2 = 32KB, 128 threads -> 16 uint4 each)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int chunk = i * 128 + tid;
        int row = chunk / 16;
        int d_bf16 = (chunk % 16) * 8;
        int grow = q_start + row;
        if (grow < S)
            *reinterpret_cast<uint4*>(&Q_smem[row * 128 + d_bf16]) =
                *reinterpret_cast<const uint4*>(&Q_bh[(size_t)grow * 128 + d_bf16]);
        else
            *reinterpret_cast<uint4*>(&Q_smem[row * 128 + d_bf16]) = make_uint4(0,0,0,0);
    }
    __syncthreads();

    // Issue first K/V load
    load_kv_tile(K_bh, V_bh, K_base[0], V_base[0], 0, S, tid);
    cp_async_commit();

    float row_m = -1e30f;
    float row_l = 0.f;
    int q_row = q_start + tid;
    int max_key_excl = min(S, q_start + BM);
    int num_kb = (max_key_excl + BN - 1) / BN;
    uint32_t phase = 0;

    // Instruction descriptors
    // QK^T: M=128, N=64, A=K-major(trans=0), B=MN-major(trans=1)
    uint32_t idesc_qk = make_idesc(128, 64, false, true);
    // PV: M=128, N=128, A from TMEM(trans=0), B=MN-major(trans=1)
    uint32_t idesc_pv = make_idesc(128, 128, false, true);

    for (int kb = 0; kb < num_kb; kb++) {
        int buf = kb % 2;
        int next_buf = 1 - buf;
        int kb_start = kb * BN;

        // Prefetch next K/V
        if (kb + 1 < num_kb) {
            load_kv_tile(K_bh, V_bh, K_base[next_buf], V_base[next_buf], (kb+1)*BN, S, tid);
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // === QK^T: S = Q @ K^T ===
        // A = Q[128, 16] K-major: LBO=16, SBO=2048
        // B = K^T[16, 64] MN-major: LBO=2048, SBO=16
        if (tid == 0) {
            for (int ks = 0; ks < 8; ks++) {
                int d = ks * 16; // 16 bf16 = 32 bytes
                uint64_t desc_a = make_smem_desc((char*)Q_smem + d * 2, 16, 2048);
                uint64_t desc_b = make_smem_desc((char*)(buf ? K_smem_1 : K_smem_0) + d * 2, 2048, 16);
                umma_cg1_smem_a(tmem_S, desc_a, desc_b, idesc_qk, (ks == 0) ? 0 : 1);
            }
            umma_commit_cg1(mbar);
        }
        mbarrier_wait(mbar, phase);
        phase ^= 1;

        // === Read S from TMEM and compute softmax ===
        float s[64];
        #pragma unroll
        for (int c = 0; c < 64; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x(tmem_S + c, &r0, &r1, &r2, &r3);
            s[c] = __uint_as_float(r0);
            s[c+1] = __uint_as_float(r1);
            s[c+2] = __uint_as_float(r2);
            s[c+3] = __uint_as_float(r3);
        }
        tmem_wait_ld();

        // Scale and causal mask
        float block_m = -1e30f;
        #pragma unroll
        for (int c = 0; c < 64; c++) {
            s[c] *= scale;
            int k_global = kb_start + c;
            if (k_global > q_row || k_global >= S) s[c] = -1e30f;
            block_m = fmaxf(block_m, s[c]);
        }

        // Online softmax
        float m_new = fmaxf(row_m, block_m);
        float factor = fast_expf(row_m - m_new);

        float sum = 0.f;
        #pragma unroll
        for (int c = 0; c < 64; c++) {
            s[c] = fast_expf(s[c] - m_new);
            sum += s[c];
        }
        row_l = row_l * factor + sum;
        row_m = m_new;

        // === Store P to TMEM (BF16 packed) ===
        #pragma unroll
        for (int c = 0; c < 32; c += 4) {
            uint32_t p0 = pack_bf16(s[c*2], s[c*2+1]);
            uint32_t p1 = pack_bf16(s[(c+1)*2], s[(c+1)*2+1]);
            uint32_t p2 = pack_bf16(s[(c+2)*2], s[(c+2)*2+1]);
            uint32_t p3 = pack_bf16(s[(c+3)*2], s[(c+3)*2+1]);
            tmem_store_4x(tmem_S + c, p0, p1, p2, p3);
        }
        tmem_wait_st();

        // === Rescale O in TMEM (read, scale, write) ===
        #pragma unroll
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x(tmem_O + c, &r0, &r1, &r2, &r3);
            tmem_wait_ld();
            float o0 = __uint_as_float(r0) * factor;
            float o1 = __uint_as_float(r1) * factor;
            float o2 = __uint_as_float(r2) * factor;
            float o3 = __uint_as_float(r3) * factor;
            tmem_store_4x(tmem_O + c, __float_as_uint(o0), __float_as_uint(o1), __float_as_uint(o2), __float_as_uint(o3));
            tmem_wait_st();
        }
        __syncthreads();

        // === PV: O += P @ V ===
        // A = P from TMEM (tmem_S), B = V[16,128] MN-major: LBO=16, SBO=2048
        if (tid == 0) {
            for (int ks = 0; ks < 4; ks++) {
                int key_off = ks * 16;
                uint32_t a_tmem = tmem_S + ks * 8; // 8 TMEM cols = 16 BF16
                uint64_t desc_b = make_smem_desc(
                    (char*)(buf ? V_smem_1 : V_smem_0) + key_off * 256, 16, 2048);
                umma_cg1_tmem_a(tmem_O, a_tmem, desc_b, idesc_pv, 1);
            }
            umma_commit_cg1(mbar);
        }
        mbarrier_wait(mbar, phase);
        phase ^= 1;
        __syncthreads();
    }

    // === Epilogue: read O from TMEM, normalize, store ===
    float inv_l = (row_l > 0.f) ? (1.0f / row_l) : 0.0f;

    // Read O and write to SMEM as BF16 (reuse Q_smem)
    __nv_bfloat16* O_smem = Q_smem;
    #pragma unroll
    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_O + c, &r0, &r1, &r2, &r3);
        tmem_wait_ld();
        float o0 = __uint_as_float(r0) * inv_l;
        float o1 = __uint_as_float(r1) * inv_l;
        float o2 = __uint_as_float(r2) * inv_l;
        float o3 = __uint_as_float(r3) * inv_l;
        O_smem[tid * 128 + c]     = __float2bfloat16_rn(o0);
        O_smem[tid * 128 + c + 1] = __float2bfloat16_rn(o1);
        O_smem[tid * 128 + c + 2] = __float2bfloat16_rn(o2);
        O_smem[tid * 128 + c + 3] = __float2bfloat16_rn(o3);
    }
    __syncthreads();

    // Coalesced store O -> global (128*128 bf16 = 256 uint4, 128 threads -> 2 each)
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int uidx = i * 128 + tid;
        int row = uidx / 16;
        int col8 = (uidx % 16) * 8;
        int grow = q_start + row;
        if (grow < S) {
            uint4 val = *reinterpret_cast<uint4*>(&O_smem[row * 128 + col8]);
            *reinterpret_cast<uint4*>(&O_bh[(size_t)grow * 128 + col8]) = val;
        }
    }

    // LSE
    if (q_row < S) {
        LSE_bh[q_row] = row_m + fast_logf(row_l);
    }

    // Deallocate TMEM
    __syncthreads();
    if (warp == 0) {
        tmem_dealloc_cg1(tmem_S, 64);
        tmem_dealloc_cg1(tmem_O, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const __nv_bfloat16* Q_p = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_p = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_p = static_cast<float*>(LSE.data_ptr());

    int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid(B * H, num_q_blocks);
    dim3 block(NUM_THREADS);
    // Q(32KB) + K[2](32KB) + V[2](32KB) + 16B overhead = ~96KB
    size_t smem_bytes = (size_t)(BM * D + 2 * BN * D + 2 * BN * D) * sizeof(__nv_bfloat16) + 16;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attnKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    attnKernel<<<grid, block, smem_bytes, stream>>>(Q_p, K_p, V_p, O_p, LSE_p, B, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_kernel::run);

}  // namespace attn_kernel