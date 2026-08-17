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
constexpr int NUM_THREADS = 256;
constexpr int ROW_STRIDE = 256; // 128 BF16 * 2 bytes = 256 bytes per row

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& b0, uint32_t& b1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(b0),"=r"(b1) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& b0, uint32_t& b1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(b0),"=r"(b1) : "r"(addr));
}

__device__ __forceinline__ uint32_t pack_bf16(float x, float y) {
    __nv_bfloat16 bx = __float2bfloat16_rn(x);
    __nv_bfloat16 by = __float2bfloat16_rn(y);
    uint16_t ux = *reinterpret_cast<uint16_t*>(&bx);
    uint16_t uy = *reinterpret_cast<uint16_t*>(&by);
    return (uint32_t)ux | ((uint32_t)uy << 16);
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

__device__ __forceinline__ void cp_async_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void st_shared_128(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ float fast_expf(float x) {
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f)); return y;
}
__device__ __forceinline__ float fast_logf(float x) {
    float y; asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y * 0.6931471805599453f;
}

// 128B swizzle: returns byte offset. Row stride = 256 bytes (128 bf16).
// Within each 128B span (64 bf16 = 8 chunks of 8 bf16), chunk index XOR'd with (row & 7)
__device__ __forceinline__ uint32_t swizzled_offset(uint32_t row, uint32_t col_bf16) {
    uint32_t span = col_bf16 >> 6;        // col / 64
    uint32_t chunk = (col_bf16 & 0x3F) >> 3; // (col % 64) / 8
    uint32_t swizzled = (row & 7) ^ chunk;
    return row * ROW_STRIDE + (span << 7) + (swizzled << 4);
}

__device__ __forceinline__ void load_q_tile(
    const __nv_bfloat16* Q_bh, uint32_t Q_base, int q_start, int S, int tid) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int uidx = i * 256 + tid;
        int eidx = uidx * 8;
        int row = eidx / D;
        int col = eidx % D;
        int grow = q_start + row;
        uint32_t addr = Q_base + swizzled_offset(row, col);
        if (grow < S) {
            uint4 v = *reinterpret_cast<const uint4*>(&Q_bh[grow * D + col]);
            st_shared_128(addr, v.x, v.y, v.z, v.w);
        } else {
            st_shared_128(addr, 0, 0, 0, 0);
        }
    }
}

__device__ __forceinline__ void load_kv_tile(
    const __nv_bfloat16* K_bh, const __nv_bfloat16* V_bh,
    uint32_t K_base, uint32_t V_base, int kb_start, int S, int tid) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int uidx = i * 256 + tid;
        int eidx = uidx * 8;
        int key = eidx / D;
        int col = eidx % D;
        int gkey = kb_start + key;
        uint32_t k_addr = K_base + swizzled_offset(key, col);
        uint32_t v_addr = V_base + swizzled_offset(key, col);
        if (gkey < S) {
            cp_async_16B(k_addr, &K_bh[gkey * D + col]);
            cp_async_16B(v_addr, &V_bh[gkey * D + col]);
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
    int lane_row = lane / 4;
    int lane_col = lane % 4;
    int local_r0 = lane_row;
    int local_r1 = lane_row + 8;
    int q_r0 = q_start + warp * 16 + local_r0;
    int q_r1 = q_start + warp * 16 + local_r1;

    size_t bh_off = (size_t)(b * H + h) * (size_t)S * (size_t)D;
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    __nv_bfloat16* O_bh = O + bh_off;
    float* LSE_bh = LSE + (size_t)(b * H + h) * (size_t)S;

    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem;
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

    const float scale = 0.08838834764831845f;

    load_q_tile(Q_bh, Q_base, q_start, S, tid);
    __syncthreads();

    load_kv_tile(K_bh, V_bh, K_base[0], V_base[0], 0, S, tid);
    cp_async_commit();

    float o_frag[16][4];
    #pragma unroll
    for (int k = 0; k < 16; k++) { o_frag[k][0]=0.f; o_frag[k][1]=0.f; o_frag[k][2]=0.f; o_frag[k][3]=0.f; }
    float m0 = -1e30f, m1 = -1e30f;
    float l0 = 0.f, l1 = 0.f;

    int max_key_excl = min(S, q_start + BM);
    int num_kb = (max_key_excl + BN - 1) / BN;
    float s_frag[8][4];

    for (int kb = 0; kb < num_kb; kb++) {
        int buf = kb % 2;
        int next_buf = 1 - buf;
        int kb_start = kb * BN;

        if (kb + 1 < num_kb) {
            load_kv_tile(K_bh, V_bh, K_base[next_buf], V_base[next_buf], (kb+1)*BN, S, tid);
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // QK^T
        #pragma unroll
        for (int j = 0; j < 8; j++) { s_frag[j][0]=0.f; s_frag[j][1]=0.f; s_frag[j][2]=0.f; s_frag[j][3]=0.f; }

        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            int d = ks * 16;
            uint32_t a0,a1,a2,a3;
            {
                int group = lane / 8;
                int row_in_tile = lane % 8;
                int row_off = (group & 1) ? 8 : 0;
                int col_off = (group & 2) ? 8 : 0;
                int a_row = warp * 16 + row_off + row_in_tile;
                int a_col = d + col_off;
                ldmatrix_x4(a0,a1,a2,a3, Q_base + swizzled_offset(a_row, a_col));
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint32_t b0,b1;
                {
                    int row_in_tile = lane % 8;
                    int mat = lane / 8;
                    int k_col_off = (mat == 1) ? 8 : 0;
                    int addr_key = j * 8 + row_in_tile;
                    int addr_d = d + k_col_off;
                    ldmatrix_x2(b0,b1, K_base[buf] + swizzled_offset(addr_key, addr_d));
                }
                mma_m16n8k16(s_frag[j][0],s_frag[j][1],s_frag[j][2],s_frag[j][3], a0,a1,a2,a3, b0,b1);
            }
        }

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            s_frag[j][0] *= scale; s_frag[j][1] *= scale;
            s_frag[j][2] *= scale; s_frag[j][3] *= scale;
        }

        // Causal mask
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int base_col = j * 8 + lane_col * 2;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                int col = base_col + (c & 1);
                int k_global = kb_start + col;
                int q_global = (c < 2) ? q_r0 : q_r1;
                if (k_global > q_global || k_global >= S) s_frag[j][c] = -1e30f;
            }
        }

        // Online softmax - compute P and rowsum together
        float block_m0 = -1e30f, block_m1 = -1e30f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            block_m0 = fmaxf(block_m0, fmaxf(s_frag[j][0], s_frag[j][1]));
            block_m1 = fmaxf(block_m1, fmaxf(s_frag[j][2], s_frag[j][3]));
        }
        block_m0 = warp_max4(block_m0);
        block_m1 = warp_max4(block_m1);

        float m_new0 = fmaxf(m0, block_m0);
        float m_new1 = fmaxf(m1, block_m1);
        float factor0 = fast_expf(m0 - m_new0);
        float factor1 = fast_expf(m1 - m_new1);

        float sum0 = 0.f, sum1 = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float p00 = fast_expf(s_frag[j][0] - m_new0);
            float p01 = fast_expf(s_frag[j][1] - m_new0);
            float p10 = fast_expf(s_frag[j][2] - m_new1);
            float p11 = fast_expf(s_frag[j][3] - m_new1);
            s_frag[j][0] = p00; s_frag[j][1] = p01;
            s_frag[j][2] = p10; s_frag[j][3] = p11;
            sum0 += p00 + p01;
            sum1 += p10 + p11;
        }
        sum0 = warp_sum4(sum0);
        sum1 = warp_sum4(sum1);

        l0 = l0 * factor0 + sum0;
        l1 = l1 * factor1 + sum1;
        m0 = m_new0; m1 = m_new1;

        #pragma unroll
        for (int k = 0; k < 16; k++) {
            o_frag[k][0] *= factor0; o_frag[k][1] *= factor0;
            o_frag[k][2] *= factor1; o_frag[k][3] *= factor1;
        }

        // PV: O += P @ V
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            uint32_t pa0 = pack_bf16(s_frag[2*p][0], s_frag[2*p][1]);
            uint32_t pa1 = pack_bf16(s_frag[2*p][2], s_frag[2*p][3]);
            uint32_t pa2 = pack_bf16(s_frag[2*p+1][0], s_frag[2*p+1][1]);
            uint32_t pa3 = pack_bf16(s_frag[2*p+1][2], s_frag[2*p+1][3]);

            #pragma unroll
            for (int k = 0; k < 16; k++) {
                uint32_t b0,b1;
                {
                    int row_in_tile = lane % 8;
                    int mat = lane / 8;
                    int addr_key = p * 16 + row_in_tile + (mat == 1 ? 8 : 0);
                    int addr_d = k * 8;
                    ldmatrix_x2_trans(b0,b1, V_base[buf] + swizzled_offset(addr_key, addr_d));
                }
                mma_m16n8k16(o_frag[k][0],o_frag[k][1],o_frag[k][2],o_frag[k][3], pa0,pa1,pa2,pa3, b0,b1);
            }
        }
        __syncthreads();
    }

    // Final normalization
    float inv_l0 = (l0 > 0.f) ? (1.0f / l0) : 0.0f;
    float inv_l1 = (l1 > 0.f) ? (1.0f / l1) : 0.0f;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        o_frag[k][0] *= inv_l0; o_frag[k][1] *= inv_l0;
        o_frag[k][2] *= inv_l1; o_frag[k][3] *= inv_l1;
    }

    // Store O to SMEM with swizzle
    __nv_bfloat16* O_smem = Q_smem;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        int col_base = k * 8 + lane_col * 2;
        int row0 = warp * 16 + local_r0;
        int row1 = warp * 16 + local_r1;
        uint32_t off0 = swizzled_offset(row0, col_base);
        uint32_t off1 = swizzled_offset(row1, col_base);
        O_smem[off0 / 2]     = __float2bfloat16_rn(o_frag[k][0]);
        O_smem[off0 / 2 + 1] = __float2bfloat16_rn(o_frag[k][1]);
        O_smem[off1 / 2]     = __float2bfloat16_rn(o_frag[k][2]);
        O_smem[off1 / 2 + 1] = __float2bfloat16_rn(o_frag[k][3]);
    }
    __syncthreads();

    // Coalesced store O -> global using O_smem array indexing
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int uidx = i * 256 + tid;
        int row = uidx / 16;
        int col8 = (uidx % 16) * 8;
        int grow = q_start + row;
        if (grow < S) {
            uint32_t off = swizzled_offset(row, col8);
            uint4 val = *reinterpret_cast<const uint4*>(&O_smem[off / 2]);
            *reinterpret_cast<uint4*>(&O_bh[grow * D + col8]) = val;
        }
    }

    if (lane % 4 == 0) {
        if (q_r0 < S) LSE_bh[q_r0] = m0 + fast_logf(l0);
        if (q_r1 < S) LSE_bh[q_r1] = m1 + fast_logf(l1);
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
    size_t smem_bytes = (size_t)(BM * D + 2 * BN * D + 2 * BN * D) * sizeof(__nv_bfloat16) + 1024;

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