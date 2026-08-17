#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_lse_d128_causal {

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int BK = 16;
constexpr int NUM_D_CHUNKS = D / BK;
constexpr int NUM_N_BLOCKS = BN / 8;
constexpr int NUM_D_BLOCKS = D / 8;
constexpr int NUM_K_CHUNKS = BN / BK;

__device__ __forceinline__ float fast_exp2f(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint16_t ha = *(uint16_t*)&a;
    uint16_t hb = *(uint16_t*)&b;
    return (uint32_t)ha | ((uint32_t)hb << 16);
}

__device__ __forceinline__ void load_a_frag_x4(
    uint32_t* a, const __nv_bfloat16* smem, int row_start, int col_start, int lda) {
    int lane = threadIdx.x % 32;
    int group = lane / 8;
    int rig = lane % 8;
    int matrix_row = (group == 0 || group == 2) ? rig : rig + 8;
    int col_offset = (group == 0 || group == 1) ? 0 : 8;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(
        smem + (row_start + matrix_row) * lda + col_start + col_offset);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(addr));
}

// QK^T: B=K^T, K stored as K[n][d] (d-fast). B[k][n]=K[n][k], needs K-major (k-fast).
// K already has k=d fast, so NO .trans needed. Group offset in K (col) dimension.
__device__ __forceinline__ void load_b_frag_qkt(
    uint32_t* b, const __nv_bfloat16* smem, int row_start, int col_start, int lda) {
    int lane = threadIdx.x % 32;
    int group = lane / 8;
    int rig = lane % 8;
    uint32_t addr = 0;
    if (group < 2) {
        int col_off = (group == 0) ? 0 : 8;
        addr = (uint32_t)__cvta_generic_to_shared(
            smem + (row_start + rig) * lda + col_start + col_off);
    }
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(addr));
}

// PV: B=V, V stored as V[k][d] (d-fast). B[k][n]=V[k][n], needs K-major (k-fast).
// V has d=n fast (N-major), so need .trans to get k-fast. Group offset in K (row) dimension.
__device__ __forceinline__ void load_b_frag_pv(
    uint32_t* b, const __nv_bfloat16* smem, int row_start, int col_start, int lda) {
    int lane = threadIdx.x % 32;
    int group = lane / 8;
    int rig = lane % 8;
    uint32_t addr = 0;
    if (group < 2) {
        int row_off = (group == 0) ? 0 : 8;
        addr = (uint32_t)__cvta_generic_to_shared(
            smem + (row_start + row_off + rig) * lda + col_start);
    }
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(addr));
}

__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
    int S, int H) {

    int q_block = blockIdx.x;
    int bh = blockIdx.y;
    int s_start = q_block * BM;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    extern __shared__ char smem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_K = smem_Q + BM * D;
    __nv_bfloat16* smem_V = smem_K + BN * D;
    __nv_bfloat16* smem_P = smem_V + BN * D;

    const __nv_bfloat16* Q_bh = Q + (uint64_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (uint64_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (uint64_t)bh * S * D;
    __nv_bfloat16* O_bh = O + (uint64_t)bh * S * D;
    float* LSE_bh = LSE + (uint64_t)bh * S;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = threadIdx.x * 8 + i;
        int row = idx / 16;
        int col = (idx % 16) * 8;
        if (s_start + row < S)
            *(int4*)(smem_Q + row * D + col) = *(int4*)(Q_bh + (s_start + row) * D + col);
        else
            *(int4*)(smem_Q + row * D + col) = make_int4(0, 0, 0, 0);
    }
    __syncthreads();

    float s_frag[NUM_N_BLOCKS][4];
    float o_frag[NUM_D_BLOCKS][4];
    float m_state[2] = {-INFINITY, -INFINITY};
    float l_state[2] = {0.f, 0.f};

    #pragma unroll
    for (int n = 0; n < NUM_D_BLOCKS; n++) {
        o_frag[n][0] = 0.f; o_frag[n][1] = 0.f;
        o_frag[n][2] = 0.f; o_frag[n][3] = 0.f;
    }

    int rows[2];
    rows[0] = warp_id * 16 + lane / 4;
    rows[1] = rows[0] + 8;
    int q_pos[2];
    q_pos[0] = s_start + rows[0];
    q_pos[1] = s_start + rows[1];

    const float scale = 0.08838834764831845f;
    const float log2e = 1.4426950408889634f;
    int num_k_blocks = min(q_block + 1, (S + BN - 1) / BN);

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        int k_start = k_block * BN;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = threadIdx.x * 4 + i;
            int row = idx / 16;
            int col = (idx % 16) * 8;
            if (k_start + row < S)
                *(int4*)(smem_K + row * D + col) = *(int4*)(K_bh + (k_start + row) * D + col);
            else
                *(int4*)(smem_K + row * D + col) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();

        #pragma unroll
        for (int n = 0; n < NUM_N_BLOCKS; n++) {
            s_frag[n][0] = 0; s_frag[n][1] = 0;
            s_frag[n][2] = 0; s_frag[n][3] = 0;
        }

        #pragma unroll
        for (int d = 0; d < NUM_D_CHUNKS; d++) {
            uint32_t a[4];
            load_a_frag_x4(a, smem_Q, warp_id * 16, d * 16, D);

            #pragma unroll
            for (int n = 0; n < NUM_N_BLOCKS; n++) {
                uint32_t b[2];
                load_b_frag_qkt(b, smem_K, n * 8, d * 16, D);

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                    : "=f"(s_frag[n][0]), "=f"(s_frag[n][1]),
                      "=f"(s_frag[n][2]), "=f"(s_frag[n][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                      "r"(b[0]), "r"(b[1]),
                      "f"(s_frag[n][0]), "f"(s_frag[n][1]),
                      "f"(s_frag[n][2]), "f"(s_frag[n][3]));
            }
        }

        #pragma unroll
        for (int n = 0; n < NUM_N_BLOCKS; n++) {
            int col_base = n * 8 + (lane % 4) * 2;
            int k_pos0 = k_start + col_base;
            int k_pos1 = k_start + col_base + 1;
            s_frag[n][0] *= scale; s_frag[n][1] *= scale;
            s_frag[n][2] *= scale; s_frag[n][3] *= scale;
            if (k_pos0 > q_pos[0] || k_pos0 >= S || q_pos[0] >= S) s_frag[n][0] = -INFINITY;
            if (k_pos1 > q_pos[0] || k_pos1 >= S || q_pos[0] >= S) s_frag[n][1] = -INFINITY;
            if (k_pos0 > q_pos[1] || k_pos0 >= S || q_pos[1] >= S) s_frag[n][2] = -INFINITY;
            if (k_pos1 > q_pos[1] || k_pos1 >= S || q_pos[1] >= S) s_frag[n][3] = -INFINITY;
        }

        float rowmax[2] = {-INFINITY, -INFINITY};
        #pragma unroll
        for (int n = 0; n < NUM_N_BLOCKS; n++) {
            rowmax[0] = fmaxf(rowmax[0], fmaxf(s_frag[n][0], s_frag[n][1]));
            rowmax[1] = fmaxf(rowmax[1], fmaxf(s_frag[n][2], s_frag[n][3]));
        }
        rowmax[0] = fmaxf(rowmax[0], __shfl_xor_sync(0xFFFFFFFF, rowmax[0], 1));
        rowmax[0] = fmaxf(rowmax[0], __shfl_xor_sync(0xFFFFFFFF, rowmax[0], 2));
        rowmax[1] = fmaxf(rowmax[1], __shfl_xor_sync(0xFFFFFFFF, rowmax[1], 1));
        rowmax[1] = fmaxf(rowmax[1], __shfl_xor_sync(0xFFFFFFFF, rowmax[1], 2));

        float m_old[2], m_new[2], alpha[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            m_old[r] = m_state[r];
            m_new[r] = fmaxf(m_old[r], rowmax[r]);
            alpha[r] = (m_new[r] == -INFINITY) ? 0.f :
                       (m_old[r] == -INFINITY) ? 0.f : fast_exp2f((m_old[r] - m_new[r]) * log2e);
            m_state[r] = m_new[r];
        }

        float rowsum[2] = {0, 0};
        #pragma unroll
        for (int n = 0; n < NUM_N_BLOCKS; n++) {
            float p0 = (s_frag[n][0] == -INFINITY) ? 0.f : fast_exp2f((s_frag[n][0] - m_new[0]) * log2e);
            float p1 = (s_frag[n][1] == -INFINITY) ? 0.f : fast_exp2f((s_frag[n][1] - m_new[0]) * log2e);
            float p2 = (s_frag[n][2] == -INFINITY) ? 0.f : fast_exp2f((s_frag[n][2] - m_new[1]) * log2e);
            float p3 = (s_frag[n][3] == -INFINITY) ? 0.f : fast_exp2f((s_frag[n][3] - m_new[1]) * log2e);
            rowsum[0] += p0 + p1;
            rowsum[1] += p2 + p3;
            s_frag[n][0] = p0; s_frag[n][1] = p1;
            s_frag[n][2] = p2; s_frag[n][3] = p3;
        }
        rowsum[0] += __shfl_xor_sync(0xFFFFFFFF, rowsum[0], 1);
        rowsum[0] += __shfl_xor_sync(0xFFFFFFFF, rowsum[0], 2);
        rowsum[1] += __shfl_xor_sync(0xFFFFFFFF, rowsum[1], 1);
        rowsum[1] += __shfl_xor_sync(0xFFFFFFFF, rowsum[1], 2);
        l_state[0] = l_state[0] * alpha[0] + rowsum[0];
        l_state[1] = l_state[1] * alpha[1] + rowsum[1];

        #pragma unroll
        for (int n = 0; n < NUM_D_BLOCKS; n++) {
            o_frag[n][0] *= alpha[0]; o_frag[n][1] *= alpha[0];
            o_frag[n][2] *= alpha[1]; o_frag[n][3] *= alpha[1];
        }

        #pragma unroll
        for (int n = 0; n < NUM_N_BLOCKS; n++) {
            int col = n * 8 + (lane % 4) * 2;
            *(uint32_t*)(smem_P + rows[0] * BN + col) = pack_bf16(
                __float2bfloat16(s_frag[n][0]), __float2bfloat16(s_frag[n][1]));
            *(uint32_t*)(smem_P + rows[1] * BN + col) = pack_bf16(
                __float2bfloat16(s_frag[n][2]), __float2bfloat16(s_frag[n][3]));
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = threadIdx.x * 4 + i;
            int row = idx / 16;
            int col = (idx % 16) * 8;
            if (k_start + row < S)
                *(int4*)(smem_V + row * D + col) = *(int4*)(V_bh + (k_start + row) * D + col);
            else
                *(int4*)(smem_V + row * D + col) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < NUM_K_CHUNKS; k++) {
            uint32_t a[4];
            load_a_frag_x4(a, smem_P, warp_id * 16, k * 16, BN);

            #pragma unroll
            for (int n = 0; n < NUM_D_BLOCKS; n++) {
                uint32_t b[2];
                load_b_frag_pv(b, smem_V, k * 16, n * 8, D);

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                    : "=f"(o_frag[n][0]), "=f"(o_frag[n][1]),
                      "=f"(o_frag[n][2]), "=f"(o_frag[n][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                      "r"(b[0]), "r"(b[1]),
                      "f"(o_frag[n][0]), "f"(o_frag[n][1]),
                      "f"(o_frag[n][2]), "f"(o_frag[n][3]));
            }
        }
        __syncthreads();
    }

    float inv_l[2];
    inv_l[0] = (l_state[0] > 0.f) ? (1.f / l_state[0]) : 0.f;
    inv_l[1] = (l_state[1] > 0.f) ? (1.f / l_state[1]) : 0.f;

    #pragma unroll
    for (int n = 0; n < NUM_D_BLOCKS; n++) {
        int col = n * 8 + (lane % 4) * 2;
        if (q_pos[0] < S) {
            O_bh[q_pos[0] * D + col] = __float2bfloat16(o_frag[n][0] * inv_l[0]);
            O_bh[q_pos[0] * D + col + 1] = __float2bfloat16(o_frag[n][1] * inv_l[0]);
        }
        if (q_pos[1] < S) {
            O_bh[q_pos[1] * D + col] = __float2bfloat16(o_frag[n][2] * inv_l[1]);
            O_bh[q_pos[1] * D + col + 1] = __float2bfloat16(o_frag[n][3] * inv_l[1]);
        }
    }

    if (q_pos[0] < S) {
        LSE_bh[q_pos[0]] = (l_state[0] > 0.f) ? (m_state[0] + logf(l_state[0])) : -INFINITY;
    }
    if (q_pos[1] < S) {
        LSE_bh[q_pos[1]] = (l_state[1] > 0.f) ? (m_state[1] + logf(l_state[1])) : -INFINITY;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp = static_cast<float*>(LSE.data_ptr());

    int nqb = (S + BM - 1) / BM;
    dim3 grid(nqb, B * H);
    dim3 block(256);
    size_t smem_sz = BM * D * 2 + BN * D * 2 + BN * D * 2 + BM * BN * 2;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);
    attention_kernel<<<grid, block, smem_sz, stream>>>(Qp, Kp, Vp, Op, Lp, (int)S, (int)H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse_d128_causal::run);

}  // namespace mha_lse_d128_causal