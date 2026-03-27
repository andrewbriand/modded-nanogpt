import torch
import triton
import triton.language as tl
import time
import ctypes
from torch.utils.cpp_extension import include_paths

@triton.jit
def fused_softcapped_entropy_fwd_kernel(
    logits_ptr, losses_ptr, lse_ptr, targets_ptr, mtp_weights_ptr,
    stride_logits_n, stride_logits_v,
    n_rows, n_cols, n_predict,
    A, B, C,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n

    max_val = -float('inf')
    sum_exp = 0.0

    inv_C = 1.0 / C
    B_div_C = B * inv_C

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=-float('inf')).to(tl.float32)
        z = A * tl.sigmoid(val * inv_C + B_div_C)
        z = tl.where(mask, z, -float('inf'))
        curr_max = tl.max(z, axis=0)
        new_max = tl.maximum(max_val, curr_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(z - new_max), axis=0)
        max_val = new_max

    lse = max_val + tl.log(sum_exp)
    tl.store(lse_ptr + row_idx, lse)

    total_loss = 0.0
    for k in range(n_predict):
        target_idx = row_idx + k
        if target_idx < n_rows:
            weight = tl.load(mtp_weights_ptr + k)
            if weight > 0:
                target = tl.load(targets_ptr + target_idx).to(tl.int32)
                if target >= 0 and target < n_cols:
                    val_target = tl.load(logits_row_ptr + target).to(tl.float32)
                    z_target = A * tl.sigmoid(val_target * inv_C + B_div_C)
                    total_loss += weight * (lse - z_target)

    tl.store(losses_ptr + row_idx, total_loss)

@triton.jit
def fused_softcapped_entropy_bwd_kernel(
    grad_input_ptr, grad_output_ptr, lse_ptr, logits_ptr, targets_ptr, mtp_weights_ptr,
    stride_logits_n, stride_logits_v, stride_grad_n, stride_grad_v,
    n_rows, n_cols, n_predict,
    A, B, C,
    grad_s,
    BLOCK_SIZE: tl.constexpr,
    N_PREDICT: tl.constexpr
):
    row_idx = tl.program_id(0).to(tl.int64)

    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    grad_row_ptr = grad_input_ptr + row_idx * stride_grad_n

    lse = tl.load(lse_ptr + row_idx)
    grad_loss = tl.load(grad_output_ptr + row_idx)

    inv_C = 1.0 / C
    B_div_C = B * inv_C
    inv_C_A = inv_C * A
    inv_grad_s = 1.0 / grad_s

    # Preload all targets and weights before the column loop
    S_w = 0.0
    t0: tl.int32 = -1
    t1: tl.int32 = -1
    t2: tl.int32 = -1
    w0: tl.float32 = 0.0
    w1: tl.float32 = 0.0
    w2: tl.float32 = 0.0

    if N_PREDICT >= 1:
        if row_idx + 0 < n_rows:
            w0 = tl.load(mtp_weights_ptr + 0)
            t0 = tl.load(targets_ptr + row_idx + 0).to(tl.int32)
            S_w += w0

    if N_PREDICT >= 2:
        if row_idx + 1 < n_rows:
            w1 = tl.load(mtp_weights_ptr + 1)
            t1 = tl.load(targets_ptr + row_idx + 1).to(tl.int32)
            S_w += w1

    if N_PREDICT >= 3:
        if row_idx + 2 < n_rows:
            w2 = tl.load(mtp_weights_ptr + 2)
            t2 = tl.load(targets_ptr + row_idx + 2).to(tl.int32)
            S_w += w2

    # Fuse all scalar multiplications
    grad_scale = grad_loss * inv_grad_s
    grad_scale_icA = grad_scale * inv_C_A

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        u = val * inv_C + B_div_C
        sigmoid_u = tl.sigmoid(u)
        z = A * sigmoid_u
        p = tl.exp(z - lse)

        term1 = S_w * p

        term2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        if N_PREDICT >= 1:
            term2 += tl.where(cols == t0, w0, 0.0)
        if N_PREDICT >= 2:
            term2 += tl.where(cols == t1, w1, 0.0)
        if N_PREDICT >= 3:
            term2 += tl.where(cols == t2, w2, 0.0)

        grad_z = term1 - term2
        grad_x = grad_scale_icA * grad_z * sigmoid_u * (1.0 - sigmoid_u)
        grad_x = grad_x.to(tl.float8e5)
        tl.store(grad_row_ptr + cols, grad_x, mask=mask)

# -----------------------------------------------------------------------------
# Tiled transpose copy kernel: dst (N, M) = src (M, N).T
# Uses coalesced reads from src and coalesced writes to dst via tl.trans().
# Replaces PyTorch's elementwise copy_ which uses a naive 75k-block kernel
# with non-coalesced writes, saturating all SMs and blocking NCCL.

@triton.jit
def _transpose_copy_kernel(
    src_ptr, dst_ptr,
    M, N,
    src_stride_m, src_stride_n,
    dst_stride_0, dst_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Coalesced read from src (M, N)
    tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask, other=0.0,
    )

    # Coalesced write to dst (N, M): dst[n, m] = src[m, n]
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(
        dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1,
        tl.trans(tile), mask=mask_T,
    )


def transpose_copy(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose copy: dst = src.T where src is (M, N) and dst is (N, M).

    Uses a 64x128 tiled Triton kernel with coalesced reads AND writes,
    achieving near memory-bandwidth-limited performance.
    """
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M), f"Expected dst shape ({N}, {M}), got {dst.shape}"

    BLOCK_M, BLOCK_N = 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _transpose_copy_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )


# -----------------------------------------------------------------------------
# Tiled transpose-add kernel: dst (M, N) += src (N, M).T
# Same tiling strategy as transpose_copy but with a fused read-add-write.
# Replaces PyTorch's .add_(src.T) which uses the same 75k-block elementwise
# kernel with non-coalesced reads from the transposed operand.

@triton.jit
def _transpose_add_kernel(
    src_ptr, dst_ptr,
    M, N,
    src_stride_m, src_stride_n,
    dst_stride_0, dst_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Coalesced read from src (M, N)
    src_tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask, other=0.0,
    )

    # Coalesced read-add-write on dst (N, M): dst[n, m] += src[m, n]
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    dst_ptrs = dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1
    dst_tile = tl.load(dst_ptrs, mask=mask_T, other=0.0)
    tl.store(dst_ptrs, dst_tile + tl.trans(src_tile), mask=mask_T)


def transpose_add(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose-add: dst += src.T where src is (M, N) and dst is (N, M).

    Uses a 32x32 tiled Triton kernel with coalesced access on both src and dst,
    replacing PyTorch's .add_(src.T) which has non-coalesced reads from the
    transposed operand.
    """
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M), f"Expected dst shape ({N}, {M}), got {dst.shape}"

    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _transpose_add_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )


class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, lm_head_weight, x_s, w_s, grad_s, A=23.0, B=5.0, C=7.5):

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = lm_head_weight.div(w_s).to(torch.float8_e4m3fn)

        w_f8_col_major = w_f8.T.contiguous().T

        logits = torch._scaled_mm(
            x_f8,
            w_f8_col_major,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )

        n_rows, n_cols = logits.shape
        if mtp_weights is None:
             mtp_weights = torch.tensor([1.0], device=logits.device, dtype=torch.float32)
        n_predict = mtp_weights.shape[0]

        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        logits = logits.contiguous()
        targets = targets.contiguous()
        mtp_weights = mtp_weights.contiguous()

        grid = (n_rows,)
        fused_softcapped_entropy_fwd_kernel[grid](
            logits, losses, lse, targets, mtp_weights,
            logits.stride(0), logits.stride(1),
            n_rows, n_cols, n_predict,
            A, B, C,
            BLOCK_SIZE=2048,
            num_warps=2
        )

        ctx.save_for_backward(logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8)
        ctx.params = (A, B, C, x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8 = ctx.saved_tensors
        A, B, C, x_s, w_s, grad_s = ctx.params
        n_rows, n_cols = logits.shape
        n_predict = mtp_weights.shape[0]

        grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)
        grad_output = grad_output.contiguous()

        grid = (n_rows,)
        fused_softcapped_entropy_bwd_kernel[grid](
            grad_input, grad_output, lse, logits, targets, mtp_weights,
            logits.stride(0), logits.stride(1), grad_input.stride(0), grad_input.stride(1),
            n_rows, n_cols, n_predict,
            A, B, C,
            grad_s,
            BLOCK_SIZE=1024,
            num_warps=4,
            N_PREDICT=n_predict,
        )

        x_scale = grad_input.new_tensor(x_s, dtype=torch.float32)
        w_scale = grad_input.new_tensor(w_s, dtype=torch.float32)
        grad_scale = grad_input.new_tensor(grad_s, dtype=torch.float32)

        grad_x = torch._scaled_mm(
            grad_input,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=w_scale,
            use_fast_accum=False,
        )

        x_f8_T = torch.empty((x_f8.shape[1], x_f8.shape[0]), dtype=x_f8.dtype, device=x_f8.device)
        transpose_copy(x_f8, x_f8_T)  # (768, n_rows) row-major

        grad_input_T = torch.empty((n_cols, n_rows), dtype=grad_input.dtype, device=grad_input.device)
        transpose_copy(grad_input, grad_input_T)  # (50304, n_rows) row-major

        grad_w = torch._scaled_mm(
            x_f8_T,            # (768, n_rows) row-major
            grad_input_T.T,    # (n_rows, 50304) column-major view
            out_dtype=torch.float32,
            scale_a=x_scale,
            scale_b=grad_scale,
            use_fast_accum=False,
        )

        return grad_x, None, None, grad_w, None, None, None

CE_KERNEL_BLOCK_SIZE = 128
CE_KERNEL_VOCAB_SIZE = 50304;

CE_KERNEL_DECLS = f"""
constexpr int VOCAB_SIZE = {CE_KERNEL_VOCAB_SIZE};
constexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};
"""

CE_KERNEL_SOURCE = """
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

struct __align__(16) __nv_bfloat168 {
    __nv_bfloat16 data[8];
    __device__ __nv_bfloat16& operator[](int i) { return data[i]; }
    __device__ const __nv_bfloat16& operator[](int i) const { return data[i]; }
};

struct __align__(8) __nv_fp8_e5m28 {
    __nv_fp8_e5m2 data[8];
    __device__ __nv_fp8_e5m2& operator[](int i) { return data[i]; }
    __device__ const __nv_fp8_e5m2& operator[](int i) const { return data[i]; }
};

template<typename T> __device__ constexpr T CEIL_DIV(T a, T b) { return (a + b - 1) / b; }

__device__ float sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

extern "C"
__launch_bounds__(128, 2)
__global__ void ce_fwd_bwd_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int* __restrict__ targets,
    const float* __restrict__ mtp_weights,
    float* __restrict__ losses,
    __nv_fp8_e5m2* grad_input,
    int batch_size,
    int n_predict,
    double A_param,
    double B_param, 
    double C_param,
    double grad_s_param,
    double grad_scale_param)
{
  constexpr int VEC_WIDTH = 8;
  constexpr int NUM_FULL_LOADS = VOCAB_SIZE / (BLOCK_SIZE * VEC_WIDTH);
  constexpr int NUM_LOADS = CEIL_DIV(VOCAB_SIZE, BLOCK_SIZE * VEC_WIDTH);

  float A = (float)A_param;
  float B = (float)B_param;
  float C = (float)C_param;
  float grad_s = (float)grad_s_param;
  float grad_scale = (float)grad_scale_param;

  extern __shared__ __nv_bfloat16 smem[];

  static_assert(VEC_WIDTH == 8);
  __nv_bfloat168 thread_logits[NUM_LOADS];

  const __nv_bfloat16 *block_logit_ptr = logits + VOCAB_SIZE * blockIdx.x;

  #pragma unroll
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      thread_logits[i] = *(__nv_bfloat168*)(&block_logit_ptr[idx]);
    }
  }

  float inv_C = 1 / C;
  float B_div_C = B * inv_C;
  float thread_max = -CUDART_INF_F;
  #pragma unroll
  for (int i = 0; i < NUM_LOADS; i++) {
    __nv_bfloat168 result;
    __nv_bfloat168 result_sigmoid;
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    #pragma unroll 
    for (int k = 0; k < VEC_WIDTH; k++) {
      float tmp = __bfloat162float(thread_logits[i][k]);
      tmp = sigmoid(tmp * inv_C + B_div_C);
      result_sigmoid[k] = __float2bfloat16(tmp);
      tmp = A * tmp;
      if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
        thread_max = max(tmp, thread_max);
      }
      result[k] = __float2bfloat16(tmp);
    }
    thread_logits[i] = result;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      *(__nv_bfloat168*)(&smem[idx]) = result_sigmoid;
    }
  }

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  int warp_id = threadIdx.x / 32;
  __shared__ float block_maxs[NUM_WARPS];
  __shared__ float block_sums[NUM_WARPS];

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));

  if (threadIdx.x % 32 == 0) {
    block_maxs[warp_id] = thread_max;
  }

  __syncthreads();

  float block_max = -CUDART_INF_F;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_max = fmaxf(block_max, block_maxs[i]);
  }

  float thread_sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    #pragma unroll 
    for (int k = 0; k < VEC_WIDTH; k++) {
      float tmp = __bfloat162float(thread_logits[i][k]);
      tmp = __expf(tmp - block_max);
      if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
        thread_sum += tmp;
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

  if (threadIdx.x % 32 == 0) {
    block_sums[warp_id] = thread_sum;
  }

  __syncthreads();

  float block_sum = 0.0f;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_sum += block_sums[i];
  }

  float lse = block_max + __logf(block_sum);

  if (threadIdx.x == 0) {
    float total_loss = 0.0f;
    for (int k = 0; k < n_predict; k++) {
      int target_idx = blockIdx.x + k;
      if (target_idx < batch_size) {
        float weight = mtp_weights[k];
        int target = targets[target_idx];
        if (target >= 0 && target < VOCAB_SIZE) {
          float z_target = A * __bfloat162float(smem[target]);
          total_loss += weight * (lse - z_target);  
        }
      }
    }
    losses[blockIdx.x] = total_loss;
  }

  float S_w = 0.0f;

  for (int i = 0; i < n_predict; i++) {
    S_w += mtp_weights[i];
  }

  int thread_targets[3];
  float thread_mtp_weights[3];
  #pragma unroll
  for (int k = 0; k < 3; k++) {
    int target_idx = blockIdx.x + k;
    if (target_idx < batch_size) {
      thread_targets[k] = targets[target_idx];
      thread_mtp_weights[k] = mtp_weights[k];
    }
  }

  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_bfloat168 sigmoid_us = *(__nv_bfloat168*)(&smem[idx]);
    __nv_fp8_e5m28 result;
          
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      #pragma unroll 
      for (int j = 0; j < VEC_WIDTH; j++) {
        float sigmoid_u = __bfloat162float(sigmoid_us[j]);
        float z = A * sigmoid_u;
        float p = __expf(z - lse);

        float term1 = S_w * p;
        float term2 = 0.0f;
        #pragma unroll
        for (int k = 0; k < 3; k++) {
          int target_idx = blockIdx.x + k;
          if (target_idx < batch_size) {
            if (thread_targets[k] == idx + j) {
              term2 += thread_mtp_weights[k];
            }
          } 
        }

        float grad_z = term1 - term2;
        float grad_x = grad_scale * (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
        auto result_tmp = __nv_cvt_float_to_fp8(grad_x, __NV_SATFINITE, __NV_E5M2);
        result[j] = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
      }
      *(__nv_fp8_e5m28*)(&grad_input[blockIdx.x * VOCAB_SIZE + idx]) = result;
    }
  }
  
}
"""

with open("ce_fwd_bwd_kernel.cu", "w+") as f:
  f.write(CE_KERNEL_DECLS + CE_KERNEL_SOURCE)

t0 = time.perf_counter()
ce_fwd_bwd_kernel = torch.cuda._compile_kernel(
    CE_KERNEL_DECLS + CE_KERNEL_SOURCE,
    "ce_fwd_bwd_kernel",
    compute_capability="89",
    cuda_include_dirs=["/usr/local/cuda/include/"],
    nvcc_options=["-lineinfo", "--use_fast_math"],
)
print(f"NVRTC compile time: {(time.perf_counter() - t0)*1e3:.1f} ms")
ce_fwd_bwd_kernel.set_shared_memory_config(CE_KERNEL_VOCAB_SIZE * 2)

class FusedSoftcappedCrossEntropyCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, lm_head_weight, x_s, w_s, grad_s, A=23.0, B=5.0, C=7.5, grad_scale=1.0):

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = lm_head_weight.div(w_s).to(torch.float8_e4m3fn)

        w_f8_col_major = w_f8.T.contiguous().T

        logits = torch._scaled_mm(
            x_f8,
            w_f8_col_major,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )

        n_rows, n_cols = logits.shape
        if mtp_weights is None:
             mtp_weights = torch.tensor([1.0], device=logits.device, dtype=torch.float32)
        n_predict = mtp_weights.shape[0]

        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        logits = logits.contiguous()
        targets = targets.contiguous()
        mtp_weights = mtp_weights.contiguous()

        grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)

        grid = (n_rows, 1, 1)
        ce_fwd_bwd_kernel(
            grid,
            (CE_KERNEL_BLOCK_SIZE, 1, 1),
            (logits, targets, mtp_weights, losses, grad_input,
             n_rows, n_predict, A, B, C, grad_s, grad_scale),
            shared_mem=CE_KERNEL_VOCAB_SIZE*2
        )
        #fused_softcapped_entropy_fwd_kernel[grid](
        #    logits, losses, lse, targets, mtp_weights,
        #    logits.stride(0), logits.stride(1),
        #    n_rows, n_cols, n_predict,
        #    A, B, C,
        #    BLOCK_SIZE=2048,
        #    num_warps=2
        #)

        ctx.save_for_backward(logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8, grad_input)
        ctx.params = (A, B, C, x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8, grad_input = ctx.saved_tensors
        A, B, C, x_s, w_s, grad_s = ctx.params
        n_rows, n_cols = logits.shape
        n_predict = mtp_weights.shape[0]

        grad_output = grad_output.contiguous()

        #grid = (n_rows,)
        #fused_softcapped_entropy_bwd_kernel[grid](
        #    grad_input, grad_output, lse, logits, targets, mtp_weights,
        #    logits.stride(0), logits.stride(1), grad_input.stride(0), grad_input.stride(1),
        #    n_rows, n_cols, n_predict,
        #    A, B, C,
        #    grad_s,
        #    BLOCK_SIZE=1024,
        #    num_warps=4,
        #    N_PREDICT=n_predict,
        #)

        x_scale = grad_input.new_tensor(x_s, dtype=torch.float32)
        w_scale = grad_input.new_tensor(w_s, dtype=torch.float32)
        grad_scale = grad_input.new_tensor(grad_s, dtype=torch.float32)

        grad_x = torch._scaled_mm(
            grad_input,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=w_scale,
            use_fast_accum=False,
        )

        x_f8_T = torch.empty((x_f8.shape[1], x_f8.shape[0]), dtype=x_f8.dtype, device=x_f8.device)
        transpose_copy(x_f8, x_f8_T)  # (768, n_rows) row-major

        grad_input_T = torch.empty((n_cols, n_rows), dtype=grad_input.dtype, device=grad_input.device)
        transpose_copy(grad_input, grad_input_T)  # (50304, n_rows) row-major

        grad_w = torch._scaled_mm(
            x_f8_T,            # (768, n_rows) row-major
            grad_input_T.T,    # (n_rows, 50304) column-major view
            out_dtype=torch.float32,
            scale_a=x_scale,
            scale_b=grad_scale,
            use_fast_accum=False,
        )

        return grad_x, None, None, grad_w, None, None, None

batch_size = 8 * 2048
vocab_size = 50304
model_dim = 768

dtype = torch.bfloat16

x = torch.randn((batch_size, model_dim), dtype=dtype, device="cuda")
targets = torch.randint(low=0, high=vocab_size+1, size=(batch_size,), dtype=torch.int32, device="cuda")
mtp_weights = torch.randn((3,), dtype=torch.float32, device="cuda").abs()
lm_head_weight = torch.randn((model_dim, vocab_size), dtype=dtype, device="cuda") / 10
x_s = 100/488
w_s = 1.6/448
grad_s = 0.75/448

x_ref = x.clone().detach().requires_grad_(True)
lm_head_weight_ref = lm_head_weight.clone().detach().requires_grad_(True)

x_kernel = x.clone().detach().requires_grad_(True)
lm_head_weight_kernel = lm_head_weight.clone().detach().requires_grad_(True)

# Correctness check

losses_ref = FusedSoftcappedCrossEntropy.apply(x_ref, targets, mtp_weights, lm_head_weight_ref, x_s, w_s, grad_s)
losses_kernel = FusedSoftcappedCrossEntropyCUDA.apply(x_kernel, targets, mtp_weights, lm_head_weight_kernel, x_s, w_s, grad_s)

print("losses_ref:", losses_ref)
print("losses_kernel:", losses_kernel)
torch.testing.assert_close(losses_ref.to(torch.bfloat16), losses_kernel.to(torch.bfloat16))
print("fwd PASS")

grad = torch.ones_like(losses_ref)

losses_ref.backward(grad)
losses_kernel.backward(grad)

print("targets:", targets)
print("x_ref.grad:", x_ref.grad)
print("x_kernel.grad:", x_kernel.grad)
torch.testing.assert_close(x_ref.grad, x_kernel.grad, atol=1e-01, rtol=.064*4)
torch.testing.assert_close(lm_head_weight_ref.grad, lm_head_weight_kernel.grad, atol=1, rtol=0.064*8)

warmups = 5
iters = 100

x_ref.grad = None
lm_head_weight_ref.grad = None
for i in range(warmups):
    losses_ref = FusedSoftcappedCrossEntropy.apply(x_ref, targets, mtp_weights, lm_head_weight_ref, x_s, w_s, grad_s)
    losses_ref.backward(grad)
    x_ref.grad = None
    lm_head_weight_ref.grad = None
torch.cuda.synchronize()

start = time.time()
for i in range(iters):
    losses_ref = FusedSoftcappedCrossEntropy.apply(x_ref, targets, mtp_weights, lm_head_weight_ref, x_s, w_s, grad_s)
    losses_ref.backward(grad)
    x_ref.grad = None
    lm_head_weight_ref.grad = None
torch.cuda.synchronize()
end = time.time()

print("Baseline (ms):", ((end - start) * 1e3) / iters)

x_kernel.grad = None
lm_head_weight_kernel.grad = None
for i in range(warmups):
    losses_kernel = FusedSoftcappedCrossEntropyCUDA.apply(x_kernel, targets, mtp_weights, lm_head_weight_kernel, x_s, w_s, grad_s)
    losses_kernel.backward(grad)
    x_kernel.grad = None
    lm_head_weight_kernel.grad = None
torch.cuda.synchronize()

start = time.time()
for i in range(iters):
    losses_kernel = FusedSoftcappedCrossEntropyCUDA.apply(x_kernel, targets, mtp_weights, lm_head_weight_kernel, x_s, w_s, grad_s)
    losses_kernel.backward(grad)
    x_kernel.grad = None
    lm_head_weight_kernel.grad = None
torch.cuda.synchronize()
end = time.time()

print("CUDA (ms):", ((end - start) * 1e3) / iters)


