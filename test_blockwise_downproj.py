import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

import time

@triton.jit
def linear_relu_square_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 M, N, K,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr,
                                 GROUP_SIZE_M: tl.constexpr,
                                 NUM_SMS: tl.constexpr,
                                 FORWARD: tl.constexpr,
                                 ):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        c0 = acc0.to(dtype)
        if not FORWARD:
            c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
            c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c], c0)

        if FORWARD:
            c0_post = tl.maximum(c0, 0)
            c0_post = c0_post * c0_post
            aux_desc.store([offs_am_c, offs_bn_c], c0_post)

        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)

        if FORWARD:
            c1_post = tl.maximum(c1, 0)
            c1_post = c1_post * c1_post
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)


def linear_relu_square(a, b, aux=None):
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    FORWARD = False
    if aux is None:
        FORWARD = True
        aux = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    num_stages = 4 if FORWARD else 3
    num_warps = 8

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    linear_relu_square_kernel[grid](
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
        NUM_SMS=NUM_SMS,
        FORWARD=FORWARD,
        num_stages=num_stages,
        num_warps=num_warps
    )

    if FORWARD:
        return c, aux
    else:
        return c

class FusedLinearReLUSquareFunction(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(ctx, x, W1, W2):
        pre, post = linear_relu_square(x.view((-1, x.shape[-1])), W1)
        x3 = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post)
        return x3.view(x.shape)

    @torch.compile
    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post = ctx.saved_tensors
        grad_output = grad_output.view((-1, grad_output.shape[-1]))
        dW2 = post.T @ grad_output
        dpre = linear_relu_square(grad_output.view((-1, grad_output.shape[-1])), W2, aux=pre)
        dW1 = dpre.T @ x
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2

def block_quantize(x, BLOCKSIZE_M, BLOCKSIZE_N, dtype=torch.float8_e4m3fn):
    x_blocked = x.reshape((x.shape[0] // BLOCKSIZE_M, BLOCKSIZE_M, x.shape[1] // BLOCKSIZE_N, BLOCKSIZE_N))
    
    
    x_blocked_scales = torch.amax(x_blocked.abs(), dim=(1,3), keepdim=True)
    x_blocked_fp8 = (x_blocked / x_blocked_scales).to(dtype)

    return x_blocked_fp8.reshape(x.shape), x_blocked_scales.squeeze().to(torch.float32)

class FusedLinearReLUSquareFunctionFp8(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(ctx, x, W1, W2):
        pre, post = linear_relu_square(x.view((-1, x.shape[-1])), W1)

        post_fp8, post_s = block_quantize(post, 128, 128)
        W2_fp8, W2_s = block_quantize(W2, 128, 1)
        #W2_fp8 = W2_fp8.T.contiguous().T

        x3 = torch._scaled_mm(
                W2_fp8.T.contiguous(),
                post_fp8.T,
                out_dtype=torch.bfloat16,
                scale_a=W2_s.T,
                scale_b=post_s.T)

        ctx.save_for_backward(x, W1, W2, pre, post, post_fp8, post_s)
        return x3.T.view(x.shape)

    @torch.compile
    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post, post_fp8, post_s = ctx.saved_tensors

        grad_output = grad_output.view((-1, grad_output.shape[-1]))

        dW2 = post.T @ grad_output

        dpre = linear_relu_square(grad_output.view((-1, grad_output.shape[-1])), W2, aux=pre)
        dW1 = dpre.T @ x
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2

dim = 768
hdim = dim * 4
batch_size = 8 * 2048
x = torch.randn((1, batch_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)
W1 = torch.randn((hdim, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)
W2 = torch.randn((hdim, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)

x_fp8 = x.clone().detach().requires_grad_(True)
W1_fp8 = W1.clone().detach().requires_grad_(True)
W2_fp8 = W2.clone().detach().requires_grad_(True)


post = FusedLinearReLUSquareFunction.apply(x, W1, W2)
post_fp8 = FusedLinearReLUSquareFunctionFp8.apply(x_fp8, W1_fp8, W2_fp8)

print("post:", post)
print("post_fp8:", post_fp8)

grad = torch.randn_like(post)

post.backward(grad)
post_fp8.backward(grad)

print("x.grad:", x.grad)
print("x_fp8.grad:", x_fp8.grad)
print("W1.grad:", W1.grad)
print("W1_fp8.grad:", W1_fp8.grad)
print("W2.grad:", W2.grad)
print("W2_fp8.grad:", W2_fp8.grad)

warmups = 5
iters = 1000


for i in range(warmups):
    post = FusedLinearReLUSquareFunction.apply(x, W1, W2)
torch.cuda.synchronize()

start = time.time()
for i in range(iters):
    post = FusedLinearReLUSquareFunction.apply(x, W1, W2)
torch.cuda.synchronize()
end = time.time()
elapsed = ((end - start) * 1e6) / iters

print("Baseline fwd (us):", elapsed)

for i in range(warmups):
    post_fp8 = FusedLinearReLUSquareFunctionFp8.apply(x, W1, W2)
torch.cuda.synchronize()

start = time.time()
for i in range(iters):
    post_fp8 = FusedLinearReLUSquareFunctionFp8.apply(x, W1, W2)
torch.cuda.synchronize()
end = time.time()
elapsed = ((end - start) * 1e6) / iters
print("Fp8 fwd (us):", elapsed)


for i in range(warmups):
    post_fp8.backward(grad, retain_graph=True)
torch.cuda.synchronize()

start = time.time()
for i in range(iters):
    post_fp8.backward(grad, retain_graph=True)
torch.cuda.synchronize()
end = time.time()
elapsed = ((end - start) * 1e6) / iters
print("Fp8 bwd (us):", elapsed)




for i in range(warmups):
    post.backward(grad, retain_graph=True)
torch.cuda.synchronize()

start = time.time()
for i in range(iters):
    post.backward(grad, retain_graph=True)
torch.cuda.synchronize()
end = time.time()
elapsed = ((end - start) * 1e6) / iters

print("Baseline bwd (us):", elapsed)
