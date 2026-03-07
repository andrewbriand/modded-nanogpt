import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def linear_relu_square_kernel(a_desc, b_desc, c_desc, aux_desc, aux_fp8, output_scale, signal, scales_used,
                                 M, N, K, num_blocks_n,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr,
                                 GROUP_SIZE_M: tl.constexpr,
                                 OUTPUT_SCALE: tl.constexpr,
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

        amax = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        c0 = acc0.to(dtype)
        if not FORWARD:
            c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
            c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c], c0)

        if FORWARD:
            c0_post = tl.maximum(c0, 0)
            c0_post = c0_post * c0_post
            if OUTPUT_SCALE:
                c0_amax = tl.max(c0_post.to(tl.float32), axis=-1)
                amax = tl.maximum(amax, c0_amax)
            aux_desc.store([offs_am_c, offs_bn_c], c0_post)

        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)

        scale = 0.0
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        if FORWARD:
            c1_post = tl.maximum(c1, 0)
            c1_post = c1_post * c1_post
            if OUTPUT_SCALE:
                c1_amax = tl.max(c1_post.to(tl.float32), axis=-1)
                amax = tl.maximum(amax, c1_amax)
                tl.atomic_max(output_scale + offs_m, amax, sem="relaxed")
                tl.atomic_add(signal + pid_m, 1, sem="release")
                flag = 0
                while flag < num_blocks_n:
                    flag = tl.atomic_add(signal + pid_m, 0, sem="acquire")
                scale = tl.load(output_scale + offs_m)
                tl.store(scales_used + pid_m * num_blocks_n * BLOCK_SIZE_M + pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), scale)

            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)

        if OUTPUT_SCALE:
            eps = 1e-5
            scale = scale + eps
            
            c0_fp8 = (c0_post / scale).to(tl.float8e4nv)
            c1_fp8 = (c1_post / scale).to(tl.float8e4nv)

            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)
            tl.store(aux_fp8 + (offs_m * N)[:,None] + offs_n, c0_fp8)
            tl.store(aux_fp8 + (offs_m * N)[:,None] + offs_n + BLOCK_SIZE_N // 2, c1_fp8)


def linear_relu_square(a, b, aux=None, output_scale=None):
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    OUTPUT_SCALE = output_scale is not None


    FORWARD = False
    if aux is None:
        FORWARD = True
        aux = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    num_stages = 3 if FORWARD else 3
    num_warps = 8

    num_blocks_n = N // BLOCK_SIZE_N
    num_blocks_m = M // BLOCK_SIZE_M
    signal = None
    if OUTPUT_SCALE:
        signal = torch.zeros((num_blocks_m,), dtype=torch.int32, device=a.device)
        aux_fp8 = torch.zeros((M, N), dtype=torch.float8_e4m3fn, device=a.device)
        scales_used = torch.zeros((num_blocks_m, num_blocks_n, BLOCK_SIZE_M), dtype=torch.float32, device=a.device)

    

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
        a_desc, b_desc, c_desc, aux_desc, aux_fp8, output_scale, signal, scales_used,
        M, N, K, num_blocks_n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
        NUM_SMS=NUM_SMS,
        FORWARD=FORWARD,
        OUTPUT_SCALE=OUTPUT_SCALE,
        num_stages=num_stages,
        num_warps=num_warps
    )

    print("signal:", signal)
    print("output_scale:", output_scale)
    print("scales_used:", scales_used)

    
    for pid_m in range(scales_used.shape[0]):
        for pid_n in range(scales_used.shape[1]):
            torch.testing.assert_close(scales_used[pid_m][pid_n], output_scale[pid_m * BLOCK_SIZE_M:pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M].squeeze())

    if FORWARD:
        return c, aux, aux_fp8
    else:
        return c

class FusedLinearReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        post_s_kernel = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)

        eps = 1e-5
        pre, post, post_fp8_kernel = linear_relu_square(x.view((-1, x.shape[-1])), W1, output_scale=post_s_kernel)

        post_s = post.to(torch.float32).abs().max(dim=-1, keepdim=True)[0]
        W2_s = W2.abs().max(dim=0, keepdim=True)[0].to(torch.float32)
        #post_s = post.abs().max().to(torch.float32)
        #W2_s = W2.abs().max().to(torch.float32)
        print("post_s:", post_s)
        print("post_s_kernel:", post_s_kernel)
        torch.testing.assert_close(post_s.to(torch.bfloat16), post_s_kernel.to(torch.bfloat16))

        post_fp8 = post.div(post_s + eps).to(torch.float8_e4m3fn)
        W2_fp8 = W2.div(W2_s + eps).to(torch.float8_e4m3fn)

        print("post_fp8:", post_fp8)
        print("post_fp8_kernel:", post_fp8_kernel)
        torch.testing.assert_close(post.div(post_s + eps).to(torch.bfloat16), post.div(post_s_kernel + eps).to(torch.bfloat16))
        torch.testing.assert_close(post_fp8_kernel, post.div(post_s_kernel + eps).to(torch.float8_e4m3fn))

        torch.testing.assert_close(post_fp8, post_fp8_kernel)

        x3 = torch._scaled_mm(
            post_fp8,
            W2_fp8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=post_s,
            scale_b=W2_s,
            use_fast_accum=True)

        #x3 = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post)
        return x3.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post = ctx.saved_tensors
        dW2 = post.T @ grad_output
        dpre = linear_relu_square(grad_output.view((-1, grad_output.shape[-1])), W2, aux=pre)
        dW1 = dpre.T @ x
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2

dim = 768
hdim = dim * 4
batch_size = 8 * 2048
x = torch.randn((batch_size, dim), dtype=torch.bfloat16, device="cuda")
W1 = torch.randn((hdim, dim), dtype=torch.bfloat16, device="cuda")
W2 = torch.randn((hdim, dim), dtype=torch.bfloat16, device="cuda")

FusedLinearReLUSquareFunction.apply(x, W1, W2)
