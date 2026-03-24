import torch
import time

dim = 768
hdim = dim * 4
batch_size = 8 * 2048
x = torch.randn((batch_size, hdim), dtype=torch.bfloat16, device="cuda")
W2 = torch.randn((hdim, dim), dtype=torch.bfloat16, device="cuda")

result = x @ W2

BLOCKSIZE = 128

def block_quantize(x, BLOCKSIZE_M, BLOCKSIZE_N):
    x_blocked = x.reshape((x.shape[0] // BLOCKSIZE_M, BLOCKSIZE_M, x.shape[1] // BLOCKSIZE_N, BLOCKSIZE_N))
    
    
    x_blocked_scales = torch.amax(x_blocked.abs(), dim=(1,3), keepdim=True)
    print("x_blocked.shape:", x_blocked.shape)
    print("x_blocked_scales.shape:", x_blocked_scales.shape)
    
    x_blocked_fp8 = (x_blocked / x_blocked_scales).to(torch.float8_e4m3fn)

    return x_blocked_fp8.reshape(x.shape), x_blocked_scales.squeeze().to(torch.float32)

x_fp8, x_s = block_quantize(x, 128, 128)
W2_fp8, W2_s = block_quantize(W2, 128, 1)

print(x_fp8.shape)
print("scale a shape:", x_s.shape)
print(W2_fp8.shape)
print("scale b shape:", W2_s.shape)

result_fp8 = torch._scaled_mm(
                x_fp8,
                W2_fp8.T.contiguous().T,
                out_dtype=torch.bfloat16,
                scale_a=x_s,
                scale_b=W2_s,
                )

W2_fp8_T = W2_fp8.T.contiguous().T

warmups = 5
for i in range(warmups):
  result_fp8 = torch._scaled_mm(
                  x_fp8,
                  W2_fp8_T,
                  out_dtype=torch.bfloat16,
                  scale_a=x_s,
                  scale_b=W2_s,
                  )
torch.cuda.synchronize()

iters = 100
start = time.time()
for i in range(iters):
  result_fp8 = torch._scaled_mm(
                  x_fp8,
                  W2_fp8_T,
                  out_dtype=torch.bfloat16,
                  scale_a=x_s,
                  scale_b=W2_s,
                  )
torch.cuda.synchronize()    
end = time.time()
elapsed = ((end - start) * 1e6) / iters

print("avg us:", elapsed)


#print(result)
#print(result_fp8)
