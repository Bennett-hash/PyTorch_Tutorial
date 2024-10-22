from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(x, y, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = x[i] + y[i]

# 初始化数据
n = 100000
x = np.arange(n, dtype=np.float32)
y = np.arange(n, dtype=np.float32)
out = np.zeros_like(x)

# 将数据拷贝到GPU
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.device_array_like(x)

# 启动CUDA kernel
threads_per_block = 256
blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block
add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)

# 将结果拷贝回主机
out = d_out.copy_to_host()

# 打印结果
print(out)