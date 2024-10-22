import torch
import numpy as np

# Initializing a Tensor
print("# Initializing a Tensor \n")

    # Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data, "\n")

    # From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np, "\n")

    # From another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

    # With random or constant values
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

shape = (1, 3, 4)   # Tuple of tensor dimensions (depth, rows, columns)
                    # Tip: If shape = (2, 3) OR shape = (2, 3, ); It represents (rows, columns)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# Attributes (Feature) of a Tensor
print("# Attributes (Feature) of a Tensor \n")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device} \n")

# Operations on Tensors
print("# Operations on Tensors \n")

print(tensor)

    # We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"Device tensor is stored on: {tensor.device} \n")

    # Transpose operation for a tensor
tensor_trans = torch.transpose(tensor, 0, 1)
print(tensor_trans, "\n")

    # Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor, "\n")

    # Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1, "\n")

    # Arithmetic operations
y1 = tensor @ tensor.T          # Matrix product, and tensor.T is the transpose of tensor
y2 = tensor.matmul(tensor.T)    # Another method of matrix product to itself

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)    # Providing an output tensor as argument

print(y1, "\n", y2, "\n", y3, "\n")

z1 = tensor * tensor    # Element-wise product (Dot product)
z2 = tensor.mul(tensor) # Another method of element-wise product

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3) # Providing an output tensor as argument

print(z1, "\n", z2, "\n", z3, "\n")

agg = tensor.sum()      # Aggregates all the values in the tensor
agg_item = agg.item()   # To get the value as a Python number from a tensor containing a single value
print(agg, "\n", agg_item, "\n", type(agg_item), "\n")

print(f"{tensor} \n")
tensor.add_(5)
print(tensor, "\n")

# Bridge with NumPy
print("# Bridge with NumPy \n")

    # Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n} \n")
    
    # A change in the tensor reflects in the NumPy array
t.add_(1)
torch.add(t, 1, out=t)  # Same as t = t + 1, and same with the above line
print(f"t: {t}")
print(f"n: {n} \n")

    # NumPy array to Tensor
        # A change in the NumPy array reflects in the tensor
np.add(n, 1, out = n)
print(f"t: {t}")
print(f"n: {n} \n")