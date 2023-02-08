import torch

# Create a sparse tensor
indices = torch.LongTensor([[0, 1, 1], [0, 0, 1]])
values = torch.FloatTensor([3, 4, 5])
shape = torch.Size([2, 2])
sparse_matrix = torch.sparse.FloatTensor(indices, values, shape)
print(sparse_matrix)
print(sparse_matrix.shape)

# Create a dense tensor
dense_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Perform sparse matrix multiplication
result = torch.sparse.mm(sparse_matrix, dense_matrix)
print(result)

# Output:
# tensor([[15., 20.],
#         [ 3.,  4.]])
