import torch
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([2, 1])
b, indices = torch.sort(b)
a = a[indices]
print(b)
print(a)