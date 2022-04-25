from random import random
import torch

torch.manual_seed(0)
random_tensor = torch.round(torch.rand((3, 3, 4)) * 10)
print(random_tensor)

indices = [0, 0, 0]

# a = torch.index_select(random_tensor, 1, torch.tensor(indices))
coords = []
coords[:] = map(list, zip(*[list(range(random_tensor.size()[0])), indices]))


print(random_tensor[[0, 0], [1, 0]])

print(random_tensor > 5)
