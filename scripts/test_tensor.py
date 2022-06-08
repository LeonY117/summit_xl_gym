import random
import torch

# torch.manual_seed(0)
random_tensor = torch.round(torch.rand((3, 3, 4)) * 10)
print(random_tensor)

indices = [0, 0, 0]

# a = torch.index_select(random_tensor, 1, torch.tensor(indices))
coords = []
coords[:] = map(list, zip(*[list(range(random_tensor.size()[0])), indices]))


print(random_tensor[[0, 0], [1, 0]])

print(random_tensor > 5)


test_tensor1 = torch.ones((4, 1))
test_tensor2 = None

# combined_tensor = torch.cat((test_tensor1, test_tensor2), dim=-1)
# print(combined_tensor.size())

a = [1, 2, 3, 4]
print(a[4:])


i = 0
while i < 10:
    if True:
        if random.uniform(0, 1) < 0.5:
            print(i)
            continue
        print(i)
    i += 1

dummy_matrix = torch.Tensor([[1, 2, 2], [4, 6, 7]])

print(torch.sum(dummy_matrix))
print(torch.sum(dummy_matrix[:, :]))


out = torch.Tensor([[0, 16, 2, 4], [1, 2, 5, 6], [4, 5, 4, 1]])
nums = torch.rand((3, 4, 3))
prob = torch.softmax(out, 0)
print(torch.softmax(out, 0))
print(nums)

# dummy_indices = torch.Tensor([[0, 1], [1, 0], [2, 2]], dtype=torch.float)

index = torch.multinomial(prob, 1, False)
index = torch.max(prob, -1).indices.unsqueeze(-1)
print(index)
index = index.repeat(1, 3).unsqueeze(1)
print(index)
print(index.shape)
print(torch.gather(nums, 1, index))

print((1-out)*0.5)


test_array = [1, 2, 3, 4, 5, 6, 7]
print(test_array[3:-1])


def myFunc(func):
    print(func)
    func()
    return 1


print(torch.Tensor([[1, 2, 2], [4, 6, 7]]) * torch.tensor([[0, 0, 1], [0, 1, 0]]))
