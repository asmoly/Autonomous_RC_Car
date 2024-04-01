import torch

test = torch.Tensor([[[1, 20, 3], [14, 5, 2], [16, 15, 8]],
                     [[6, 4, 9], [5, 10, 1], [19, 5, 4]],
                     [[7, 3, 2], [9, 11, 14], [14, 5, 3]]])

print(test)
print("_______________")
print(torch.argmax(test, dim=2))