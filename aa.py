import torch
import torch.nn as nn
# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)

output = loss(input, target.max(-1)[1])
print(output)
# Example of target with class probabilities

output = loss(input, target)
print(output)