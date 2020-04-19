import torch
from torch import nn
from torch.autograd import Variable

inp = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
inp = Variable(inp)
out = torch.Tensor([2, 4, 6, 8, 10, 12, 14, 16, 18])
net = nn.Linear(9, 9)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())
for _ in range(2000):
    hat = net(inp)
    mse = loss(out, hat)
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()

print(net(inp))
