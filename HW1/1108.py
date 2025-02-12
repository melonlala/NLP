import numpy as np
x = [[1,1,1,1],
[2,2,2,2],
[3,3,3,3],
[4,4,4,4]]
k = [[1,1],
[0,0]]
y = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        y[i][j] = x[i][j]*k[0][0] + x[i][j+1]*k[0][1] + x[i+1][j]*k[1][0] + x[i+1][j+1]*k[1][0]
# print(y)
import torch
from torch import nn
model = nn.Sequential(nn.Linear(10,20),nn.ReLU(),
                      nn.Linear(20,40), nn.ReLU(),
                      nn.Linear(40,5), nn.ReLU())

input = torch.ones(100,10)
output = model(input)
print(output)