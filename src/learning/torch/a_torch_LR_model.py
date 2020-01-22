'''
Created on 2020年1月21日

@author: lipy
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as Data

class LR(nn.Module):
    
    def __init__(self):
        super(LR, self).__init__()
        self.w = nn.Linear(1, 1)

    def forward(self, X):
        pred_Y = self.w(X)
        return pred_Y

import matplotlib.pyplot as plt 
if __name__ == '__main__':
    x1 = list(range(100, 200))
    x2 = list(range(0, 10000, 100))
#     X = [[x1[i], x2[i]] for i in range(100)]
    X = [[i + 0.01] for i in range(100)]
    Y = [[i + 0.01] for i in range(100)]
#     plt.plot(X, Y)
#     plt.show()
#     print(X)
    X = Variable(torch.Tensor(X))
    Y = Variable(torch.Tensor(Y))
    torch_dataset = Data.TensorDataset(X, Y)
    loader = Data.DataLoader(
         dataset=torch_dataset,      # torch TensorDataset format
         batch_size=1,      # mini batch size
         shuffle=True,               # random shuffle for training
         num_workers=2,              # subprocesses for loading data
     )
#     X = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
#     Y = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
    model = LR()
#     optimizer = optim.SGD(model.parameters(), lr=0.01)
#     criterion = nn.MSELoss(size_average=False)
    criterion = nn.MSELoss(size_average=False) # Defined loss function
    optimizer = optim.SGD(model.parameters(), lr=0.00001) # Defined optimizer
#     print("数据量是" ,len(X), X)
    for _ in range(10):
        for batch_x, batch_y in loader:
#             print("范围是", i, i+10)
            pred_Y = model(batch_x)
#             print( Y[i: i + 10 ,:])
            loss = criterion(pred_Y, batch_y)
            print("损失值是", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Forward pass
#         y_pred = model(X)
#         loss = criterion(y_pred, Y)
#         print(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    