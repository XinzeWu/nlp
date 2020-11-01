import torch
import torch.nn as nn
import numpy as np

##LSTM使用
lstm = nn.LSTM(5,1,2)
lista = [[[1,2,3,4,5],[1,2,3,4,5]],[[1,2,3,4,5],[1,2,3,4,5]]] #含义：?
npa = np.array(lista)
b = torch.from_numpy(npa).float() #输入必须是浮点数
output,h = lstm(b)
