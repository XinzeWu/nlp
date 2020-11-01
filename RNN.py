import torch
import torch.nn as nn
import numpy as np

##rnn的使用
rnn = nn.RNN(5,1,20) ##分别为输入序列个数，输出个数，RNN神经元个数
lista = [[[1,2,3,4,5],[1,2,3,4,5]],[[1,2,3,4,5],[1,2,3,4,5]]] #含义：?
npa = np.array(lista)
b = torch.from_numpy(npa).float() #输入必须是浮点数
output,h = rnn(b)
