# -*- coding: utf-8 -*-
# @File  : net.py
# @Author: 汪畅
# @Time  : 2022/5/11  18:29
from torch import nn
from torchvision.models import vgg16
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=4):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.4)  # utilize the LSTM model in torch.nn
        self.forwardCalculation = nn.Linear(hidden_dim, output_dim)

    def forward(self, _x):
        _x = _x.view(1, -1, _x.size(1))
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1).squeeze(0)
        return x




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.output = nn.Linear(in_features=64 * 25 * 25, out_features=3)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # [batch, 32,7,7]
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
        output = self.output(x)  # 输出[batch,10]
        output = self.softmax(output)
        return output