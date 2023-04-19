# -*- coding: utf-8 -*-
'''
Multi-Dimension Feature Transformer
'''
# @Time : 2023/4/19 10:14
# @Author : LINYANZHEN
# @File : MDFT.py
import torch
import torch.nn as nn
import os
import shutil

import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import tqdm
from torch.optim import lr_scheduler
from sklearn import metrics
import dataset


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self, in_channels=1, embedding_size=256, k=5):
        super(context_embedding, self).__init__()
        self.causal_convolution = CausalConv1d(in_channels, embedding_size, kernel_size=k)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.causal_convolution(x)
        return self.tanh(x)


class MDFT(torch.nn.Module):
    """
    Time Series application of transformers based on paper
    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel
    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)
    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector
    """

    def __init__(self):
        super(MDFT, self).__init__()
        # 一维卷积，对数据进行升维
        self.input_embedding = context_embedding(79, 256, 9)
        # 使用全连接层升维
        self.input_fc = torch.nn.Linear(79, 256)
        # 创建一个长度为512的编码表作为位置编码，接受输入维度为256
        self.positional_embedding = torch.nn.Embedding(512, 256)
        # 定义解码层，输入维度256，输出维度8
        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8)
        # 解码器，包含多层解码层
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)
        # 全连接层，用于最后的回归
        self.output_fc = torch.nn.Linear(256, 1)

    def forward(self, x, attention_masks):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        # print(z)
        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        x_embedding = self.input_embedding(x).permute(2, 0, 1)
        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1, 0, 2)
        # print(z_embedding)
        # print(positional_embeddings)
        input_embedding = x_embedding + positional_embeddings
        # 接受2个参数
        # 输入：(序列长度,batch_size,数据维度)
        # 掩码：(序列长度，序列长度)
        # print(input_embedding.size())
        # print(attention_masks)
        transformer_embedding = self.transformer_decoder(input_embedding, attention_masks)
        # print(transformer_embedding.size())
        output = self.fc1(transformer_embedding.permute(1, 0, 2))

        return output


if __name__ == '__main__':
    train_loader, val_loader = dataset.get_dataset(0.7)
    model = MDFT()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 100], gamma=0.5)
    progress = tqdm.tqdm(train_loader, total=len(train_loader))
    for (x, y, masks) in progress:
        masks = masks
        out = model(x, masks[0]).squeeze()
        print(out)
        optimizer.zero_grad()
        loss = criterion(out, y[:, :-1])
        loss.backward()
        optimizer.step()
        break
