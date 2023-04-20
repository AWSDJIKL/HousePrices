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
        self.input_embedding = context_embedding(1, 256, 9)
        # 创建一个长度为512的编码表作为位置编码，接受输入维度为256
        self.positional_embedding = torch.nn.Embedding(512, 256)
        # 定义解码层，输入维度256，输出维度8
        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8)
        # 解码器，包含多层解码层
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)
        # 全连接层，用于最后的回归
        self.fc1 = torch.nn.Linear(256 * 79, 1)

    def forward(self, x, attention_masks):
        z = x.unsqueeze(1)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)
        transformer_embedding = self.transformer_decoder(z_embedding, attention_masks)
        output = self.fc1(transformer_embedding.view(-1))
        return output


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 全连接层，用于最后的回归
        self.input_fc = torch.nn.Linear(79, 256)
        self.act = nn.ReLU()
        self.body = nn.Sequential(*[nn.Linear(256, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, 2048),
                                    nn.ReLU(),
                                    nn.Linear(2048, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    ])
        self.output_fc = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.act(x)
        x = self.body(x)
        x = self.act(x)
        x = self.output_fc(x)
        return x


def train_and_val(model, train_loader, val_loader, epoch, criterion, optimizer, scheduler):
    train_loss_list = []
    val_loss_list = []
    for i in range(epoch):
        print("[Epoch {}/{}] lr={}".format(i + 1, epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        train_epoch_loss = 0
        progress = tqdm.tqdm(train_loader, total=len(train_loader))
        for (x, y, masks) in progress:
            x = x.to(device)
            y = y.to(device)
            masks = masks.to(device)
            out = model(x)
            # out = model(x, masks[0])
            optimizer.zero_grad()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        train_epoch_loss /= len(train_loader)
        train_loss_list.append(train_epoch_loss)

        val_epoch_loss = 0
        progress = tqdm.tqdm(val_loader, total=len(val_loader))
        with torch.no_grad():
            for (x, y, masks) in progress:
                x = x.to(device)
                y = y.to(device)
                masks = masks.to(device)
                out = model(x)
                # out = model(x, masks[0])
                loss = criterion(out, y)
                val_epoch_loss += loss.item()
        val_epoch_loss /= len(val_loader)
        val_loss_list.append(val_epoch_loss)
        print("train loss={}".format(train_epoch_loss))
        print("val loss={}".format(val_epoch_loss))
        scheduler.step()
    return train_loss_list, val_loss_list


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == '__main__':
    train_loader, val_loader = dataset.get_dataset(0.7)
    # device = "cuda:0"
    device = "cpu"
    # model = MDFT().to(device)
    model = LinearModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)
    train_loss_list, val_loss_list = train_and_val(model, train_loader, val_loader, 20, criterion, optimizer,
                                                   scheduler)
    torch.save(model.state_dict(), "TTS.pth")
    y_list = []
    pred_list = []
    progress = tqdm.tqdm(val_loader, total=len(val_loader))
    with torch.no_grad():
        for (x, y, masks) in progress:
            x = x.to(device)
            y = y.to(device)
            masks = masks.to(device)
            out = model(x)
            # out = model(x, masks[0])
            y_list.append(y.cpu().item())
            pred_list.append(out.cpu().item())
    y_list = np.array(y_list)
    pred_list = np.array(pred_list)
    test_loss = mean_absolute_percentage_error(y_list, pred_list)
    print("val loss={}".format(test_loss))
    fig = plt.figure()
    fig.suptitle('TTS\nval loss={0:.2f}%'.format(test_loss))
    ax_train_loss = fig.add_subplot(2, 2, 1)
    ax_val_loss = fig.add_subplot(2, 2, 2)
    ax_pred_y = fig.add_subplot(2, 1, 2)

    # 画loss
    ax_train_loss.plot(train_loss_list)
    ax_train_loss.set_title("train loss")
    ax_val_loss.plot(val_loss_list)
    ax_val_loss.set_title("val loss")
    ax_pred_y.plot(y_list, "blue", label="y")
    ax_pred_y.plot(pred_list, "red", label="pred")

    fig.savefig("TTS.png")
