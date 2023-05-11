# -*- coding: utf-8 -*-
'''
Multi-Linear Network
'''
# @Time : 2023/5/3 11:29
# @Author : LINYANZHEN
# @File : MLN.py
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


class MLN(nn.Module):
    def __init__(self):
        super(MLN, self).__init__()
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
            out = model(x)
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
                out = model(x)
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


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.sum(np.power((y_true - y_pred), 2))))


if __name__ == '__main__':
    train_loader, val_loader = dataset.get_dataset(0.7)
    # device = "cuda:0"
    device = "cpu"
    model = MLN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)
    train_loss_list, val_loss_list = train_and_val(model, train_loader, val_loader, 20, criterion, optimizer,
                                                   scheduler)
    torch.save(model.state_dict(), "MLN.pth")

    # 导入模型，在测试集中
    model.load_state_dict(torch.load("MLN.pth"))
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
    fig ,ax= plt.subplots()
    fig.suptitle('MLN\nval loss={0:.2f}'.format(test_loss))

    # ax.plot(y_list, "blue", label="y")
    # ax.plot(pred_list, "red", label="pred")
    # fig.savefig("MLN_test.png")

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
    fig.savefig("MLN.png")
