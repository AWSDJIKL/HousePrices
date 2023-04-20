# -*- coding: utf-8 -*-
'''
数据集处理
'''
# @Time : 2023/4/19 10:28
# @Author : LINYANZHEN
# @File : dataset.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class HousePriceDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.x = torch.tensor(np.array(df.iloc[:, 1:-1])).type(torch.float)
        self.y = torch.tensor(np.array(df.iloc[:, -1])).type(torch.float)
        self.masks = self._generate_square_subsequent_mask(79 - 1)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y, self.masks

    def __len__(self):
        return len(self.y)

    def _generate_square_subsequent_mask(self, t0):
        mask = torch.zeros(t0 + 1, t0 + 1)
        for i in range(0, t0):
            mask[i, t0:] = 1
        for i in range(t0, t0 + 1):
            mask[i, i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))  # .masked_fill(mask == 1, float(0.0))
        return mask


def get_dataset(train_percent=0.7):
    train_data = pd.read_csv("train_pre-process.csv").sample(frac=1.0)
    test_data = pd.read_csv("test_pre-process.csv")
    train_loader = DataLoader(
        HousePriceDataset(train_data[:int(train_percent * len(train_data))].reset_index(drop=True)), batch_size=10,
        shuffle=True)
    val_loader = DataLoader(HousePriceDataset(train_data[int(train_percent * len(train_data)):].reset_index(drop=True)))
    return train_loader, val_loader


if __name__ == '__main__':
    train_data = pd.read_csv("train_pre-process.csv").sample(frac=1.0)
