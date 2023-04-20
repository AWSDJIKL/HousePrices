# -*- coding: utf-8 -*-
'''
处理缺失值
'''
# @Time : 2023/4/17 11:08
# @Author : LINYANZHEN
# @File : missing_data_processing.py
import pandas as pd


def missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percentage = round(df.isnull().sum().sort_values(ascending=False) * 100 / len(df), 2)[
        df.isnull().sum().sort_values(ascending=False) * 100 / len(df) != 0]
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])


# 读取csv
train = pd.read_csv("house-prices-advanced-regression-techniques/train.csv", header=0, index_col=0)
test = pd.read_csv("house-prices-advanced-regression-techniques/test.csv", header=0, index_col=0)
print("train")
print(missing_percentage(train))
print("test")
print(missing_percentage(test))

# 这部分属性的缺失值是有其中含义，因此使用None代替
missing_val_col = ["Alley",
                   "PoolQC",
                   "MiscFeature",
                   "Fence",
                   "FireplaceQu",
                   "GarageType",
                   "GarageFinish",
                   "GarageQual",
                   "GarageCond",
                   'BsmtQual',
                   'BsmtCond',
                   'BsmtExposure',
                   'BsmtFinType1',
                   'BsmtFinType2',
                   'MasVnrType']

for i in missing_val_col:
    train[i] = train[i].fillna('None')
    test[i] = test[i].fillna('None')

# 这部分连续特征的缺失值使用0代替
missing_val_col2 = ['BsmtFinSF1',
                    'BsmtFinSF2',
                    'BsmtUnfSF',
                    'TotalBsmtSF',
                    'BsmtFullBath',
                    'BsmtHalfBath',
                    'GarageYrBlt',
                    'GarageArea',
                    'GarageCars',
                    'MasVnrArea']

for i in missing_val_col2:
    train[i] = train[i].fillna(0)
    test[i] = test[i].fillna(0)

# 使用物理位置的均值填充确实的房子周围的街道数
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

for df in [train, test]:
    # 有些类别特征以数字形式给出，所以要转变成类别变量
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    # mode指的是最常出现的值
    df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    # 重要的年份和月份应当是类别变量而不是连续变量
    df['YrSold'] = df['YrSold'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)
    # 剩余变量填充
    df['Functional'] = df['Functional'].fillna('Typ')
    df['Utilities'] = df['Utilities'].fillna('AllPub')
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['KitchenQual'] = df['KitchenQual'].fillna("TA")
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
    df['Electrical'] = df['Electrical'].fillna("SBrkr")

print("train pre-process")
print(missing_percentage(train))
print("test pre-process")
print(missing_percentage(test))

# # 进一步预处理，将文本特征转为离散型数字特征
all_data = pd.concat([train.iloc[:, :-1], test])
print(all_data)
non_numeric_cols = all_data.select_dtypes(exclude=['number']).columns
all_data[non_numeric_cols] = all_data[non_numeric_cols].apply(lambda x: pd.factorize(x)[0])
# 分别对每一列做归一化
all_data = all_data.apply(lambda x: (x - x.mean()) / x.std())
print(all_data)
train.iloc[:, :-1] = all_data.iloc[:len(train), :]
test.iloc[:, :] = all_data.iloc[len(train):, :]

# # 将处理后的数据进行保存
train.to_csv("train_pre-process.csv")
test.to_csv("test_pre-process.csv")
