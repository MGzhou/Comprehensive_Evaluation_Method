#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File   : EWM.py
@Time   : 2024/04/27 17:29:25
@Desc   : 熵权法 Entropy Weight method
'''
import numpy as np
import pandas as pd


class EWM:
    def __init__(self) -> None:
        """
        初始化
        """
        self.W = None  # 保存权重
        self.max_val_list = None
        self.min_val_list = None


    def fit(self, X):
        """
        模型计算权重
        @X: 训练数据样本, np.ndarray
        @normal: 是否需要标准化X数据，默写进行标准化
        @max_val_list: 限定样本指标标准化时最大值
        @min_val_list: 限定样本指标标准化时最小值
        """
        [m, n] = X.shape  # m 个样本， n 个指标
        # 计算比重
        P = X / np.sum(X, axis=0)
        P = np.clip(P, a_min=1e-100, a_max=None)
        # 计算熵值
        e = -1 / np.log(m) * np.sum(P * np.log(P), axis=0)
        # 差异系数
        d = 1-e
        # 计算权重
        self.W = d / sum(d)
        return self.W


    def predict(self, X, normal=True):
        """
        预测函数
        """
        return X @ self.W
    
    def min_max_scaler_pos(self, X, min_val=None, max_val = None):
        """
        正向指标标准化
        """
        if min_val is None:
            min_val = np.min(X, axis=0)  # 获取每一列的最小值
        if max_val is None:
            max_val = np.max(X, axis=0)
        X = (X - min_val) / (max_val - min_val)
        return X, min_val, max_val


    def min_max_scaler_neg(self, X, min_val=None, max_val = None):
        """
        负向指标标准化
        """
        if min_val is None:
            min_val = np.min(X, axis=0)  # 获取每一列的最小值
        if max_val is None:
            max_val = np.max(X, axis=0)
        X = (max_val - X) / (max_val - min_val)
        return X, min_val, max_val



def test():
    import os
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(root)
    root = os.path.dirname(root)
    df = pd.read_excel(os.path.join(root, "data/综合评价数据.xlsx"), index_col="地区")
    print(f"部分数据: \n{df.head()}\n")

    X = df.to_numpy()
    print(X[0])

    ewm = EWM()
    # 标准化
    X, min_val, max_val = ewm.min_max_scaler_pos(X)

    ewm.fit(X)

    print(f"权重 = {ewm.W}")  # [0.0631 0.05947 0.0554 0.0633 0.1908 0.2255 0.2053 0.1372]

    test_X = X

    # 需要手动标准化
    test_X, _, _ = ewm.min_max_scaler_pos(test_X, ewm.min_val_list, ewm.max_val_list)

    # 预测
    target = ewm.predict(test_X)
    # target = list(target)  # np转为list
    print(f"预测结果: {target}")



if __name__ == '__main__':
    test()

