#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File   : critic.py
@Time   : 2024/04/29 21:21:30
@Desc   : critic 法
'''
import numpy as np
import pandas as pd

class Critic:

    def __init__(self) -> None:
        self.W = None
        self.S = None  # 对比强调，标准差
        self.corr = None  # 皮尔逊相关系数
        self.R = None  # 冲突性，相关系数
    
    def fit(self, X):
        """
        @X: 输入数据,输入前需要进行标准化, 格式np.ndarray
        """
        # 计算对比强调，标准差
        self.S = np.std(X, axis=0)

        # 计算冲突性
        self.corr = np.corrcoef(X.T)
        self.R = np.sum(1-self.corr, axis=0)

        # 计算信息量
        self.C = self.S * self.R

        # 权重
        self.W = self.C / np.sum(self.C)

        return self.W
    
    def predict(self, X):
        """
        预测
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



def main():
    # A1 到 A4 4个指标
    data = {'A1': [7, 6, 5, 7],
        'A2': [9, 7, 6, 8],
        'A3': [6, 5, 8, 6],
        'A4': [8, 7, 7, 9]}

    df = pd.DataFrame(data)
    X = df.to_numpy()
    critic = Critic()
    X_norm, min_val, max_val = critic.min_max_scaler_pos(X)
    W = critic.fit(X_norm)

    print(f"权重= {W}")

    test_X = [8, 7, 8, 9]
    test_X_norm, _, _ = critic.min_max_scaler_pos(test_X, min_val, max_val)

    y = critic.predict(test_X_norm)

    print(f"预测值= {y}")

if __name__ == '__main__':
    main()

