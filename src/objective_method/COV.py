#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File   : COV.py
@Time   : 2024/04/29 21:48:30
@Desc   : None
'''

import numpy as np
import pandas as pd


class COV:
    def __init__(self) -> None:
        self.W = None
        self.S = None  # 标准差
        self.V = None  # 变异系数
    
    def fit(self, X):
        
        # 标准差
        self.S = np.std(X, axis=0)
        # 变异系数
        self.V = self.S / np.mean(X, axis=0)
        # 权重
        self.W = self.V / np.sum(self.V)

        return self.W
    
    def predict(self, X):

        return X @ self.W

    def min_max_scaler_pos(self, X, min_val=None, max_val=None):
        """
        正向指标标准化
        """
        if min_val is None:
            min_val = np.min(X, axis=0)  # 获取每一列的最小值
        if max_val is None:
            max_val = np.max(X, axis=0)
        X = (X - min_val) / (max_val - min_val)
        return X, min_val, max_val


    def min_max_scaler_neg(self, X, min_val=None, max_val=None):
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
    # 地区生产总值	人均可支配收入	人均消费支出
    data = {
        "广西壮族自治区":[27202, 29514, 19749],
        "重庆市":[30146,37595,	26515],
        "四川省":[60133,32514,	23550],
        "贵州省":[20913,27098,	20161],
        "云南省":[30021,28421,	20995],
    }

    df = pd.DataFrame(data)
    print(df)
    X = df.to_numpy().T

    cov = COV()  # 创建变异系数法对象

    X_norm, min_val, max_val = cov.min_max_scaler_pos(X)

    w = cov.fit(X_norm)

    print(f"权重={w}")

    # 预测
    y = cov.predict(X_norm)
    print(f"预测结果={y}")

if __name__ == '__main__':
    main()

