#!usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


def generateData(show=True):
    # 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
    X1, y1 = make_gaussian_quantiles(cov=2.0, n_samples=500, n_features=2, n_classes=2, random_state=1)
    # 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3，协方差系数为2
    X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=400, n_features=2, n_classes=2, random_state=1)
    # 讲两组数据合成一组数据
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, - y2 + 1))
    if show:
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
        plt.show()
    return X, y


def adaboostClassification(X, y):
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=200, learning_rate=0.8)
    bdt.fit(X, y)
    return bdt


def pltResult(adaboost, X, y, show=True):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 列中最大最小值
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 行中最大最小值
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))  # 从坐标向量中返回矩阵

    Z = adaboost.predict(np.c_[xx.ravel(), yy.ravel()])  # 按行连接两个矩阵 i.e. 两矩阵左右相加 要求行数相等 (r - 按列)
    Z = Z.reshape(xx.shape)
    if show:
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)  # 等高线
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
        plt.show()


def showResult(adaboost, X, y):
    score = adaboost.score(X, y)
    print('score: {}'.format(score))


if __name__ == '__main__':
    X, y = generateData(show=False)
    adaboost = adaboostClassification(X, y)
    pltResult(adaboost, X, y, show=False)
    showResult(adaboost, X, y)