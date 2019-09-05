#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:16:34 2019

@author: ymguo
"""

'''
2. Logistic regression:
    
Logistic regression is widely used.
We derived the cost function and it's gradient in class. 
Please complete the logistic regression code in "Python's way" as well.
Tips: It's almost like the linear regression code. 
      The only difference is you need to complete a sigmoid function and use the result of that as your "new X" and also you need to generate your own training data.

'''

# %%

import random
import matplotlib.pyplot as plt
import numpy as np

def inference(X, Theta):
    '''
    X: (nb_samples, n+1), `nb_samples`: 样本数
    Theta: (n+1,), `n`: 特征数
    retval: (nb_samples,), 'retval': 返回值
    '''
    z = X @ Theta
    return 1 / (1 + np.exp(-z))

def eval_loss(X, Y_gt, Theta):
    '''
    X: (nb_samples, n+1), `nb_samples`: 样本数
    Y_gt: (nb_samples,)
    Theta: (n+1,), `n`: 特征数
    '''
    Y_pred = inference(X, Theta)  # (m, 1)
    loss = -sum(Y_gt * np.log(Y_pred) + (1 - Y_gt) * np.log(1 - Y_pred))
    avg_loss = loss / len(X)
    return avg_loss

def gradient(Y_pred, Y_gt, X):
    """
    Y_pred: (nb_samples,), `nb_samples`: 样本数
    Y_gt: (nb_samples,)
    X: (nb_samples, n+1)
    retval: (n+1,)
    """
    nb_samples = len(X)
    dTheta = X.transpose() @ (Y_pred - Y_gt) / nb_samples
    # transpose 作用是改变序列,不指定参数是默认矩阵转置
    # dw = diff * x
    return dTheta

def cal_step_gradient(X, Y_gt, Theta, lr):
    """
    X: (nb_samples, n+1): `nb_samples`: 样本数, `n`: 特征数
    Y_gt: (nb_samples,)
    Theta: (n+1,)
    lr: learn rate
    retval: (n+1,)
    """
    Y_pred = inference(X, Theta)
    dTheta = gradient(Y_pred, Y_gt, X)
    Theta = Theta - lr * dTheta
    return Theta

def train(X, Y_gt, batch_size, lr, max_iter):
    """
    X: (nb_samples, n+1), `nb_samples`: 样本数, `n`: 特征数
    Y_gt: (nb_samples,)
    """
    Theta = np.zeros((3,), dtype=X.dtype)
    nb_samples = len(X)
    loss_list = []
    for _ in range(max_iter):
        batch_idxs = np.random.choice(nb_samples, batch_size)
        X_batch = X[batch_idxs, :]
        Y_batch = Y_gt[batch_idxs]
        Theta = cal_step_gradient(X_batch, Y_batch, Theta, lr)
        loss = eval_loss(X, Y_gt, Theta)
        loss_list.append(loss)
    return Theta, loss_list

def gen_sample_data(nb_samples=100):
    b = 5 + random.random() * 2 - 1  # [4,6)
#    k = -1 + random.random() * 1 - 0.5  # [-1.5,1.5)
    k = -1 + random.random() * 3 - 0.5  # [-1.5,1.5)
    # random.random()生成0和1之间的随机浮点数float
    xrange = 10
    bottom = b
    top = k * xrange + b

    xs1 = [random.random() * xrange for _ in range(nb_samples)]
    xs2 = [(top - bottom) * random.random() +
           bottom for _ in range(nb_samples)]

    def random_y(x1, x2):
        delta = 0.5
        boundary = b + k * x1 + random.random() * delta - delta * 0.5
        return x2 >= boundary and 1 or 0

    ys = [random_y(xs1[i], xs2[i]) for i in range(nb_samples)]

    return xs1, xs2, ys, k, b


def gen_sample_matrix(nb_samples=100):
    xs1, xs2, ys, k, b = gen_sample_data(nb_samples)

    dtype = np.float
    X = np.array([xs1, xs2], dtype=dtype).transpose()
    One = np.ones((nb_samples, 1))
    X = np.hstack((One, X))
    # np.hstack():在水平方向上平铺
    Y = np.array(ys, dtype=dtype)
#   Theta = np.array([b, k], dtype=np.float)
#   return X, Y, Theta
    return X, Y, k, b

def draw(X, Y, model, loss):
    '''
    X: (nb_samples, n+1): `nb_samples`: 样本数, `n`: 特征数
    Y: (nb_samples,)
    model: (n+1,): 参数
    loss: (iter,), `iter`: 迭代次数
    '''
    fig, [plt1, plt2] = plt.subplots(1, 2)
    fig.suptitle("Logistic Regression")

    plt1.set_xlabel("x1")
    plt1.set_ylabel("x2")
    pos_samples = (Y == 1)
    neg_samples = pos_samples == False
    pos = X[pos_samples, :]
    neg = X[neg_samples, :]
#   绘制散点图
    plt1.scatter(pos[:, 1], pos[:, 2], label="positive", color="red")
    plt1.scatter(neg[:, 1], neg[:, 2], label="negative", color="yellow")

    theta0 = model[0]
    theta1 = model[1]
    theta2 = model[2]
    xs1 = np.linspace(0, 10, 100)
    xs2 = []

    if theta2 != 0:
        xs2 = [-(theta0+theta1*x)/theta2 for x in xs1]
    plt1.plot(xs1, xs2, label="h_theta", color='blue')

    plt2.plot(list(range(len(loss))), loss, label='loss')
    plt2.set_xlabel("iter")
    plt2.set_ylabel("loss")
    plt.savefig("./logistic_regression_train_loss.jpg", dpi=1000)
    plt.show()

def run():
    nb_samples = 500
    lr = 0.03
    max_iter = 3000

    X, Y, k, b = gen_sample_matrix(nb_samples)
    print("X: {}, Y: {}, k: {}, b: {}".format(X.shape, Y.shape, k, b))

    model, loss = train(X, Y, 50, lr, max_iter)

    draw(X, Y, model, loss)

# %%
if __name__ == "__main__":
# 跑.py的时候，跑main下面的；
# 被导入当模块时，main下面不跑，其他当函数调.
    run()