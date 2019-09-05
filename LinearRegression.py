#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 08:58:39 2019

@author: ymguo
"""

'''
1. Reorganize Linear Regression in Python mode.

You have seen what we are doing in class to do linear regression. 
That is not bad in C++. 
But it's not a good idea in Python because we were not using Python's features at all.
So, your first task is: rewrite linear regression code in Python.
You are not allowed to use "Too Many For Loops", especially when doing calculations.
Write the code in "Python's way". Go ahead and good luck.

'''
import random
import numpy as np
import matplotlib.pyplot as plt

def inference(X, Theta):
    '''
    X: (nb_samples, n+1), `nb_samples`: 样本数
    Theta: (n+1,), `n`: 特征数
    retval: (nb_samples,), 'retval': 返回值
    '''
    return X @ Theta

def eval_loss(X, Y_gt, Theta):
    '''
    X: (nb_samples, n+1), `nb_samples`: 样本数
    Y_gt: (nb_samples,)
    Theta: (n+1,), `n`: 特征数
    '''
    nb_samples = len(X)
    diff = inference(X, Theta) - Y_gt
    avg_loss = 0.5 * sum(diff ** 2) / nb_samples
    return avg_loss

def gradient(Y_pred, Y_gt, X):
    '''
    Y_pred: (nb_samples,), `nb_samples`: 样本数
    Y_gt: (nb_samples,)
    X: (nb_samples, n+1)
    retval: (n+1,)
    '''
    nb_samples = len(X)
    dTheta = X.transpose() @ (Y_pred - Y_gt) / nb_samples
    # transpose 作用是改变序列,不指定参数是默认矩阵转置
    # dw = diff * x
    return dTheta



def cal_step_gradient(X, Y_gt, Theta, lr):
    '''
    X: (nb_samples, n+1): `nb_samples`: 样本数, `n`: 特征数
    Y_gt: (nb_samples,)
    Theta: (n+1,)
    lr: learning rate
    retval: (n+1,)
    '''
    Y_pred = inference(X, Theta)
    dTheta = gradient(Y_pred, Y_gt, X)
    Theta = Theta - lr * dTheta
    return Theta

def train(X, Y_gt, batch_size, lr, max_iter):
    '''
    X: (nb_samples, n+1), `nb_samples`: 样本数, `n`: 特征数
    Y_gt: (nb_samples,)
    '''
    Theta = np.zeros((2,), dtype=X.dtype)
    nb_samples = len(X)
    loss_list = []
    for _ in range(max_iter):
        batch_idxs = np.random.choice(nb_samples, batch_size)
        X_batch = X[batch_idxs, :]
        Y_batch = Y_gt[batch_idxs]
        Theta = cal_step_gradient(X_batch, Y_batch, Theta, lr)
        loss = eval_loss(X, Y_gt, Theta)
        loss_list.append(loss)
#        print(loss)
#        if loss > loss_list[_ - 1]:
#            break
#    if draw: 
#        plt.plot(loss_list)
#        plt.xlabel("iteration")
#        plt.ylabel("loss")
#        plt.title("Train Loss")
#        plt.savefig("./train_loss.jpg", dpi=1000)
    return Theta, loss_list

def gen_sample_data(nb_samples=100):
    theta_0 = random.random() * 5
    theta_1 = random.random() * 10

    xs = [random.random() * 10 for _ in range(nb_samples)]
    ys = [theta_0 + theta_1 * x + random.random() * 2 - 1 for x in xs]

    return xs, ys, theta_0, theta_1

def gen_sample_matrix(nb_samples=100):
    xs, ys, theta_0, theta_1 = gen_sample_data(nb_samples)

    X = np.array([xs], dtype=np.float).transpose()
    One = np.ones((nb_samples, 1))
    X = np.hstack((One, X))
    # np.hstack():在水平方向上平铺
    
    Y = np.array(ys, dtype=np.float)

    Theta = np.array([theta_0, theta_1], dtype=np.float)
    return X, Y, Theta

def draw(X, Y, Theta, loss):
    '''
    X: (nb_samples, n+1): `nb_samples`: 样本数, `n`: 特征数
    Y_gt: (nb_samples,)
    Theta: (n+1,)
    loss: (iter,), `iter`: 迭代次数
    '''
    fig, [plt1, plt2] = plt.subplots(1, 2)
    fig.suptitle("Linear Regression")

    plt1.set_xlabel("x")
    plt1.set_ylabel("y")
    plt1.scatter(X[:, 1], Y, label="samples")

    x = np.linspace(0, 10, 100)
    y = Theta[0] + x * Theta[1]
    plt1.plot(x, y, label="pred", color='red')
    plt1.legend()

    plt2.plot(list(range(len(loss))), loss, label="loss")
    plt2.set_xlabel("iter")
    plt2.set_ylabel("loss")
    plt.savefig("./linear_regression_train_loss.jpg", dpi=1000)
    plt.show()
    
def run():
    nb_samples = 100
    lr = 0.001
    max_iter = 100
#    draw = True
    X, Y, Theta = gen_sample_matrix(nb_samples)
    print("X: {}, Y: {}, Theta: {}".format(X.shape, Y.shape, Theta.shape))

    model, loss = train(X, Y, 50, lr, max_iter)
    # model就是参数Theta
    print('Theta:{} \n{}\ntrain:{} \n{}'.format(
        Theta.shape, Theta, model.shape, model))

    draw(X, Y, model, loss)

if __name__ == "__main__":
# 跑.py的时候，跑main下面的；
# 被导入当模块时，main下面不跑，其他当函数调.
    run()












