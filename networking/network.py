# coding: utf-8
# author: vonng(fengruohang@outlook.com)
# ctime: 2017-05-10

import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.L = len(sizes)
        #神经网络层数
        self.layers = range(0, self.L - 1)
        #神经网络权重
        self.w = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        #神经网络偏值
        self.b = [np.random.randn(x, 1) for x in sizes[1:]]

    #前馈a = &(wx + b),返回结果a
    def feed_forward(self, a):
        for l in self.layers:
            a = 1.0 / (1.0 + np.exp(-np.dot(self.w[l], a) - self.b[l]))
        return a

    #梯度下降
    def gradient_descent(self, train, test, epoches=30, m=10, eta=3.0):
        for round in range(epoches):
            #用mini batch方法训练
            random.shuffle(train)
            for batch in [train_data[k:k + m] for k in range(0, len(train), m)]:
                x = np.array([item[0].reshape(784) for item in batch]).transpose()
                y = np.array([item[1].reshape(10) for item in batch]).transpose()
                n, r, a = len(batch), eta / len(batch), [x]
                
                # forward & save activations
                for l in self.layers:
                    a.append(1.0 / (np.exp(-np.dot(self.w[l], a[-1]) - self.b[l]) + 1))

                # back propagation
                d = (a[-1] - y) * a[-1] * (1 - a[-1])    #BP1
                for l in range(1, self.L):  # l is reverse index since last layer
                    if l > 1:    #BP2
                        d = np.dot(self.w[-l + 1].transpose(), d) * a[-l] * (1 - a[-l])
                    self.w[-l] -= r * np.dot(d, a[-l - 1].transpose()) #BP3
                    self.b[-l] -= r * np.sum(d, axis=1, keepdims=True) #BP4
            
            # evaluate
            acc_cnt = sum([np.argmax(self.feed_forward(x)) == y for x, y in test])
            print("Round {%d}: {%s}/{%d}" % (round, acc_cnt, len(test_data)))


if __name__ == '__main__':
    import mnist_loader

    train_data, valid_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 100, 50, 10])
    net.gradient_descent(train_data, test_data, epoches=100, m=10, eta=2.0)