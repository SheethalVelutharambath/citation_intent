import numpy as np
from tqdm import tqdm.

import numpy as np
from tqdm import tqdm

class Perceptron(object):
    def __init__(self, lr=0.2, itrs=100):
        self.lr = lr
        self.itrs = itrs
        self.W = []
        self.cost = []

    def fit_onevsall(self, X, Y):
        count = 0
        for i in tqdm(range(len(Y[0]))):
            print('Training: Class', count, 'vs Rest Classes')
            W = np.zeros(X.shape[1])
            print(W)
            print(W.shape)
            cost = []
            Y_ovr = Y[:, count]
            print(Y_ovr)
            for iteration in range(self.itrs):
                Z = X.dot(W)
                H = self.sigmoid(Z)
                W = self.gradient_desc(X, H, W, Y_ovr)
                cost.append(self.cal_cost(H, W, Y_ovr))
                if (iteration % 50 == 0):
                    print(self.cal_cost(H, W, Y_ovr))
            self.W.append((i, W))
            self.cost.append((i, cost))
            count = count + 1
        return self

    def gradient_desc(self, X, H, W, Y):
        gradient = np.dot((H - Y), X) / Y.size
        W = W - self.lr * gradient
        return W

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def cal_cost(self, H, W, Y):
        n = len(Y)
        cost = (np.sum(-Y.T.dot(np.log(H)) - (1 - Y).T.dot(np.log(1 - H)))) / n
        return cost

    def predict_ovr(self, X):
        Y_pred = [max((self.sigmoid(i.dot(W)), c) for c, W in self.W)[1] for i in X]
        return Y_pred

    def cal_score_ovr(self, X, Y):
        print(self.predict_ovr(X))
        score = sum(self.predict_ovr(X) == Y) / len(Y)
        return score