from classModel import Model
import numpy as np

class Perceptron(Model):


    def __init__(self, lr):
        self.performance = []
        self.lr = lr
        self.weights = None


    def prediction(self, X):
        net = np.dot(self.weights[1:], X) + self.weights[0]
        return 1 if net > 0.0 else 0


    def _fit_epoch(self, X, Y, epoch, mode, verbose):
        error = 0
        for x, y in zip(X, Y):
            upd = self.lr * (y - self.prediction(x))
            self.weights[0] += upd
            self.weights[1:] += upd * x
            erroe += int(upd != 0.0)
        if epoch % 10 == 0 and verbose:
            print('Epoch %d : %d errors' % (epoch, error))
        self.performance.append((epoch, error, self.weights[1:], self.weights[0]))
        return error
