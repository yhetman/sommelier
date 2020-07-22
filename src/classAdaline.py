# ADAptive LInear NEuron

from classModel import Model
from normalization import normalization
from dataVisualization import read_data
from fitPerceptron import dataCleaning
from dataAnimation import save_animation
from plotVisualization import plot_performance
from math import exp

import pandas as pd
import numpy as np

class Adaline(Model):


    def __init__(self, lr):
        self.lr = lr
        self.weights = None
        self.performance = []


    def prediction(self, X):
        return 1 if self._activation(X) > 0.5 else 0


    def _activation(self, X):
        output = np.dot(self.weights[1:], X) + self.weights[0]
        return (1 / (1 + exp(- output)))


    def _evaluation(self, X, Y, epoch):
        mistakes = 0
        for x, y in zip(X, Y):
            mistakes += int(self.prediction(x) != int(y))
        return mistakes


    def _fit_epoch(self, X, Y, epoch, mode, verbose):
        errors = []
        for x, y in zip(X, Y):
            error = y - self._activation(x)
            errors.append(error)

            if mode == 'stochastic':
                self.weights[0] += self.lr * error
                self.weights[1:] += self.lr * error * x
        if mode == 'batch':
            self.weights[0] += self.lr * sum(errors)
            self.weights[1:] += self.lr * np.dot(errors, X)

        mistakes = self._evaluation(X, Y, epoch)
        if epoch % 10 == 0 and verbose:
            print('Epoch %d : %d errors' % (epoch, mistakes))
        self.performance.append((epoch, mistakes, self.weights[1:], self.weights[0]))
        return errors
