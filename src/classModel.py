import abc
import random

class Model(abc.ABC):

    def __init__(self, lr):
        self.performance = []
        self.lr = lr
        self.weights = None


    @abc.abstractmethod
    def prediction(self, X):
        pass


    @abc.abstractmethod
    def _fit_epoch(self, X, Y, epoch, mode, verbose):
        pass


    def evaluation(self, X, Y):
        accuracy = 0
        for x, y in zip(X, Y):
            accuracy += int(int(y) == self.prediction(x))
        accuracy /= len(Y)
        print("The accuracy of Model: %.6f" % (accuracy))
        return accuracy


    def fit(self, X, Y, epochs, mode = 'batch', verbose = False, seed = None):
        if self.weights is None:
            self.weights = [0.0001 * random.uniform(-1, 1) for i in range(X.shape[1] + 1)]
        if mode not in ['batch', 'stochastic']:
            raise ValueError("invalid training mode")
        if epochs < 0:
            raise ValueError("invlid number of epochs")
        if seed:
            random.seed(seed)
        curr_epoch = 0
        while True:
            epoch_error = self._fit_epoch(X, Y, curr_epoch, mode, verbose)
            curr_epoch += 1
            if epochs != 0 and curr_epoch == epochs:
                break
            if epochs == 0 and epoch_error == 0:
                break
        return self.performance


