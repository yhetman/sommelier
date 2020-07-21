from  dataVisualization import read_data
import pandas as pd
import numpy as np
from classModel import Model
from classPerceptron import Perceptron
import dataAnimation

def dataCleaning(data):
    data = data.assign(highQ = pd.Series(data['quality'] > 5))
    features = ['quality', 'alcohol', 'highQ', 'volatile acidity']
    samples = data[(data['quality'] < 4) | (data['quality'] > 7)][features]
    print(samples)
    Y = samples['highQ'].values
    X = samples.loc[:,['alcohol', 'volatile acidity']]
    return X, Y, samples


def main():
    data = read_data('../data/winequality-red.csv')
    X, Y, samples = dataCleaning(data)
    model = Perceptron(lr = 0.001)
    training_statistics = model.fit(X.values, Y, epochs = 5000, verbose = True, seed = 29)
    for row in training_statistics:
        print(row)
    accuracy = model.evaluation(X.values, Y)
    figure = dataAnimation.plot_performance(training_statistics, samples, ['alcohol', 'volatile acidity'], 7, 4, -1, False)
    plt.show(figure)
    

if __name__ == "__main__":
    main()
