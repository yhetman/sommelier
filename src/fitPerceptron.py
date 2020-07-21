from  dataVisualization import read_data
import pandas as pd
from classPerceptron import Perceptron
from plotVisualization import plot_performance
import matplotlib.pyplot as plt
import matplotlib.animation as anima
from dataAnimation import save_animation

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
    training_statistics = model.fit(X.values, Y, epochs = 5001, verbose = True, seed = 29)
    for row in training_statistics:
        print(row)
    accuracy = model.evaluation(X.values, Y)
    save_animation(training_statistics, samples, ['alcohol', 'volatile acidity'], 7, 4)


if __name__ == "__main__":
    main()
