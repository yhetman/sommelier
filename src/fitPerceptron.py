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
    samples = samples.reset_index(drop=True)
    Y = samples['highQ'].values
    X = samples.loc[:,['alcohol', 'volatile acidity']]
    return X, Y, samples


def main():
    data = read_data('../data/winequality-red.csv')
    X, Y, samples = dataCleaning(data)
    model = Perceptron(lr = 0.005)
    training_statistics = model.fit(X.values, Y, epochs = 1000, verbose = True, seed = 29)
   # plot_performance(model.performance, sampVes, ['volatile acidity', 'alcohol'], 7, 4, 300, True)
    save_animation(model.performance, samples, ['volatile acidity', 'alcohol'], 7 , 4 )
   # accuracy = model.evaluation(X.values, Y)


if __name__ == "__main__":
    main()
