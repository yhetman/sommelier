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
    model = Perceptron(lr = 0.005)
    training_statistics = model.fit(X.values, Y, epochs = 1000, verbose = True, seed = 29)
    print("|length og statistics| %d |" % (len(training_statistics)))
    i = 0
    for row in training_statistics:
        i += 1
        print("|%d|" %(i), row)
    plot_performance(training_statistics, samples, ['alcohol', 'volatile acidity'], 7, 4, 999, True)
    save_animation(training_statistics, samples, ['alcohol', 'volatile acidity'], 7, 4)
   # accuracy = model.evaluation(X.values, Y)


if __name__ == "__main__":
    main()
