import pandas as pd
import matplotlib.pyplot as plt

from  dataVisualization import read_data
from classPerceptron import Perceptron
from fitPerceptron import dataCleaning
from dataAnimation import save_animation
from plotVisualization import plot_performance

def normalization(series):
    if not isinstance(series, pd.Series):
        raise ValueError('input argument is not an instance of pandas.Series class')
    return (series - series.mean()) / (series.max() - series.min())


def main():
    data = read_data('../data/winequality-red.csv')
    X, Y, normeDB = dataCleaning(data)
    normeDB['alcohol'] = normalization(normeDB['alcohol'])
    normeDB['volatile acidity'] = normalization(normeDB['volatile acidity'])
    X = normeDB.loc[ :,['volatile acidity', 'alcohol' ]]
    Y = normeDB['highQ'].values
    model  = Perceptron(lr = 0.001)
    stats = model.fit(X.values, Y,  epochs=1000, verbose = True, seed=2929)
    #fig = plot_performance(model.performance, normeDB, ['volatile acidity', 'alcohol'], 7, 4, 300, True)
    #save_animation(model.performance, normeDB, ['volatile acidity', 'alcohol'], 7 , 4 )
    #plt.show(fig)

if __name__ == "__main__":
    main()
 
