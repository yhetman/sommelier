import numpy as np
import pandas as pd
import random

from  dataVisualization import read_data
from classAdaline import Adaline
from dataAnimation import save_animation
from plotVisualization import plot_performance
from normalization import normalization





data = read_data('../data/winequality-red.csv')

data = data.assign(highQ = pd.Series(data['quality'] > 5))
features = ['volatile acidity', 'alcohol', 'quality', 'highQ']
cleaned_data = data[(data['quality'] > 6) | (data['quality'] < 5)][features]
cleaned_data = cleaned_data.reset_index(drop=True)

cleaned_data['alcohol'] = normalization(cleaned_data['alcohol'])
cleaned_data['volatile acidity'] = normalization(cleaned_data['volatile acidity'])
X = cleaned_data.loc[:,['volatile acidity', 'alcohol']]
Y = cleaned_data['highQ']

def best_lr_Adaline():
    best_lr = 0
    min_errors = 26

    for n in range(100):
        lr = round(random.uniform(0.001, 0.05), 5)
        model = Adaline(lr)
        stats = model.fit(X.values, Y, 1000, 'batch', verbose=False)
        min_num_errors = min([elem[1] for elem in stats])
        last_error = stats[-1][1]
        print('tring : %d,  lr : %f,  min_num_errors : %d,  min errors %d,' %(n, lr, min_num_errors, min_errors))

        if min_num_errors < min_errors:
            min_errors = min_num_errors
            best_lr = lr

    print("Best learnig rate: %f, num errors: %d" % (best_lr, min_errors))
    return best_lr


def main():
    best_lr = best_lr_Adaline()
    model = Adaline(lr=best_lr)
    stats = model.fit(X.values, Y, 3001, mode='batch', verbose=True)
    figure = plot_performance(stats, cleaned_data, ['alcohol', 'volatile acidity'], 6, 5, 3000, False)
    plt.show(figure)
    plt.savefig('../images/adaline-performance-plot.png')

if __name__ == "__main__":
    main()



