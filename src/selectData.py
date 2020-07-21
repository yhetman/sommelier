from  dataVisualization import read_data
import pandas as pd
import numpy as np


def data_cleaning(data):
    data = data.assign(highQ = pd.Series(data['quality'] > 5))
    features = ['quality', 'alcohol', 'highQ', 'volatile acidity']
    samples = data[(data['quality'] < 4) | (data['quality'] > 7)][features]
    print(samples)
    Y = samples['highQ'].values
    X = samples.loc[:,['alcohol', 'volatile acidity']]
    return X, Y

def main():
    data = read_data('../data/winequality-red.csv')
    X, Y = data_cleaning(data)
    print(X)
    print(Y)
    
if __name__ == "__main__":
    main()
