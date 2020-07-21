import os
import csv
import pandas as pd
import matplotlib.pyplot as plt


def read_data(fpath):
    try:
        df = pd.DataFrame(pd.read_csv(fpath, sep = ';'))
    except FileNotFoundError:
        print('Double check the file path')
    return df


def check_quality(df):
    good_treshold = 7
    bad_treshold = 4
    print('OK')
    g_wines = df[(df['quality'] > good_treshold)]
    b_wines = df[(df['quality'] < bad_treshold)]
    print('Quality checked')
    return g_wines, b_wines


def plot_scatter_matrix(df_wine, good_wines, bad_wines, save_plot=False):
    samples, feats = df_wine.shape
    print('samples and feats done')
    figure, axes = plt.subplots(nrows = feats, ncols = feats, figsize=(18,18))
    figure.subplots_adjust(hspace=0, wspace=0)
    print('subplots generated')
    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    for i, title in enumerate(df_wine.columns):
        axes[i, i].annotate(title, (0.5, 0.5), xycoords = 'axes fraction', va = 'center', ha = 'center')
    for i in range(feats):
        for j in range(i + 1, feats):
            axes[i, j].scatter(good_wines.iloc[:, j], good_wines.iloc[:, i], c = 'blue', marker = '.')
            axes[j, i].scatter(good_wines.iloc[:, i], good_wines.iloc[:, j], c = 'blue', marker = '.')
            axes[i, j].scatter(bad_wines.iloc[:, j], bad_wines.iloc[:, i], c = 'red', marker = '.')
            axes[j, i].scatter(bad_wines.iloc[:, i], bad_wines.iloc[:, j], c = 'red', marker = '.')
    print('markers created')
    if save_plot :
        plt.savefig('wine-quality-scatter-matrix.png')
    return figure


def main():
    wine_data = read_data("../data/winequality-red.csv")
    print(wine_data.head())
    good_wines, bad_wines = check_quality(wine_data)
    f = plot_scatter_matrix(wine_data, good_wines, bad_wines, False)
    #plt.show(f)


if __name__ == "__main__":
    main()
