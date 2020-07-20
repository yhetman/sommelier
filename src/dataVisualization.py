import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

def read_data(fpath):
    try:
        df = pd.read_csv(fpath)
    except FileNotFoundError:
        print('Double check the file path')
    return df

def check_quality(df):
    good_treshold = 7
    bad_treshold = 4
    g_wines = df[(df['quality'] > good_treshold)]
    b_wines = df[(df['quality'] < bad_treshold)]
    return g_wines, b_wines


def plot_scatter_matrix(df_wine, good_wines, bad_wines, save_plot=False):
    samples, feats = df_wine.shape
    figure, axes = plt.subplots(nrows = samples, ncols = feats, figsize=(18,18))
    figure.subplots_adjust(hspace=0, wspace=0)
    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    for i, title in enumerate(df_wine.columns):
        axes[i, i].annotate(title, (0.5, 0.5), xycoords = 'axes fraction', va = 'center', ha = 'center')


def main():
    wine_data = read_data("../data/winequality-red.csv")
    good_wines, bad_wines = check_quality(wine_data)
    plot_scatter_matrix(wine_data, good_wines, bad_wines, True)


if __name__ = "__main__":
    main()
