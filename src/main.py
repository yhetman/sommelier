import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from  dataVisualization import read_data
from classAdaline import Adaline
from dataAnimation import save_animation
from plotVisualization import plot_performance
from normalization import normalization
import Kfold
from defineLearningRate import create_training_data


if __name__ == '__main__':
    main()
