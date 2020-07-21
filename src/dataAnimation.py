import matplotlib.image as plimg
from matplotlib.animation import ArtistAnimation, PilloWriter
import os
from plotVisualization import plot_performance, array_init
import matplotlib.pyplot as plt


def save_animation(performance, data, features, good_thresh, bad_thresh):
    frames = []
    figure = plt.figure(figsize = (8, 5))
    
    for i in range(len(performance)):
        plot = plot_performance(performance, data, features,
                good_thresh, bad_thresh, i, save_plot=True)
        img = plimg.imread('../images/performance-plot.png')
        plt.axis('off')
        frame = plt.imshow(img, animated=True)
        frames.append([frame])
        print("Frame %d" %(i))

    anim = ArtistAnimation(figure, frames, interval= 100, blit=True, repeat_delay=1000)
    writergif = PillowWriter(fps=30)
    anim.save('plot-animation.gif', writer=writergif)
    print('| DONE | Animation created and saved! ')
