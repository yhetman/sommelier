import matplotlib.image as img
import matplotlib.animation as anima
import os
from plotVisualization import plot_performance, array_init
import matplotlib.pyplot as plt


def save_animation(performance, data, features, good_thresh, bad_thresh,
        name='../images/performance_animation.gif'):
    frames = []
    figure = plt.figure(figsize=(40,20), dpi=150)
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for i in range(len(performance)):
        plot = plot_performance(performance, data, features,
                good_thresh, bad_thresh, i, save_plot=True)
        image = img.imread('../images/performance-plot.png')
        plt.axis('off')
        print("Frame %d" %(i))
        frame = plt.imshow(image, animated=True)
        frames.append([frame])
        #if image != None :
        image = 0
    animation = anima.ArtistAnimation(figure, frames, interval=120, blit=True, repeat_delay=1000)
    animation.save(name)
    print('| DONE | Animation created and saved as %s' %(name))

