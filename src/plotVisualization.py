from dataVisualization import check_quality
import matplotlib.pyplot as plt
import pandas as pd

def array_init(shape, mode='rand'):
    if mode not in ['rand', 'zeros', 'ones']:
        raise ValueError('invalid mode')
    new_shape = shape
    if isinstance(shape, int):
        new_shape = (shape, 1) 
    if isinstance(shape, tuple) and len(shape) == 1:
        new_shape = (shape[0], 1)
    base_dict = dict()
    for i in range(new_shape[1]):
        if mode == 'rand':
            base_dict[i] = [0.0001 * random.uniform(-1, 1) for i in range(new_shape[0])]
        elif mode == 'zeros':
            base_dict[i] = new_shape[0] * [0.0]
        elif mode == 'ones':
            base_dict[i] = new_shape[0] * [1.0]
    df = pd.DataFrame.from_dict(base_dict)
    if isinstance(shape, int) or isinstance(shape, tuple) and len(shape) == 1:
        return df.values.squeeze()
    return df.values


def draw_num_errors(ax, performance, epoch):
    epochs = [elem[0] for elem in performance[:epoch + 1]]
    epoch_errors = [elem[1] for elem in performance[:epoch + 1]]
    ax.plot(epochs, epoch_errors)
    ax.set_xlim([0, len(performance)])
    ax.set_title('Errors as a function of epochs')
    ax.set_xlabel('epoch')
    ax.set_ylabel('classification errors')


def draw_decision_boundary(ax, performance, epoch, data, features):
    x_min = data.loc[:, features[0]].min() - 0.15
    x_max = data.loc[:, features[0]].max() + 0.15
    y_min = data.loc[:, features[1]].min() - 0.15
    y_max = data.loc[:, features[1]].max() + 0.15
    
    ax.set_title('Decision boundary at epoch %d' %(epoch))
    ax.set_xlabel(features[0])
    ax.set_xlim([x_min, x_max])
    ax.set_ylabel(features[1])
    ax.set_ylim([y_min, y_max])
    
    w2, w1 = performance[epoch][2]
    b = performance[epoch][3]
    slope = -w1/w2
    intercept = -b/w2
    x_coords = range(int(x_min) - 1, int(x_max) + 2)
    y_coords = slope * x_coords + intercept

    ax.plot(x_coords, y_coords, 'b--', label='Decision boundary')
    ax.fill_between(x_coords, y_coords, y_min, color='#99ff99')
    ax.fill_between(x_coords, y_coords, y_max, color='#ff9999')


def draw_scatter(ax, data, features, good_thresh, bad_thresh):
    good_wines, bad_wines = check_quality(data)
    ax.scatter(good_wines.loc[:, features[0]], good_wines.loc[:, features[1]],
                    c='blue', label='good wines (> %d score)' % (good_thresh))
    ax.scatter(bad_wines.loc[:, features[0]], bad_wines.loc[:, features[1]],
                    c='red', label='bad wines (< %d score)' % (bad_thresh))
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2)

    
def plot_performance(performance, data, features, good_thresh, bad_thresh,
                     epoch=-1, save_plot=False, save_name='../images/performance-plot.png'):
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    if epoch > len(performance) - 1:
        raise ValueError('number of epochs should be less than %d' % (len(performance)))
    if len(features) != 2:
        raise ValueError('number of features should be 2')
    if epoch == -1:
        epoch = len(performance) - 1
    draw_num_errors(axes[0], performance, epoch)    
    draw_decision_boundary(axes[1], performance, epoch, data, features)
    draw_scatter(axes[1], data, features, good_thresh, bad_thresh)
    if save_plot:
        plt.savefig(save_name)
    plt.close(figure)
    return figure

