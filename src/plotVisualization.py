import matplotlib.pyplot as plt

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
    ax.set_ylabel(features[1])
        
    w2, w1 = performance[epoch][2]
    b = performance[epoch][3]
    
    x_coords = range(int(x_min) - 1, int(x_max) + 2)
    y_coords = [-b/ w2 + -w1 / w2 * item for item in x_coords]

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.plot(x_coords, y_coords, 'b--', label='Decision boundary')
    ax.fill_between(x_coords, y_coords, y2=y_max, color='#ff9999')
    ax.fill_between(x_coords, y_coords, y2=y_min, color='#9999ff')


def draw_scatter(ax, data, features, good_treshold, bad_treshold):
    good_wines = data[data['quality'] > good_treshold]
    bad_wines = data[data['quality'] < bad_treshold]
    ax.scatter(good_wines.loc[:, features[0]], good_wines.loc[:, features[1]],
                    c='blue', label='good wines (> %d score)' % (good_treshold))
    ax.scatter(bad_wines.loc[:, features[0]], bad_wines.loc[:, features[1]],
                    c='red', label='bad wines (< %d score)' % (bad_treshold))
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2)

    
def plot_performance(performance, data, features, good_treshold, bad_treshold,
                     epoch=-1, save_plot=False, save_name='../images/performance-plot.png'):
    figure, axes = plt.subplots(1, 2, figsize=(8, 5))
    
    if epoch > len(performance) - 1:
        raise ValueError('number of epochs should be less than %d' % (len(performance)))
    
    if len(features) != 2:
        raise ValueError('number of features should be 2')
    
    if epoch == -1:
        epoch = len(performance) - 1
    
    draw_num_errors(axes[0], performance, epoch)    
    
    draw_decision_boundary(axes[1], performance, epoch, data, features)
    
    draw_scatter(axes[1], data, features, good_treshold, bad_treshold)
    
    if save_plot:
        plt.savefig(save_name)
    plt.close(figure)
    return figure

