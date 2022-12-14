import matplotlib.pyplot as plt
import numpy as np


def plot_metric(metrics, metric, save_path=None):
    '''Plot (and save) a metric that has been recorded during training, validation and testing.'''
    train_epochs = len(metrics[f'train_{metric}_epoch'])
    val_epochs = len(metrics[f'train_{metric}_epoch'])
    batches = len(metrics[f'train_{metric}_step'])
    val_offset = val_epochs - train_epochs
    val_epoch_steps = np.linspace(1 - val_offset, val_epochs, val_epochs)
    train_epoch_steps = np.linspace(1, train_epochs, train_epochs)
    batch_steps = np.linspace(1, train_epochs, batches)

    plt.plot(batch_steps, metrics[f'train_{metric}_step'], linestyle='-', marker='', color='blue', label=f'train {metric} (batches)')
    plt.plot(train_epoch_steps, metrics[f'train_{metric}_epoch'], linestyle='-', marker='.', color='red', label=f'train {metric} (epochs)')
    plt.plot(val_epoch_steps, metrics[f'val_{metric}_epoch'], linestyle='-', marker='.', color='orange', label=f'val {metric} (epochs)')
    plt.plot([train_epochs], [metrics[f'test_{metric}_epoch'][-1]], linestyle='', marker='o', markersize=3, color='cyan', label=f'test {metric}')

    plt.title(metric)
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.xticks(val_epoch_steps)
    plt.legend(loc='best')

    if save_path is not None:
        plt.savefig(save_path)
    #plt.show()
    plt.close()
