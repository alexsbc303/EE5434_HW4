import utils
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

def plot_separate(data, plot=True, save=True):
    # Create plot in separate format
    fig, axis = plt.subplots(2, 5)
    for i, digits_list in enumerate(data):
        for d in digits_list:
            if i < 5:
                axis[0, i].scatter(d['decimal'], d['symm'], alpha=0.5, c=utils.COLORS[i], edgecolors='none', s=20, label=i)
                axis[0, i].set_xlim([0, 0.7])
                axis[0, i].set_ylim([-8, 0])
                axis[0, i].set_title('%s' % i)
            else:
                axis[1, i-5].scatter(d['decimal'], d['symm'], alpha=0.5, c=utils.COLORS[i], edgecolors='none', s=20, label=i)
                axis[1, i-5].set_xlim([0, 0.7])
                axis[1, i-5].set_ylim([-8, 0])
                axis[1, i-5].set_title('%s' % i)

    fig.suptitle('10 Labels (Separated)')
    for ax in axis.flat:
        ax.set(xlabel='Intensity', ylabel='Symmetry')
    for ax in axis.flat:
        ax.label_outer()
    if save:
        pic = os.path.join('figure', 'separate.png')
        plt.savefig(pic)
        print('%s is saved.' % pic)
    if plot:
        plt.show()

def plot_all_labels(data, plot=True, save=True):
    # Create plot in concentrated format
    fig, ax = plt.subplots()
    for i, digits_list in enumerate(data):
        for d in digits_list:
            ax.scatter(d['decimal'], d['symm'], alpha=0.5, c=utils.COLORS[i], edgecolors='none', s=20, label=i)

    plt.title('10 Labels (Concentrated)')
    plt.xlabel("Average Intensity")
    plt.ylabel("Symmetry")
    plt.xlim([0, 0.7])
    plt.ylim([-8, 0])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    if save:
        pic = os.path.join('figure', 'all_labels.png')
        plt.savefig(pic)
        print('%s is saved.' % pic)
    if plot:
        plt.show()

if __name__== "__main__":
    utils.download_data()
    if not os.path.isdir('figure'):
        os.mkdir('figure')

    # Read data
    dataset = utils.Dataset()
    data = dataset.read_data(path='features.train')

    # Plot Graph
    plot_all_labels(data, plot=False, save=True)
    plot_separate(data, plot=False, save=True)
