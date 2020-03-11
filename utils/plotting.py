from matplotlib import pyplot as plt
from numpy import array
import numpy as np
from sklearn.model_selection import learning_curve


def best_fit(x: array, y: array) -> tuple:
    x_bar = sum(x) / len(x)
    y_bar = sum(y) / len(y)
    n = len(x)
    numerator = sum([xi * yi for xi, yi in zip(x, y)]) - n * x_bar * y_bar
    denum = sum([xi ** 2 for xi in x]) - n * x_bar ** 2

    b = numerator / denum
    a = y_bar - b * x_bar
    return a, b


def plot_residual(y_true: array, predictions: array, name: str = '', ax=None) -> tuple:
    """
    A function that plots the residual on a matplotlib axis
    :param y_true:
    :param predictions:
    :param name:
    :param ax:
    :return:
    """
    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax[0]
        ax = ax[1]
    a, b = best_fit(y_true, predictions)
    best_fit_eq = [a + b * xi for xi in y_true]
    ax.scatter(x=y_true, y=predictions)
    ax.plot(y_true, best_fit_eq, color='red', linewidth=1)
    ax.title.set_text(name)
    ax.set_xlabel('Observed Critical')
    ax.set_ylabel('Predicted Critical')
    return fig, ax


def plot_learning_curve(train_scores, test_scores, title, train_sizes=None, axes=None, ylim=None):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    Parameters
    ----------

    title : string
        Title for the chart.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    return axes
