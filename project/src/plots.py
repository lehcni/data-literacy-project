import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def plot_regression(observation, prediction, x, title, ylabel='rent index'):

    fig = plt.figure(figsize=(7, 5))
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x, observation, color='darkred', label='observation')
    plt.plot(x, prediction, color='blue', label='prediction')
    plt.legend()
    plt.xlabel('year')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show();


def plot_coefficients(feature_names, coefs):

    plt.figure(figsize=(7, 5))
    y = np.arange(1, len(feature_names)+1, 1)
    plt.scatter(coefs, y, s=20, color='darkred')
    plt.yticks(y, labels=feature_names)
    plt.vlines(0, 0, 4, color='pink')
    plt.xlabel('regression coefficient size')
    plt.title('regression coefficients')
    plt.show();