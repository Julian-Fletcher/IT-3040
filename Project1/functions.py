import numpy as np
import matplotlib.pyplot as plt


def min_max(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def mean_normalization(x):
    return (x - np.mean(x)) / (np.max(x) - np.min(x))


def z_score(x):
    return (x - np.mean(x)) / np.std(x)

# sig function from Dr. Culmer
def sig(z):
    return 1 / (1 + np.exp(-z))


# Code from Dr. Culmer
def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0

    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)


# Map feature function from Dr. Culmer
def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)
