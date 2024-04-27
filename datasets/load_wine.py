# quantum_ml_library/datasets/load_wine.py

import numpy as np

def load_wine():
    # Load the wine dataset
    wine = np.loadtxt('wine.csv', delimiter=',', skiprows=1)

    # Separate the features and labels
    X = wine[:, :13]
    y = wine[:, 13].astype(int)

    # Normalize the features to range [0, 1]
    X = (X - np.min(X)) / (np.max(X) - np.min(X))

    return X, y
