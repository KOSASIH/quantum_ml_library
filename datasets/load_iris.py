# quantum_ml_library/datasets/load_iris.py

import numpy as np

def load_iris():
    # Load the iris dataset
    iris = np.loadtxt('iris.csv', delimiter=',', skiprows=1)

    # Separate the features and labels
    X = iris[:, :4]
    y = iris[:, 4].astype(int)

    # Normalize the features to range [0, 1]
    X = (X - np.min(X)) / (np.max(X) - np.min(X))

    return X, y
