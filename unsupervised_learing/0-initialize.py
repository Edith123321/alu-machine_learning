#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def initialize(X, k):
    """
    Initializes cluster centroids for K-means using a multivariate uniform distribution.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) containing n data points of dimension d.
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Centroid array of shape (k, d) drawn from a uniform distribution
                       over the range of X along each dimension, or None on failure.
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    if X.shape[0] == 0:  # no data points
        return None

    # Compute per-dimension min and max
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # Generate centroids using a single call to uniform
    # Broadcasting: low and high of shape (d,) are expanded to (k, d)
    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, X.shape[1]))

    return centroids
