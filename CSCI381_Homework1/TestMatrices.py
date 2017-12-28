# script to test your computation code
# do not change this file

from ComputeMatrices import compute_distance_naive, \
    compute_distance_smart, compute_correlation_naive, \
    compute_correlation_smart
import numpy as np
from sklearn.datasets import load_iris

# my computation
def my_comp_distance(X):
    N = X.shape[0]
    D = X[0].shape[0]

    M = np.zeros([N,N])

    return M

# an example code for testing
def main():
    iris = load_iris()
    X = iris.data

    distance_true = my_comp_distance(X)
    distance_loop = compute_distance_naive(X)
    distance_cool = compute_distance_smart(X)

    print np.allclose(distance_true, distance_loop)
    print np.allclose(distance_true, distance_cool)

if __name__ == "__main__": main()
