# script to test your computation code
# do not change this file

from ComputeMatrices import compute_distance_naive, \
    compute_distance_smart, compute_correlation_naive, \
    compute_correlation_smart
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits

# an example code for testing
def main():
    iris = load_iris()
    breastcancer = load_breast_cancer()
    digits = load_digits()
    X = iris.data
    Y = breastcancer.data
    Z = digits.data

    iris_loop_start = time.time()
    distance_loop = compute_distance_naive(X)
    iris_loop_end = time.time()
    iris_cool_start = time.time()
    distance_cool = compute_distance_smart(X)
    iris_cool_end = time.time()

    breastcancer_loop_start = time.time()
    distance_loop = compute_distance_naive(Y)
    breastcancer_loop_end = time.time()
    breastcancer_cool_start = time.time()
    distance_cool = compute_distance_smart(Y)
    breastcancer_cool_end = time.time()

    digits_loop_start = time.time()
    distance_loop = compute_distance_naive(Z)
    digits_loop_end = time.time()
    digits_cool_start = time.time()
    distance_cool = compute_distance_smart(Z)
    digits_cool_end = time.time()

    duration_loop_iris = iris_loop_end - iris_loop_start
    duration_loop_bc = breastcancer_loop_end - breastcancer_loop_start
    duration_loop_digits = digits_loop_end - digits_loop_start

    duration_cool_iris = iris_cool_end - iris_cool_start
    duration_cool_bc = breastcancer_cool_end -  breastcancer_cool_start
    duration_cool_digits =  digits_cool_end - digits_cool_start

    # data for plotting
    n_groups = 3
    loop_data = (duration_loop_iris, duration_loop_bc, duration_loop_digits)
    cool_data = (duration_cool_iris, duration_cool_bc, duration_cool_digits)

    # drawing the plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.20
    opacity = 0.70

    rects1 = plt.bar(index, loop_data, bar_width,
                     alpha=opacity,
                     color='r',
                     label='loop')

    rects2 = plt.bar(index + bar_width, cool_data, bar_width,
                     alpha=opacity,
                     color='b',
                     label='cool')

    plt.xlabel('Data Set')
    plt.ylabel('Compute Time - Distance')
    plt.title('Compute-Time Comparison')
    plt.xticks(index + bar_width, ('Iris', 'Breast Cancer', 'Digits'))
    plt.legend()

    plt.tight_layout()

    plt.savefig('DistanceTimeComparison.pdf')
    print "result is written to DistanceTimeComparison.pdf"

    iris_loop_start = time.time()
    distance_loop = compute_correlation_naive(X)
    iris_loop_end = time.time()
    iris_cool_start = time.time()
    distance_cool = compute_correlation_smart(X)
    iris_cool_end = time.time()

    breastcancer_loop_start = time.time()
    distance_loop = compute_correlation_naive(Y)
    breastcancer_loop_end = time.time()
    breastcancer_cool_start = time.time()
    distance_cool = compute_correlation_smart(Y)
    breastcancer_cool_end = time.time()

    digits_loop_start = time.time()
    distance_loop = compute_correlation_naive(Z)
    digits_loop_end = time.time()
    digits_cool_start = time.time()
    distance_cool = compute_correlation_smart(Z)
    digits_cool_end = time.time()

    duration_loop_iris = iris_loop_end - iris_loop_start
    duration_loop_bc = breastcancer_loop_end - breastcancer_loop_start
    duration_loop_digits = digits_loop_end - digits_loop_start

    duration_cool_iris = iris_cool_end - iris_cool_start
    duration_cool_bc = breastcancer_cool_end - breastcancer_cool_start
    duration_cool_digits = digits_cool_end - digits_cool_start

    # data for plotting
    n_groups = 3
    loop_data = (duration_loop_iris, duration_loop_bc, duration_loop_digits)
    cool_data = (duration_cool_iris, duration_cool_bc, duration_cool_digits)

    # drawing the plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.20
    opacity = 0.70

    rects1 = plt.bar(index, loop_data, bar_width,
                     alpha=opacity,
                     color='r',
                     label='loop')

    rects2 = plt.bar(index + bar_width, cool_data, bar_width,
                     alpha=opacity,
                     color='b',
                     label='cool')

    plt.xlabel('Data Set')
    plt.ylabel('Compute Time - Correlation')
    plt.title('Compute-Time Comparison')
    plt.xticks(index + bar_width, ('Iris', 'Breast Cancer', 'Digits'))
    plt.legend()

    plt.tight_layout()
    plt.savefig('CovarianceTimeComparison.pdf')
    print "result is written to CovarianceTimeComparison.pdf"

if __name__ == "__main__": main()
