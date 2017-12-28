#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `synthetic` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the variance score are also
calculated.

"""
# print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause

import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


def f(x, nlevel):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x) + np.random.normal(0, nlevel, len(x))


#    return np.sin(x) + np.random.normal(0,nlevel,len(x))


# assert to guarantee at least one argument
assert len(sys.argv) > 1

# the 1st argument is the maximal feature degree
# if not 1, feature index must be positive (so that only one feature is selected)
str = "Max Degree = "
print str, sys.argv[1]
max_feature_degree = int(sys.argv[1])

# the second parameter is the noise level
nlevel = 0.5
if len(sys.argv) > 2:
    nlevel = float(sys.argv[2])
str = "Noise = "
print str, nlevel

# generate points and keep a subset of them
synthetic_X = np.linspace(0, 12, 100)
np.random.seed(1)
rng = np.random.RandomState(0)
rng.shuffle(synthetic_X)
# synthetic_X = np.sort(synthetic_X[:50])
synthetic_y = f(synthetic_X, nlevel)

num_samples = len(synthetic_X)
synthetic_X = synthetic_X.reshape((num_samples, 1))
print "Data size = ", num_samples
if max_feature_degree > 1:
    synthetic_X = np.hstack((synthetic_X, np.zeros((num_samples, max_feature_degree - 1))))
    for i in range(0, num_samples):
        x = synthetic_X[i, 0]
        tmparr = synthetic_X[i].tolist()
        for d in range(1, max_feature_degree):
            tmparr[d] = (tmparr[d - 1] * x)
        synthetic_X[i, :] = tmparr

# print "data 0-4: ", synthetic_X[0:5]
# print "target 0-4: ", synthetic_y[0:5]

test_error_list = np.zeros(max_feature_degree)
validation_error_mean_list = np.zeros(max_feature_degree)
validation_error_std_list = np.zeros(max_feature_degree)
training_error_list = np.zeros(max_feature_degree)

tsize = -75
# Split the data into training/testing sets
synthetic_X_train_all = synthetic_X[:tsize]
synthetic_X_test_all = synthetic_X[tsize:]

# Split the targets into training/testing sets
synthetic_y_train = synthetic_y[:tsize]
synthetic_y_test = synthetic_y[tsize:]

for d in range(1, max_feature_degree + 1):
    synthetic_X_train = synthetic_X_train_all[:, 0:d]
    synthetic_X_test = synthetic_X_test_all[:, 0:d]

    k = 5
    k_fold_val_error_list = np.zeros(k)
    block_size = 5
    for fold_id in range(0, k):
        train_idx = range(25)
        validation_idx = range(fold_id * block_size, (fold_id + 1) * block_size)
        train_idx = list(set(train_idx) - set(validation_idx))

        regr = linear_model.LinearRegression()
        regr.fit(synthetic_X_train[train_idx], synthetic_y_train[train_idx])
        k_fold_val_error_list[fold_id] = np.mean(
            (regr.predict(synthetic_X_train[validation_idx]) - synthetic_y_train[validation_idx]) ** 2)

    validation_error_mean_list[d - 1] = np.mean(k_fold_val_error_list)
    validation_error_std_list[d - 1] = np.std(k_fold_val_error_list)
    regr = linear_model.LinearRegression()
    regr.fit(synthetic_X_train, synthetic_y_train)

    training_error_list[d - 1] = np.mean((regr.predict(synthetic_X_train) - synthetic_y_train) ** 2)
    test_error_list[d - 1] = np.mean((regr.predict(synthetic_X_test) - synthetic_y_test) ** 2)

degree_list = range(1, max_feature_degree + 1)
plt.plot(degree_list, training_error_list, color='blue', linewidth=2, label="Training error")
# plt.errorbar(degree_list, validation_error_mean_list, yerr=validation_error_std_list, color='red', linewidth=2, label="Validataion error")
plt.plot(degree_list, validation_error_mean_list, color='red', linewidth=2, label="Validataion error")
plt.plot(degree_list, test_error_list, color='black', linewidth=2, label="Test error")
legend = plt.legend(loc='upper left', shadow=True)
plt.ylim([-5, 60])
plt.xlabel('Degrees')
plt.ylabel('Error')

plt.show()

