#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cpt
import time
from matplotlib.colors import ListedColormap
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# acquire data, split it into training and testing sets (50% each)
# nc -- number of classes for synthetic datasets
def acquire_data(data_name, nc = 2):
    if data_name == 'synthetic-easy':
        print 'Creating easy synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 0 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-medium':
        print 'Creating medium synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 3 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-hard':
        print 'Creating hard easy synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 5 * rng.uniform(size=X.shape)
    elif data_name == 'moons':
        print 'Creating two moons dataset'
        X, y = datasets.make_moons(noise=0.2, random_state=0)
    elif data_name == 'circles':
        print 'Creating two circles dataset'
        X, y = datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
    elif data_name == 'iris':
        print 'Loading iris dataset'
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    elif data_name == 'digits':
        print 'Loading digits dataset'
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target
    elif data_name == 'breast_cancer':
        print 'Loading breast cancer dataset'
        bcancer = datasets.load_breast_cancer()
        X = bcancer.data
        y = bcancer.target
    else:
        print 'Cannot find the requested data_name'
        assert False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test

# compare the prediction with grount-truth, evaluate the score
def myscore(y, y_gt):
    assert len(y) ==  len(y_gt)
    return np.sum(y == y_gt)/float(len(y))

# plot data on 2D plane
# use it for debugging
def draw_data(X_train, X_test, y_train, y_test, nclasses):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    cm = plt.cm.jet
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, edgecolors='k', marker='x', linewidth = 3, label='Test Data')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()

# draw results on 2D plan for binary classification
# this is a fake version (using a random linear classifier)
# modify it for your own usage (pass in parameter etc)
def draw_result_binary_fake(X_train, X_test, y_train, y_test):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Put the result into a color plot
    tmpX = np.c_[xx.ravel(), yy.ravel()]

    Z_class, Z_pred_val = get_prediction_fake(tmpX)

    Z_clapped = np.zeros(Z_pred_val.shape)
    Z_clapped[Z_pred_val>=0] = 1.5
    Z_clapped[Z_pred_val>=1.0] = 2.0
    Z_clapped[Z_pred_val<0] = -1.5
    Z_clapped[Z_pred_val<-1.0] = -2.0

    Z = Z_clapped.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu, alpha = .4)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    #    ax = plt.figure(1)
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', marker='x', linewidth=3,
                label='Test Data')

    y_train_pred_class, y_train_pred_val = get_prediction_fake(X_train)
    sv_list_bool = np.logical_and(y_train_pred_val >= -1.0, y_train_pred_val <= 1.0)
    sv_list = np.where(sv_list_bool)[0]
    plt.scatter(X_train[sv_list, 0], X_train[sv_list, 1], s=100, facecolors='none', edgecolors='orange', linewidths = 3, label='Support Vectors')

    y_test_pred_class, y_test_pred_val = get_prediction_fake(X_test)
    score = myscore(y_test_pred_class, y_test)
    plt.text(xx.max() - .3, yy.min() + .3, ('Score = %.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

    plt.legend()
    plt.show()

# predict labels using a random linear classifier
# returns a list of length N, each entry is either 0 or 1
def get_prediction_fake(X):
    np.random.seed(100)
    nfeatures = X.shape[1]
    # w = np.random.rand(nfeatures + 1) * 2.0
    w = [-1,0,0]

    assert len(w) == X.shape[1] + 1
    w_vec = np.reshape(w,(-1,1))
    X_extended = np.hstack([X, np.ones([X.shape[0],1])])
    y_pred_value = np.ravel(np.dot(X_extended,w_vec))
    y_pred_class = np.maximum(np.zeros(y_pred_value.shape), y_pred_value)
    return y_pred_class, y_pred_value
    print 'Finished. Took:', time.time() - startTime


####################################################
# binary label classification

def linear_kernel (X, y):
    return np.dot(X,y)

def poly_kernel(X, y, kpar):
    return (1 + np.dot(X,y))**kpar

def gaussian_kernel (X, y, sigma):
    y = np.transpose(y)
    return np.exp(-np.linalg.norm(X-y)**2 / (2*(sigma**2)))

def gram_matrix(X, Xi, ker, kpar):
    n = X.shape[0]
    m = Xi.shape[0]
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if ker == 'linear':
                K[i, j] = linear_kernel(X[i], Xi[j])
            if ker == 'polynomial':
                K[i, j] = poly_kernel(X[i], Xi[j], kpar)
            if ker == 'gaussian':
                K[i, j] = gaussian_kernel(X[i], Xi[j], kpar)
    return K

#change sign of y_train
def changeSign(s):
    if s <= 0.0:
        return -1
    else:
        return 1

def compute_multipliers(X, y, c, ker, kpar):
    n_samples, n_features = X.shape
    K = gram_matrix(X, X, ker, kpar)

    P = cpt.matrix(np.outer(y, y) * K)
    q = cpt.matrix(np.ones(n_samples) * -1)

    # -a_i \leq 0

    if c is None:
        G = cpt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cpt.matrix(np.zeros(n_samples))

    # a_i \leq c
    else:
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cpt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * c
        h = cpt.matrix(np.hstack((tmp1, tmp2)))

    y = y.astype(np.double)
    A = cpt.matrix(y, (1, n_samples))

    b = cpt.matrix(0.0)
    solution = cpt.solvers.qp(P, q, G, h, A, b)
    return np.ravel(solution['x'])


# training kernel svm
# return sv_list: list of surport vector IDs
# alpha: alpha_i's
# b: the bias
def mytrain_binary(X_train, y_train, C, ker, kpar):
    print 'Start training ...'
    alpha = compute_multipliers(X_train, y_train, C, ker, kpar)
    sv_list1 = np.nonzero(alpha > 1.0e-3)
    sv_list = np.ravel(sv_list1)
    # bias = y_k - \sum z_i y_i  K(x_k, x_i)

    alphaVals = alpha[sv_list]
    yVals = y_train[sv_list]
    sv_values = X_train[sv_list]
    kernelVals = gram_matrix(sv_values,sv_values, ker, kpar)

    b = np.average(yVals - np.dot(np.transpose(alphaVals * yVals), kernelVals))
    print 'Finished training.'
    return sv_list, alpha, b

# predict given X_test data,
# need to use X_train, ker, kpar_opt to compute kernels
# need to use sv_list, y_train, alpha, b to make prediction
# return y_pred_class as classes (convert to 0/1 for evaluation)
# return y_pred_value as the prediction score
def mytest_binary(X_test, X_train, y_train, sv_list, alpha, b, ker, kpar):
    n = X_test.shape[0]
    alphaVals = alpha[sv_list]
    alphaVals = np.transpose(np.reshape(alphaVals, (-1, 1)))
    alphaVals = np.ravel(alphaVals)
    yVals = y_train[sv_list]
    yVals = np.transpose(np.reshape(yVals, (-1, 1)))
    yVals = np.ravel(yVals)
    xVals = X_train[sv_list]

    K = gram_matrix(xVals, X_test, ker, kpar)

    y_pred_value = np.dot((alphaVals * yVals), K) + b
    y_pred_value = np.ravel(y_pred_value)
    change = np.vectorize(changeSign)
    y_pred_class = np.ravel(change(y_pred_value))

    return y_pred_class, y_pred_value


def split_data(X_train, y_train, n, k):
    N, D = X_train.shape
    folds = N / k

    start = n * folds
    end = start + folds

    values = X_train[start:end]
    labels = y_train[start:end]
    rest = np.concatenate((X_train[:start], X_train[end:]))
    train_label = np.concatenate((y_train[:start], y_train[end:]))
    return values, labels, rest, train_label

def my_cross_validation(X_train, y_train, ker, k = 5):
    assert ker == 'linear' or ker == 'polynomial' or ker == 'gaussian'

    C_opt, kpar_opt, max_score = 0.0, 0.0, 0.0

    result = []
    scores = []
    for C in range(1, 5, 1):
        for kpar in range(1, 5, 1):
            for fold in range(k):
                X_test_new, y_test_new, X_train_new, y_train_new = split_data(X_train,y_train,fold,k)
                sv_list, alpha, b = mytrain_binary(X_train_new, y_train_new, C, ker, kpar)
                y_pred_class, y_pred_value = mytest_binary(X_test_new, X_train_new, y_train_new, sv_list, alpha, b,
                ker, kpar)
                test_score = myscore(y_pred_class, y_test_new)
                scores.append(test_score)
                average_score = np.average(scores)
                result.append((C, kpar, average_score))

    for c, kpar,score in result:
        if score > max_score:
            max_score = score
            C_opt = c
            kpar_opt = kpar

    print 'C_opt: ', C_opt
    print 'kpar_opt: ', kpar_opt
    print 'max_score: ', max_score

    return C_opt, kpar_opt

################

def main():

    #######################
    # get data
    # only use binary labeled

    X_train, X_test, y_train, y_test = acquire_data('synthetic-easy')
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-medium')
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-hard')
    # X_train, X_test, y_train, y_test = acquire_data('moons')
    # X_train, X_test, y_train, y_test = acquire_data('circles')
    # X_train, X_test, y_train, y_test = acquire_data('breast_cancer')

    change = np.vectorize(changeSign)
    y_train = change(y_train)
    y_test = change(y_test)
    nfeatures = X_train.shape[1]    # number of features
    ntrain = X_train.shape[0]   # number of training data
    ntest = X_test.shape[0]     # number of test data
    y = np.append(y_train, y_test)
    nclasses = len(np.unique(y)) # number of classes

    # only draw data (on the first two dimension)
    # draw_data(X_train, X_test, y_train, y_test, nclasses)
    # a face function to draw svm results
    # draw_result_binary_fake(X_train, X_test, y_train, y_test)

    ker = 'linear'
    # ker = 'polynomial'
    # ker = 'gaussian'

    C_opt, kpar_opt = my_cross_validation(X_train, y_train, ker)

    start_time = time.time()
    sv_list, alpha, b = mytrain_binary(X_train, y_train, C_opt, ker, kpar_opt)
    end_time = time.time()
    print ('Training Time = ', end_time - start_time)
    y_test_pred_class, y_test_pred_val = mytest_binary(X_test, X_train, y_train, sv_list, alpha, b, ker, kpar_opt)

    test_score = myscore(y_test_pred_class, y_test)

    print 'Test Score:', test_score

if __name__ == "__main__": main()
