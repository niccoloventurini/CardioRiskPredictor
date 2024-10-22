import numpy as np
from helpers import *
from data_preprocessing import *
from implementations import *
from predictions import *
from evaluation import *


# This function given a seed, N and k_fold returns an array of k_fold indices which belong to the train set (taken from ML labs)
def _build_k_indices(N, k_fold, seed):
    """
    This function builds k indices for the K-fold cross-validation.

    Args:
        N: shape=(N,1)
        k_fold: K in K-fold, i.e. the fold num
        seed: the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
        
    """
    num_row = N
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)] #dived the index in k group randomly
    return np.array(k_indices)

#cross validation step that is repeated for every lambda
def cross_val_step(x, y, k_indices, k, degree, gamma_, max_iters, loss_bias):
    """
    This function does one step of the cross-validation. This function is called k_fold times and repeted for every parameters to test.

    Args:
        x: shape(N,D)
        y: shape(N,1)
        k_indices: 2D array returned by build_k_indices()
        k: scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        degree: scalar
        gamma_: scalar
        loss_bias: scalar
        max_iters: scalar
        
    Returns:
        f1_score : a scalar
        
    """
    x , _ , y= preprocess(x, degree, True, True, True, y = y, x_test = None)
    train_set = np.delete(x, k_indices[k], axis=0)
    y_train = np.delete(y, k_indices[k], axis=0) #selecting the train sets
    N,D= get_dim(train_set)
    w0= np.zeros(D)
    w,_= logistic_regression(y_train, train_set, w0, max_iters, gamma_, loss_bias)
    test_set = x[k_indices[k], :]
    y_test = y[k_indices[k]]
    y_prediction = logistic_prediction(test_set, w)
    y_prediction = zero_to_minone(y_prediction)
    return f1_score(y_test,y_prediction)

#cross validation step that is repeated for every lambda
def cross_val_step_lasso(x, y, k_indices, k, degree, gamma_, max_iters, loss_bias, lambda_):
    """
    This function does one step of the cross-validation. This function is called k_fold times and repeted for every parameters to test.
    This function is used to do the cross-validation step using the regularized logistic regression with Lasso.

    Args:
        x: shape(N,D)
        y: shape(N,1)
        k_indices: 2D array returned by build_k_indices()
        k: scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        degree: scalar
        gamma_: scalar
        loss_bias: scalar
        max_iters: scalar
        lambda_: a scalar, the penalty term. 
        
    Returns:
        f1_score : a scalar
        
    """
    x , _ , y= preprocess(x, degree, True, True, True, y = y, x_test = None)
    train_set = np.delete(x, k_indices[k], axis=0)
    y_train = np.delete(y, k_indices[k], axis=0) #selecting the train sets
    # train_set= preprocess(train_set, degree)
    N,D= get_dim(train_set)
    w0= np.zeros(D)
    w,_= reg_logistic_regression_lasso(y_train, train_set, lambda_, w0, max_iters, gamma_, loss_bias)
    test_set = x[k_indices[k], :]
    # test_set = feature_adding_constant(test_set)
    y_test = y[k_indices[k]]
    y_prediction = logistic_prediction(test_set, w)
    y_prediction = zero_to_minone(y_prediction)
    return f1_score(y_test,y_prediction)

#cross-validation data for lambda 1
def cross_val_lambda(lambda1s, k_fold, seed, x, y, degree, gamma_, max_iters,loss_bias): #lambda1s is an array with the selected value, k_fold is the number of set created, random seed to make it reproducable
    """
    Cross-validation over the regularization parameter lambda.

    Args:
        lambdas: shape (p,) where p is the number of values of lambda to test 
        k_fold: K in K-fold, i.e. the fold num
        seed: the random seed
        x: shape(N,D)
        y: shape(N,1)
        degree: scalar
        gamma_: scalar
        loss_bias: scalar
        max_iters: scalar
        
    Returns:
        final_parameter: scalar, value of the best lambda
        best_performance: scalar, value of the best F1 score corresponding to the best lambda
        
    """

    k_indices= _build_k_indices(np.shape(x)[0], k_fold, seed)
    best_performance = -1
    final_parameter = -1 #to update with best value
    for lambda1 in lambda1s:
        average_value=0
        for k in range(k_fold): #do the cross-validation step for each k
            performance= cross_val_step_lasso(x, y, k_indices, k, degree, gamma_, max_iters, loss_bias, lambda1)
            average_value += performance
        average_value /= k_fold
        if average_value > best_performance: #check if is the best performance
            best_performance = average_value
            final_parameter = lambda1
    return final_parameter , best_performance

#cross-validation data for degree
def cross_val_degree(degrees, k_fold, seed, x, y, gamma_, max_iters,loss_bias): #lambda1s is an array with the selected value, k_fold is the number of set created, random seed to make it reproducable
    """
    Cross-validation over the parameter degree.

    Args:
        degrees: shape (p,) where p is the number of values of degree to test 
        k_fold: K in K-fold, i.e. the fold num
        seed: the random seed
        x: shape(N,D)
        y: shape(N,1)
        gamma_: scalar
        loss_bias: scalar
        max_iters: scalar
        
    Returns:
        final_parameter: scalar, value of the best degree
        best_performance: scalar, value of the best F1 score corresponding to the best degree
        
    """

    k_indices= _build_k_indices(np.shape(x)[0], k_fold, seed)
    best_performance = -1
    final_parameter = -1 #to update with best value
    for degree in degrees:
        average_value=0
        for k in range(k_fold): #do the cross-validation step for each k
            performance= cross_val_step(x, y, k_indices, k, degree, gamma_, max_iters,loss_bias)
            average_value += performance
        average_value /= k_fold
        if average_value > best_performance:#check if is the best performance
            best_performance = average_value
            final_parameter = degree
    return final_parameter , best_performance


#cross-validation data for gamma_
def cross_val_gamma_(gammas, k_fold, seed, x, y, degree,  max_iters,loss_bias): #lambda1s is an array with the selected value, k_fold is the number of set created, random seed to make it reproducable
    """
    Cross-validation over the parameter gamma.

    Args:
        gammas: shape (p,) where p is the number of values of gamma to test 
        k_fold: K in K-fold, i.e. the fold num
        seed: the random seed
        x: shape(N,D)
        y: shape(N,1)
        degree: scalar
        loss_bias: scalar
        max_iters: scalar
        
    Returns:
        final_parameter: scalar, value of the best gamma
        best_performance: scalar, value of the best F1 score corresponding to the best gamma
        
    """

    k_indices= _build_k_indices(np.shape(x)[0], k_fold, seed)
    best_performance = -1
    final_parameter = -1 #to update with best value
    for gamma_ in gammas:
        average_value=0
        for k in range(k_fold): #do the cross-validation step for each k
            performance= cross_val_step(x, y, k_indices, k, degree, gamma_, max_iters, loss_bias)
            average_value += performance
        average_value /= k_fold
        if average_value > best_performance: #check if is the best performance
            best_performance = average_value
            final_parameter = gamma_
    return final_parameter , best_performance

def cross_val_loss_bias(loss_biases, k_fold, seed, x, y, degree , gamma_, max_iters):
    """
    Cross-validation over the parameter loss-bias.

    Args:
        loss_baises: shape (p,) where p is the number of values of loss_bias to test 
        k_fold: K in K-fold, i.e. the fold num
        seed: the random seed
        x: shape(N,D)
        y: shape(N,1)
        degree: scalar
        gamma_: scalar
        max_iters: scalar
        
    Returns:
        final_parameter: scalar, value of the best loss_bias
        best_performance: scalar, value of the best F1 score corresponding to the best loss_bias
        
    """

    k_indices= _build_k_indices(np.shape(x)[0], k_fold, seed)
    best_performance = -1
    final_parameter = -1 #to update with best value
    for loss_bias in loss_biases:
        average_value=0
        for k in range(k_fold): #do the cross-validation step for each k
            performance= cross_val_step(x, y, k_indices, k, degree, gamma_, max_iters, loss_bias)
            average_value += performance
        average_value /= k_fold
        if average_value > best_performance: #check if is the best performance
            best_performance = average_value
            final_parameter = loss_bias
    return final_parameter , best_performance