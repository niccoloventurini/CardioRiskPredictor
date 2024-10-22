import numpy as np
from helpers import *
import os
import csv

def load_data(data_path_x,data_path_y, sub_sample=False):
    """
        This function loads data, of the train dataset, and returns yb (class labels), input_data (features) and ids (event ids)

        Args:
            data_path_x: path of the x_train dataset (str)
            data_path_y: path of the y_train dataset (str)
            sub_sample (bool, optional): If True the data will be subsempled. Default to False.

        Returns:
            yb: labels of train data
            input_data: train data
            ids: ids of train data
    """
    y = np.genfromtxt(data_path_y, delimiter=",", skip_header=1, dtype=str)
    y = y[:,1] #eliminate the index of the matrix
    x = np.genfromtxt(data_path_x, delimiter=",", skip_header=1, missing_values=np.nan)#max rows load only the first 50 coloumns
    ids = x[:, 0].astype(int) #index of the matrix
    input_data = x[:, 2:]
    # input_data = input_data[:,-91:] #uncomment to use only the last 91 features

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    for i, yi in enumerate(y):
        if yi == '-1' :
            yb[i] = 0
        else:
            yb[i] = 1

    # sub-sample
    if sub_sample:
        yb = yb[:5000]
        input_data = input_data[:5000,:]
        ids = ids[:5000]
        print(np.shape(yb))
        print(np.shape(input_data))
        print(np.shape(ids))

    return yb, input_data, ids

def load_test_data(data_path_x):
    """
        This function loads data, of the test dataset, and returns input_data (features) and ids (event ids)

        Args:
            data_path_x: path of the x_train dataset (str)
            data_path_y: path of the y_train dataset (str)
            sub_sample (bool, optional): If True the data will be subsempled. Default to False.

        Returns:
            yb: labels of train data
            input_data: train data
            ids: ids of train data
    """
    x = np.genfromtxt(data_path_x, delimiter=",", skip_header=1, missing_values=np.nan)#max rows load only the first 50 coloumns
    ids = x[:, 0].astype(int) #index of the matrix
    input_data = x[:, 2:]
    # input_data = input_data[:,-91:] #uncommet to use only the last 91 features
    return input_data, ids

# leave out the column that are not relevant
def manage_col (x, x_test): 
    """
        This function calls the other functions that implies the delete of some columns

        Args:
            x: train data
            x_test: test data 

        Returns:
             x: train data with new columns
             x_test: test data with new columns
    """
    x, x_test=std_col_delete(x,x_test) #delete columns with variance equal to 0
    corr_mat= column_correlation(x) #calculate correlation matrix
    x, x_test=drop_correlated_col(x,corr_mat, 0.8,x_test) #delete columns that have a correlation higher than 0.8
    return x, x_test

# replace null value with the mean of the column
def replace_empty_with_mean(x):
    """
        This function replace the empty value with  the mean of the other attributes in the column if it isn't categoric feature

        Args:
            x: data matrix
        
        Returns:
            new_x: data matrix without nan
    """
    new_x=np.where(np.isnan(x), np.ma.array(x, mask=np.isnan(x)).mean(axis=0), x)
    return new_x

# remove the column that have std=0
def std_col_delete (x, x_test):
    """
        This function delete the columns that have standard deviation equal 0

        Args:
            x: data matrix train
            x_test: data matrix test
        
        Returns:
            x: data matrix train without columns with std = 0
            x_test: data matrix test without columns with std = 0
    """
    if x_test is not None: #check if there is the x test
        x_test= x_test[:, x.std(axis=0)!=0]
    x= x[:, x.std(axis=0)!=0] #delete col
    return x, x_test

#normalize all the feature
def feature_normalization(x):
    """
        This function normalize the attributes in the columns

        Args:
            x: data matrix 
        
        Returns:
            x: data matrix normalized
    """
    # get dimension D
    _, D = get_dim(x)
    # for each feature
    for i in range(D):
        # set each x[:,i] to itself minus the mean divided standard deviation
        x[:, i] = (x[:, i] - np.mean(x[:, i])) / np.std(x[:, i])
    return x

def replace_empty_with_mean_for_category(x_j):
    """
        This function replace nan values with the mode of the column

        Args:
            x_j: a column of the matrix
        
        Returns:
            new_x_j: column without nan value
    """
    unique_elements, unique_counts = np.unique(x_j, return_counts=True, equal_nan=False)
    num_unique_values = len(unique_elements) #count the different values
    if num_unique_values <= 15: #check if there are more than 15 different values if yes it's not categorica
        most_frequent_index = np.argmax(unique_counts)
        most_frequent = unique_elements[most_frequent_index]
        new_x_j=np.where(np.isnan(x_j),most_frequent, x_j) #find and replace nan
    else:
        new_x_j=replace_empty_with_mean(x_j) #if is not categorical call the function tha utilize the mean
    return new_x_j

def iterations_for_mean(x):
    """
        This function replace nan values with the mode of the column by calling the replace_empty_with_mean_for_category function

        Args:
            x: data matrix 
        
        Returns:
            x: data matrix train without nan values
    """
    for i in range( get_dim(x)[1]):
        x[:,i]= replace_empty_with_mean_for_category(x[:,i]) #change nan values
    return x


# Performs feature expansion
def build_poly(x, degree):
    """
        This function build the polyomial feature expansion, based on the selected degree

        Args:
            x: data matrix 
            degree: a scalar, define the degree of the polynomial
        
        Returns:
            poly: new matrix after polynomial feature expansion
    """
    poly = np.ones((len(x), 1)) #build the new matrix
    for deg in range(1, degree + 1): 
        poly = np.c_[poly, np.power(x, deg)] #make the polynomial features expansion
    return poly


#find the correlation between columns
def column_correlation(x):
    """
        This function calculate the correlation matrix

        Args:
            x: data matrix 
            
        Returns:
            correlation_matrix: correlation matrix
    """
    correlation_matrix = np.corrcoef(x, rowvar=False) #calculate correlation coefficient and build the matrix
    correlation_matrix = np.round(correlation_matrix, 2)
    return correlation_matrix

#after have found the correlation matrix we decide which column to drop 
def drop_correlated_col(x, correlation_matrix, threshold, x_test):
    """
        This function delete the high correleted columns

        Args:
            x: data train matrix 
            correlation_matrix: correletion matrix
            threshold: threshold to determinate the high correlation value
            x_test: data test matrix
        
        Returns:
            x_filtered: data train without correleted columns 
            x_test: data test without correleted columns 
            
    """
    # Create a mask to identify highly correlated columns
    mask = np.triu(np.abs(correlation_matrix) > threshold, k=1)
    # The np.triu function creates an upper triangular matrix with True for highly correlated pairs
    # The k=1 argument excludes the main diagonal (k=0) to avoid self-correlation

    # Find the indices of columns to drop
    columns_to_drop = np.where(mask.any(axis=1))[0]
    if x_test is not None:
        x_test = np.delete(x_test,columns_to_drop,axis=1)
    # Create a list of column names to drop
    x_filtered = np.delete(x,columns_to_drop,axis=1)

    return x_filtered, x_test

#oversampling
def copy_and_concatenate_rows(x, y, num_times):
    """
        This function find and copy the rows correspondent to a 1 value in the y train matrix

        Args:
            x: data train matrix 
            y: labels of the train matrix
            num_times: a scalar, repetition factor
        
        Returns:
            x: train matrix with oversampling the 1 value
            y: labels of the train with oversampling the 1 value
            
    """

    data_0 = x[(y == 0).reshape(y.shape[0])]
    data_1 = x[(y == 1).reshape(y.shape[0])]
    print('no of 0', len(data_0))
    print('no of 1', len(data_1))

    # duplicate data_1 to match 80% of quantity of data_0
    data_1 = np.repeat(data_1, num_times, axis=0)

    

    # get corresponding y
    y_0 = y[y == 0]
    y_1 = y[y == 1]

    # duplicate y_1 to match 80% of quantity of y_0
    y_1 = np.repeat(y_1, num_times, axis=0)

    # concatenate data_0 and data_1
    x = np.concatenate((data_0, data_1), axis=0)
    y = np.concatenate((y_0, y_1), axis=0) 

    #shuffle the position of the new ones
    permutation = np.random.permutation(np.shape(x)[0])
    x = x[permutation]
    y = y[permutation]

    print(x.shape)
    print(y.shape)
    print(y)

    return x, y

# Preprocess the data then augments features
def preprocess(x, degree=1, normalize=True, replace_empty=True, manage_column=True, x_test = None, y=None, repetition = 6):
    """
        This function calls the previous define function to do the preprocessing

        Args:
            x: data train matrix
            degree: a scalar, define the degree of the polynomial
            normalize: boolean, if is True the function makes the normalization of the data
            replace_empty: boolean, if is True the function replace the nan values with the mean or the mode
            manage_column: boolean, if is True  the function calls the manage_column function
            x_test: data test matrix
            y = labels of train data
            repetition = a scalar, rapresent the factor of the oversampling
        
        Returns:
            x: data train matrix preprocessed
            x_test: data test matrix preprocessed
            y: labels of the train data preprocessed 
    """
    #calls all the function to preprocess the data, previous describe
    if replace_empty:
        x = iterations_for_mean(x)
        if x_test is not None:
            x_test = iterations_for_mean(x_test)
    if manage_column:
        x, x_test = manage_col(x, x_test)
    if normalize:
        x = feature_normalization(x)
        if x_test is not None:
            x_test = feature_normalization(x_test)
    print('shape before poly', x.shape)
    x = build_poly(x, degree)
    print('shape after poly', x.shape)
    if x_test is not None:
        x_test = build_poly(x_test, degree)
    x,y = copy_and_concatenate_rows(x,y,repetition)
    return x, x_test,y

