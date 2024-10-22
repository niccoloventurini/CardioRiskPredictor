from data_preprocessing import *
from implementations import *
from validations import *
from helpers import *
from evaluation import *

base_path = os.path.dirname(__file__)
x_train_path = os.path.join(base_path, 'data', 'x_train.csv') #load the path of x_train
y_train_path = os.path.join(base_path, 'data', 'y_train.csv') #load the path of y_train
y,x,ind = load_data(x_train_path,y_train_path) #load x_train and y _train
#sub_sample true to test only first 5000 people

x_test_path = os.path.join(base_path, 'data', 'x_test.csv') #load the path of x_test
x_test, ids_test = load_test_data(x_test_path) #load x test

# default values of gamma, lambda, degree and loss bias
gamma_ = 0.8
# lambda_ = 0.01 #uncomment to use lambda
degree = 2
loss_bias= 0.01

# search space for hyper-parameters
#create vector to find the best values of gamma, lambda, degree and loss biasgamma, lambda, degree and loss bias
degrees = [1, 2, 3]

#uncomment to use lambda
# lambdas = [
#     0.5,
#     0.1,
#     0.01,
#     0.001
# ]

gammas = [
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]

loss_biases = [
    0.001,
    0.01,
    0.1
]

#set k_fold, max iteration and create a seed
k_fold = 5
seed = 381
max_iters = 5000
print(np.shape(x))
print(np.shape(x_test))

validation = False #set it True to do the cross-validation

if validation:
    cross_deg,f1_deg = cross_val_degree(degrees, k_fold, seed, x, y, gamma_, max_iters, loss_bias)

    print(cross_deg)
    print(f1_deg)
    # cross_deg = 2
    cross_gamma = 0.8
    cross_loss_bias= 0.01

    #uncomment to do the cross-validation of lambda
    # cross_lambda,f1_lamda = cross_val_lambda(lambdas, k_fold, seed, x, y, cross_deg, cross_gamma, max_iters, cross_loss_bias)

    # print(cross_lambda)
    # print(f1_lamda)

    cross_gamma,f1_gamma = cross_val_gamma_(gammas, k_fold, seed, x, y, cross_deg, max_iters, loss_bias)

    print(cross_gamma)
    print(f1_gamma)

    cross_loss_bias,f1_bias = cross_val_loss_bias(loss_biases, k_fold, seed, x, y, cross_deg, cross_gamma, max_iters)

    print(cross_loss_bias)
    print(f1_bias) 
else:
    #obtain the default values
    # cross_lambda = lambda_ #uncomment when the lambda is present
    cross_gamma = gamma_
    cross_deg = degree
    cross_loss_bias = loss_bias

print(cross_deg)
# print(cross_lambda) #uncomment when the lambda is present
print(cross_gamma)
print(cross_loss_bias)

#preprocess the data
x_preproc, x_test_preproc,y_preproc = preprocess(x, cross_deg, True, True, True, x_test=x_test, y = y,repetition=6)
print(np.shape(x_preproc))

#reduce the size of the matrix to do the logistic regression
x_preproc_train = x_preproc[0:100000,:]
y_preproc_train = y_preproc[0:100000]

print(np.shape(x_preproc_train))
print(np.shape(y_preproc_train))
print(np.shape(x_test_preproc))

N,D= get_dim(x_test_preproc) #get the dimension of x_test_preproc
w0= np.zeros(D) #get a vector of shape D, the number of the columns of x_test_preproc

max_iters = 5000 #number of iteration of logistic regression

w,loss = logistic_regression(y_preproc_train, x_preproc_train, w0, max_iters, cross_gamma,cross_loss_bias) #Logistic regression
# w,loss = reg_logistic_regression_lasso(y_preproc_train, x_preproc_train, cross_lambda, w0, max_iters, cross_gamma, cross_loss_bias) #uncomment this function to do the regularized logistic regression with Lasso 

y_hat= logistic_prediction(x_test_preproc,w) #logistic prediction
print(y_hat)
y_predictions = zero_to_minone(y_hat) #change from 0 to -1

print(y_predictions)

create_csv_submission(ids_test,y_predictions, "submission_exp3.csv") #create submission

#test the model on the last 50.000 subject of the train dataset
try_test = x_preproc[-50000:,:]
try_y_preproc = y_preproc[-50000:]

try_y_hat = logistic_prediction(try_test,w)

try_y_predictions = zero_to_minone(try_y_hat)

f1_score(try_y_preproc,try_y_predictions)