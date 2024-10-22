import numpy as np
import matplotlib.pyplot as plt
from helpers import batch_iter
from predictions import _sigmoid
from helpers import get_dim
# from tqdm import tqdm
"""1. Linear regression using gradient descent"""

def compute_gradient(y, tx, w):
    """
    This function computes the gradient and the error.
    
     Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, 1)
    
    Returns :
        grad: shape (D, 1)
        err : shape=(N, 1)
        
    """
    err = y - tx.dot(w) #calculate the error
    grad = -tx.T.dot(err) / len(err) #calculate the gradient
    return grad, err

def calculate_mse(e):
    """ This function calculates the mean squared error for vector e. """
    return 1/2*np.mean(e**2) 

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma) :
    """
    Linear regression using gradient descent.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        initial_w: the initial guess (or the initialization) for the model parameters, shape=(D,1). 
        max_iters: a scalar denoting the total number of iterations of gradient descent.
        gamma: a scalar denoting the stepsize. 

    Returns:
        w: a scalar, the last weight of the iterations. 
        loss: a scalar, the last loss value of the iterations.
        
    """

    #set initial values and initial list of loss and w
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
    grad, err = compute_gradient(y, tx, w)
    loss = calculate_mse(err)
    return w, loss
    

"""2. Linear regression using stochastic gradient descent"""

def compute_stoch_gradient(y, tx, w):
    """
    This function computes the stochastic gradient and the error.
    
     Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, 1)
    
    Returns :
        grad: shape (D, 1)
        err : shape=(N, 1)
        
    """

    err = y - tx.dot(w) #calculate error
    grad = -tx.T.dot(err) / len(err) #calculate gradient
    return grad, err  

def compute_loss(y, tx, w):
    """
    This function calculates the loss using mse.
    
    Args:
        y: shape=(N, 1)
        tx: shape=(N, D)
        w: shape=(D, 1)
    
    Return :
        loss calculated with mse. 
 
    """
    e = y - tx.dot(w) #calculate error
    return calculate_mse(e)
   
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma) :
    """
    Linear regression using stochastic gradient descent.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        initial_w: the initial guess (or the initialization) for the model parameters, shape=(D,1). 
        max_iters: a scalar denoting the total number of iterations of gradient descent.
        gamma: a scalar denoting the stepsize. 

    Returns:
        w: a scalar, the last weight of the iterations. 
        loss: a scalar, the last loss value of the iterations.
        
    """
    #set initial values and initial list of loss and w
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
    loss = compute_loss(y, tx, w)
    return w , loss
    
    
"""3. Least squares regression using normal equations."""

def least_squares(y, tx) :
    """
    Calculate the least squares solution.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)

    Returns:
        w: optimal weights, numpy array of shape(D,1)
        loss: a scalar, calculated with mse.
    """
    
    # calculate w
    w=np.dot(np.dot(np.linalg.inv(np.dot(tx.T,tx)), tx.T), y)
    # calculate loss
    loss = compute_loss(y, tx, w)
    return w, loss

    
"""4. Ridge regression using normal equations."""

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.
    
    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        lambda_ : a scalar, the penalty term. 

    Returns:
        w: optimal weights, numpy array of shape(D,1)
        loss: a scalar, calculated with mse.
        
    """
    # get the dimensions of tx
    N, D = get_dim(tx)
    # transpose tx
    tx_t = np.transpose(tx)
    # obtain A = (tx_t)t + 2*N*lambda*(D*I)
    A = tx_t.dot(tx) + ((2 * N) * (lambda_)) * np.eye(D)
    # solve the linear system Aw = (tx_t)y for w
    w = np.linalg.solve(A, tx_t.dot(y))
    # calculate loss
    loss = compute_loss(y, tx, w)
    return w, loss


"""5. Logistic regression using gradient descent or SGD (y ∈ {0, 1})."""


def calculate_loss(y, tx, w):
    """
    This function computes the cost by negative log likelihood.
    
    Args:
        y: shape=(N, 1)
        tx: shape=(N, D)
        w: shape=(D, 1)
    
    Return :
        loss: a scalar
 
    """
    eps = 1e-15 #value to avoid division by 0
    N = y.shape[0]
    pred = _sigmoid(tx.dot(w))
    loss = (1.0 / N) * np.sum(-y*np.log(pred+eps)-(1-y)*np.log(1-pred+eps)) 
    return loss

def calculate_gradient(y, tx, w):
    """
    This function computes the gradient of loss.
    
     Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, 1)
    
    Returns :
        grad: shape (D, 1)
        
    """
    pred = _sigmoid(tx.dot(w)) #call the sigmoid
    N = y.shape[0]
    grad = (1.0 / N) * np.dot(np.transpose(tx),(pred-y))
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma,loss_bias =0):
    """
    Logistic regression using gradient descent (y ∈ {0, 1}).

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        initial_w: the initial guess (or the initialization) for the model parameters, shape=(D,1). 
        max_iters: a scalar denoting the total number of iterations of gradient descent.
        gamma: a scalar denoting the stepsize. 
        loss_bias: a scalar, a parameter used to treat unbalanced dataset

    Returns:
        w: a scalar, the last weight of the iterations. 
        loss: a scalar, the last loss value of the iterations.
        
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute loss, gradient
        loss=calculate_loss(y, tx, w)
        grad=calculate_gradient(y, tx, w)   
        # update w by gradient descent
        if loss_bias == 0:
            w = w - gamma * grad
        else:
            w = w - gamma * grad * loss_bias
        # store w and loss
        ws.append(w)
        losses.append(loss)

        if n_iter % 100 == 0:
            print(n_iter,loss)
    
    loss=calculate_loss(y, tx, w)
    return w, loss
    
   
  
"""6. Regularized logistic regression using gradient descent or SGD (y ∈ {0, 1}, with regularization term λ∥w∥2)"""

def regularized_loss_gradient(y, tx, w, lambda_):
    """
    This function return the loss and the gradient with regularization term λ∥w∥2.
    
    Args :
        y: shape=(N, 1)
        tx: shape=(N, D)
        w: shape=(D, 1)
        lambda_: a scalar, parameter of the regularization term
    
    Returns :
        loss: a scalar, calculated by negative log likelihood
        grad: shape (D, 1), gradient with regularization term
      
    """
    #Moreover, the loss returned by the regularized methods (ridge regression and reg logistic regression) should not include the penalty term:
    loss=calculate_loss(y,tx,w) 
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent (y ∈ {0, 1}), with regularization term λ∥w∥2.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        lambda_: a scalar, parameter of the regularization term
        initial_w: shape=(D,1), the initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of gradient descent
        gamma: a scalar denoting the stepsize
        

    Returns:
        w : a scalar, the last weight of the iterations. 
        loss: a scalar, the last loss value of the iterations.
    """ 
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # compute loss, gradient
        loss,grad=regularized_loss_gradient(y, tx, w,lambda_)
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if n_iter % 100 == 0:
            print(n_iter,loss)
    loss,grad=regularized_loss_gradient(y, tx, w,lambda_)
    return w, loss

"""7. Regularized logistic regression with lasso using gradient descent or SGD (y ∈ {0, 1}, with regularization terms λ∥w∥2, λ∥w∥1)"""

def regularized_loss_gradient_lasso(y, tx, w, lambda_):
    """
    This function return the loss and the gradient with regularization term λ∥w∥1 (Lasso regularization).
    
    Args :
        y: shape=(N, 1)
        tx: shape=(N, D)
        w: shape=(D, 1)
        lambda_: a scalar, parameter of the Lasso regularization term
    
    Returns :
        loss: a scalar, calculated by negative log likelihood
        grad: shape (D, 1), gradient with regularization term
      
    """
    #Moreover, the loss returned by the regularized methods (ridge regression and reg logistic regression) should not include the penalty term:
    loss=calculate_loss(y,tx,w)
    gradient = calculate_gradient(y, tx, w) +  lambda_ * np.sign(w)
    return loss, gradient

#regularize logistic regression with lasso
def reg_logistic_regression_lasso(y, tx, lambda_ , initial_w, max_iters, gamma, loss_bias=0):
    """
    Regularized logistic regression using gradient descent (y ∈ {0, 1}), with regularization term λ∥w∥1 (Lasso regularization).

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        lambda_: a scalar, parameter of the regularization term
        initial_w: shape=(D,1), the initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of gradient descent
        gamma: a scalar denoting the stepsize
        loss_bias: a scalar, a parameter used to treat unbalanced dataset

    Returns:
        w : a scalar, the last weight of the iterations. 
        loss: a scalar, the last loss value of the iterations.
        
    """ 
    ws = [initial_w]
    initial_loss,_ = regularized_loss_gradient_lasso(y, tx, initial_w,lambda_)
    losses = [initial_loss]
    w = initial_w

    for n_iter in range(max_iters):
        # compute loss, gradient
        loss,grad=regularized_loss_gradient_lasso(y, tx, w,lambda_)
        # update w by gradient descent
        if loss_bias ==0:
            w = w - gamma * grad 
        else :
            w = w - gamma * grad * loss_bias
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if n_iter % 100 == 0:
           print(n_iter,loss)
    loss,grad=regularized_loss_gradient_lasso(y, tx, w,lambda_)
    return w, loss

