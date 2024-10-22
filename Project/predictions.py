import numpy as np

# Calculates sigmoid(x)
def _sigmoid(x, clip=1e-8):
    """
    This function applies the logistic function (a.k.a. sigmoid) on x.
    
    Args:
        x : scalar or numpy array
        clip : limit the values in the returns array

    Return:
        scalar or numpy array
        
    """
    return np.clip(1/(1 + np.exp(-x)), clip, 1 - clip) #calculate and return the sigmoid

def logistic_prediction(x, weights):
    """
    This function calculates the linear combination of the input features and the model parameters and applies the logistic (sigmoid)   
    function to this linear combination.
    
    Args:
        x : Input features, shape=(N, D)
        weights : Model parameters, shape=(D, 1)

    Return:
        y_hat : The prediction, shape=(N, 1)
    
    """
    z = x.dot(weights)
    y_hat = _sigmoid(z) #calculate the prediction thanks to the sigmoid funcrion
    return y_hat
