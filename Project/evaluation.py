import numpy as np

def f1_score(y_true,y_pred):
    """
    This function calculates the F1 score in order to evalute the performance of the model. 
    
    Args:
        y_true : The true data to test the model, shape=(N, 1)
        y_pred : The predictions obtained with the model, shape=(N, 1)

    Return:
        F1 : The F1 score, a scalar 
        
    """

     # Transform y_true so that it contains the correct labels -1 and 1 (label 0 is replaced by label -1)
    for j in range(len(y_true)):
        if y_true[j] == 0:
            y_true[j] = -1

    # Split the true labels into two arrays
    y_true_0 = y_true[y_true == -1]
    y_true_1 = y_true[y_true == 1]
    # Split the predictions into two arrays
    y_0 = y_pred[y_pred == -1]
    y_1 = y_pred[y_pred == 1]

    # Calculate the confusion matrix :
    y_corr = y_true[y_true==y_pred]
     # true negatives,
    tn = len(y_corr[y_corr==-1])
    # true positives,
    tp = len(y_corr[y_corr==1])
    # false negatives and
    fn = len(y_0)-tn
    # false positives. 
    fp = len(y_1)-tp
    
    # Calculate F1 score
    F1 = 2*tp/(2*tp+fp+fn)

    # Print the accuracy, the percentage of correct predictions for class -1 and class 1, and the F1 score.
    print('Accuracy: ', np.sum(y_pred == y_true) / y_true.shape[0] * 100, '%')
    print("Percentage of correct -1: " + str(tn/len(y_true_0)*100))
    print("Percentage of correct 1: " + str(tp/len(y_true_1)*100))
    print("F1 score: "+str(F1))

    return F1
