import numpy as np
from copy import deepcopy

import sklearn.linear_model as lm
import sklearn.cross_validation as cv

def make_model(X, y):
    """
    runs a regression and returns the model
    
    Parameters:
    -----------
    X: array with values for each subject (beh + clustermeans)
    
    y: responsevar targetvalues for all subjects
    
    Returns:
    --------
    reg: the fitted model
    """
    reg = lm.LinearRegression()
    reg.fit(X, y)
    return reg

def determine_model_all(X, y, behvars=2):
    """
    outer function for the model feature selection. used to itterate through all possible
    feature combinations based on the variable values given in X
    
    Parameters:
    -----------
    X: array with variable (feature) values for each subject
    
    y: target values (important for model fitting)
    
    behvars: how many of the (first) variables of X should be included in all models by default
    
    Returns:
    --------
    idx_final: a vector of the same length as the total number of available features,
               filled with boolean values depending on weather the feature was chosen for
               the final model or not
    
    model: a sklearn.linear_model fitted on X&y using only the variables of X that are set
           to True in idx_final
    """
    num_vars = X.shape[1]
    # behvars first variables will always be included in the model (forced)
    # beh as default vars
    idx_final = np.array([False for i in xrange(num_vars)])
    idx_final[:behvars] = True #use all behvars in the model 
    # compute the first base error
    _, _, _, error = nested_crossval(X[:,idx_final], y)
    # idx array used for keeping track of the best idx combination
    idx_final_temp = deepcopy(idx_final)
    # find the model that has the lowest error
    for col in range(behvars, num_vars):
        # start with the given idx and 1 additional
        idx_temp = deepcopy(idx_final)
        idx_temp[col] = True
        # use recursion for other combinations, only best idx is returned
        idx_temp, error_temp = allcomb_rec(y, X, idx_temp, col+1)
        # update error + idx
        if error_temp < error:
            error = error_temp
            idx_final_temp = deepcopy(idx_temp)
    
    idx_final = idx_final_temp       
    # make the final model
    model = make_model(X[:,idx_final], y)
    print "lowest RMSE achived: %s with formula:"%error
    print idx_final
    return idx_final, model
    
def allcomb_rec(y, X, idx, col):
    """
    recursive heart of determine_model_all()
    
    Parameters:
    -----------
    y: target values
    
    X: matrix with feature values
    
    idx: current index
    
    col: indicates which features can still be taken (needed so that every feature
         combination indep of order is tried only once)
         
    Returns:
    --------
    idx_final: the best feature combination as determined in the whole recursion so far
    
    error: the error that this feature combination yields (to measure the goodness of fit)
    """
    num_vars = len(idx)
    # compute current error (the one with the least variables acts as base error)
    idx_final = deepcopy(idx)
    _, _, _, error = nested_crossval(X[:,idx_final], y)
    # if there are no more variables to take, end
    if num_vars == col:
        return idx_final, error
    # else call the function again with one additional variable of the remaining ones
    for col2 in range(col, num_vars):
        idx_temp = deepcopy(idx)
        idx_temp[col2] = True
        # recursion!
        idx_temp, error_temp = allcomb_rec(y, X, idx_temp, col2+1)
        # update error
        if error_temp < error:
            error = error_temp
            idx_final = deepcopy(idx_temp)
    return idx_final, error

def nested_crossval(X, y):
    """
    runs a complete LOO cross-validation on the given data
    
    Parameters:
    -----------
    X: variable values (all features in here will be used)
    
    y: target values
    
    Returns:
    --------
    predscores_alpha: the predicted scores
    
    acutalscores: the actual y values
    
    meanerr: the (abs) mean error between the predicted and actual scores
    
    rmserr: root mean squared error of pred vs actual scores
    """    
    predscores = []
    actualscores = []
    for trainidx, testidx in cv.LeaveOneOut(len(y)):
        # make desmats
        X_train = X[trainidx,:]
        X_test = X[testidx,:]
        # fit the model
        model = make_model(X_train, y[trainidx])
        # predict
        prediction = model.predict(X_test)
        predscores.append(prediction)
        actualscores.append(y[testidx][0])
    # some resturcturing, otherwise the error is not calculated correctly...
    actualscores = np.array(actualscores)
    predscores_beta = []
    for y in xrange(len(predscores)):
        [predscores_beta.append(x) for x in predscores[y]]
    predscores_alpha = np.array(predscores_beta)
    # compute errors
    prederrors = predscores_alpha - actualscores
    meanerr = np.mean(np.abs(prederrors))
    rmsqerr = np.sqrt(np.mean(prederrors**2))     
    return predscores_alpha, actualscores, meanerr, rmsqerr
