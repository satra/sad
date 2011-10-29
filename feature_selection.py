import numpy as np

from itertools import combinations

import sklearn.linear_model as lm
import sklearn.cross_validation as cv
import sklearn.metrics as metrics

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
    n_samples, n_features = X.shape

    if n_features == behvars:
        idx_final = np.ones((n_features)).astype(np.bool)
        print "no imaging clusters"
    else:
        rem_features = n_features - behvars
        combos = []
        for i in range(1, rem_features+1):
            combos.extend(combinations(range(rem_features), i))

        min_error = np.inf
        best_features = None
        for idx, feature_idx in enumerate(combos):
            features = np.hstack((range(behvars),
                                  np.array(feature_idx) + behvars)).astype(np.int)
            error = sum(cv.cross_val_score(lm.LinearRegression(),
                                           X[:, features],
                                           y.ravel(),
                                           metrics.mean_square_error,
                                           cv.KFold(n_samples, 5)))
            if error < min_error:
                min_error = error
                best_features = features
        idx_final = np.zeros((n_features)).astype(np.bool)
        idx_final[best_features] = True
        print "lowest RMSE achived: %s with formula:" % np.sqrt(error)
    print idx_final
    model = lm.LinearRegression().fit(X[:,idx_final], y.ravel())
    return idx_final, model
