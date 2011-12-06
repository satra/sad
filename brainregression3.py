import os
import shutil
import time

# prevent lengthy SPM output
from nipype.utils.logger import logging, logger, fmlogger, iflogger
logger.setLevel(logging.getLevelName('CRITICAL'))
fmlogger.setLevel(logging.getLevelName('CRITICAL'))
iflogger.setLevel(logging.getLevelName('CRITICAL'))

import numpy as np
from sklearn.linear_model.base import BaseEstimator, RegressorMixin
import sklearn.metrics as skm
import sklearn.cross_validation as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#from nipype.utils.config import config
#config.enable_debug_mode()

import nipype.pipeline.engine as pe

from spm_2lvl import do_spm         #spm workflow --> give directory + confiles
from feature_selection import determine_model_all
from cluster_tools import get_labels, get_clustermeans
from cfutils import get_subjects, get_subject_data

## INITIAL SETUP
outdir = os.path.join(os.getcwd(),'figures_scatterpreds_skl')
if not os.path.isdir(outdir):
    os.makedirs(outdir)
    
tempspmdir = '/mindhive/scratch/satra/tempspm'
if os.path.isdir(tempspmdir):
    shutil.rmtree(tempspmdir)
os.makedirs(tempspmdir)


def setup_spm(subjects, y):
    """
    runs the SPM analysis workflow within a LOO crossvalidation
    
    Parameters:
    -----------
    subjects: vector with a subset of range(subject_num)
    
    select_conf: list of confiles from all subjects in the subjects index vector
    
    Returns:
    --------
    analdirs: the list of paths to the directories where the SPM output is saved
    """

    metawf = pe.Workflow(name='fitloo')
    np.random.seed(int(time.time()*100000))
    metawf.base_dir = os.path.join(tempspmdir, '%f_%d' % (time.time(),
                                                          np.random.randint(10000)))
    count = 0
    _, pdata = get_subject_data(subjects)
    max_folds = np.min(np.histogram(pdata.classtype, bins=2)[0])
    #print subjects, pdata.classtype, max_folds
    for trainidx, testidx in cv.StratifiedKFold(pdata.classtype, max_folds):
        # workflow
        count += 1
        analname='anal%02d' % count
        wf = do_spm(subjects[trainidx], y[trainidx],
                    analname=analname,
                    run_workflow=False)
        metawf.add_nodes([wf])
    #print count
    #print metawf._graph.nodes()
    metawf.run(plugin='PBS', plugin_args={'qsub_args': '-o /dev/null -e /dev/null',
                                          'max_tries': 5,
                                          'retry_timeout': 1})
    metawf.run(plugin='MultiProc', plugin_args={'n_procs': 24})
    #metawf.run()
    return os.path.join(metawf.base_dir, metawf.name)

def _fit(X, y, behav_data=None):
    # run the SPM analysis (external workflow) in a LOO crossval
    # and save the directories in which the threshold images are located
    print "Fitting"
    print X
    print y
    print "doing spm cv"
    analdir = setup_spm(X, y)
    # get labels & clustermeans
    labels, nlabels = get_labels(analdir)
    # delete all the workflow directories again
    shutil.rmtree(os.path.realpath(os.path.join(analdir, '..')))
    clustermeans = get_clustermeans(X, labels, nlabels)
    #print "finding model"
    # make new design matrix (first behvars, then clustermeans)
    if behav_data is not None:
        X_new = np.hstack((behav_data, clustermeans))
        varidx, model = determine_model_all(X_new, y, behav_data.shape[1])
    else:
        X_new = clustermeans
        varidx, model = determine_model_all(X_new, y, 0)
    return model, varidx, labels, nlabels


class BrainReg(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model_ = None
        self.labels_ = None
        self.nlabels_ = None
        
    def fit(self,X,y):
        """
        fits the model using linear regression
        
        Parameters:
        -----------
        X:  the design matrix; assumes first column is the subject id (number 
            between 0 and n) according to which, SPM 2lvl is done and clusters
            and clustermeans are computed; remaining columns are beh data
            
        y:  target scores for each subject
        """

        _, pdata = get_subject_data(X)
        features = np.hstack((pdata.lsas_pre[:, None],
                              pdata.classtype[:, None] - 2))
        model, varidx, labels, nlabels = _fit(X, y, features)
        self.labels_ = labels
        self.nlabels_ = nlabels
        self.model_ = model
        self.varidx_ = varidx
        return self
        
    def predict(self,X):
        """
        predicts from the linear regression model
        
        Parameters:
        -----------
        X:  test samples used for prediction. Assumes same structure as above
        """
        if self.model_ is not None:
            # get the confiles + clustermeans of the given subjects
            clustermeans = get_clustermeans(X, self.labels_, self.nlabels_)
            _, pdata = get_subject_data(X)
            features = np.hstack((pdata.lsas_pre[:, None],
                                  pdata.classtype[:, None] - 2))
            # make new matrix (first behvars, then clustermeans)
            X_new = np.hstack((features, clustermeans))
            prediction = self.model_.predict(X_new[:,self.varidx_])
            return prediction
        else:
            raise Exception('no model')

if __name__ == "__main__":
    X = get_subjects()
    _, pdata = get_subject_data(X)
    X = pdata.subject
    y = pdata.lsas_pre - pdata.lsas_post
    n_subjects, = X.shape

    """
    result = []
    for train, test in cv.StratifiedKFold(pdata.classtype, 18):
        model = BrainReg().fit(X[train], y[train])
        result.append((y[test], model.predict(X[test])))
    """

    value, distribution, pvalue = cv.permutation_test_score(BrainReg(), X, y,
                                                            skm.mean_square_error,
                                                            cv=cv.StratifiedKFold(
                                                                pdata.classtype,
                                                                18),
                                                            n_permutations=1000,
                                                            n_jobs=1)
 
    print distribution
    print value
    print pvalue
    plt.figure()
    plt.hist(distribution, 128)
    plt.plot([value, value], [0, 50], color='r')
    plt.title('p = %.3f' % pvalue)
    plt.savefig(os.path.join(outdir,"permtest_hist.png"),dpi=100,format="png")
    #model, varidx, labels, nlabels = _fit(X, y, pdata.lsas_pre[:,None])

    
