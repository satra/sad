import sys
import os
import shutil
import numpy as np
from glob import glob
from sklearn.linear_model.base import BaseEstimator, RegressorMixin
import sklearn.linear_model as lm
import sklearn.metrics as skm
import sklearn.cross_validation as cv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spm_2lvl import do_spm         #spm workflow --> give directory + confiles
from feature_selection import determine_model_all
from cluster_tools import get_labels, get_clustermeans

## INITIAL SETUP
outdir = '/mindhive/gablab/u/fhorn/Sad/testmodels/better/figures_scatterpreds_skl'
if not os.path.isdir(outdir):
    os.mkdir(outdir)
    
# in this folder are all sym links to con files
spmdir = '/mindhive/scratch/fhorn/model_spminp_l2o/con1'
if not os.path.isdir(spmdir):
    sys.exit("please run the clustersmodel_l2ocrossval.py script first to generate the necessary input files")
# here the spm files created for getting the clustermask are temporarily saved 
tempspmdir = '/mindhive/scratch/fhorn/tempspm'
if not os.path.isdir(tempspmdir):
    os.mkdir(tempspmdir)
    
confiles = sorted(glob(os.path.join(spmdir,'*_con1.nii')))

# original input file with test scores etc for every subject + all amygdala activations
pdata = np.recfromcsv('/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/l2output/social/split_halves/regression/lsasDELTA/6mm/allsubs.csv',names=True)
# put here either lsas_delta or lsas_post
responsevar = pdata.lsas_pre - pdata.lsas_post
subject_num = len(pdata.subject)
behvars = 2 # don't forget to update this when changeing the vars

def setup_spm(subjects, select_conf):
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
    analdirs = []
    for trainidx, testidx in cv.LeaveOneOut(len(subjects)):
        trainconfiles = np.array(select_conf)[trainidx]
        leftout = subjects[testidx][0]
        analdir = os.path.join(tempspmdir,"anal%02d_%s"%(leftout, np.random.randint(100000)))
        # workflow
        do_spm(analdir, trainconfiles)
        analdirs.append(os.path.join(analdir,'thresh'))
    return analdirs


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
        # get the confiles of the given subjects
        select_conf = [confiles[i] for i in X[:,0]]
        # run the SPM analysis (external workflow) in a LOO crossval
        # and save the directories in which the threshold images are located
        analdirs = setup_spm(X[:,0], select_conf)
        # get labels & clustermeans
        labels, nlabels = get_labels(analdirs)
        # delete all the workflow directories again
        [shutil.rmtree(os.path.dirname(analdir)) for analdir in analdirs]
        self.labels_ = labels
        self.nlabels_ = nlabels
        clustermeans = get_clustermeans(labels, nlabels, select_conf)
        # make new design matrix (first behvars, then clustermeans)
        X_new = np.hstack((X[:,1:], clustermeans))
        varidx, model = determine_model_all(X_new, y, behvars) 
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
            select_conf = [confiles[i] for i in X[:,0]]
            clustermeans = get_clustermeans(self.labels_, self.nlabels_, select_conf)
            # make new matrix (first behvars, then clustermeans)
            X_new = np.hstack((X[:,1:], clustermeans))
            prediction = self.model_.predict(X_new[:,self.varidx_])
            return prediction
        else:
            raise Exception('no model')
            
if __name__ == "__main__":
    X = np.array([range(subject_num),pdata.classtype-2,pdata.lsas_pre]).T
    Y = responsevar
    value, distribution, pvalue = cv.permutation_test_score(BrainReg(), X, Y,
                                                            skm.explained_variance_score,
                                                            cv=cv.KFold(subject_num,subject_num),
                                                            n_permutations=1, n_jobs=16)
 
    print distribution
    print value
    print pvalue
    plt.figure()
    plt.hist(distribution, 128)
    plt.plot([value, value], [0, 50], color='r')
    plt.title('p = %.3f' % pvalue)
    plt.savefig(os.path.join(outdir,"permtest_hist.png"),dpi=100,format="png")

    
