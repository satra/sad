import numpy as np
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
from glob import glob
import sklearn.linear_model as lm
import sklearn.metrics as skm
import sklearn.cross_validation as cv
import nibabel as nib
from nipy.labs import viz

from feature_selection import determine_model_all
from cluster_tools import get_labels, get_clustermeans

#for debugging: to print big arrays (and hopefully save big arrays too...):
np.set_printoptions(threshold='nan')

## INITIAL SETUP
outdir = '/mindhive/gablab/u/fhorn/Sad/testmodels/better/figures_scatterpreds_skl'
if not os.path.isdir(outdir):
    os.mkdir(outdir)
# original input file with test scores etc for every subject
pdata = np.recfromcsv('/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/l2output/social/split_halves/regression/lsasDELTA/6mm/allsubs.csv',names=True)
# put here either lsas_delta or lsas_post
responsevar = pdata.lsas_pre - pdata.lsas_post
subject_num = len(pdata.subject)
desmat = np.array([pdata.classtype-2,pdata.lsas_pre]).T
behvars = 2 # don't forget to update this when changeing the vars in desmat

# in this folder, all sym links to con files and all the SPM output will be loaded from if already exist
spmdir = '/mindhive/scratch/fhorn/model_spminp_l2o/con1'
if not os.path.isdir(spmdir):
    sys.exit("please run the clustersmodel_l2ocrossval.py script first to generate the necessary input files") 
    
confiles = sorted(glob(os.path.join(spmdir,'*_con1.nii')))
imgshape = nib.load(confiles[0]).get_data().shape
imgaff = nib.load(confiles[0]).get_affine()
imghead = nib.load(confiles[0]).get_header()

anat_img = nib.load('/software/mricron/templates/ch2.nii.gz')
anat_data, anat_aff = anat_img.get_data(), anat_img.get_affine()

def brainplot(brainmat, savepath):
    """
    takes a matrix (e.g. from loading an image file) and plots the activation
    the figure is saved at 'savepath'
    """
    # savepath should end in .png
    plt.figure()
    osl = viz.plot_map(np.asarray(brainmat), imgaff, anat=anat_data, anat_affine=anat_aff, 
                       threshold=0.0001, black_bg=True, draw_cross=False)
    pylab.savefig(savepath)

def crossval():
    """
    perform a crossvalidation on the data (beh + precomputed brain) of all subjects
    """ 
    predscores = []
    actualscores = []
    clust_disj = np.zeros(imgshape)
    for trainidx, testidx in cv.LeaveOneOut(subject_num):
        # n-p training files
        trainconfiles = [cf for i, cf in enumerate(confiles) if trainidx[i]]
        # left out subjects to test with
        testconfiles = [cf for i, cf in enumerate(confiles) if testidx[i]]
        ### get all the files from a leave2out crossval and get clusters
        _, name = os.path.split(testconfiles[0])
        sid = name.split('con')[0][:-1]
        # sidx is the row# of the sid in our pdata variable
        sidx = np.nonzero(pdata.subject == sid)[0][0]
        analysisdirs = []
        for idx in range(subject_num):
            if not idx == sidx:
                left_out = [sidx, idx]
                left_out.sort()
                analysisdirs.append(os.path.join(spmdir,'analysis_lo_%02d_%02d'%(left_out[0],left_out[1]),'thresh_h01_f05'))
        # get labels and clustermeans
        labels, nlabels = get_labels(analysisdirs)
        clustermeans_train = get_clustermeans(labels, nlabels, trainconfiles)
        clustermeans_test = get_clustermeans(labels, nlabels, testconfiles)
        # make desmats
        X_train = np.hstack((desmat[trainidx], clustermeans_train))
        X_test = np.hstack((desmat[testidx], clustermeans_test))
        # fit the model (by determining the best model first)
        varsidx, model = determine_model_all(X_train, responsevar[trainidx])
        # save location of _selected_ clusters
        for clust in range(nlabels):
            if varsidx[behvars+clust]:
                idx = np.where(labels == clust+1)
                clust_disj[idx] += 1
        # and save scores
        prediction = model.predict(X_test[:,varsidx])
        predscores.append(prediction)
        actualscores.append(responsevar[testidx][0])
    # rearrange vectors for error computation
    actualscores = np.array(actualscores)
    predscores_beta = []
    for y in xrange(len(predscores)):
        [predscores_beta.append(x) for x in predscores[y]]
    predscores_alpha = np.array(predscores_beta)
    # compute errors
    prederrors = predscores_alpha - actualscores
    meanerr = np.mean(np.abs(prederrors))
    rmsqerr = np.sqrt(np.mean(prederrors**2))
    # save + plot cluster distribution in brain
    brainplot(clust_disj, os.path.join(outdir,"cluster_disj_crossval.png"))
    outimg = os.path.join(outdir,'clusterdisj_crossval.nii')
    nib.Nifti1Image(clust_disj,imgaff,imghead).to_filename(outimg)
    return predscores_alpha, actualscores, meanerr, rmsqerr

def actvspred(modelname, predmodel):
    """
    plot the predicted vs. the actual score
    """
    predscores, actualscores, meanerr, rmsqerr = predmodel
    axmax = int(round(np.max([predscores,actualscores])))
    axmin = int(round(np.min([predscores,actualscores])))
    # fit line through the scores
    actualscores2 = actualscores.reshape(subject_num,1)
    model = lm.LinearRegression()
    model.fit(actualscores2, predscores)
    # get explained variance  
    rsqrd = skm.explained_variance_score(actualscores, predscores)
    x = np.array(range(axmin-5, axmax+6))
    y = model.coef_[0]*x+model.intercept_
    # plot scatterplot and lines
    plt.figure()
    plt.scatter(actualscores,predscores,s=70)
    plt.plot(x,x,'g',label='optimal model')  
    plt.plot(x,y,'k',label='our model',linewidth=2)
    plt.xlabel("actual lsas delta")
    plt.ylabel("predicted lsas delta")
    plt.title(modelname)
    plt.axis([axmin-5,axmax+5,axmin-5,axmax+5])
    axes = plt.axes()
    axes.grid(b=True)
    axes.text(0.05,0.8,"meanerr: %.2f\nrmse: %.2f\nexpl. var: %.2f"%(meanerr,rmsqerr,rsqrd),transform=axes.transAxes)
    #plt.legend()
    plt.savefig(os.path.join(outdir,"%s_crossval.png"%modelname),dpi=100,format="png")

   
if __name__=="__main__":
    # plot the actual vs. predicted scores from a crossval where in each run the optimal model is determined
    actvspred("full",crossval())

    
