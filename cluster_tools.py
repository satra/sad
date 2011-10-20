from glob import glob
import os
import numpy as np
from scipy.ndimage import label
import nibabel as nib

from cfutils import get_subject_data
#imgshape = nib.load('/mindhive/scratch/fhorn/model_spminp_l2o/con1/SAD_P03_con1.nii').get_data().shape

def get_labels(analdir):
    """
    conputes the conjunction of clusters based on the spm thr images from all given analysis directories
    --> since this was created with spm_2lvl.py, the thresholded image is in a 'thresh' folder. this must be
        appended before giving the function the directory paths

    Parameters:
    -----------
    analdirs: list of directories created in the SPM workflow in which the spm_thr image can be found

    Returns:
    --------
    labels: array with clusters labeled as '1', '2', etc.

    nlabels: number of clusters/labels: max(labels)
    """
    confiles = glob(os.path.join(analdir, '*', 'thresh', 'spmT_0001_thr.img'))
    conjunc = np.zeros(nib.load(confiles[0]).get_data().shape)

    for fname in confiles:
        #print fname
        data = nib.load(fname).get_data()
        #print np.nonzero(data>0)[0].shape
        #print np.prod(data.shape)
        # get clusters
        conjunc += (data>0).astype(np.int)
    # label the remaining clusters
    return label(conjunc==len(confiles))

def get_clustermeans(X, labels, nlabels):
    """
    finds the cluster means of each of the nlabels clusters for each subject in the confiles
    
    Parameters:
    -----------
    labels: array with labeled clusters
    
    nlabels: number of clusters
    
    confiles: links to the confiles of various subjects, for each of the files, the clustermask
              for a specifically labeled cluster gets applied and then of the selected values,
              the mean is computed
              
    Returns:
    --------
    clustermeans: array of shape (len(confiles), nlabels) with the clustermeans for each subject
                  for each cluster
    """
    confiles, _ = get_subject_data(X)
    clustermeans = np.zeros([len(confiles), nlabels])
    for sub, conf in enumerate(confiles):
        data = nib.load(conf).get_data()
        mean_val = np.zeros(nlabels)

        for clusteridx in range(1,nlabels+1):
            #for each cluster, find the average value of voxel intensity from data
            idx = np.where(labels == clusteridx)
            mean_val[clusteridx-1] = np.mean(data[idx])

        clustermeans[sub,:] = mean_val
    return clustermeans 

