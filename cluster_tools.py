import os
import numpy as np
from scipy.ndimage import label
import nibabel as nib

imgshape = nib.load('/mindhive/scratch/fhorn/model_spminp_l2o/con1/SAD_P03_con1.nii').get_data().shape

def get_labels(analdirs):
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
    conjunc = np.ones(imgshape)
    for analdir in analdirs:  
        # get clusters
        fname = os.path.join(analdir,'spmT_0001_thr.img')
        img = nib.load(fname)
        data = img.get_data()
        # do the conjunction
        idx = np.where(data == 0)
        conjunc[idx] = 0
    # label the remaining clusters
    labels, nlabels = label(conjunc)    
    return labels, nlabels
    
def get_clustermeans(labels, nlabels, confiles):
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

