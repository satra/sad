import os
import numpy as np
from cfutils import get_subjects, get_subject_data

X = get_subjects()
_, pdata = get_subject_data(X)
X = pdata.subject
y = pdata.lsas_pre - pdata.lsas_post

lgroup,_ = get_subject_data(X[y<=np.median(y)])
hgroup,_ = get_subject_data(X[y>np.median(y)])

import nipype.interfaces.spm as spm

from nipype.caching import Memory
os.makedirs('/mindhive/scratch/satra/sadfigures/nipype_mem')
mem = Memory('/mindhive/scratch/satra/sadfigures')

designer = mem.cache(spm.OneSampleTTestDesign)
estimator = mem.cache(spm.EstimateModel)
cestimator = mem.cache(spm.EstimateContrast)

ldesres =  designer(in_files = lgroup)
lestres = estimator(spm_mat_file=ldesres.outputs.spm_mat_file, estimation_method={'Classical':None})
lcestres = cestimator(spm_mat_file=lestres.outputs.spm_mat_file, beta_images=lestres.outputs.beta_images, residual_image=lestres.outputs.residual_image, group_contrast=True, contrasts=[('LGroup', 'T', ['mean'], [1])])

hdesres =  designer(in_files = hgroup)
hestres = estimator(spm_mat_file=hdesres.outputs.spm_mat_file, estimation_method={'Classical':None})
hcestres = cestimator(spm_mat_file=hestres.outputs.spm_mat_file, beta_images=hestres.outputs.beta_images, residual_image=hestres.outputs.residual_image, group_contrast=True, contrasts=[('LGroup', 'T', ['mean'], [1])])
