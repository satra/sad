from mayavi import mlab
from surfer import Brain, io
subject = 'fsaverage' # SAD_017
surface = 'pial'

br1 = Brain(subject, 'rh', surface, title='lgroup')
l_surf_data = io.project_volume_data('/mindhive/scratch/satra/sadfigures/nipype_mem/nipype-interfaces-spm-model-EstimateContrast/cf535c8b3e6380c2c8512307f3c294ad/spmT_0001.img',
                                     'rh', subject_id='fsaverage',
                                     target_subject=subject)
br1.add_overlay(l_surf_data, min=2, max=3.7, sign='pos', name='lgroup')

br2 = Brain(subject, 'rh', surface, title='hgroup')
h_surf_data = io.project_volume_data('/mindhive/scratch/satra/sadfigures/nipype_mem/nipype-interfaces-spm-model-EstimateContrast/d1fbe9c9d24038d7d1e16b16936f52ed/spmT_0001.img',
                                     'rh', subject_id='fsaverage',
                                     target_subject=subject)
br2.add_overlay(h_surf_data, min=2, max=3.7, sign='pos', name='hgroup')
mlab.sync_camera(br1._f, br2._f)
mlab.sync_camera(br2._f, br1._f)

"""
br1.save_image('lgroup.png')
br2.save_image('hgroup.png')
"""