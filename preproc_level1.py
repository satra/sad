"""
1. Tell python where to find the appropriate functions.
"""

import nipype.interfaces.io as nio           # Data i/o 
import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.matlab as mlab      # how to run matlab
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.utility as util     # misc. modules
import nipype.algorithms.rapidart as ra	     # ra

#####################################################################
"""
2. Setup any package specific configuration. The output file format
   for FSL routines is being set to uncompressed NIFTI and a specific
   version of matlab is being used. The uncompressed format is
   required because SPM does not handle compressed NIFTI.
"""

# Tell freesurfer what subjects directory to use
subjects_dir = '/mindhive/gablab/sad/PY_STUDY_DIR/Block/data'
fs.FSCommand.set_default_subjects_dir(subjects_dir)

# Set the way matlab should be called
mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
# If SPM is not in your MATLAB path you should add it here
# mlab.MatlabCommand.set_default_paths('/path/to/your/spm8')

from volsurf import l1pipeline

"""
3. The following lines of code sets up the necessary information
   required by the datasource module. It provides a mapping between
   run numbers (nifti files) and the mnemonic ('struct', 'func',
   etc.,.)  that particular run should be called. These mnemonics or
   fields become the output fields of the datasource module. In the
   example below, run 'f3' is of type 'func'. The 'f3' gets mapped to
   a nifti filename through a template '%s.nii'. So 'f3' would become
   'f3.nii'.
"""

subject_list = ['SAD_P56','SAD_P57','SAD_P58']


# Map field names to individual subject runs.
info = dict(func=[['subject_id', 'f3']],
            struct=[['subject_id','struct']])
infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")

"""Here we set up iteration over all the subjects. The following line
is a particular example of the flexibility of the system.  The
``datasource`` attribute ``iterables`` tells the pipeline engine that
it should repeat the analysis on each of the items in the
``subject_list``. In the current example, the entire first level
preprocessing and estimation will be repeated for each subject
contained in subject_list.
"""

infosource.iterables = ('subject_id', subject_list)

######################################################################
# Setup preprocessing pipeline nodes

"""
Now we create a :class:`nipype.interfaces.io.DataGrabber` object and
fill in the information from above about the layout of our data.  The
:class:`nipype.pipeline.NodeWrapper` module wraps the interface object
and provides additional housekeeping and pipeline specific
functionality.
"""

datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=['func', 'struct']),
                     name = 'datasource')
datasource.inputs.base_directory = subjects_dir
datasource.inputs.template = '%s/%s.nii'
datasource.inputs.template_args = info

#######################################################################
# setup analysis components
#######################################################################
"""
   a. Setup a function that returns subject-specific information about
   the experimental paradigm. This is used by the
   :class:`nipype.interfaces.model.SpecifyModel` to create the
   information necessary to generate an SPM design matrix. In this
   tutorial, the same paradigm was used for every participant. Other
   examples of this function are available in the `doc/examples`
   folder. Note: Python knowledge required here.
"""
from nipype.interfaces.base import Bunch
from copy import deepcopy
def subjectinfo(subject_id):
    print "Subject ID: %s\n"%str(subject_id)  
    output = []
    names = ['AngryFaces','NeutralFaces','PotentScenes','EmotionalScenes','NeutralScenes']
    regular = ['SAD_017','SAD_019','SAD_021','SAD_023','SAD_025','SAD_028','SAD_030','SAD_032','SAD_034','SAD_036','SAD_038','SAD_040',
'SAD_043','SAD_045','SAD_047','SAD_049','SAD_051','SAD2_019','SAD2_025','SAD2_028','SAD2_032','SAD2_030','SAD2_036','SAD2_038',
'SAD2_047','SAD2_049','SAD_P03','SAD_P04','SAD_P07','SAD_P09','SAD_P11','SAD_P13','SAD_P15','SAD_P17','SAD_P19','SAD_P21',
'SAD_P24','SAD_P26','SAD_P28','SAD_P30','SAD_P32','SAD_P33','SAD_P35','SAD_P37','SAD_P39','SAD_P41','SAD_P43','SAD_P46',
'SAD_P48','SAD_P49','SAD_P53','SAD_P54','SAD_P57','SAD_P58','SAD_POST09','SAD_POST07','SAD_POST04','SAD_POST13','SAD_POST11','SAD_POST21','SAD_POST24',
'SAD_POST26','SAD_POST28','SAD_POST30','SAD_POST35','SAD_POST39','SAD_POST41','SAD_POST46','SAD_POST53']

    cb = ['SAD_018','SAD_020','SAD_022','SAD_024','SAD_027','SAD_029','SAD_031','SAD_033','SAD_035','SAD_037','SAD_039','SAD_041','SAD_044',
'SAD_046','SAD_048','SAD_050','SAD2_020','SAD2_022','SAD2_024','SAD2_027','SAD2_029','SAD2_031','SAD2_033','SAD2_039','SAD2_046',
'SAD2_050','SAD2_044','SAD_P05','SAD_P06','SAD_P08','SAD_P10','SAD_P12','SAD_P14','SAD_P16','SAD_P18','SAD_P20','SAD_P22','SAD_P23','SAD_P25',
'SAD_P27','SAD_P29','SAD_P31','SAD_P34','SAD_P36','SAD_P38','SAD_P40','SAD_P42','SAD_P44','SAD_P45',
'SAD_P47','SAD_P50','SAD_P51','SAD_P52','SAD_P55','SAD_P56','SAD_POST05','SAD_POST06','SAD_POST08','SAD_POST10','SAD_POST12','SAD_POST14',
'SAD_POST16','SAD_POST20','SAD_POST22','SAD_POST27','SAD_POST31','SAD_POST34','SAD_POST38','SAD_POST36','SAD_POST44','SAD_POST45','SAD_POST47',
'SAD_POST50','SAD_POST51','SAD_POST52']
 
    for r in range(1):
	if subject_id in regular:
		onsets = [[45,120,240,315,405,465],[60,135,195,285,420,495],[30,105,255,330,375,525],[15,165,210,300,390,510],[75,150,225,345,435,480]]
	elif subject_id in cb:
		onsets = [[75,135,225,300,420,495],[45,120,255,345,405,480],[15,165,210,285,435,510],[30,150,240,330,375,525],[60,105,195,315,390,465]]
	else: 
		raise Exception('%s unknown' %subject_id)
	durations = [[15],[15],[15],[15],[15]]
        output.insert(r,
                      Bunch(conditions=names,
                            onsets=deepcopy(onsets),
                            durations=deepcopy(durations),
                            amplitudes=None,
                            tmod=None,
                            pmod=None,
                            regressor_names=None,
                            regressors=None))
    return output
"""
   b. Setup the contrast structure that needs to be evaluated. This is
   a list of lists. The inner list specifies the contrasts and has the
   following format - [Name,Stat,[list of condition names],[weights on
   those conditions]. The condition names must match the `names`
   listed in the `subjectinfo` function described above. 
"""
cont1 = ['AngryFaces>NeutralFaces','T', ['AngryFaces','NeutralFaces'],[1,-1]]
cont2 = ['EmotionalScenes>NeutralScenes','T', ['EmotionalScenes','NeutralScenes'],[1,-1]]
cont3 = ['NeutralFaces>NeutralScenes','T', ['NeutralFaces','NeutralScenes'],[1,-1]]
cont4 = ['AngryFaces>EmotionalScenes','T', ['AngryFaces','EmotionalScenes'],[1,-1]]
cont5 = ['AFES>NFNS','T', ['AngryFaces','EmotionalScenes','NeutralFaces','NeutralScenes'],[.5,.5,-.5,-.5]]
cont6 = ['NFAF>NSES','T', ['NeutralFaces','AngryFaces','NeutralScenes','EmotionalScenes'],[.5,.5,-.5,-.5]]
cont7 = ['AFNS>NFES','T', ['AngryFaces','NeutralScenes','NeutralFaces','EmotionalScenes'],[.5,.5,-.5,-.5]]
cont8 = ['All>Fix','T', ['AngryFaces','NeutralFaces','PotentScenes','EmotionalScenes','NeutralScenes'],[.2,.2,.2,.2,.2]]
cont9 = ['AngryFaces>NeutralScenes','T', ['AngryFaces','NeutralScenes'],[1,-1]]
cont10 = ['PotentScenes>NeutralScenes','T', ['PotentScenes','NeutralScenes'],[1,-1]]
cont11 = ['Faces>Fixation', 'T', ['AngryFaces','NeutralFaces'], [.5,.5]]
cont12 = ['Places>Fixation', 'T', ['EmotionalScenes','NeutralScenes','PotentScenes'],[.33,.33,.33]]
cont13 = ['AngryFaces>Fix', 'T', ['AngryFaces'], [1]]
cont14 = ['NeutralFaces>Fix', 'T', ['NeutralFaces'], [1]]
cont15 = ['PotentScenes>Fix', 'T', ['PotentScenes'], [1]]
cont16 = ['EmotionalScenes>Fix', 'T', ['EmotionalScenes'], [1]]
cont17 = ['NeutralScenes>Fix', 'T', ['NeutralScenes'], [1]]

contrasts = [cont1,cont2,cont3,cont4,cont5,cont6,cont7,cont8,cont9,cont10,cont11,cont12,cont13,cont14,cont15,cont16,cont17]


"""
Set preprocessing parameters
----------------------------
"""

l1pipeline.inputs.preproc.fssource.subjects_dir = subjects_dir

volsmoothnode = l1pipeline.get_node('preproc.volsmooth')
volsmoothnode.iterables = ('fwhm',[0 6])

l1pipeline.inputs.preproc.surfsmooth.surface_fwhm = 4
l1pipeline.inputs.preproc.surfsmooth.vol_fwhm = 4

art = l1pipeline.inputs.preproc.art
art.use_differences      = [True,False]
#first is use scan to scan for motion, second is use overall mean for intensity
art.use_norm             = True
#composite measure of motion
art.norm_threshold       = 1
#in mm
art.zintensity_threshold = 3
#in standard dev
art.mask_type            = 'file'
art.parameter_source = 'SPM'

"""
Set up node specific inputs
---------------------------

We replicate the modelspec parameters separately for the surface- and
volume-based analysis.
"""

tr = 2.5
hpcutoff = 128
modelspecref = l1pipeline.inputs.volanalysis.modelspec
modelspecref.input_units             = 'secs'
modelspecref.output_units            = 'secs'
modelspecref.time_repetition         = tr
modelspecref.high_pass_filter_cutoff = hpcutoff

modelspecref = l1pipeline.inputs.surfanalysis.modelspec
modelspecref.input_units             = 'secs'
modelspecref.output_units            = 'secs'
modelspecref.time_repetition         = tr
modelspecref.high_pass_filter_cutoff = hpcutoff

l1designref = l1pipeline.inputs.volanalysis.level1design
l1designref.timing_units       = modelspecref.output_units
l1designref.interscan_interval = modelspecref.time_repetition

l1designref = l1pipeline.inputs.surfanalysis.level1design
l1designref.timing_units       = modelspecref.output_units
l1designref.interscan_interval = modelspecref.time_repetition

l1pipeline.inputs.inputnode.contrasts = contrasts


"""
   e. Use :class:`nipype.algorithms.rapidart` to determine if stimuli are correlated with motion or intensity parameters (STIMULUS CORRELATED MOTION).
"""
stimcorr = pe.Node(interface=ra.StimulusCorrelation(),name='stimcorr')
stimcorr.inputs.concatenated_design             = True

"""
f.  Smooth contrasts images to be used for second level smoothing.
"""
consmooth = pe.Node(interface=spm.Smooth(), name = 'consmooth')
consmooth.inputs.fwhm = [6,6,6]

#################################################################################
# Setup pipeline
#################################################################################

"""
   The nodes setup above do not describe the flow of data. They merely
   describe the parameters used for each function. In this section we
   setup the connections between the nodes such that appropriate
   outputs from nodes are piped into appropriate inputs of other
   nodes.  

   a. Use :class:`nipype.pipeline.engine.Pipeline` to create a
   graph-based execution pipeline for first level analysis. The config
   options tells the pipeline engine to use `workdir` as the disk
   location to use when running the processes and keeping their
   outputs. The `use_parameterized_dirs` tells the engine to create
   sub-directories under `workdir` corresponding to the iterables in
   the pipeline. Thus for this pipeline there will be subject specific
   sub-directories. 

   The ``nipype.pipeline.engine.Pipeline.connect`` function creates the
   links between the processes, i.e., how data should flow in and out
   of the processing nodes. 
"""
"""
Setup the pipeline
------------------

The nodes created above do not describe the flow of data. They merely
describe the parameters used for each function. In this section we
setup the connections between the nodes such that appropriate outputs
from nodes are piped into appropriate inputs of other nodes.

Use the :class:`nipype.pipeline.engine.Workfow` to create a
graph-based execution pipeline for first level analysis. 
"""

level1 = pe.Workflow(name="level1")
level1.base_dir = '/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/workingdir/social'

level1.connect([(infosource, datasource, [('subject_id', 'subject_id')]),
                (datasource,l1pipeline,[('func','inputnode.func')]),
                (infosource,l1pipeline,[('subject_id','inputnode.subject_id'),
                                       (('subject_id', subjectinfo),
                                        'inputnode.session_info')]),
		(l1pipeline,stimcorr,[('preproc.realign.realignment_parameters','realignment_parameters'),
		                      ('preproc.art.intensity_files','intensity_values'),
		                      ('volanalysis.level1design.spm_mat_file','spm_mat_file')]),
		(l1pipeline,consmooth,[('volnormconimages.norm2mni.normalized_files','in_files')]),
                ])

"""
Setup the datasink
"""
datasink = pe.Node(interface=nio.DataSink(container='social'), name="datasink")
datasink.inputs.base_directory = '/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/l1output'

# store relevant outputs from various stages of the 1st level analysis
level1.connect([(l1pipeline, datasink,[('volanalysis.contrastestimate.spmT_images','contrasts.@T'),
                                       ('volanalysis.contrastestimate.con_images','contrasts.@con'),
				       ('volanalysis.level1estimate.beta_images','model.@betas'),
				       ('volanalysis.level1estimate.spm_mat_file','model.@spm'),
				       ('volanalysis.level1estimate.mask_image','model.@mask'),
				       ('volanalysis.level1estimate.residual_image','model.@res'),
			  	       ('volanalysis.level1estimate.RPVimage','model.@rpv'),
				       ('preproc.realign.realignment_parameters','realign.@param'),
				       ('preproc.realign.mean_image','realign.@mean'),
				       ('preproc.art.outlier_files','art.@outliers'),
				       ('preproc.art.statistic_files','art.@stats'),
				       ('preproc.volsmooth.smoothed_files','smooth.@file'),
				       ('preproc.surfregister.out_reg_file','bbreg_file.@reg'),
				       ('preproc.surfregister.min_cost_file','mincost.@file'),
				       ('volnormconimages.norm2mni.normalized_files','norm_contrasts.@T'), 
				       ])	
	 	])

level1.connect([(consmooth, datasink,[('smoothed_files','smoothed_contrasts.@T'),
				       ])	
	 	])

level1.connect([(stimcorr, datasink,[('stimcorr_files','stimcorr.@stats'),
				       ])	
	 	])

##########################################################################
# Execute the pipeline
##########################################################################

"""
   The code discussed above sets up all the necessary data structures
   with appropriate parameters and the connectivity between the
   processes, but does not generate any output. To actually run the
   analysis on the data the ``nipype.pipeline.engine.Pipeline.Run``
   function needs to be called. 
"""
if __name__ == '__main__':
    level1.run()
    level1.write_graph(graph2use='flat')
