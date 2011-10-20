from tempfile import mkdtemp
import sys

import nipype.pipeline.engine as pe         # the workflow and node wrappers
import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as mlab
mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
mlab.MatlabCommand.set_default_paths('/software/spm8_4290')


from cfutils import get_subject_data

def do_spm(subjects, y, analdir=None, analname=None, run_workflow=True):
    """
    workflow function that does all spm analysis (2 sample t-test, estimate model, estimate contrasts, threshold image)
    
    Parameters
    ----------

    subjects : list of subjects on whom to do regression
    y : dependent variable corresponding to the list of subjects
    
    """

    confiles, pdata = get_subject_data(subjects)

    # MAKE COVARIATES
    # containes lsas_delta and lsas_pre for group2/3
    covariates_group2 = [[],[]]
    covariates_group3 = [[],[]]
    # initialize the classtype lists (contains subjects nii files) 
    classtype2 = []
    classtype3 = []
    # list of subject ids belonging to respective group
    group2 = []
    group3 = []
    for sidx, con_file in enumerate(confiles):
        # add subject files to lists
        if pdata.classtype[sidx] == 2:
            # add .nii file
            classtype2.append(con_file)
            # add subject id
            group2.append(sidx)
            # add lsas_delta score for the subject
            covariates_group2[0].append(y[sidx])
            # and at the same time fill up the matrix of the other group with zeros from the beginning
            covariates_group3[0].insert(0,0)
            # do the same for lsas_pre scores
            covariates_group2[1].append(pdata.lsas_pre[sidx])
            covariates_group3[1].insert(0,0)

        elif pdata.classtype[sidx] == 3:
            # add .nii file
            classtype3.append(con_file)
            # add subject id
            group3.append(sidx)
            # add lsas_delta and lsas_pre to group3's covariates
            covariates_group3[0].append(y[sidx])
            covariates_group3[1].append(pdata.lsas_pre[sidx])
            
    # fill up the rest of group2's covariates with zeros 
    for n in range(len(group3)):
        covariates_group2[0].append(0)
        covariates_group2[1].append(0)

    # AT THIS POINT: covariates_group2 has 2 columns (lsas_delta and lsas_pre) 
    # with values in the first half of the vector and zeros in the later half
    # and covariats_group3 looks the same only that the first half are zeros and the 2nd half contain values

    # TWO SAMPLE T-TEST

    ttester = pe.Node(interface = spm.TwoSampleTTestDesign(), name = 'ttestdes')

    # group1: classtype 2, group2: classtype 3
    ttester.inputs.group1_files = classtype2
    ttester.inputs.group2_files = classtype3
    ttester.inputs.covariates = [dict(vector=covariates_group2[0], name='Group2LSAS_delta'),
                                 dict(vector=covariates_group3[0], name='Group3LSAS_delta'), 
                                 dict(vector=covariates_group2[1], name='Group2LSAS_pre'),
                                 dict(vector=covariates_group3[1], name='Group3LSAS_pre')]

    # ESTIMATE MODEL

    estimator = pe.Node(interface = spm.EstimateModel(), name = 'ttestest')
    estimator.inputs.estimation_method = {'Classical':1}

    # ESTIMATE CONTRAST

    conest = pe.Node(interface = spm.EstimateContrast(), name = 'conest')
    conest.inputs.group_contrast = True
    conest.group_contrast = True
    con1 = ('Group2>Group3','T',['Group_{1}','Group_{2}'],[1,-1])
    con2 = ('Group3>Group2','T',['Group_{1}','Group_{2}'],[-1,1])
    con3 = ('Group2LSAS_delta>Group3LSAS_delta','T',['Group2LSAS_delta','Group3LSAS_delta'],[1,-1])
    con4 = ('Group2LSAS_delta<Group3LSAS_delta','T',['Group2LSAS_delta','Group3LSAS_delta'],[-1,1])
    con5 = ('LSAS Delta Response','T',['Group2LSAS_delta','Group3LSAS_delta'],[.5,.5])
    con5 = ('LSAS Delta Response','T',['Group2LSAS_delta','Group3LSAS_delta'],[.5,.5])
    conest.inputs.contrasts = [con5]


    # THRESHOLD
    thresh = pe.Node(interface = spm.Threshold(), name = 'thresh')
    thresh.inputs.contrast_index = 1
    thresh.inputs.height_threshold = 0.001
    thresh.inputs.use_fwe_correction = False
    thresh.inputs.use_topo_fdr = True
    thresh.inputs.extent_fdr_p_threshold = 0.05

    # WORKFLOW
    if analname is None:
        analname = 'some_name'

    workflow = pe.Workflow(name = analname)
    workflow.connect(ttester,'spm_mat_file',estimator,'spm_mat_file')
    workflow.connect(estimator,'spm_mat_file',conest,'spm_mat_file')
    workflow.connect(estimator,'residual_image',conest,'residual_image')
    workflow.connect(estimator,'beta_images',conest,'beta_images')
    workflow.connect(conest,'spm_mat_file',thresh,'spm_mat_file')
    workflow.connect(conest,'spmT_images',thresh,'stat_image')
    if run_workflow:
        if analdir is None:
            analdir = mkdtemp()
        workflow.base_dir = analdir
        workflow.write_graph()
        workflow.run()
    return workflow

    # DONE WITH SPM <3
    
if __name__=="__main__":
    """ this assumes that you give it as arguments first the directory where the stuff should be saved,
    then the list of confiles """
    if len(sys.argv) == 1:
        sys.exit("please call the script by giving it the analysisdir & confiles arguments")
    analdir = sys.argv[1]
    confiles = [str(conf)[:-1] for conf in sys.argv[3:-1]]
    do_spm(analdir, confiles)
