from glob import glob
import numpy as np

con_template = ('/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/l1output/'
                'social/norm_contrasts/_subject_id_%s/_fwhm_6/wcon_0001_out_warped.nii')

def get_subjects():
    """Returns names of all subjects
    """
    pdata = np.recfromcsv(('/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/'
                           'l2output/social/split_halves/regression/'
                           'lsasDELTA/6mm/allsubs.csv'), names=True)
    return pdata.subject.tolist()

def get_subject_data(subjects):
    """Returns contrast_files and behavioral data for a given list of subjects

    Parameters
    ----------

    subjects : list of strings (names of subjects)

    Returns
    -------

    confiles : list of strings (names of contrast files)
    pdata : recarray containing behavioral information for given subject order

    """
    # original input file with test scores etc for every subject
    pdata = np.recfromcsv(('/mindhive/gablab/sad/PY_STUDY_DIR/Block/volsurf/'
                           'l2output/social/split_halves/regression/'
                           'lsasDELTA/6mm/allsubs.csv'), names=True)

    sidx = []
    confiles = []
    for s in subjects:
        try:
            idx = np.nonzero(pdata.subject==s)[0][0]
        except IndexError, e:
            raise IndexError('subject %s not found' % s)
        sidx.append(idx)
        confile = glob(con_template % s)
        if not confile:
            raise ValueError('no confile found for subject %s' % s)
        confiles.extend(confile)
        
    # put here either lsas_delta or lsas_post
    #responsevar = pdata.lsas_pre - pdata.lsas_post
    #group = pdata.classtype - 2
    #lsas_pre = pdata.lsas_pre
    #subject_num = len(pdata.subject)
    return confiles, pdata[sidx]
