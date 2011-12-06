# <nbformat>2</nbformat>

# <codecell>

%config InlineBackend.figure_format = 'svg'

# <codecell>

import os
import shutil
import time

# prevent lengthy SPM output
from nipype.utils.logger import logging, logger, fmlogger, iflogger
#logger.setLevel(logging.getLevelName('CRITICAL'))
#fmlogger.setLevel(logging.getLevelName('CRITICAL'))
#iflogger.setLevel(logging.getLevelName('CRITICAL'))

import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
from scipy.stats import wilcoxon
import sklearn as sk
from sklearn.linear_model.base import BaseEstimator, RegressorMixin
import sklearn.metrics as skm
import sklearn.cross_validation as cv
import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt


#from nipype.utils.config import config
#config.enable_debug_mode()

import nipype.pipeline.engine as pe

from spm_2lvl import do_spm         #spm workflow --> give directory + confiles
from feature_selection import determine_model_all
from cluster_tools import get_clustermeans
from cfutils import get_subjects, get_subject_data

# <codecell>

X = get_subjects()
_, pdata = get_subject_data(X)
X = pdata.subject
y = pdata.lsas_pre - pdata.lsas_post
dcsidx = np.nonzero(pdata.classtype==2)[0]
pcbidx = np.nonzero(pdata.classtype==3)[0]

# <codecell>

#wf = do_spm(X, y, analname='all_subjects', run_workflow=False)
#wf.base_dir = os.path.realpath('..')
#wf.run()

# <markdowncell>

# ###get cluster coordinates

# <codecell>

def get_coords(img, affine):
    coords = []
    labels = np.setdiff1d(np.unique(img.ravel()), [0])
    cs = []
    for label in labels:
        cs.append(np.sum(img==label))
    for label in labels[argsort(cs)[::-1]]:
        coords.append(np.dot(affine, 
                             np.hstack((np.mean(np.asarray(np.nonzero(img==label)), 
                                                axis = 1),
                                        1)))[:3].tolist())
    return coords

# <codecell>

from nipy.labs import viz
from nibabel import load
def show_slices(img, coords=None, threshold=0.1, cmap=None, prefix=None,
                show_colorbar=None, formatter='%.2f'):
    if cmap is None:
        cmap = pylab.cm.hot
    data, aff = img.get_data(), img.get_affine()
    anatimg = load('/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz')
    anatdata, anataff = anatimg.get_data(), anatimg.get_affine()
    anatdata = anatdata.astype(np.float)
    anatdata[anatdata<10.] = np.nan
    outfile = 'cluster.svg'
    if prefix:
        outfile = '_'.join((prefix, outfile))
    outfile = os.path.join('figures', outfile)
    if coords is None:
        osl = viz.plot_map(np.asarray(data), aff, threshold=threshold, 
                           cmap=cmap, black_bg=False)
        osl.frame_axes.figure.savefig(outfile, transparent=True)
    else:
        for idx,coord in enumerate(coords):
            outfile = 'cluster%02d' % idx
            if prefix:
                outfile = '_'.join((prefix, outfile))
            outfile = os.path.join('figures', outfile)
            osl = viz.plot_map(np.asarray(data), aff, anat=anatdata, anat_affine=anataff,
                               threshold=threshold, cmap=cmap,
                               black_bg=False, cut_coords=coord)
            if show_colorbar:
                cb = colorbar(gca().get_images()[1], cax=axes([0.4, 0.075, 0.2, 0.025]), 
                         orientation='horizontal', format=formatter)
                cb.set_ticks([cb._values.min(), cb._values.max()])
                show()
            osl.frame_axes.figure.savefig(outfile+'.svg', bbox_inches='tight', transparent=True)
            osl.frame_axes.figure.savefig(outfile+'.png', dpi=600, bbox_inches='tight', transparent=True)

# <codecell>

def plot_regression_line(x,y, xlim, color='r'):
    model=sk.linear_model.LinearRegression().fit(x[:,None],y)
    xplot = np.arange(xlim[0], xlim[1])[:,None]
    plot(xplot, model.predict(xplot), color=color)

# <codecell>

import os
from scipy.ndimage import label
import scipy.stats as ss
def get_labels(data, min_extent=5):
    labels, nlabels = label(data)
    for idx in range(1, nlabels+1):
        if sum(labels==idx)<min_extent:
            labels[labels==idx] = 0
    return labels, nlabels

# <codecell>

base_dir = '/mindhive/gablab/satra/sad/'
filename = os.path.join(base_dir, 'scripts', 'clustermean.nii.gz')
img=load(filename)
labels, nlabels = label(abs(img.get_data())>0)
coords = get_coords(labels, img.get_affine())
show_slices(img, coords, cmap=pylab.cm.jet, prefix='overlap', show_colorbar=True,
            formatter='%d')

# <codecell>

base_dir = '/mindhive/gablab/satra/sad/'
filename = os.path.join(base_dir, 'all_subjects', 'conest', 'spmT_0001.img')
img=load(filename)
labels, nlabels = get_labels(img.get_data()>ss.t.ppf(1-0.001,33), 20)
data = img.get_data()
data[labels==0] = 0
#cmeans = get_clustermeans(X, labels, nlabels)
coords = get_coords(labels, img.get_affine())
show_slices(img, coords, threshold=0.5, prefix='uncorrected', show_colorbar=True)

# <codecell>

import os
from scipy.ndimage import label
base_dir = '/mindhive/gablab/satra/sad/'
filename = os.path.join(base_dir, 'all_subjects', 'thresh', 'spmT_0001_thr.img')
img=load(filename)
labels, nlabels = label(abs(img.get_data())>0)
cmeans = get_clustermeans(X, labels, nlabels)
coords = get_coords(labels, img.get_affine())
show_slices(img, coords, prefix='topocorrect', show_colorbar=True)

# <codecell>

close('all')
axes([0.1,0.1,0.7,0.8])
plot(y, cmeans[:,0], 'o', color=[0.2,0.2,0.2])
plot(y, cmeans[:,1], 'o', color=[0.6,0.6,0.6])
xlim([-5, 84])
xlabel('LSAS Delta')
ylabel('contrast activation')
legend(('Cluster 1', 'Cluster 2'), 'best', numpoints=1)
plot_regression_line(y, cmeans[:,0], [-4,85], color=[0.2,0.2,0.2])
plot_regression_line(y, cmeans[:,1], [-4,85], color=[0.6,0.6,0.6])
grid()
axes([0.8,0.1,0.15,0.8])
boxplot([cmeans[dcsidx,0], cmeans[pcbidx,0], cmeans[dcsidx,1], cmeans[pcbidx,1]])
yticks([])
xticks([1,2,3,4], ['D1', 'P1', 'D2', 'P2'])
savefig('figures/scatter_means_all.svg')
savefig('figures/scatter_means_all.png', dpi=600)
print 'r: C1', pearsonr(cmeans[:,0], y)
print 'r: C2', pearsonr(cmeans[:,1], y)

# <codecell>

close('all')
axes([0.1,0.1,0.7,0.8])
plot(pdata.lsas_pre, cmeans[:,0], 'o', color=[0.2,0.2,0.2])
plot(pdata.lsas_pre, cmeans[:,1], 'o', color=[0.6,0.6,0.6])
xlim([58, 124])
xlabel('LSAS Pre')
ylabel('contrast activation')
legend(('Cluster 1', 'Cluster 2'), 'best', numpoints=1)
plot_regression_line(pdata.lsas_pre, cmeans[:,0], [57, 125], color=[0.2,0.2,0.2])
plot_regression_line(pdata.lsas_pre, cmeans[:,1], [57, 125], color=[0.6,0.6,0.6])
grid()
axes([0.8,0.1,0.15,0.8])
boxplot([cmeans[dcsidx,0], cmeans[pcbidx,0], cmeans[dcsidx,1], cmeans[pcbidx,1]])
yticks([])
xticks([1,2,3,4], ['D1', 'P1', 'D2', 'P2'])
savefig('figures/scatter_means_all_lsaspre.svg')
savefig('figures/scatter_means_all_lsaspre.png', dpi=600)
print 'r: C1', pearsonr(cmeans[:,0], pdata.lsas_pre)
print 'r: C2', pearsonr(cmeans[:,1], pdata.lsas_pre)
print 'r: C1D', pearsonr(cmeans[dcsidx,0], pdata.lsas_pre[dcsidx])
print 'r: C2D', pearsonr(cmeans[dcsidx,1], pdata.lsas_pre[dcsidx])
print 'r: C1P', pearsonr(cmeans[pcbidx,0], pdata.lsas_pre[pcbidx])
print 'r: C2P', pearsonr(cmeans[pcbidx,1], pdata.lsas_pre[pcbidx])

# <codecell>

def Rmodel(y_true, y_pred):
    robjects.globalenv['y_true'] = robjects.FloatVector(y_true)
    robjects.globalenv['y_pred'] = robjects.FloatVector(y_pred)
    robjects.r("model = lm('y_true~y_pred')")
    print robjects.r("summary(model)")

# <codecell>

close('all')
a1 = axes([0.05, 0.2, 0.15, 0.75])
boxplot([y[dcsidx], y[pcbidx]])
ylim([-5, 82])
ylabel('LSAS Delta')
xticks([1,2],('D','P'))
a2 = axes([0.2, 0.05, 0.75, 0.15])
boxplot([pdata.lsas_pre[dcsidx], pdata.lsas_pre[pcbidx]],
        vert=False)
xlim([58, 124])
xlabel('LSAS Pre')
yticks([1,2],('D','P'))
a3 = axes([0.2, 0.2, 0.75, 0.75]) #, sharex=a2, sharey=a1)
plot(pdata.lsas_pre[dcsidx], y[dcsidx], 'o', color=(0.2, 0.2, 0.2))
plot(pdata.lsas_pre[pcbidx], y[pcbidx], 'o', color=(0.6, 0.6, 0.6))
plot_regression_line(pdata.lsas_pre, y, [57, 125])
a3.set_xticks([])
a3.set_yticks([])
ylim([-5, 82])
xlim([58, 124])
grid()
legend(('DCS', 'Placebo'), 'lower right', numpoints=1)
title('r=%.2f, p=%.2f' % pearsonr(y, pdata.lsas_pre))
savefig('figures/corr_pre_delta.svg')
savefig('figures/corr_pre_delta.png', dpi=600)
print 'D', mean(pdata.lsas_pre[dcsidx]), '+-', std(pdata.lsas_pre[dcsidx])
print 'P', mean(pdata.lsas_pre[pcbidx]), '+-', std(pdata.lsas_pre[pcbidx])

# <codecell>

from rpy2 import robjects
from rpy2.robjects.packages import importr
stats = importr('stats')
base = importr('base')

# <codecell>

c1 = robjects.FloatVector(cmeans[:,0])
c2 = robjects.FloatVector(cmeans[:,1])
lsasd = robjects.FloatVector(y)
robjects.globalenv['c1'] = c1
robjects.globalenv['c2'] = c2
robjects.globalenv['lsaspre'] = robjects.FloatVector(pdata.lsas_pre)
robjects.globalenv['group'] = robjects.IntVector(pdata.classtype-2)
robjects.globalenv['lsasd'] = lsasd
m1 = robjects.r("model1 = lm('lsasd~c1 + c2 + lsaspre + lsaspre:group +c1:group + c2:group')")
m2 = robjects.r("model2 = lm('lsasd~lsaspre + lsaspre:group')")
m3 = robjects.r("model3 = lm('lsasd~lsaspre')")

# <codecell>

print robjects.r("summary(model1)")
print robjects.r("summary(model2)")
print robjects.r("summary(model3)")
print robjects.r("anova(model3, model2)")

# <codecell>

from sklearn.linear_model import LinearRegression
import sklearn.cross_validation as cv
result = []
Xnew = np.vstack((pdata.lsas_pre, pdata.lsas_pre*(pdata.classtype-2))).T
for train, test in cv.StratifiedKFold(pdata.classtype, 18):
    model = LinearRegression()
    model.fit(Xnew[train], y[train])
    result.append([y[test], model.predict(Xnew[test])])
result_lsas = result
y_true = []; y_pred = []
for a,b in result:
    y_true.extend(a.tolist())
    y_pred.extend(b.tolist())
result = np.array(np.vstack((y_true, y_pred))).T

# <codecell>

value, distribution, pvalue = cv.permutation_test_score(LinearRegression(), Xnew, y,
                                                        score_func=skm.mean_square_error,
                                                        cv=cv.StratifiedKFold(pdata.classtype, 18),
                                                        n_permutations=2000,
                                                        )

# <codecell>

hist(distribution, 32, alpha=0.5, color='gray')
plot([value, value], [0,200], 'r')
title('p=%.2f' % (1-pvalue))
xlabel('Mean square error')

# <codecell>

print np.corrcoef(result.T)
Rmodel(result.T[0], result.T[1])

# <codecell>

plot(result[:,0], result[:,1], 'o', color=[0.6,0.6,0.6])
minv = np.min(result)-5
maxv = np.max(result)+5
plot_regression_line(result[:,0], result[:,1], [minv-1, maxv+1], color='r')
xlabel('LSAS Delta actual')
ylabel('LSAS Delta predicted')
axis('scaled')
ylim([minv, maxv])
xlim([minv, maxv])
grid()
title('r=%.2f' % np.corrcoef(result.T)[0,1])
savefig('figures/loo_lsaspre.svg')
savefig('figures/loo_lsaspre.png', dpi=600)

# <codecell>

result = []
Xnew = np.hstack((np.vstack((pdata.lsas_pre, pdata.lsas_pre*(pdata.classtype-2))).T, 
                  cmeans))
for train, test in cv.StratifiedKFold(pdata.classtype, 18):
    model = LinearRegression()
    model.fit(Xnew[train], y[train])
    result.append([y[test], model.predict(Xnew[test])])
y_true = []; y_pred = []
for a,b in result:
    y_true.extend(a.tolist())
    y_pred.extend(b.tolist())
result = np.array(np.vstack((y_true, y_pred))).T

# <codecell>

np.corrcoef(result.T)

# <codecell>

value, distribution, pvalue = cv.permutation_test_score(LinearRegression(), Xnew, y,
                                                        score_func=skm.mean_square_error,
                                                        cv=cv.StratifiedKFold(pdata.classtype, 18),
                                                        n_permutations=2000,
                                                        )

# <codecell>

pvalue = min(pvalue, 1-1./2000)
hist(distribution, 32, alpha=0.5, color='gray')
plot([value, value], [0,200], 'r')
title('p=%.4f' % (1-pvalue))
xlabel('Mean square error')

# <codecell>

plot(result[:,0], result[:,1], 'o', color=[0.6,0.6,0.6])
xlabel('LSAS Delta actual')
ylabel('LSAS Delta predicted')
minv = np.min(result)-5
maxv = np.max(result)+5
plot_regression_line(result[:,0], result[:,1], [minv-1, maxv+1], color='r')
axis('scaled')
ylim([minv, maxv])
xlim([minv, maxv])
grid()
title('r=%.2f' % np.corrcoef(result.T)[0,1])
savefig('figures/loo_group_cluster.svg')
savefig('figures/loo_group_cluster.png', dpi=600)

# <codecell>

cvres = np.load('result_cv.npz')
minv = np.min(cvres['aout'])-5
maxv = np.max(cvres['aout'])+5
plot(cvres['aout'][:,0], cvres['aout'][:,1], 'o', color=[0.6,0.6,0.6])
plot_regression_line(cvres['aout'][:,0], cvres['aout'][:,1], [minv-1, maxv+1], color='r')
xlabel('LSAS Delta actual')
ylabel('LSAS Delta predicted')
axis('scaled')
ylim([minv, maxv])
xlim([minv, maxv])
grid()
title('r=%.2f' % np.corrcoef(cvres['aout'].T)[0,1])
savefig('figures/fullcv_results.svg')
savefig('figures/fullcv_results.png', dpi=600)

# <codecell>

skm.explained_variance_score(cvres['aout'][:,0], cvres['aout'][:,1])
Rmodel(cvres['aout'][:,0], cvres['aout'][:,1])

# <codecell>

permdata = np.load('100iter.npz')
hist(permdata['distribution'], 64, color=[0.6,0.6,0.6])
plot([permdata['value'], permdata['value']], [0, 12], color='r', linewidth=2)
title('p = %.3f' % max(1./100, (1-permdata['pvalue'])))
xlim([390, 1100])
xlabel('Mean square error (lower=better)')
savefig("figures/permtest_hist.svg")
savefig("figures/permtest_hist.png", dpi=600)

# <codecell>

msedata = []
for idx, res in enumerate(result_lsas):
    msedata.append((skm.mean_square_error(res[0], res[1]), 
                    skm.mean_square_error(cvres['result'][idx][0],
                                          cvres['result'][idx][1])))

# <codecell>

print wilcoxon(np.diff(msedata, axis=1).ravel())
boxplot(np.diff(msedata, axis=1))

# <markdowncell>

# ##Amygdala responses

# <codecell>

amygdata = recfromcsv('AmygdalaResponses.csv', names=True)
amygX = amygdata.view(np.float64).reshape(39,8)
names = []
for name in amygdata.dtype.names:
    if '_1' in name:
        names.append(name.replace('_1','_R'))
    else:
        names.append(name+'_L') 

# <codecell>

bp = boxplot(amygX)
xticks(arange(1,9), names, rotation=75)
ylabel("Contrast value (% signal change)")
grid()
title('Amygdala response')
savefig('figures/amygdala_response.svg', bbox_inches='tight')
savefig('figures/amygdala_response.png', dpi=600, bbox_inches='tight')

# <codecell>

robjects.globalenv['y_true'] = robjects.FloatVector(y)
robjects.globalenv['lsaspre'] = robjects.FloatVector(pdata.lsas_pre)
robjects.globalenv['group'] = robjects.IntVector(pdata.classtype-2)
for i,name in enumerate(names):
    robjects.globalenv[name] = robjects.FloatVector(amygX[:,i])
m1str = 'y_true~lsaspre + lsaspre:group + %s + %s' % ('+'.join(names), ':group +'.join(names))
m1 = robjects.r("m1 = lm(%s)" % m1str)
print robjects.r("summary(m1)")
m2 = robjects.r("m2 = lm('y_true~lsaspre + lsaspre:group + angry_faces_R + angry_faces_R:group + neutral_faces_R + neutral_faces_R:group')")
print robjects.r("summary(m2)")
m3 = robjects.r("m3 = lm('y_true~lsaspre + lsaspre:group')")
print robjects.r("anova(m3,m1)")

# <codecell>

imshow(corrcoef(amygX.T), interpolation='nearest')

# <codecell>

result = []
Xnew = np.hstack((np.vstack((pdata.lsas_pre, pdata.lsas_pre*(pdata.classtype-2))).T, 
                  amygX))
for train, test in cv.StratifiedKFold(pdata.classtype, 18):
    model = LinearRegression()
    model.fit(Xnew[train], y[train])
    result.append([y[test], model.predict(Xnew[test])])
y_true = []; y_pred = []
for a,b in result:
    y_true.extend(a.tolist())
    y_pred.extend(b.tolist())
result = np.array(np.vstack((y_true, y_pred))).T

# <codecell>

value, distribution, pvalue = cv.permutation_test_score(LinearRegression(), Xnew, y,
                                                        score_func=skm.mean_square_error,
                                                        cv=cv.StratifiedKFold(pdata.classtype, 18),
                                                        n_permutations=2000,
                                                        )

# <codecell>

hist(distribution, 32, alpha=0.5, color='gray')
plot([value, value], [0,200], 'r')
title('p=%.2f' % (1-pvalue))
xlabel('Mean square error')

# <codecell>

plot(result[:,0], result[:,1], 'o', color=[0.6, 0.6, 0.6])
xlabel('LSAS Delta actual')
ylabel('LSAS Delta predicted')
minv = np.min(result)-5
maxv = np.max(result)+5
plot_regression_line(result[:,0], result[:,1], [minv-1, maxv+1], color='r')
axis('scaled')
ylim([minv, maxv])
xlim([minv, maxv])
grid()
title('r=%.2f' % np.corrcoef(result.T)[0,1])
savefig('figures/loo_amygdala.svg')
savefig('figures/loo_amygdala.png', dpi=600)

# <codecell>

print pearsonr(pdata.lsas_pre,pdata.lsas_post)
print pearsonr(pdata.lsas_pre,pdata.lsas_pre-pdata.lsas_post)
print 'DP:', pearsonr(pdata.lsas_pre[dcsidx],pdata.lsas_post[dcsidx])
print 'PP:', pearsonr(pdata.lsas_pre[pcbidx],pdata.lsas_post[pcbidx])
print 'DD:',pearsonr(pdata.lsas_pre[dcsidx],pdata.lsas_pre[dcsidx]-pdata.lsas_post[dcsidx])
print 'PD:',pearsonr(pdata.lsas_pre[pcbidx],pdata.lsas_pre[pcbidx]-pdata.lsas_post[pcbidx])
print spearmanr(pdata.lsas_pre,pdata.lsas_post)
print spearmanr(pdata.lsas_pre,pdata.lsas_pre-pdata.lsas_post)
plot(pdata.lsas_pre[dcsidx], pdata.lsas_post[dcsidx], 'o', color=(0.2,0.2,0.2))
plot(pdata.lsas_pre[pcbidx], pdata.lsas_post[pcbidx], 'o', color=(0.6,0.6,0.6))
legend(['DCS', 'PCB'], 'best', numpoints=1)
plot_regression_line(pdata.lsas_pre[dcsidx], pdata.lsas_post[dcsidx], [55, 125], 
                     color=[0.2,0.2,0.2])
plot_regression_line(pdata.lsas_pre[pcbidx], pdata.lsas_post[pcbidx], [55, 125], 
                     color=[0.6,0.6,0.6])
xlabel('LSAS Pre')
ylabel('LSAS Post')
xlim([58,124])
ylim([8,85])
grid()

# <markdowncell>

# ## LSAS delta

# <codecell>

filename = os.path.join(base_dir, 'lsasdelta_all', 'conest', 'spmT_0001.img')
img=load(filename)
print img.get_header()['descrip']
labels, nlabels = get_labels(abs(img.get_data())>ss.t.ppf(1-0.001,35), 20)
data = img.get_data()
data[labels==0] = 0
#cmeans = get_clustermeans(X, labels, nlabels)
coords = get_coords(labels, img.get_affine())
show_slices(img, coords, threshold=0.5, prefix='uncorrected_lsasdelta', show_colorbar=True,
            cmap=cm.jet)

# <markdowncell>

# ## LSAS Post

# <codecell>

filename = os.path.join(base_dir, 'lsaspost_all', 'conest', 'spmT_0001.img')
img=load(filename)
print img.get_header()['descrip']
labels, nlabels = get_labels(abs(img.get_data())>ss.t.ppf(1-0.001,35), 20)
data = img.get_data()
data[labels==0] = 0
#cmeans = get_clustermeans(X, labels, nlabels)
coords = get_coords(labels, img.get_affine())
show_slices(img, coords, threshold=0.5, prefix='uncorrected_lsaspost', show_colorbar=True,
            cmap=cm.jet)

# <markdowncell>

# ## LSAS pre

# <codecell>

filename = os.path.join(base_dir, 'lsaspre_all', 'conest', 'spmT_0001.img')
img=load(filename)
print img.get_header()['descrip']
labels, nlabels = get_labels(abs(img.get_data())>ss.t.ppf(1-0.001,35), 20)
data = img.get_data()
data[labels==0] = 0
#cmeans = get_clustermeans(X, labels, nlabels)
coords = get_coords(labels, img.get_affine())
show_slices(img, coords, threshold=0.5, prefix='uncorrected_lsaspre', show_colorbar=True,
            cmap=cm.jet)

# <codecell>


