import os.path as op
from os.path import join as pjoin
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from base import gat_stats
from tqdm import tqdm
from scipy.stats import spearmanr
import mne
from datetime import datetime
import matplotlib.colors as colors
from joblib import Parallel, delayed
import argparse


'''
# How to use this file
1. set environment variables 
$DEMARCHI_DATA_PATH and $DEMARCHI_FIG_PATH

run file with arguments 
'''

# python plot_reproduce_and_reorder.py --force_recalc 0 --suffix _small_window --suffix_data _small_window --plot_kind rd_to_all --n_timebins 67 --tmin "-0.33" --tmax 0.33
# Example command line:
# python plot_reproduce_and_reorder.py --force_recalc 0 --suffix _small_window --suffix_data _small_window --plot_kind rd_to_all --n_timebins 67 --tmin "-0.33" --tmax 0.33

# VSCode launch.json "args" format:
# "args": [
#     "--force_recalc", "0",
#     "--suffix", "_small_window",
#     "--suffix_data", "_small_window",
#     "--plot_kind", "rd_to_all",
#     "--n_timebins", "67",
#     "--tmin", "-0.33",
#     "--tmax", "0.33"
# ]

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument("--plot_kind", type=str, default = 'rd_to_all', required=True)
parser.add_argument("--force_recalc", type=int, default = 0)
parser.add_argument("--suffix", type=str, default = '')
parser.add_argument("--suffix_data", type=str, default = '')
parser.add_argument("--data_subdir", type=str, default = 'results')
parser.add_argument("--results_folder", type=str, default = 'reorder_random')
parser.add_argument("--plots_to_make", type=str, default = 'orig,reord,diff')
parser.add_argument("--inc_neg_time", type=int, default = 1)
parser.add_argument("--tmin", type=float, default=-0.7)
parser.add_argument("--tmax", type=float, default=0.7)
parser.add_argument("--n_jobs", type=int, default=-1)
parser.add_argument("--n_timebins", type=int, default=141, help='number of time bins')
 

args = parser.parse_args()
results_folder = args.results_folder

plt.rcParams['axes.titlesize'] = 10

# suffix, sp determine parts of filenames to get data from and figure names to save plots to
sp = ''; suffix = args.suffix #suffix = '';
#path_results = '/home/demitau/data_Quentin/data_demarchi/results'
#path_results = '/p/project/icei-hbp-2022-0017/demarchi/data_demarchi/MEG_demarchi/results'; suffix = ''
#path_results = '/p/project/icei-hbp-2022-0017/demarchi/data_demarchi/MEG_demarchi/results0'; suffix = '_0'
#path_results = '/p/project/icei-hbp-2022-0017/demarchi/data_demarchi/MEG_demarchi/results_Romain'; suffix='_R'
#path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'; suffix = ''; sp = '__sp'  
#path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'; suffix = ''
path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/' + args.data_subdir; 

#path_fig = '/p/project/icei-hbp-2022-0017/demarchi/output_plots'
path_fig = os.path.expandvars('$DEMARCHI_FIG_PATH')

stimtype = results_folder[len('reorder_random'):]

suffix += sp

print('Results folder = ',path_results,  results_folder, 'stimtype = ',stimtype)

force_recalc = args.force_recalc
plot_kind    = args.plot_kind
print('force_recalc = {}, plot_kind ={} '.format(force_recalc, plot_kind) )

# list all participants
participants = [f for f in os.listdir(path_results) if f[:1] == '1']
#order = ['Random', 'Midminus', 'Midplus', 'Ordered']

if plot_kind == 'rd_to_all':
    # ans1 should be in in increasing order 
    suffix += ''        
    ans1 = ['rd_to_rd', 'rd_to_mm', 'rd_to_mp', 'rd_to_or']; 
    ans2 = [ an + '_reord' for an in ans1]
elif plot_kind == 'self':
    suffix += '_self' 
    ans1 = ['rd_to_rd', 'mm_to_mm', 'mp_to_mp', 'or_to_or']; 
    ans2 = ['rd_reord_to_rd_reord', 'mm_reord_to_mm_reord', 'mp_reord_to_mp_reord', 'or_reord_to_or_reord']; 
elif plot_kind == 'simple_pred':
    suffix += '_sp' 
    # train on Xrd1,yrd1, test on X to y_sp
    # Xrd1 = epochs_rd_init[orig_nums_reord][:minl]
    # X = ordered raw, y_sp = simple pred of ordered
    ans1 = ['rd_to_rd', 'rd_to_mm_sp', 'rd_to_mp_sp', 'rd_to_or_sp'];  
    #Xreord, yreord_sp = simple pred after reord
    ans2 = ['rd_to_rd_reord_sp', 'rd_to_mm_reord_sp', 'rd_to_mp_reord_sp', 'rd_to_or_reord_sp'];  

#if stimtype == 'omission':
#    ans2 = [ an.replace('_reord','rd') for an in ans2]

##suffix += '_selfsp' 
##ans1 = ['rd_to_rd', 'mm_to_mm', 'mp_to_mp', 'or_to_or']; 
##ans2 = ['rd_to_rd', 'mm_reord_to_reord_mm', 'mp_reord_to_reord_mp', 'or_reord_to_reord_or']; 

## train on Xrd1,yrd1, test on Xreord to yreord_sp
## Xreord = orderd epochs_reord, yreord_sp = take oredered events make 'simple pred' then reord
##ans2 = [ an + '_reord' for an in ans1]

ans = ans1 + ans2
an2scores = dict( zip(ans, [ 0 ] * len(ans) ))
for an in an2scores: # I need independent lists
    an2scores[an] = []

lbl2ans = dict( zip(['orig', 'reord'], [ans1,ans2] ) )

# Loop accross participants
print('Num particiapnts = ',len(participants) )
dts = []
#scs = []
for participant in participants:
    #print(participant)
    # Append scores (with cross validation)

    p = ['rd_to_rd', 'rd_to_or', 'rd_to_or_sp', 'rd_to_or_reord', 'rd_to_or_sp_reord', 
            'or_to_or', 'or_to_or_reord', 'or_to_or_sp_fromsp', 'or_to_or_reord_fromsp' ]
    p+= ['rd_to_mp', 'rd_to_mp_sp', 'rd_to_mp_reord', 'rd_to_mp_sp_reord', 
            'mp_to_mp', 'mp_to_mp_reord', 'mp_to_mp_sp_fromsp', 'mp_to_mp_reord_fromsp' ]
    p+= ['rd_to_mm', 'rd_to_mm_sp', 'rd_to_mm_reord', 'rd_to_mm_sp_reord', 
            'mm_to_mm', 'mm_to_mm_reord', 'mm_to_mm_sp_fromsp', 'mm_to_mm_reord_fromsp' ]
    p+= ['mm_reord_to_mm_reord', 'mp_reord_to_mp_reord', 'or_reord_to_or_reord' ]
    p+= [ 'rd_to_mm_reord_sp', 'rd_to_mp_reord_sp', 'rd_to_or_reord_sp']
    #p+= [ 'rd_to_mmrd_sp', 'rd_to_mprd_sp', 'rd_to_or_orrd_sp']
    fnfz = zip(p, [op.join(path_results, participant, results_folder, f'cv_{s}_scores{args.suffix_data}.npy') for s in p] )
    fnfd = dict(fnfz)

    
    for an in ans:
        #print(participant, an, len(an2scores[an] ), len(an2scores[ans[0]] ), len(an2scores[ans[1]] ) )
        if an in ['rd_to_rd_reord', 'rd_reord_to_rd_reord',
                'rd_to_rd_reord_sp'] :
            continue

        fnf = fnfd[an]

        #if stimtype == '_omission':
        #    fnf = fnf.replace('mm_reord','mmrd')
        #    fnf = fnf.replace('mp_reord','mprd')
        #    fnf = fnf.replace('or_reord','orrd')

        #print(fnf)
        if not os.path.exists(fnf):
            print('File does not exist',fnf)
            raise ValueError(participant)
        sc_ = np.load(fnf)
        #scs += [sc_]
        an2scores[an] += [sc_]
        if len(an2scores[an]) > 33:
            raise ValueError('aa')

        #del sc_

        dt = datetime.fromtimestamp(os.stat(fnf).st_mtime)
        dts += [dt]

# some special scores are just random to random
an2scores['rd_to_rd_reord']       = an2scores['rd_to_rd'].copy()
an2scores['rd_reord_to_rd_reord'] = an2scores['rd_to_rd'].copy()
an2scores['rd_to_rd_reord_sp']    = an2scores['rd_to_rd'].copy()

an2scores2 = {}
for an in ans:
    an2scores2[an] = np.array( an2scores[an] )
    print(an,'len = ',len(an2scores2[an]))

print(f'Earliest computed data = ',str(np.min(dts) ) )
print(f'Latest computed data = ',str(np.max(dts) ) )

# cut the figure to plot a smaller time window
# all times
tmin,tmax = args.tmin,args.tmax
times = np.linspace(tmin,tmax, sc_.shape[-1])
wh_x = np.where((times >= tmin) & (times <= tmax))[0] # 141
if args.inc_neg_time:
    wh_y = np.where((times >= -0.33) & (times <= 0.33))[0]  
else:
    wh_y = np.where((times >= 0) & (times <= 0.33))[0]  # 33
# this ix_ is needed only because numpy (some versions of) are picky about slicing in two dimensions simultaneosly
ixgrid = np.ix_(wh_y, wh_x)

ndims = [ an2scores2[an].ndim for an in ans ]
# make sure all arrays have consisten dimensions
assert len(set(ndims)) == 1
if stimtype == '':
    assert ndims[0] == 4
    an2scores3 = {}
    for an in ans:
        # shape of scores file is (nsubjects,nsamples,nestimators,nslices)
        #print(an2scores2[an].shape)
        an2scores3[an] = np.array( an2scores2[an][:,:,ixgrid[0], ixgrid[1]].mean(1)   )  # mean on the second dimension because we kept scores on each fold 
else:
    assert ndims[0] == 3
    an2scores3 = {}
    for an in ans:
        an2scores3[an] = np.array( an2scores2[an][:,ixgrid[0], ixgrid[1]]   )  # mean on the second dimension because we kept scores on each fold 


an2scores = an2scores3

# plotting parameters

if args.inc_neg_time:
    xlim0 = -0.33
else:
    xlim0 = 0.0
matshow_pars = dict(origin='lower', extent=[tmin, tmax, xlim0, 0.33],
                    cmap='inferno')
#color_cluster = 'k'
color_cluster = 'white' # color of cluster boundaries
xlines = [-0.33,0,0.33] # times [s] where to draw vertical lines
color_vline = 'grey' # vertical lines color
vmin_rhos, vmax_rhos = -0.6, 0.6  # plotting correlation colorbar limits
lims_diff = -0.02, 0.02             # plotting diff colorbar limits 
# plotting scores colorbar limits  
#lims_scores = 0.23, 0.27; s = ''
sh = 0.032
lims_scores = 0.25 - sh, 0.25 + sh; s = ''
#lims_scores = 0.23, 0.30; s = f'_max_{lims_scores[1]:.1f}'
suffix += s
fsz = (7,13)

# which sets of plots to generate (of 3 in total)
plots_to_make = args.plots_to_make.split(',') #['orig', 'reord', 'diff']
#plots_to_make = ['orig', 'reord'] 
#plots_to_make = ['reord']
cleanup = 0

n_jobs = args.n_jobs # n threads for computing correlations

#nsamples = 33
nsamples = len(wh_y)
nsubj = 33
n_timebins = args.n_timebins #141


# function to compute correlations
def ff(i, rand2rd, rand2mm, rand2mp, rand2or):
    # takes scores
    rhos_ = np.zeros([nsamples, n_timebins])
    for row in range(rhos_.shape[0]):
        for column in range(rhos_.shape[1]):
            corr_values = [rand2rd[row, column], 
                    rand2mm[row, column],
                    rand2mp[row, column], 
                    rand2or[row, column] ]
            r,p = spearmanr([0, 1, 2, 3], corr_values) 
            rhos_[row, column] = r
            #print(i,rhos_.shape)
    return (i,rhos_)

lbl2rhos = {}
lbl2allsig = {}
for lbl in plots_to_make:
    if lbl == 'diff':
        continue
    anscur = lbl2ans[lbl] # orig or reord

    # commute spearman correlation as in Demarchi et al. across entropies
    # Initialize spearman rho results
    fn = f'{lbl}_rhos{stimtype}{suffix}.npz'
    if not os.path.exists(fn) or force_recalc:
        print('Compute rhos')
        rhos = np.zeros([nsubj, nsamples, n_timebins])

        args_ = []
        for i in range(nsubj):
            arg = [i]
            for an in anscur:
                arg += [ an2scores[an][i] ]
            arg = tuple(arg)
            args_.append(arg)

        #args = [ (i, all_cv_rd_to_rd_scores[i], all_cv_rd_to_mm_scores[i], 
        #    all_cv_rd_to_mp_scores[i], all_cv_rd_to_or_scores[i] ) for i in range(nsubj) ]

        plr = Parallel(n_jobs=n_jobs, backend='multiprocessing',)(delayed(ff)(*arg) for arg in args_)
        for i,rhos_ in plr:
            rhos[ i, :,: ] = rhos_
        np.savez(fn, rhos)
    else:
        dtstr = datetime.fromtimestamp(os.stat(fn).st_mtime)
        print('Load rhos',fn, dtstr)
        f = np.load(fn, allow_pickle=True)
        rhos = f['arr_0'][()]
    lbl2rhos[lbl] = rhos

    fn = f'{lbl}_allsig{stimtype}{suffix}.npz'
    # Plot main figure of Demarchi et al. (Fig3a)
    vmin, vmax = lims_scores
    norm1 = colors.Normalize(vmin=vmin, vmax=vmax)
    chance = 0.25
    # get permutations clusters
    if not os.path.exists(fn) or force_recalc:
        print('Compute clustering')
        all_sig = list()
        for an in anscur[::-1]:
        #for scores in tqdm([all_cv_rd_to_or_scores, all_cv_rd_to_mp_scores, all_cv_rd_to_mm_scores, all_cv_rd_to_rd_scores, rhos]):
            scores = an2scores[an]
            print('iteration')
            gat_p_values = gat_stats(np.array(scores) - chance)
            sig = np.array(gat_p_values < 0.05)
            all_sig.append(sig)

        print('iteration')
        gat_p_values = gat_stats(rhos)
        sig = np.array(gat_p_values < 0.05)
        all_sig.append(sig)

        np.savez(fn, all_sig)
    else:
        dtstr = datetime.fromtimestamp(os.stat(fn).st_mtime)
        print(f'Load clustering {fn}', dtstr )
        f = np.load(fn, allow_pickle=True)
        all_sig = f['arr_0'][()]
    lbl2allsig[lbl] = all_sig


for lbl in plots_to_make:
    if lbl == 'diff':
        continue
    anscur = lbl2ans[lbl] # orig or reord
    
    all_sig  = lbl2allsig[lbl]  
    rhos = lbl2rhos[lbl] 

    # plot the 4 conditions
    fig, axs = plt.subplots(5, 1, figsize=fsz, constrained_layout=True)
    fig.set_layout_engine('tight')

    #for i in np.arange(0,4)[::-1]:
    for ani,an in enumerate(anscur):
        dat = an2scores[an]
        i = 3 - ani
        im = axs[i].matshow(dat.mean(0), **matshow_pars, norm=norm1)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[i].contour(xx, yy, all_sig[i], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[i].title.set_text(an )

    norm_unif = colors.Normalize(vmin=vmin_rhos, vmax=vmax_rhos)
    if sp == '':
        im2 = axs[4].matshow(rhos.mean(0), **matshow_pars, norm=norm_unif)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[4].contour(xx, yy, all_sig[4], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[4].title.set_text('Correlation across entropies')

    for x in xlines:
        for ax in axs:
            ax.axvline(x, c=color_vline, ls='-')
            ax.xaxis.set_ticks_position("bottom")

    #plt.tight_layout()
    for ax in axs[:4]:
        fig.colorbar(im, ax=ax,  norm=norm1, label='Score')
    fig.colorbar(im2, ax=axs[4], norm=norm_unif, label='Spearman rho')

    plt.savefig(path_fig + f'/{lbl}_{stimtype}{suffix}.png')
    plt.close()

######################################################################

# Plot main figure of Demarchi et al. (Fig3a) with differences between classic and reordered trials
###################################################################

if 'diff' in plots_to_make:

    an2scores_diff = {}
    for i in range(len(ans1)):
        an = ans1[i]
        anr = ans2[i]
        an2scores_diff[an] = an2scores[an] - an2scores[anr]

    vmin, vmax = lims_diff
    norm1_diff = colors.Normalize(vmin=vmin, vmax=vmax)
    chance = 0

    # compute spearman correlation as in Demarchi et al. across entropies
    # Initialize spearman rho results
    fn = f'diff_rhos{stimtype}{suffix}.npz'
    if not os.path.exists(fn) or force_recalc:
        print('Compute rhos diff')
        rhos_diff = np.zeros([nsubj, nsamples, n_timebins])

        args_ = []
        for i in range(nsubj):
            arg = [i]
            for an in ans1:
                arg += [ an2scores_diff[an][i] ]
            arg = tuple(arg)
            args_.append(arg)

        plr = Parallel(n_jobs=n_jobs, backend='multiprocessing',
                                                )(delayed(ff)(*arg) for arg in args_)
        for i,rhos_diff_ in plr:
            rhos_diff[ i, :,: ] = rhos_diff_
        np.savez(fn, rhos_diff)
    else:
        dtstr = datetime.fromtimestamp(os.stat(fn).st_mtime)
        print('Load rhos diff ', fn, dtstr)
        f = np.load(fn, allow_pickle=True)
        rhos_diff = f['arr_0'][()]


    # get permutations clusters
    diff_all_sig = list()
    print('Compute clustering diff')
    fn = f'diff_allsig{stimtype}{suffix}.npz'
    if not os.path.exists(fn) or force_recalc:
        for an in ans1[::-1]:
        #for scores in tqdm([diff_rd_to_orrd, diff_rd_to_mprd, diff_rd_to_mmrd, diff_rd_to_rd, rhos_diff]):
            scores = an2scores_diff[an]

            scores = an2scores[an]
            print('iteration')
            gat_p_values = gat_stats(np.array(scores) - chance)
            sig = np.array(gat_p_values < 0.05)
            diff_all_sig.append(sig)

        print('iteration')
        gat_p_values = gat_stats(rhos_diff)
        sig = np.array(gat_p_values < 0.05)
        diff_all_sig.append(sig)

        np.savez(fn, diff_all_sig)
    else:
        dtstr = datetime.fromtimestamp(os.stat(fn).st_mtime)
        print('Load all_sig diff ', fn, dtstr)
        f = np.load(fn, allow_pickle=True)
        diff_all_sig = f['arr_0'][()]

    # plot the 4 conditions
    fig, axs = plt.subplots(5, 1, figsize=fsz, constrained_layout=True)
    fig.set_layout_engine('tight')


    for ani,an in enumerate(ans1):
        dat = an2scores_diff[an]
        i = 3 - ani
        im = axs[i].matshow(dat.mean(0), **matshow_pars, norm=norm1_diff)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[i].contour(xx, yy, diff_all_sig[i], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[i].title.set_text(an )

    norm_unif = colors.Normalize(vmin=vmin_rhos, vmax=vmax_rhos)
    im2 = axs[4].matshow(rhos_diff.mean(0), **matshow_pars, norm=norm_unif)
    xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
    axs[4].contour(xx, yy, diff_all_sig[4], colors=color_cluster, levels=[0],
                linestyles='dashed', linewidths=1)
    axs[4].title.set_text('Correlation across entropies')

    for ax in axs[:4]:
        fig.colorbar(im, ax=ax,  norm=norm1, label='Score diff')
    fig.colorbar(im2, ax=axs[4], norm=norm_unif, label='Spearman of diff')

    for x in xlines:
        for ax in axs:
            ax.axvline(x, c=color_vline, ls='-')
            ax.xaxis.set_ticks_position("bottom")

    #plt.tight_layout()
    plt.savefig(path_fig + f'/diff_with_reord_{stimtype}{suffix}.png')
    plt.close()


    # plot diagonals
    if args.inc_neg_time:
        inds = np.where((times >= -0.33) & (times <= 0.33))[0] 
    else:
        inds = np.where((times >= 0) & (times <= 0.33))[0]  # 33

    #inds = np.where((times >= -0.33) & (times <= 0.33))[0]
    titles = ['Normal order','Reordered']
    ttl0 = suffix.replace("test_",'')

    fig,axs = plt.subplots(3,1, figsize=(12,10))
    for ansi,anscur in enumerate([ans1,ans2]):
        ax = axs[ansi]
        #plt.figure()
        for an in anscur:
        #for an in an2scores.keys():
            #an2scores[an][:,:,inds].shape
            diags = np.diagonal(an2scores3[an][:,:,inds], axis1=1,axis2=2).mean(0)

            ax.plot(times[inds],diags, label=an)
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.set_ylim(0.25,0.365)
        ax.set_ylabel(f'Decoding acc,\n average across participants')
        ax.set_xlabel('Train=test time')
        ax.set_title(f'{plots_to_make[ansi]} {ttl0}')
        
    #plt.figure()
    for ani,an in enumerate(ans1):    
        ax = axs[2]
        diags = np.diagonal(an2scores_diff[an][:,:,inds], axis1=1,axis2=2).mean(0)
        ax.plot(times[inds], diags, label=f'{an} - {ans2[ani]}')
        
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.set_ylim(-0.025,0.025)
        ax.set_ylabel(f'Diff of decoding acc,\n average across participants')
        ax.set_xlabel('Train=test time')
        ax.set_title(f'Diffs {ttl0}')
    plt.tight_layout()
    figfnf = pjoin(path_fig,f'diag_{stimtype}{suffix}')
    plt.savefig(figfnf)
    print('fig saved to ',figfnf)


    print('Finished successfully')

