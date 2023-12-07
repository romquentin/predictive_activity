import os.path as op
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

'''
# How to use this file
1. set environment variables 
$DEMARCHI_DATA_PATH and $DEMARCHI_FIG_PATH



'''

plt.rcParams['axes.titlesize'] = 10

# suffix, sp determine parts of filenames to get data from and figure names to save plots to
sp = ''; suffix = '';
#path_results = '/home/demitau/data_Quentin/data_demarchi/results'
#path_results = '/p/project/icei-hbp-2022-0017/demarchi/data_demarchi/MEG_demarchi/results'; suffix = ''
#path_results = '/p/project/icei-hbp-2022-0017/demarchi/data_demarchi/MEG_demarchi/results0'; suffix = '_0'
#path_results = '/p/project/icei-hbp-2022-0017/demarchi/data_demarchi/MEG_demarchi/results_Romain'; suffix='_R'
#path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'; suffix = ''; sp = '__sp'  
#path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'; suffix = ''
path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results_test'; suffix = 'test'

#path_fig = '/p/project/icei-hbp-2022-0017/demarchi/output_plots'
path_fig = os.path.expandvars('$DEMARCHI_FIG_PATH')

results_folder = 'reorder_random'; stimtype=''  # stimtype is omission or sound
#results_folder = 'reorder_random_omission'; stimtype='_omission'

suffix += sp

print('Results folder = ',path_results,  results_folder)

force_recalc = int(sys.argv[1])
print('force_recalc = ',force_recalc)

# list all participants
participants = [f for f in os.listdir(path_results) if f[:1] == '1']
# Initialize list of scores (with cross_validation)
all_cv_rd_to_rd_scores = list()
all_cv_rd_to_mm_scores = list()
all_cv_rd_to_mp_scores = list()
all_cv_rd_to_or_scores = list()
all_cv_rd_to_mmrd_scores = list()
all_cv_rd_to_mprd_scores = list()
all_cv_rd_to_orrd_scores = list()

# Loop accross participants
print('Num particiapnts = ',len(participants) )
dts = []
for participant in participants:
    #print(participant)
    # Append scores (with cross validation)
    fnf = op.join(path_results, participant, results_folder, f'cv_rd_to_rd{sp}_scores.npy')
    dt = datetime.fromtimestamp(os.stat(fnf).st_mtime)
    dts += [dt]

    # read scores from files
    all_cv_rd_to_rd_scores.append(np.load(fnf))
    all_cv_rd_to_mm_scores.append(np.load(op.join(path_results, participant, results_folder,   'cv_rd_to_mm_scores.npy')))
    all_cv_rd_to_mp_scores.append(np.load(op.join(path_results, participant, results_folder,   'cv_rd_to_mp_scores.npy')))
    all_cv_rd_to_or_scores.append(np.load(op.join(path_results, participant, results_folder,   f'cv_rd_to_or{sp}_scores.npy')))
    all_cv_rd_to_mmrd_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_mmrd_scores.npy')))
    all_cv_rd_to_mprd_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_mprd_scores.npy')))
    all_cv_rd_to_orrd_scores.append(np.load(op.join(path_results, participant, results_folder, f'cv_rd_to_orrd{sp}_scores.npy')))

print(f'Earliest (for rd_to_rd{sp}) computed data = ',str(np.min(dts) ) )
print(f'Latest (for rd_to_rd{sp}) computed data = ',str(np.max(dts) ) )

# create arrays with cross-validated scores
all_cv_rd_to_rd_scores = np.array(all_cv_rd_to_rd_scores)
all_cv_rd_to_mm_scores = np.array(all_cv_rd_to_mm_scores)
all_cv_rd_to_mp_scores = np.array(all_cv_rd_to_mp_scores)
all_cv_rd_to_or_scores = np.array(all_cv_rd_to_or_scores)
all_cv_rd_to_mmrd_scores = np.array(all_cv_rd_to_mmrd_scores)
all_cv_rd_to_mprd_scores = np.array(all_cv_rd_to_mprd_scores)
all_cv_rd_to_orrd_scores = np.array(all_cv_rd_to_orrd_scores)


all_cv_rd_to_rd_scores_ = np.array(all_cv_rd_to_rd_scores)
all_cv_rd_to_mmrd_scores_ = np.array(all_cv_rd_to_mmrd_scores)
all_cv_rd_to_mprd_scores_ = np.array(all_cv_rd_to_mprd_scores)
all_cv_rd_to_orrd_scores_ = np.array(all_cv_rd_to_orrd_scores)

# cut the figure to plot a smaller time window
# all times
times = np.linspace(-0.7, 0.7, all_cv_rd_to_or_scores.shape[-1])
wh_x = np.where((times >= -0.7) & (times <= 0.7))[0] # 141
wh_y = np.where((times >= 0) & (times <= 0.33))[0]  # 33
# this ix_ is needed only because numpy (some versions of) are picky about slicing in two dimensions simultaneosly
ixgrid = np.ix_(wh_y, wh_x)

ndims = [ all_cv_rd_to_rd_scores   .ndim, all_cv_rd_to_mm_scores   .ndim, all_cv_rd_to_mp_scores   .ndim,  all_cv_rd_to_or_scores .ndim,  all_cv_rd_to_mmrd_scores.ndim,  all_cv_rd_to_mprd_scores        .ndim,  all_cv_rd_to_orrd_scores                .ndim ]
# make sure all arrays have consisten dimensions
assert len(set(ndims)) == 1

if all_cv_rd_to_rd_scores.ndim == 4:
    # nestimators is X.shape[-1] = n_tasks
    # shape of scores file is (nsubjects,nsamples,nestimators,nslices)
    all_cv_rd_to_rd_scores = all_cv_rd_to_rd_scores[:, :, ixgrid[0], ixgrid[1]].mean(1)  # mean on the second dimension because we kept scores on each fold
    all_cv_rd_to_mm_scores = all_cv_rd_to_mm_scores[:, :, ixgrid[0], ixgrid[1]].mean(1)
    all_cv_rd_to_mp_scores = all_cv_rd_to_mp_scores[:, :, ixgrid[0], ixgrid[1]].mean(1)
    all_cv_rd_to_or_scores = all_cv_rd_to_or_scores[:, :, ixgrid[0], ixgrid[1]].mean(1)
    all_cv_rd_to_mmrd_scores = all_cv_rd_to_mmrd_scores[:, :, ixgrid[0], ixgrid[1]].mean(1)
    all_cv_rd_to_mprd_scores = all_cv_rd_to_mprd_scores[:, :, ixgrid[0], ixgrid[1]].mean(1)
    all_cv_rd_to_orrd_scores = all_cv_rd_to_orrd_scores[:, :, ixgrid[0], ixgrid[1]].mean(1)
else:
    all_cv_rd_to_rd_scores = all_cv_rd_to_rd_scores[:, ixgrid[0], ixgrid[1]]  # mean on the second dimension because we kept scores on each fold
    all_cv_rd_to_mm_scores = all_cv_rd_to_mm_scores[:,  ixgrid[0], ixgrid[1]]
    all_cv_rd_to_mp_scores = all_cv_rd_to_mp_scores[:,  ixgrid[0], ixgrid[1]]
    all_cv_rd_to_or_scores = all_cv_rd_to_or_scores[:,  ixgrid[0], ixgrid[1]]
    all_cv_rd_to_mmrd_scores = all_cv_rd_to_mmrd_scores[:,  ixgrid[0], ixgrid[1]]
    all_cv_rd_to_mprd_scores = all_cv_rd_to_mprd_scores[:,  ixgrid[0], ixgrid[1]]
    all_cv_rd_to_orrd_scores = all_cv_rd_to_orrd_scores[:,  ixgrid[0], ixgrid[1]]

# plotting parameters

matshow_pars = dict(origin='lower', extent=[-0.7, 0.7, 0, 0.33],
                    cmap='inferno')
#color_cluster = 'k'
color_cluster = 'white' # color of cluster boundaries
xlines = [-0.33,0,0.33] # times [s] where to draw vertical lines
color_vline = 'grey' # vertical lines color
vmin_rhos, vmax_rhos = -0.6, 0.6  # plotting correlation colorbar limits
lims_diff = -0.02, 0.02             # plotting diff colorbar limits 
# plotting scores colorbar limits  
#lims_scores = 0.23, 0.27; s = ''
lims_scores = 0.23, 0.285; s = ''
#lims_scores = 0.23, 0.30; s = f'_max_{lims_scores[1]:.1f}'
suffix += 's'

# which sets of plots to generate (of 3 in total)
plots_to_make = ['orig', 'reord', 'diff']
#plots_to_make = ['reord']
cleanup = 0

n_jobs = -1 # n threads for computing correlations

nsamples = 33
nsubj = 33
nt = 141


# function to compute correlations
def f(i, rand2rd, rand2mm, rand2mp, rand2or):
    # takes scores
    rhos_ = np.zeros([nsamples, nt])
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

# first plot reproduction 
if 'orig' in plots_to_make:
    # compute spearman correlation as in Demarchi et al. across entropies
    # Initialize spearman rho results
    fn = f'orig_rhos{stimtype}{suffix}.npz'
    if not os.path.exists(fn) or force_recalc:
        print('Compute rhos')
        rhos = np.zeros([nsubj, nsamples, nt])
        #for i in tqdm(range(rhos.shape[0])):
        #    for row in range(rhos.shape[1]):
        #        for column in range(rhos.shape[2]):
        #            corr_values = [all_cv_rd_to_rd_scores[i, row, column], all_cv_rd_to_mm_scores[i, row, column],
        #                           all_cv_rd_to_mp_scores[i, row, column], all_cv_rd_to_or_scores[i, row, column]]
        #            rhos[i, row, column], _ = spearmanr([0, 1, 2, 3], corr_values)

        args = [ (i, all_cv_rd_to_rd_scores[i], all_cv_rd_to_mm_scores[i], 
            all_cv_rd_to_mp_scores[i], all_cv_rd_to_or_scores[i] ) for i in range(nsubj) ]

        plr = Parallel(n_jobs=n_jobs, backend='multiprocessing',)(delayed(f)(*arg) for arg in args)
        for i,rhos_ in plr:
            rhos[ i, :,: ] = rhos_
        np.savez(fn, rhos)
    else:
        print('Load rhos')
        f = np.load(fn, allow_pickle=True)
        rhos = f['arr_0'][()]

    fn = f'orig_allsig{stimtype}{suffix}.npz'
    # Plot main figure of Demarchi et al. (Fig3a)
    vmin, vmax = lims_scores
    norm1 = colors.Normalize(vmin=vmin, vmax=vmax)
    chance = 0.25
    # get permutations clusters
    if not os.path.exists(fn) or force_recalc:
        print('Compute clustering')
        all_sig = list()
        for scores in tqdm([all_cv_rd_to_or_scores, all_cv_rd_to_mp_scores, all_cv_rd_to_mm_scores, all_cv_rd_to_rd_scores, rhos]):
            print('iteration')
            if scores.all() == rhos.all():
                gat_p_values = gat_stats(np.array(scores))
            else:
                gat_p_values = gat_stats(np.array(scores) - chance)
            sig = np.array(gat_p_values < 0.05)
            all_sig.append(sig)
        np.savez(fn, all_sig)
    else:
        print('Load clustering')
        f = np.load(fn, allow_pickle=True)
        all_sig = f['arr_0'][()]



    # plot the 4 conditions
    fig, axs = plt.subplots(5, 1, figsize=(7,8), constrained_layout=True)
    fig.set_layout_engine('tight')

    if sp == '':
        im = axs[3].matshow(all_cv_rd_to_rd_scores.mean(0), **matshow_pars, norm=norm1)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[3].contour(xx, yy, all_sig[3], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[3].title.set_text('Random')

        axs[2].matshow(all_cv_rd_to_mm_scores.mean(0), **matshow_pars, norm=norm1)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[2].contour(xx, yy, all_sig[2], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[2].title.set_text('Midminus')
        axs[1].matshow(all_cv_rd_to_mp_scores.mean(0), **matshow_pars, norm=norm1)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[1].contour(xx, yy, all_sig[1], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[1].title.set_text('Midplus')

    xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
    axs[0].matshow(all_cv_rd_to_or_scores.mean(0), **matshow_pars, norm=norm1)
    axs[0].contour(xx, yy, all_sig[0], colors=color_cluster, levels=[0],
                linestyles='dashed', linewidths=1)
    axs[0].title.set_text(f'Ordered{sp}')

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

    plt.savefig(path_fig + f'/main_fig_demarchi{stimtype}{suffix}.png')

#sys.exit(0)

# now plot reordered version 
if 'reord' in plots_to_make:
    ###################################################################
    # compute spearman correlation as in Demarchi et al. across entropies
    # Initialize spearman rho results
    ###################################################################

    #del rhos, reord_all_sig

    fn = f'reord_rhos{stimtype}{suffix}.npz'
    if not os.path.exists(fn) or force_recalc:
        print('Compute rhos reord')
        rhos_reord = np.zeros([nsubj, nsamples, nt])
        #for i in tqdm(range(rhos_reord.shape[0])):
        #    for row in range(rhos_reord.shape[1]):
        #        for column in range(rhos_reord.shape[2]):
        #            corr_values = [all_cv_rd_to_rd_scores[i, row, column],
        #               all_cv_rd_to_mmrd_scores[i, row, column],
        #               all_cv_rd_to_mprd_scores[i, row, column],
        #               all_cv_rd_to_orrd_scores[i, row, column]]
        #            rhos_reord[i, row, column], _ = spearmanr([0, 1, 2, 3], corr_values)
        args = [ (i,all_cv_rd_to_rd_scores[i], all_cv_rd_to_mmrd_scores[i],
                all_cv_rd_to_mprd_scores[i], all_cv_rd_to_orrd_scores[i] ) for i in range(nsubj) ]

        plr = Parallel(n_jobs=n_jobs, backend='multiprocessing',
                                                )(delayed(f)(*arg) for arg in args)
        for i,rhos_reord_ in plr:
            rhos_reord[ i, :,: ] = rhos_reord_
        np.savez(fn, rhos_reord)
    else:
        print('Load rhos reord')
        f = np.load(fn, allow_pickle=True)
        rhos_reord = f['arr_0'][()]

    # Plot main figure of Demarchi et al. (Fig3a) with reordered events from only random
    vmin, vmax = lims_scores
    norm1 = colors.Normalize(vmin=vmin, vmax=vmax)
    chance = 0.25
    fn = f'reord_allsig{stimtype}{suffix}.npz'
    if not os.path.exists(fn) or force_recalc:
        print('Compute clustering reord')
        # get permutations clusters
        reord_all_sig = list()
        for scores in tqdm([all_cv_rd_to_orrd_scores, all_cv_rd_to_mprd_scores, all_cv_rd_to_mmrd_scores, all_cv_rd_to_rd_scores, rhos_reord]):
            print('iteration')
            if scores.all() == rhos_reord.all():
                gat_p_values = gat_stats(np.array(scores))
            else:
                gat_p_values = gat_stats(np.array(scores) - chance)
            sig = np.array(gat_p_values < 0.05)
            reord_all_sig.append(sig)
        np.savez(fn, reord_all_sig)
    else:
        print('Load clustering reord')
        f = np.load(fn, allow_pickle=True)
        reord_all_sig = f['arr_0'][()]

    # plot the 4 conditions
    fig, axs = plt.subplots(5, 1, figsize=(7,8), constrained_layout=True)
    fig.set_layout_engine('tight')
    if sp == '':
        im = axs[3].matshow(all_cv_rd_to_rd_scores.mean(0), **matshow_pars, norm=norm1)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[3].contour(xx, yy, reord_all_sig[3], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[3].title.set_text('Random')

        axs[2].matshow(all_cv_rd_to_mmrd_scores.mean(0), **matshow_pars, norm=norm1)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[2].contour(xx, yy, reord_all_sig[2], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[2].title.set_text('Midminus rd2mmrd')

        axs[1].matshow(all_cv_rd_to_mprd_scores.mean(0), **matshow_pars, norm=norm1)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[1].contour(xx, yy, reord_all_sig[1], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[1].title.set_text('Midplus rd2mprd')

    xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
    axs[0].matshow(all_cv_rd_to_orrd_scores.mean(0), **matshow_pars, norm=norm1)
    axs[0].contour(xx, yy, reord_all_sig[0], colors=color_cluster, levels=[0],
                linestyles='dashed', linewidths=1)
    axs[0].title.set_text(f'Ordered{sp} rd2orrd')

    norm_unif = colors.Normalize(vmin=vmin_rhos, vmax=vmax_rhos)
    if sp == '':
        im2 = axs[4].matshow(rhos_reord.mean(0), **matshow_pars, norm=norm_unif)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[4].contour(xx, yy, reord_all_sig[4], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[4].title.set_text('Correlation across entropies')

    for ax in axs:
        ax.set_ylabel('Train time')
    axs[-1].set_xlabel('Test time')


    for ax in axs[:4]:
        fig.colorbar(im, ax=ax,  norm=norm1, label='Score')
    fig.colorbar(im2, ax=axs[4], norm=norm_unif, label='Spearman rho')

    for x in xlines:
        for ax in axs:
            ax.axvline(x, c=color_vline, ls='-')
            ax.xaxis.set_ticks_position("bottom")

    plt.savefig(path_fig + f'/main_fig_reorder_demarchi{stimtype}{suffix}.png')


######################################################################

# Plot main figure of Demarchi et al. (Fig3a) with differences between classic and reordered trials
###################################################################

if 'diff' in plots_to_make:
    vmin, vmax = lims_diff
    norm1_diff = colors.Normalize(vmin=vmin, vmax=vmax)
    chance = 0
    # get permutations clusters
    diff_all_sig = list()
    diff_rd_to_orrd = all_cv_rd_to_or_scores - all_cv_rd_to_orrd_scores
    diff_rd_to_mprd = all_cv_rd_to_mp_scores - all_cv_rd_to_mprd_scores
    diff_rd_to_mmrd = all_cv_rd_to_mm_scores - all_cv_rd_to_mmrd_scores
    diff_rd_to_rd = all_cv_rd_to_rd_scores - all_cv_rd_to_rd_scores

    # compute spearman correlation as in Demarchi et al. across entropies
    # Initialize spearman rho results
    fn = f'diff_rhos{stimtype}{suffix}.npz'
    if not os.path.exists(fn) or force_recalc:
        print('Compute rhos diff')
        rhos_diff = np.zeros([nsubj, nsamples, nt])
        #for i in tqdm(range(rhos_diff.shape[0])):
        #    for row in range(rhos_diff.shape[1]):
        #        for column in range(rhos_diff.shape[2]):
        #            corr_values = [diff_rd_to_rd[i, row, column],
        #               diff_rd_to_mmrd[i, row, column], diff_rd_to_mprd[i, row, column], diff_rd_to_orrd[i, row, column]]
        #            rhos_diff[i, row, column], _ = spearmanr([0, 1, 2, 3], corr_values)

        args = [ (i,diff_rd_to_rd[i], diff_rd_to_mmrd[i], diff_rd_to_mprd[i],
                diff_rd_to_orrd[i] ) for i in range(nsubj) ]

        plr = Parallel(n_jobs=n_jobs, backend='multiprocessing',
                                                )(delayed(f)(*arg) for arg in args)
        for i,rhos_diff_ in plr:
            rhos_diff[ i, :,: ] = rhos_diff_
        np.savez(fn, rhos_diff)
    else:
        print('Load rhos diff')
        f = np.load(fn, allow_pickle=True)
        rhos_diff = f['arr_0'][()]


    print('Compute clustering diff')
    fn = f'diff_allsig{stimtype}{suffix}.npz'
    if not os.path.exists(fn) or force_recalc:
        for scores in tqdm([diff_rd_to_orrd, diff_rd_to_mprd, diff_rd_to_mmrd, diff_rd_to_rd, rhos_diff]):
            print('iteration')
            if scores.all() == rhos_diff.all():
                gat_p_values = gat_stats(np.array(scores))
            else:
                gat_p_values = gat_stats(np.array(scores) - chance)
            sig = np.array(gat_p_values < 0.05)
            diff_all_sig.append(sig)
        np.savez(fn, diff_all_sig)
    else:
        f = np.load(fn, allow_pickle=True)
        diff_all_sig = f['arr_0'][()]

    # plot the 4 conditions
    fig, axs = plt.subplots(5, 1, figsize=(7,8), constrained_layout=True)
    fig.set_layout_engine('tight')

    if sp == '':
        im =axs[3].matshow(diff_rd_to_rd.mean(0), **matshow_pars, norm=norm1_diff)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[3].contour(xx, yy, diff_all_sig[3], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[3].title.set_text('Random')

        axs[2].matshow(diff_rd_to_mmrd.mean(0), **matshow_pars, norm=norm1_diff)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[2].contour(xx, yy, diff_all_sig[2], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[2].title.set_text('Midminus')
        axs[1].matshow(diff_rd_to_mprd.mean(0), **matshow_pars, norm=norm1_diff)
        xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
        axs[1].contour(xx, yy, diff_all_sig[1], colors=color_cluster, levels=[0],
                    linestyles='dashed', linewidths=1)
        axs[1].title.set_text('Midplus')

    xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
    axs[0].matshow(diff_rd_to_orrd.mean(0), **matshow_pars, norm=norm1_diff)
    axs[0].contour(xx, yy, diff_all_sig[0], colors=color_cluster, levels=[0],
                linestyles='dashed', linewidths=1)
    axs[0].title.set_text(f'Ordered{sp}')

    norm_unif = colors.Normalize(vmin=vmin_rhos, vmax=vmax_rhos)
    if sp == '':
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
    plt.savefig(path_fig + f'/main_fig_diff_with_reorder_demarchi{stimtype}{suffix}.png')
    print('Finished successfully')

