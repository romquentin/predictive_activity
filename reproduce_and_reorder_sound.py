# %%
import os.path as op
import os, sys
import numpy as np
import argparse
import pandas as pd

import mne
from mne.decoding import GeneralizingEstimator, LinearModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from base import * 

path_data = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/MEG'
path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'

###  for debug only
print(sys.argv)
# -0.33
# sys.argv = sys.argv[:1] + [ "-s", "4", "--add_epind_channel=1", "--add_sampleind_channel=1", 
#             "--n_jobs=1", "--tmin=\"-0.33\"", "--tmax=\"0.33\"",
#               "--save_suffix_scores=only_pre_window", "--leakage_report_only=0","--nfolds=2",
#                           "--shift_orig_inds=-1", "--remove_leak_folds=1"]\
#--tmin=\"-0.33\" --tmax=\"0.33\" --add_epind_channel=1 --add_sampleind_channel=1 --remove_leak_folds=1 --save_suffix_scores=small_window -s 0

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# tmin,tmax=-0.7,0.7
# Add the arguments to the parser
parser.add_argument('-s',"--subject", type=int, default =-1, required=True, help='subject index (zero-based integer)')
parser.add_argument("--nfolds", type=int, default=5, help='num CV folds')
parser.add_argument("--force_refilt", type=int, default=0, help='force recalc of filtered raws')
parser.add_argument("--extract_filters_patterns", type=int, default =0)
parser.add_argument("--shuffle_cv", type=int, default=0)
parser.add_argument("--shift_orig_inds", type=int, default=0)
parser.add_argument("--remove_leak_folds", type=int, default=1)
parser.add_argument("--exit_after", type=str, default='end')
parser.add_argument("--reord_narrow_test", type=int, default=0, help='restrict test to only reord events (currently not working, but also not necessary)')
parser.add_argument("--add_epind_channel", type=int, default=1, help='add channel with epoch index')
parser.add_argument("--add_sampleind_channel", type=int, default=1, help='add channel with sample index to raw')
parser.add_argument("--tmin", type=str, default="-0.33", help='tmin as string (need quotes)')
parser.add_argument("--tmax", type=str, default="0.33", help='tmax as string')
parser.add_argument("--n_jobs", type=int, default=-1, help='number of jobs to run in parallel')
parser.add_argument("--save_suffix_scores", type=str, default="", help='suffix to add to the save filenames')
parser.add_argument("--leakage_report_only", type=int, default=0, help='only report leakage, do not run the decoding')

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to the variables
extract_filters_patterns = args.extract_filters_patterns
nfolds = args.nfolds
force_refilt = args.force_refilt
shuffle_cv = bool(args.shuffle_cv)

# %%
# define tmin and tmax
#tmin, tmax = -0.7, 0.7

tmin,tmax = args.tmin,args.tmax
if tmin[0]  == '"':
    assert tmin[-1] == '"'
    tmin = tmin[1:-1]
if args.tmax[0]  == '"':
    assert tmax[-1] == '"'
    tmax = tmax[1:-1]
tmin,tmax= float(tmin),float(tmax)
#assert args.tmin[0] == '"' and args.tmax[0] == '"'
#assert args.tmin[-1] == '"' and args.tmax[-1] == '"'

events_all = events_sound + events_omission # event_ids to be used to select events with MNE
del_processed = 1  # determines the version of the reordering algorithm. Currently only = 1 works
cut_fl = 0 # whether we cut out first and last events from the final result           
#gen_est_verbose = True
gen_est_verbose = False # def True, argument of GeneralizingEstimator
dur = 200 # duration (in samples) of pre-task and post-task data  
nsamples = 33 # trial duration in samples
max_num_epochs_per_cond = 5000 


# parse directory names from the data directory
rows = [ dict(zip(['subj','block','cond','path'], v[:-15].split('_') + [v])  ) for v in os.listdir(path_data)]
df0 = pd.DataFrame(rows)
df = df0.copy()
# simple readable subject id
df['sid'] = df['subj'].apply(lambda x: corresp.get(x, 'unknown'))

# which subject IDs to use
sids_to_use = [args.subject]
df = df.query('sid.isin(@sids_to_use)')

# check we have complete data (all conditions for every subject)
grp = df.groupby(['subj'])
assert grp.size().min() == grp.size().max()
assert grp.size().min() == 4

assert len(grp.groups) == 1

# %%
# iterating over subjects (if we selected one, then process one subject)
for g,inds in grp.groups.items(): # recall we made sure we just have one subject above
    subdf = df.loc[inds]

# get paths to datasets for each entropy condition per subject
subdf= subdf.set_index('cond')
subdf = subdf.drop(columns=['subj','block','sid'])
meg_rd = subdf.to_dict('index')['random']['path']
meg_or = subdf.to_dict('index')['ordered']['path']
print(meg_rd, meg_or)

# results folder where to save the scores for one participant
ps = [p[:12] for p in [meg_rd, meg_or] ]
assert len(set(ps) ) == 1

p0 = op.join( os.path.expandvars('$TEMP_DATA_DEMARCHI') , meg_rd[:-15] )

participant = meg_or[:12]
print('------------------------------------------------------')
print('---------------------- Starting participant', participant)
print('------------------------------------------------------')
results_folder = op.join(path_results, participant, 'reorder_random')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# load or recalc filtered epochs
cond2epochs, cond2raw   = read_raws(p0,force_refilt, tmin, tmax, 
        max_num_epochs_per_cond, args.add_sampleind_channel, subdf, path_data, events_all, args.n_jobs)
raw_random = cond2raw['random'] # raw_rd is used for reorder later

if args.add_epind_channel:
    for cond,epochs in cond2epochs.items():
        print('Adding epind channel to ',cond)

        # at every time point we put the epoch number
        eps_full = addEpindChan(epochs, np.arange(len(epochs)) )

        cond2epochs[cond] = eps_full

#### remove omission and following trials in random trials
lens_ext = []
cond2counts = {}
# cycle over four entropy conditions
for cond,epochs in cond2epochs.items():
    # just in case save numbers before removing omission trials
    lens_ext += [(cond+'_keepomission',len(epochs))  ]
    cond2counts[cond+'_keepomission'] = Counter(epochs.events[:,2])

    # get indices of omission events
    om = np.where(np.isin(epochs.events, events_omission))[0]
    # take next indices after them and sort indices
    om_fo = np.sort(np.concatenate([om, om+1]))
    # if the last one is not an index, remove it
    if om_fo[-1] == len(epochs.events):
        om_fo = np.delete(om_fo, -1)
    # remove these indices from random epochs
    cond2epochs[cond] = epochs.drop(om_fo)
    cond2counts[cond] = Counter(cond2epochs[cond].events[:,2])

################################################################
# reorder random as ...
################################################################
# %%

epochs_random_init = cond2epochs['random'].copy()

cond2epochs_reord = {}
cond2orig_inds_reord = {}

cond2epochs_sp_reord = {}
cond2orig_inds_sp_reord = {}

reorder_pars = dict(del_processed= del_processed, cut_fl=cut_fl, 
    tmin=tmin, tmax=tmax, dur=dur, nsamples=nsamples)
# cycle over four entropy conditions (targets of reordering)
for cond,epochs in cond2epochs.items():
    # original random events
    random_events = epochs_random_init.events.copy()
    # target events
    events0 = epochs.events.copy()
    
    # reorder random events to another entropy condition
    epochs_reord0, orig_inds_reord0 = reorder(random_events, events0, raw_random, **reorder_pars) 
    if args.add_epind_channel:
        epochs_reord_ext0 = addEpindChan(epochs_reord0, orig_inds_reord0 )
        cond2epochs_reord[cond] = epochs_reord_ext0
    else:
        cond2epochs_reord[cond] = epochs_reord0
    cond2orig_inds_reord[cond] = orig_inds_reord0

    cond2counts[cond+'_reord'] = Counter(cond2epochs_reord[cond].events[:,2])

    #########################
    ####   reorder simple prediction
    #########################

    # first we transform events from the current entropy condtion into it's "simple prediction" (most probable next event) verion 
    events = events_simple_pred(epochs.events.copy(), cond2code[cond])
    # then we do the reorderig like before, but in this case the target events are the transformed events, not the true ones
    epochs_reord, orig_inds_reord = reorder(random_events, events, raw_random, **reorder_pars) 

    if args.add_epind_channel:
        epochs_reord_ext = addEpindChan(epochs_reord, orig_inds_reord )
        cond2epochs_sp_reord[cond] = epochs_reord_ext
    else:
        cond2epochs_sp_reord[cond] = epochs_reord
    cond2orig_inds_sp_reord[cond] = orig_inds_reord

    cond2counts[cond+'_sp_reord'] = Counter(cond2epochs_sp_reord[cond].events[:,2])


# save counts of all classes to process later (not in this script)
fnf = op.join(results_folder, f'cond2counts.npz' )
print('Saving ',fnf)
np.savez(fnf , cond2counts )

###################################################################
########################     cross validation
###################################################################
print("------------   Starting CV")
cv = StratifiedKFold(nfolds, shuffle=shuffle_cv)

# we need to know minimum number of trials to use it always (they don't actually differ that much but it reduces headache with folds correspondance)
lens = [ len(ep) for ep in cond2epochs.values() ]
lens += [ len(ep) for ep in cond2epochs_reord.values() ]
lens += [ len(ep) for ep in cond2epochs_sp_reord.values() ]
minl = np.min(lens)
print('epochs lens = ',lens, ' minl = ',minl)

lens_ext += [ (cond,len(ep) ) for cond,ep in cond2epochs.items() ]
lens_ext += [ (cond+'_reord',len(ep) ) for cond,ep in cond2epochs_reord.items() ]
lens_ext += [ (cond+'sp_reord',len(ep) ) for cond,ep in cond2epochs_sp_reord.items() ]

if args.exit_after == 'prep_dat':
    sys.exit(0)

# Initialize classifier
if extract_filters_patterns:
    clf = LinearModel(LinearDiscriminantAnalysis() ); 
else:
    clf = make_pipeline(LinearDiscriminantAnalysis())
clf = GeneralizingEstimator(clf, n_jobs=args.n_jobs, scoring='accuracy', verbose=gen_est_verbose)

cond2ms={}

# %%
# cycle over entropies
for cond,epochs in cond2epochs.items():
    print(f"-----  CV for {cond}")
    cond2ms[cond] = []  # condition 2 leakage table

    # keep only the same number of trials for all conditions
    epochs = epochs[:minl]  
    # get the X and Y for each condition in numpy array
    X = epochs.get_data() # for the given condition

    y_sp_ = events_simple_pred(epochs.events.copy() , cond2code[cond]) # not reordered
    y_sp = y_sp_[:, 2] 

    #----------
    epochs_reord = cond2epochs_reord[cond][:minl]
    orig_inds_reord = cond2orig_inds_reord[cond] + args.shift_orig_inds # here we shift the indices of random events
    # TODO: find way to use both sp and not sp, reord and not

    # keep same trials in epochs_rd and epochs_reord
    epochs_rd1 = epochs_random_init[orig_inds_reord][:minl]
    Xrd1 = epochs_rd1.get_data()
    yrd1 = epochs_rd1.events[:, 2]

    epochs_sp_reord = cond2epochs_sp_reord[cond]
    #orig_nums_sp_reord = cond2orig_nums_sp_reord[cond] 
    #
    #epochs_rd2 = epochs_rd_init[orig_nums_sp_reord][:minl]
    #Xrd2 = epochs_rd2.get_data()
    #yrd2 = epochs_rd2.events[:, 2]

    y0_ = epochs.events.copy()[:minl]
    y0 = y0_[:, 2] 

    Xreord = epochs_reord.get_data()[:minl]

    # simple prediction after reordering
    yreord_ = epochs_reord.events
    yreord = yreord_[:, 2]
    yreord_sp = events_simple_pred(yreord_, cond2code[cond])[:, 2]

    #Xsp_reord = epochs_sp_reord.get_data()[:minl]
    ysp_reord_ = epochs_sp_reord.events
    ysp_reord = ysp_reord_[:, 2]

    # get short entropy condition code to generate save filenames
    s = cond2code[cond]
    scores = {} # score type 2 score

    filters  = []
    patterns = []
    for foldi, (train_inds, test_inds) in enumerate(cv.split(Xrd1, yrd1) ):
        print(f"##############  Starting {cond} fold {foldi}")
        print('Lens of train and test are :',len(train_inds), len(test_inds) )

        valchans = np.arange(Xrd1.shape[1])
        if args.add_epind_channel:
            ind_dim_epind = -1
            valchans = valchans[:-1]
            
            epinds_train  = Xreord[train_inds][:,ind_dim_epind,0].astype(int)
            epinds_test   = Xreord[test_inds][:,ind_dim_epind,0].astype(int)
            print(Xreord.shape, epinds_train.shape, epinds_test.shape)
            print('intersection size epinds_train and epinds_test = ', 
                len(set(epinds_train) & set(epinds_test) ) ) # equal to 0            

        if args.add_sampleind_channel:
            if args.add_epind_channel:
                ind_dim_sampleind = -2
            else:
                ind_dim_sampleind = -1
            valchans = valchans[:-1]
            
            # sampleinds_train = Xreord[train_inds][:,ind_dim_sampleind,:].astype(int)
            # sampleinds_test  = Xreord[test_inds][:,ind_dim_sampleind,:].astype(int)
            # inum = len(set(sampleinds_train.flatten()) & set(sampleinds_test.flatten() ) )
            # tnum = len(set(sampleinds_test.flatten()) )
            # print(f'intersection size sampleinds_train and sampleinds_test = {inum} = {100* inum/tnum:.2f}% of test indices ')             
            train_sample_inds  = Xreord[train_inds][:,ind_dim_sampleind,:]
            test_sample_inds   = Xreord[test_inds] [:,ind_dim_sampleind,:]
            m = calc_leackage_sampleinds(train_sample_inds, test_sample_inds, verbose=0)
            cond2ms[cond] += [(m,len(test_sample_inds) )]        
            ll = len(test_inds) * Xreord.shape[-1]
            print('max={}, sum={}, pct={:.3f}%\n'.format(np.max(m), np.sum(m), np.sum(m)*100/ll ) )#, np.where(m > 0) )

        if args.leakage_report_only:
            continue
        # Run cross validation for the ordered (and reorder-order) (and keep the score on the random too only here)
        # Train and test with cross-validation
        Xrd1   = Xrd1[:,valchans,:]
        X      = X[:,valchans,:]
        Xreord = Xreord[:,valchans,:]
        #print(f'{Xrd1.shape=}, {X.shape=}, {Xreord.shape=}')

        if args.remove_leak_folds:
            # m.shape = (train_inds.shape[0], test_inds.shape[0] )
            test_inds_clean = test_inds[ np.max(m, axis=0) <= 2 ]
            print('len cleaned test {}, len all test inds {}, pct={:.2f}%'.format(len(test_inds_clean), len(test_inds), len(test_inds_clean)*100/len(test_inds) ) )
        else:
            test_inds_clean = test_inds

        print('Start training of clf')
        clf.fit(Xrd1[train_inds], yrd1[train_inds])  # fit on random

        # to plot patterns later... not very useful in the end, they are too inconsistent
        if extract_filters_patterns:
            filters_, patterns_ = getFiltPat(clf)
            filters  += [filters_]
            patterns += [patterns_]

        # fit on random, test on random
        cv_rd_to_rd_score = clf.score(Xrd1[test_inds_clean], yrd1[test_inds_clean])
        # fit on random, test on order
        cv_rd_to__score = clf.score(X[test_inds_clean], y0[test_inds_clean])
        # fit on random, test on simple pred (not reordered)
        cv_rd_to_sp_score = clf.score(X[test_inds_clean], y_sp[test_inds_clean])
            
        ## does not work because the size of resulting array is not the same as the size of Xreord  
        if args.reord_narrow_test:
            #test_reord = np.isin(orig_nums_reord, test_rd)  # why sum(test_reord) != len(test_rd)
            print('{} test_rd among orig_nums_reord. Total = {} '.format( len(test_inds), len(test_inds) ) )
            cv_rd_to_reord_score = clf.score(Xreord[test_inds], yreord[test_inds])
        else:
            cv_rd_to_reord_score    = clf.score(Xreord[test_inds_clean], yreord[test_inds_clean])
            # simple prediction after reordering
            cv_rd_to_reord_sp_score = clf.score(Xreord[test_inds_clean], yreord_sp[test_inds_clean])
            # reordered simpled pred
            cv_rd_to_sp_reord_score = clf.score(Xreord[test_inds_clean], ysp_reord[test_inds_clean])

        dadd(scores,'rd_to_rd',cv_rd_to_rd_score      )
        dadd(scores,f'rd_to_{s}',cv_rd_to__score        )
        dadd(scores,f'rd_to_{s}_sp',cv_rd_to_sp_score        )

        dadd(scores,f'rd_to_{s}_reord',cv_rd_to_reord_score   )
        dadd(scores,f'rd_to_{s}_reord_sp',cv_rd_to_reord_sp_score   )
        dadd(scores,f'rd_to_{s}_sp_reord',cv_rd_to_sp_reord_score   )
        #'cv'
    if extract_filters_patterns:
        filters_rd,patterns_rd = np.array(filters), np.array(patterns)

    m = None
    for cond,ms in cond2ms.items():
        for foldi,(m,l) in enumerate(ms):
            ll = l * Xreord.shape[-1]
            print('-- ', cond,foldi,'max={}, sum={}, pct={:.1f}%'.format(np.max(m), np.sum(m), np.sum(m)*100/ll ) )
    if args.leakage_report_only:
        continue

    # train on non-random and test on same or reord (to make "self" plots)
    filters  = []
    patterns = []
    # train on NOT (only) random and test on itself
    for train, test in cv.split(X, y0):
        print(f"##############  Starting {cond} fold")
        clf.fit(X[train], y0[train])  
        if extract_filters_patterns:
            filters_, patterns_ = getFiltPat(clf)
            filters  += [filters_]
            patterns += [patterns_]

        cv__to__score = clf.score(X[test], y0[test])
        cv__to_reord_score = clf.score(Xreord[test], yreord[test])
        dadd(scores,f'{s}_to_{s}', cv__to__score )
        dadd(scores,f'{s}_to_{s}_reord', cv__to_reord_score )
    filters_cond,patterns_cond = np.array(filters), np.array(patterns)

    filters  = []
    patterns = []
    for train, test in cv.split(Xreord, yreord):
        print(f"##############  Starting {cond} fold reord")
        clf.fit(Xreord[train], yreord[train])  
        if extract_filters_patterns:
            filters_, patterns_ = getFiltPat(clf)
            filters  += [filters_]
            patterns += [patterns_]

        cv_reord_to__score = clf.score(X[test], y0[test])
        cv_reord_to_reord_score = clf.score(Xreord[test], yreord[test])
        dadd(scores,f'{s}_reord_to_{s}', cv_reord_to__score )
        dadd(scores,f'{s}_reord_to_{s}_reord', cv_reord_to_reord_score )
    filters_cond_reord,patterns_cond_reord = np.array(filters), np.array(patterns)

    if extract_filters_patterns:
        for name,(filters_,patterns_) in zip( ['fit_rd0', f'fit_{cond}',f'fit_{cond}_reord'], 
                [ (filters_rd,patterns_rd), (filters_cond,patterns_cond), (filters_cond_reord,patterns_cond_reord) ] ) :
            fnf = op.join(results_folder, f'cv_{name}_filters.npy' )
            print('Saving ',fnf)
            np.save(fnf , filters_ )     # folds x times x classes x channels 

            fnf = op.join(results_folder, f'cv_{name}_patterns.npy' )
            print('Saving ',fnf)
            np.save(fnf , patterns_ )    # folds x times x classes x channels

    # save everything
    for k,v in scores.items():
        scores[k] = np.array(v)
        fnf = op.join(results_folder, f'cv_{k}_scores{args.save_suffix_scores}.npy' )
        print('Saving ',fnf)
        np.save(fnf , v )

    # clean
    import gc; gc.collect()

# %%
cond2avpct = {}
for cond,ms in cond2ms.items():
    cond2avpct[cond] = 0.
    for foldi,(m,l) in enumerate(ms):
        ll = l * Xreord.shape[-1]
        pct = np.sum(m)*100/ll 
        print(cond,foldi,'max={}, sum={}, pct={:.2f}%'.format(np.max(m), np.sum(m), pct ) )
        cond2avpct[cond] += pct / len(ms)

for cond,ms in cond2ms.items():
    print(cond,'avpct={:.2f}%'.format(cond2avpct[cond] ) )

fnf = op.join(results_folder, f'leakage{args.save_suffix_scores}.npz' )
print('Saving ',fnf)
np.save(fnf , cond2ms )
