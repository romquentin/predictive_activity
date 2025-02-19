import os.path as op
import os, sys
import numpy as np
import argparse
import pandas as pd

import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator, LinearModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from base import corresp, events_simple_pred, cond2code, events_omission, events_sound, reorder, getFiltPat, dadd

os.environ['DEMARCHI_DATA_PATH'] ='/p/project/icei-hbp-2022-0017/demarchi/data_demarchi/MEG_demarchi'
so.environ['TEMP_DATA_DEMARCHI'] = '$SCRATCH/memerr/demarchi'

path_data = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/MEG'
path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add the arguments to the parser
parser.add_argument("--subject", type=int, default =-1, required=True)
parser.add_argument("--extract_filters_patterns", type=int, default =1)
parser.add_argument("--nfolds", type=int, default=5)
parser.add_argument("--force_refilt", type=int, default=0)
parser.add_argument("--shuffle_cv", type=int, default=0)
parser.add_argument("--exit_after", type=str, default='end')

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to the variables
extract_filters_patterns = args.extract_filters_patterns
nfolds = args.nfolds
force_refilt = args.force_refilt
shuffle_cv = bool(args.shuffle_cv)

# define tmin and tmax
tmin, tmax = -0.7, 0.7
events_all = events_sound + events_omission # event_ids to be used to select events with MNE
del_processed = 1  # determines the version of the reordering algorithm. Currently only = 1 works
cut_fl = 0 # whether we cut out first and last events from the final result           
reord_narrow_test = 0 
#gen_est_verbose = True
gen_est_verbose = False # def True, argument of GeneralizingEstimator
dur = 200 # duration (in samples) of pre-task and post-task data  
nsamples = 33 # trial duration in samples


# parse directory names from the data directory
rows = [ dict(zip(['subj','block','cond','path'], v[:-15].split('_') + [v])  ) for v in os.listdir(path_data)]
df0 = pd.DataFrame(rows)
df = df0.copy()
# simple readable subject id
df['sid'] = df['subj'].apply(lambda x: corresp[x])

# which subject IDs to use
sids_to_use = [args.subject]
df = df.query('sid.isin(@sids_to_use)')

# check we have complete data (all conditions for every subject)
grp = df.groupby(['subj'])
assert grp.size().min() == grp.size().max()
assert grp.size().min() == 4

# iterating over subjects (if we selected one, then process one subject)
for g,inds in grp.groups.items():
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

    participant = meg_or[:12]
    print('------------------------------------------------------')
    print('---------------------- Starting participant', participant)
    print('------------------------------------------------------')
    results_folder = op.join(path_results, participant, 'reorder_random')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    cond2epochs = {}
    cond2raw   = {}

    # load or recalc filtered epochs
    p0 = op.join( os.path.expandvars('$TEMP_DATA_DEMARCHI') , meg_rd[:-15] )
    if op.exists(op.join(p0, 'flt_rd-epo.fif')) and (not force_refilt):
        print('!!!!!   Loading precomputed filtered raws and epochs from ',p0)
        raw_rd = mne.io.read_raw_fif(op.join(p0,'flt_rd-raw.fif'), preload=True)
        # keep only MEG channels
        raw_rd.pick_types(meg=True, eog=False, ecg=False,
                      ias=False, stim=False, syst=False)

        # actually read epochs and filtered raws
        for cond,condcode in cond2code.items():
            s = condcode
            cond2epochs[cond] = mne.read_epochs( op.join(p0, f'flt_{s}-epo.fif')) 

            raw_ = mne.io.read_raw_fif(op.join(p0,f'flt_{s}-raw.fif'), preload=True) 
            # keep only MEG channels
            raw_.pick_types(meg=True, eog=False, ecg=False,
                          ias=False, stim=False, syst=False)
            cond2raw[cond] = raw_

    else:
        print('!!!!!   (Re)compute filtered raws from ',p0)
        for cond,condcode in cond2code.items():
            fnf = op.join(path_data, subdf.to_dict('index')[cond]['path'] )
            # Read raw file
            raw = mne.io.read_raw_fif(fnf, preload=True)
            print(f'Filtering raw {fnf}')
            raw.filter(0.1, 30, n_jobs=-1)
            if not op.exists(p0):
                os.makedirs(p0)
            raw.save( op.join(p0, f'flt_{condcode}-raw.fif'), overwrite = True )
            # Get events
            events = mne.find_events(raw, shortest_event=1)
            # keep only MEG channels
            raw.pick_types(meg=True, eog=False, ecg=False,
                          ias=False, stim=False, syst=False)
            cond2raw[cond] = raw

            # Create epochs
            epochs = mne.Epochs(raw, events,
                                event_id=events_all,
                                tmin=tmin, tmax=tmax, baseline=None, preload=True)
            epochs.save( op.join(p0, f'flt_{condcode}-epo.fif'), overwrite=True)
            cond2epochs[cond] = epochs

        raw_or = cond2raw['ordered']
        raw_rd = cond2raw['random']


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

    epochs_rd_init = cond2epochs['random'].copy()

    cond2epochs_reord = {}
    cond2orig_nums_reord = {}

    cond2epochs_sp_reord = {}
    cond2orig_nums_sp_reord = {}

    reorder_pars = dict(del_processed= del_processed, cut_fl=cut_fl, tmin=tmin, tmax=tmax, dur=dur, nsamples=nsamples)
    # cycle over four entropy conditions (targets of reordering)
    for cond,epochs in cond2epochs.items():
        # original random events
        random_events = epochs_rd_init.events.copy()
        # target events
        events0 = epochs.events.copy()
        
        # reorder random events to another entropy condition
        epochs_reord0, orig_nums_reord0 = reorder(random_events, events0, raw_rd, **reorder_pars) 
        cond2epochs_reord[cond] = epochs_reord0
        cond2orig_nums_reord[cond] = orig_nums_reord0

        cond2counts[cond+'_reord'] = Counter(cond2epochs_reord[cond].events[:,2])

        #########################
        ####   reorder simple prediction
        #########################

        # first we transform events from the current entropy condtion into it's "simple prediction" (most probable next event) verion 
        events = events_simple_pred(epochs.events.copy(), cond2code[cond])
        # then we do the reorderig like before, but in this case the target events are the transformed events, not the true ones
        epochs_reord, orig_nums_reord = reorder(random_events, events, raw_rd, **reorder_pars) 
        cond2epochs_sp_reord[cond] = epochs_reord
        cond2orig_nums_sp_reord[cond] = orig_nums_reord

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
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=gen_est_verbose)

    # cycle over entropies
    for cond,epochs in cond2epochs.items():
        print(f"-----  CV for {cond}")

        # keep only the same number of trials for all conditions
        epochs = epochs[:minl]  
        # get the X and Y for each condition in numpy array
        X = epochs.get_data()
        y_sp_ = events_simple_pred(epochs.events.copy() , cond2code[cond])
        y_sp = y_sp_[:, 2] 

        #----------
        epochs_reord = cond2epochs_reord[cond][:minl]
        orig_nums_reord = cond2orig_nums_reord[cond] 
        # TODO: find way to use both sp and not sp, reord and not

        # keep same trials in epochs_rd and epochs_reord
        epochs_rd1 = epochs_rd_init[orig_nums_reord][:minl]
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
        for train_rd, test_rd in cv.split(Xrd1, yrd1):
            print(f"##############  Starting {cond} fold")
            print('Lens of train and test are :',len(train_rd), len(test_rd) )
            # Run cross validation for the ordered (and reorder-order) (and keep the score on the random too only here)
            # Train and test with cross-validation
            clf.fit(Xrd1[train_rd], yrd1[train_rd])  # fit on random

            # to plot patterns later... not very useful in the end, they are too inconsistent
            if extract_filters_patterns:
                filters_, patterns_ = getFiltPat(clf)
                filters  += [filters_]
                patterns += [patterns_]


            # fit on random, test on random
            cv_rd_to_rd_score = clf.score(Xrd1[test_rd], yrd1[test_rd])
            # fit on random, test on order
            cv_rd_to__score = clf.score(X[test_rd], y0[test_rd])
            # fit on random, test on order simple pred
            cv_rd_to_sp_score = clf.score(X[test_rd], y_sp[test_rd])

            # DQ: is it good to restrict test number so much?
            if reord_narrow_test:
                test_reord = np.isin(orig_nums_reord, test_rd)  # why sum(test_reord) != len(test_rd)
                print('{} test_rd among orig_nums_reord. Total = {} '.format( len(test_reord), len(test_rd) ) )
                cv_rd_to_reord_score = clf.score(Xreord[test_reord], yreord[test_reord])
            else:
                cv_rd_to_reord_score = clf.score(Xreord[test_rd], yreord[test_rd])
                cv_rd_to_reord_sp_score = clf.score(Xreord[test_rd], yreord_sp[test_rd])

                # not used so far
                cv_rd_to_sp_reord_score = clf.score(Xreord[test_rd], ysp_reord[test_rd])

            dadd(scores,'rd_to_rd',cv_rd_to_rd_score      )
            dadd(scores,f'rd_to_{s}',cv_rd_to__score        )
            dadd(scores,f'rd_to_{s}_sp',cv_rd_to_sp_score        )

            dadd(scores,f'rd_to_{s}_reord',cv_rd_to_reord_score   )
            dadd(scores,f'rd_to_{s}_reord_sp',cv_rd_to_reord_sp_score   )
            dadd(scores,f'rd_to_{s}_sp_reord',cv_rd_to_sp_reord_score   )
            #'cv'
        filters_rd,patterns_rd = np.array(filters), np.array(patterns)

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
            fnf = op.join(results_folder, f'cv_{k}_scores.npy' )
            print('Saving ',fnf)
            np.save(fnf , v )

        # clean
        import gc; gc.collect()
