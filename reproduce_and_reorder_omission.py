import os.path as op
import os, sys
import numpy as np
import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
from tqdm import tqdm
from base import corresp

from base import corresp
from base import events_simple_pred
from base import cond2code

path_data = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/MEG'
path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'
# define tmin and tmax
tmin, tmax = -0.7, 0.7
crop_twice = 0
v2only = 1

# Import the argparse module
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add the arguments to the parser
parser.add_argument("--subject", type=int, default =-1, required=True)
parser.add_argument("--force_refilt", type=int, default=0)
parser.add_argument("--exit_after", type=str, default='end')

# Parse the arguments
args = parser.parse_args()
force_refilt = args.force_refilt

sids_to_use = [args.subject]

import pandas as pd
rows = [ dict(zip(['subj','block','cond','path'], v[:-15].split('_') + [v])  ) for v in os.listdir(path_data)]
df = pd.DataFrame(rows)
df['sid'] = df['subj'].apply(lambda x: corresp[x])

df = df.query('sid.isin(@sids_to_use)')
#TODO: run with arg of bad subject

grp = df.groupby(['subj'])
assert grp.size().min() == grp.size().max()
assert grp.size().min() == 4

for g,inds in grp.groups.items():
    subdf = df.loc[inds]

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
    results_folder = op.join(path_results, participant, 'reorder_random_omission')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    cond2epochs = {}
    cond2raw   = {}

    p0 = op.join( os.path.expandvars('$SCRATCH/memerr/demarchi') , meg_rd[:-15] )
    if op.exists(op.join(p0, 'flt_rd-epo.fif')) and (not force_refilt):
        print('!!!!!   Loading precomputed filtered epochs from ',p0)
        raw_rd = mne.io.read_raw_fif(op.join(p0,'flt_rd-raw.fif'), preload=True)
        raw_rd.pick_types(meg=True, eog=False, ecg=False,
                      ias=False, stim=False, syst=False)

        for cond,condcode in cond2code.items():
            s = condcode
            cond2epochs[cond] = mne.read_epochs( op.join(p0, f'flt_{s}-epo.fif')) 

            raw_ = mne.io.read_raw_fif(op.join(p0,f'flt_{s}-raw.fif'), preload=True) 
            raw_.pick_types(meg=True, eog=False, ecg=False,
                          ias=False, stim=False, syst=False)
            cond2raw[cond] = raw_

    else:
        print('!!!!!   (Re)compute filtered raws from ',p0)
        for cond,condcode in cond2code.items():
            fnf = op.join(path_data, subdf.to_dict('index')[cond] )
            # Read raw files
            raw = mne.io.read_raw_fif(fnf, preload=True)
            print('Filtering ')
            raw.filter(0.1, 30, n_jobs=-1)
            if not op.exists(p0):
                os.makedirs(p0)
            raw.save( op.join(p0, f'flt_{condcode}-raw.fif'), overwrite = True )
            raw.pick_types(meg=True, eog=False, ecg=False,
                          ias=False, stim=False, syst=False)
            cond2raw[cond] = raw
            # Get events
            events = mne.find_events(raw, shortest_event=1)

            # Create epochs
            epochs = mne.Epochs(raw, events,
                                event_id=events_all,
                                tmin=tmin, tmax=tmax, baseline=None, preload=True)
            epochs.save( op.join(p0, f'flt_{condcode}-epo.fif'), overwrite=True)
            cond2epochs[cond] = epochs

        raw_rd = cond2raw['random']

    # Save an epochs_rd to start from at each iteration on the decoding the reorders
    epochs_rd = cond2epochs['random']
    epochs_or = cond2epochs['ordered']
    epochs_mm = cond2epochs['midminus']
    epochs_mp = cond2epochs['midplus']
    raw_rd = cond2raw['random']

    epochs_rd_init = epochs_rd.copy()
    
    # reorder random raw as ordered
    # get events for random and ordered and initialize new events for reordered data
    random_events = epochs_rd.events.copy()
    ordered_events = list(epochs_or.events)
    events_orrd = list()  # the reordered events (based on yor)
    # prepare raw data
    raw_Xrd = raw_rd.get_data()
    raw_Xorrd = list()  # the reordered X (based on yor)
    new_sample = 0  # keep track of the current sample to create the reordered events
    raw_Xorrd.append(raw_Xrd[:, :200])  # start the reorderd random with the 2 first seconds of the random raw
    first_samp = raw_rd.first_samp
    # keep the original trial numbers in the random (for correct cross-validation and also comparison with the same not-reordered random trials)
    random_events_numbers = np.arange(len(random_events))
    orig_nums = list()
    new_sample+=200
    for event in tqdm(ordered_events):
        if event[2] in random_events[:, 2]:
            index = random_events[:, 2].tolist().index(event[2])
            if event[2] in [10, 20, 30, 40]:
                orig_nums.append(random_events_numbers[index])
            samp = random_events[index, 0] - first_samp
            raw_Xorrd.append(raw_Xrd[:, samp:samp+33])
            random_events = np.delete(random_events, index, axis=0)
            random_events_numbers = np.delete(random_events_numbers, index, axis=0)
            events_orrd.append([new_sample, 0, event[2]])
            new_sample+=33
        else:
            pass
    raw_Xorrd.append(raw_Xrd[:, -200:])  # end the reorderd random with the 2 last seconds of the random raw
    orig_nums_orrd = np.array(orig_nums)#[1:-1]  # removing the first and last trials
    events_orrd = np.array(events_orrd)#[1:-1]  # removing the first and last trials
    raw_Xorrd = np.concatenate(raw_Xorrd, axis=1)
    raw_orrd = mne.io.RawArray(raw_Xorrd, raw_rd.info)

    # reorder random raw as midplus
    # get events for random and ordered and initialize new events for reordered data
    random_events = epochs_rd.events.copy()
    midplus_events = list(epochs_mp.events)
    events_mprd = list()  # the reordered events (based on yor)
    # prepare raw data
    raw_Xrd = raw_rd.get_data()
    raw_Xmprd = list()  # the reordered X (based on yor)
    new_sample = 0  # keep track of the current sample to create the reordered events
    raw_Xmprd.append(raw_Xrd[:, :200])  # start the reorderd random with the 2 first seconds of the random raw
    first_samp = raw_rd.first_samp
    # keep the original trial numbers in the random (for correct cross-validation and also comparison with the same not-reordered random trials)
    random_events_numbers = np.arange(len(random_events))
    orig_nums = list()
    new_sample+=200
    for event in tqdm(midplus_events):
        if event[2] in random_events[:, 2]:
            index = random_events[:, 2].tolist().index(event[2])
            if event[2] in [10, 20, 30, 40]:
                orig_nums.append(random_events_numbers[index])
            samp = random_events[index, 0] - first_samp
            raw_Xmprd.append(raw_Xrd[:, samp:samp+33])
            random_events = np.delete(random_events, index, axis=0)
            random_events_numbers = np.delete(random_events_numbers, index, axis=0)
            events_mprd.append([new_sample, 0, event[2]])
            new_sample+=33
        else:
            pass
    raw_Xmprd.append(raw_Xrd[:, -200:])  # end the reorderd random with the 2 last seconds of the random raw
    orig_nums_mprd = np.array(orig_nums)#[1:-1]  # removing the first and last trials
    events_mprd = np.array(events_mprd)#[1:-1]  # removing the first and last trials
    raw_Xmprd = np.concatenate(raw_Xmprd, axis=1)
    raw_mprd = mne.io.RawArray(raw_Xmprd, raw_rd.info)

    # reorder random raw as midminus
    # get events for random and ordered and initialize new events for reordered data
    random_events = epochs_rd.events.copy()
    midminus_events = list(epochs_mm.events)
    events_mmrd = list()  # the reordered events (based on yor)
    # prepare raw data
    raw_Xrd = raw_rd.get_data()
    raw_Xmmrd = list()  # the reordered X (based on yor)
    new_sample = 0  # keep track of the current sample to create the reordered events
    raw_Xmmrd.append(raw_Xrd[:, :200])  # start the reorderd random with the 2 first seconds of the random raw
    first_samp = raw_rd.first_samp
    # keep the original trial numbers in the random (for correct cross-validation and also comparison with the same not-reordered random trials)
    random_events_numbers = np.arange(len(random_events))
    orig_nums = list()
    new_sample+=200
    for event in tqdm(midminus_events):
        if event[2] in random_events[:, 2]:
            index = random_events[:, 2].tolist().index(event[2])
            if event[2] in [10, 20, 30, 40]:
                orig_nums.append(random_events_numbers[index])
            samp = random_events[index, 0] - first_samp
            raw_Xmmrd.append(raw_Xrd[:, samp:samp+33])
            random_events = np.delete(random_events, index, axis=0)
            random_events_numbers = np.delete(random_events_numbers, index, axis=0)
            events_mmrd.append([new_sample, 0, event[2]])
            new_sample+=33
        else:
            pass
    raw_Xmmrd.append(raw_Xrd[:, -200:])  # end the reorderd random with the 2 last seconds of the random raw
    # DQ: why removing first and last?
    orig_nums_mmrd = np.array(orig_nums)#[1:-1]  # removing the first and last trials
    events_mmrd = np.array(events_mmrd)#[1:-1]  # removing the first and last trials
    raw_Xmmrd = np.concatenate(raw_Xmmrd, axis=1)
    raw_mmrd = mne.io.RawArray(raw_Xmmrd, raw_rd.info)

    ##################################################################
    ####################  V2
    ##################################################################
    print('Starting decoding orrd')

    # Initialize classifier 
    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    # Train classifiers on random sounds
    epochs_rd_sound = epochs_rd_init['1', '2', '3', '4']
    Xrd_sound = epochs_rd_sound.get_data()
    yrd_sound = epochs_rd_sound.events[:, 2]
    clf.fit(Xrd_sound, yrd_sound)
    
    # Test for order (and reorder-order)
    # create an epoch from the reordered raw random 
    epochs_orrd = mne.Epochs(raw_orrd, events_orrd,
                             event_id=[10, 20, 30, 40],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # keep same trials in epochs_rd and epochs_orrd
    epochs_rd = epochs_rd_init[orig_nums_orrd]
    # keep only the same number of trials in ordered
    epochs_or = epochs_or['10', '20', '30', '40'][:len(epochs_rd)]
    # get the X and Y for each condition in numpy array
    # DQ: why / 10 ?
    Xor = epochs_or.get_data()
    yor = (epochs_or.events[:, 2]/10).astype(int)
    Xrd = epochs_rd.get_data()
    yrd = (epochs_rd.events[:, 2]/10).astype(int)
    Xorrd = epochs_orrd.get_data()
    yorrd = (epochs_orrd.events[:, 2]/10).astype(int)
    # test classifiers
    cv_rd_to_rd_score = clf.score(Xrd, yrd)
    cv_rd_to_or_score = clf.score(Xor, yor)
    cv_rd_to_orrd_score = clf.score(Xorrd, yorrd)
    # save scores
    np.save(op.join(results_folder, 'cv_rd_to_rd_scores.npy'), cv_rd_to_rd_score)
    np.save(op.join(results_folder, 'cv_rd_to_or_scores.npy'), cv_rd_to_or_score)
    np.save(op.join(results_folder, 'cv_rd_to_or_reord_scores.npy'), cv_rd_to_orrd_score)


    ##################################################################
    # Test for midminus (and reorder-midminus)
    # create an epoch from the reordered raw random 
    print('Starting decoding mmrd')

    epochs_mmrd = mne.Epochs(raw_mmrd, events_mmrd,
                             event_id=[10, 20, 30, 40],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # keep same trials in epochs_rd and epochs_mmrd
    epochs_rd = epochs_rd_init[orig_nums_mmrd]
    # keep only the same number of trials in ordered
    epochs_mm = epochs_mm['10', '20', '30', '40'][:len(epochs_rd)]
    # get the X and Y for each condition in numpy array
    Xmm = epochs_mm.get_data()
    ymm = (epochs_mm.events[:, 2]/10).astype(int)
    Xmmrd = epochs_mmrd.get_data()
    ymmrd = (epochs_mmrd.events[:, 2]/10).astype(int)

    # test classifiers (trained on random)
    cv_rd_to_mm_score = clf.score(Xmm, ymm)
    cv_rd_to_mmrd_score = clf.score(Xmmrd, ymmrd)
    # save scores
    np.save(op.join(results_folder, 'cv_rd_to_mm_scores.npy'), cv_rd_to_mm_score)
    np.save(op.join(results_folder, 'cv_rd_to_mm_reord_scores.npy'), cv_rd_to_mmrd_score)

    ##################################################################
    # Test for midplus (and reorder-midplus)
    # create an epoch from the reordered raw midplus 
    print('Starting decoding mprd')

    epochs_mprd = mne.Epochs(raw_mprd, events_mprd,
                             event_id=[10, 20, 30, 40],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # keep same trials in epochs_rd and epochs_mmrd
    epochs_rd = epochs_rd_init[orig_nums_mprd]
    # keep only the same number of trials in ordered
    epochs_mp = epochs_mp['10', '20', '30', '40'][:len(epochs_rd)]
    # get the X and Y for each condition in numpy array
    Xmp = epochs_mp.get_data()
    ymp = (epochs_mp.events[:, 2]/10).astype(int)
    Xmprd = epochs_mprd.get_data()
    ymprd = (epochs_mprd.events[:, 2]/10).astype(int)
    # test classifiers
    cv_rd_to_mp_score = clf.score(Xmp, ymp)
    cv_rd_to_mprd_score = clf.score(Xmprd, ymprd)
    # save scores
    np.save(op.join(results_folder, 'cv_rd_to_mp_scores.npy'), cv_rd_to_mp_score)
    np.save(op.join(results_folder, 'cv_rd_to_mp_reord_scores.npy'), cv_rd_to_mprd_score)

    if v2only:
        print("V2 (no CV) Finished sucessfully")
        sys.exit(0)

    ##################################################################
    ####################  CV
    ##################################################################

    # Run cross validation for the ordered (and reorder-order) (and keep the score on the random too only here)
    # create an epoch from the reordered raw random 
    epochs_orrd = mne.Epochs(raw_orrd, events_orrd,
                             event_id=[10, 20, 30, 40],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)
    if crop_twice:
        epochs_orrd = epochs_orrd[1:-1]
    # keep same trials in epochs_rd and epochs_orrd
    epochs_rd = epochs_rd_init[np.sort(orig_nums_orrd)]
    # keep only the same number of trials in ordered
    epochs_or = epochs_or[:len(epochs_rd)]
    # get the X and Y for each condition in numpy array
    Xor = epochs_or.get_data()
    yor = epochs_or.events[:, 2]
    Xrd = epochs_rd.get_data()
    yrd = epochs_rd.events[:, 2]
    Xorrd = epochs_orrd.get_data()
    yorrd = epochs_orrd.events[:, 2]
    # Initialize classifier 
    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    # Train and test with cross-validation
    cv_rd_to_rd_scores = list()
    cv_rd_to_or_scores = list()
    cv_rd_to_orrd_scores = list()
    cv = StratifiedKFold(5)
    for train_rd, test_rd in cv.split(Xrd, yrd):
        clf.fit(Xrd[train_rd], yrd[train_rd])
        cv_rd_to_rd_score = clf.score(Xrd[test_rd], yrd[test_rd])
        cv_rd_to_or_score = clf.score(Xor[test_rd], yor[test_rd])
        test_orrd = np.isin(orig_nums_orrd, test_rd)  # why sum(test_orrd) != len(test_rd) 
        cv_rd_to_orrd_score = clf.score(Xorrd[test_orrd], yorrd[test_orrd])
        cv_rd_to_rd_scores.append(cv_rd_to_rd_score)
        cv_rd_to_or_scores.append(cv_rd_to_or_score)
        cv_rd_to_orrd_scores.append(cv_rd_to_orrd_score)
    cv_rd_to_rd_scores = np.array(cv_rd_to_rd_scores)
    cv_rd_to_or_scores = np.array(cv_rd_to_or_scores)
    cv_rd_to_orrd_scores = np.array(cv_rd_to_orrd_scores)
    # save scores (cross-validation)
    np.save(op.join(results_folder, 'cv_rd_to_rd_scores.npy'), cv_rd_to_rd_scores)
    np.save(op.join(results_folder, 'cv_rd_to_or_scores.npy'), cv_rd_to_or_scores)
    np.save(op.join(results_folder, 'cv_rd_to_orrd_scores.npy'), cv_rd_to_orrd_scores)

    # Run cross validation for the midminus (and reorder-midminus)
    # create an epoch from the reordered raw random 
    epochs_mmrd = mne.Epochs(raw_mmrd, events_mmrd,
                             event_id=[10, 20, 30, 40],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)
    if crop_twice:
        epochs_mmrd = epochs_mmrd[1:-1]
    # keep same trials in epochs_rd and epochs_mmrd
    epochs_rd = epochs_rd_init[np.sort(orig_nums_mmrd)]
    # keep only the same number of trials in midminus
    epochs_mm = epochs_mm[:len(epochs_rd)]
    # get the X and Y for each condition in numpy array
    Xmm = epochs_mm.get_data()
    ymm = epochs_mm.events[:, 2]
    Xrd = epochs_rd.get_data()
    yrd = epochs_rd.events[:, 2]
    Xmmrd = epochs_mmrd.get_data()
    ymmrd = epochs_mmrd.events[:, 2]
    # Initialize classifier 
    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    # Train and test with cross-validation
    cv_rd_to_rd_scores = list()
    cv_rd_to_mm_scores = list()
    cv_rd_to_mmrd_scores = list()
    cv = StratifiedKFold(5)
    for train_rd, test_rd in cv.split(Xrd, yrd):
        clf.fit(Xrd[train_rd], yrd[train_rd])
        cv_rd_to_mm_score = clf.score(Xmm[test_rd], ymm[test_rd])
        test_mmrd = np.isin(orig_nums_mmrd, test_rd)  # why sum(test_mmrd) != len(test_rd) 
        cv_rd_to_mmrd_score = clf.score(Xmmrd[test_mmrd], ymmrd[test_mmrd]) # !error here beucae size of test_mmrd is 398 whereas Xmmrd.shape[0] is 397
        cv_rd_to_mm_scores.append(cv_rd_to_mm_score)
        cv_rd_to_mmrd_scores.append(cv_rd_to_mmrd_score)
    cv_rd_to_mm_scores = np.array(cv_rd_to_mm_scores)
    cv_rd_to_mmrd_scores = np.array(cv_rd_to_mmrd_scores)
    # save scores (cross-validation)
    np.save(op.join(results_folder, 'cv_rd_to_mm_scores.npy'), cv_rd_to_mm_scores)
    np.save(op.join(results_folder, 'cv_rd_to_mmrd_scores.npy'), cv_rd_to_mmrd_scores)

    # Run cross validation for the midplus (and reorder-midplus)
    # create an epoch from the reordered raw random 
    epochs_mprd = mne.Epochs(raw_mprd, events_mprd,
                             event_id=[10, 20, 30, 40],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)
    if crop_twice:
        epochs_mprd = epochs_mprd[1:-1]
    # keep same trials in epochs_rd and epochs_mprd
    epochs_rd = epochs_rd_init[np.sort(orig_nums_mprd)]
    # keep only the same number of trials in midplus
    epochs_mp = epochs_mp[:len(epochs_rd)]
    # get the X and Y for each condition in numpy array
    Xmp = epochs_mp.get_data()
    ymp = epochs_mp.events[:, 2]
    Xrd = epochs_rd.get_data()
    yrd = epochs_rd.events[:, 2]
    Xmprd = epochs_mprd.get_data()
    ymprd = epochs_mprd.events[:, 2]
    # Initialize classifier 
    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    # Train and test with cross-validation
    cv_rd_to_rd_scores = list()
    cv_rd_to_mp_scores = list()
    cv_rd_to_mprd_scores = list()
    cv = StratifiedKFold(5)
    for train_rd, test_rd in cv.split(Xrd, yrd):
        clf.fit(Xrd[train_rd], yrd[train_rd])
        cv_rd_to_mp_score = clf.score(Xmp[test_rd], ymp[test_rd])
        test_mprd = np.isin(orig_nums_mprd, test_rd)  # why sum(test_mprd) != len(test_rd) 
        cv_rd_to_mprd_score = clf.score(Xmprd[test_mprd], ymprd[test_mprd])
        cv_rd_to_mp_scores.append(cv_rd_to_mp_score)
        cv_rd_to_mprd_scores.append(cv_rd_to_mprd_score)
    cv_rd_to_mp_scores = np.array(cv_rd_to_mp_scores)
    cv_rd_to_mprd_scores = np.array(cv_rd_to_mprd_scores)
    # save scores (cross-validation)
    np.save(op.join(results_folder, 'cv_rd_to_mp_scores.npy'), cv_rd_to_mp_scores)
    np.save(op.join(results_folder, 'cv_rd_to_mprd_scores.npy'), cv_rd_to_mprd_scores)
