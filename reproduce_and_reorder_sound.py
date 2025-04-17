import os.path as op
import os
import sys
import numpy as np
import pandas as pd
import mne
from mne.decoding import GeneralizingEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

from base import (canon_subj_order, corresp, events_simple_pred,
                        events_omission, events_sound, reorder, check_intersection)
# Define the path to the data
os.environ['DEMARCHI_DATA_PATH'] = '/Users/romainquentin/Desktop/data/MEG_demarchi'
os.environ['TEMP_DATA_DEMARCHI'] = '/Users/romainquentin/Desktop/data/MEG_demarchi'
path_data = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/MEG'
path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'

# define cross-validation and classifier
cv = KFold(5)
clf = make_pipeline(LinearDiscriminantAnalysis())
clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy')

# define tmin and tmax, events to be used, duration of pre-task and post-task data and window duration
tmin_train, tmax_train = 0, 0.32
tmin_test, tmax_test = -0.666, 0.666
events_all = events_sound + events_omission  # event_ids to be used
reord_narrow_test = 0
dur = 200  # duration (in samples) of pre-task and post-task data  
nsamples = 33  # window duration in samples
# define the participants to be used
rows = [dict(zip(['subj', 'block', 'cond', 'path'], v[:-15].split('_') + [v])) for v in os.listdir(path_data) if v.endswith('.fif')]
df0 = pd.DataFrame(rows)
df = df0.copy()
df['sid'] = df['subj'].apply(lambda x: corresp.get(x, 'unknown'))
# which participant IDs to use
sids_to_use = np.arange(len(corresp))
df = df.query('sid.isin(@sids_to_use)')
# check we have complete data (all conditions for every participant)
grp = df.groupby(['subj'])
assert grp.size().min() == grp.size().max()
assert grp.size().min() == 4

# iterating over participants
for participant, inds in list(grp.groups.items()):
    subdf = df.loc[inds]
    # get paths to datasets for each entropy condition per subject
    subdf = subdf.set_index('cond')
    subdf = subdf.drop(columns=['subj', 'block', 'sid'])
    print('---------------------- Starting participant', participant)
    results_folder = op.join(path_results, participant)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    all_raws = {}
    all_reord_raws = {}
    all_events = {}
    all_events_reord = {}
    all_orig_nums = {}
    # load raw data and create epochs for each entropy condition
    preprocessed_folders = op.join(os.path.expandvars('$TEMP_DATA_DEMARCHI'), 'preprocessed', participant)
    if not op.exists(preprocessed_folders):
        os.makedirs(preprocessed_folders)
    for cond, condcode in zip(['random', 'midminus', 'midplus', 'ordered'], ['rd', 'mm', 'mp', 'or']):
        fnf = op.join(path_data, subdf.to_dict('index')[cond]['path'])
        # Read raw file
        raw = mne.io.read_raw_fif(fnf, preload=True)
        raw.filter(0.1, 30, n_jobs=-1)
        raw.save(op.join(preprocessed_folders, f'flt_{condcode}-raw.fif'), overwrite=True)
        # Get events
        events = mne.find_events(raw, shortest_event=1)
        # Remove occurrences of 10, 16, 20, 30, 32, 40 and the following event
        to_remove = np.where(np.isin(events[:, 2], [10, 16, 20, 30, 32, 40]))[0]
        to_remove = np.sort(np.concatenate([to_remove, to_remove + 1]))
        if to_remove[-1] >= len(events):  # Ensure no out-of-bounds index
            to_remove = to_remove[to_remove < len(events)]
        events = np.delete(events, to_remove, axis=0)
        all_raws[cond] = raw
        all_events[cond] = events
    # reorder raws
    for cond, condcode in zip(['random', 'midminus', 'midplus', 'ordered'], ['rd', 'mm', 'mp', 'or']):
        if cond == 'random':
            # no reordering for random condition
            all_reord_raws[cond] = all_raws['random']
            all_orig_nums['random'] = np.arange(len(all_events['random']))
            all_events_reord[cond] = all_events['random']
        else:
            raw_reord, orig_num, events_reord = reorder(all_events['random'], all_events[cond], all_raws['random'])
            all_reord_raws[cond] = raw_reord
            all_orig_nums[cond] = orig_num
            all_events_reord[cond] = events_reord

    # use the same number of trials for all conditions
    min_len = min(len(events) for events in all_orig_nums.values())
    # Cut all conditions of all_orig_nums and all_events_reord accordingly
    for cond in all_reord_raws:
        all_orig_nums[cond] = all_orig_nums[cond][:min_len]
        all_events_reord[cond] = all_events_reord[cond][:min_len]
        all_events[cond] = all_events[cond][:min_len]

    # get the simple prediction events for each condition
    all_events_sp = {}
    all_events_reord_sp = {}
    for cond, condcode in zip(['random', 'midminus', 'midplus', 'ordered'], ['rd', 'mm', 'mp', 'or']):
        all_events_sp[cond] = events_simple_pred(all_events[cond].copy(), condcode)
        all_events_reord_sp[cond] = events_simple_pred(all_events_reord[cond].copy(), condcode)

# ----------------  Demarchi reproduction
    # train on random and test on random and other conditions
    for cond, condcode in zip(['random', 'midminus', 'midplus', 'ordered'], ['rd', 'mm', 'mp', 'or']):
        scores = list()  # list to store the scores
        scores_sp = list()  # list to store the single prediction scores
        for train, test in cv.split(np.arange(len(all_events['random']))):
            # Adjust boundaries for test fold to avoid overlap with train when doing large epochs (i.e., tmin=-0.666, tmax=0.666)
            test_min = test.min()
            test_max = test.max()
            if test[0] == 0:  # First trial in test is the first one
                test = test[test <= test_max - 1]
            elif test[-1] == len(all_events['random']) - 1:  # Last trial in test is the last one
                test = test[test >= test_min + 2]
            else:  # General case
                test = test[(test <= test_max - 1) & (test >= test_min + 2)]
            # create epoch for test and train separately
            epochs_train = mne.Epochs(all_raws['random'], all_events['random'][train],
                                      event_id=[1, 2, 3, 4],
                                      tmin=tmin_train, tmax=tmax_train,
                                      baseline=None, preload=True)
            epochs_test = mne.Epochs(all_raws[cond], all_events[cond][test],
                                     event_id=[1, 2, 3, 4],
                                     tmin=tmin_test, tmax=tmax_test,
                                     baseline=None, preload=True)
            epochs_test_sp = mne.Epochs(all_raws[cond], all_events_sp[cond][test],
                                        event_id=[1, 2, 3, 4],
                                        tmin=tmin_test, tmax=tmax_test,
                                        baseline=None, preload=True)
            # keep only MEG channels
            epochs_train.pick_types(meg=True, eog=False, ecg=False,
                                    ias=False, stim=False, syst=False)
            epochs_test.pick_types(meg=True, eog=False, ecg=False,
                                   ias=False, stim=False, syst=False)
            epochs_test_sp.pick_types(meg=True, eog=False, ecg=False,
                                      ias=False, stim=False, syst=False)
            # fit and score under normal hypothesis (trying to decode the true label at time 0)
            clf.fit(epochs_train.get_data(), epochs_train.events[:, 2])
            score = clf.score(epochs_test.get_data(), epochs_test.events[:, 2])
            scores.append(score)
            # score under single prediction hypothesis (trying to decode the most probable label at time 0 based on the previous label)
            score_sp = clf.score(epochs_test_sp.get_data(), epochs_test_sp.events[:, 2])
            scores_sp.append(score_sp)
        scores = np.mean(scores, axis=0)
        scores_sp = np.mean(scores_sp, axis=0)
        # save the scores
        np.save(op.join(results_folder, 'newcv_rd_to_%s_scores.npy' % condcode), scores)
        np.save(op.join(results_folder, 'newcv_rd_to_%s_sp_scores.npy' % condcode), scores_sp)

# ----------------  Empirical Null (using random data reordered into more ordered sequences)
    # train on random and test on random and reorder conditions (here the only data present are the random data)
    for cond, condcode in zip(['random', 'midminus', 'midplus', 'ordered'], ['rd', 'mm', 'mp', 'or']):
        scores = list()  # list to store the scores
        scores_sp = list()  # list to store the single prediction scores
        for train, test in cv.split(np.arange(len(all_events_reord[cond]))):
            # Adjust boundaries for test fold to avoid overlap with train when doing large epochs (i.e., tmin=-0.666, tmax=0.666)
            test_min = test.min()
            test_max = test.max()
            if test[0] == 0:  # First trial in test is the first one
                test = test[test <= test_max - 1]
            elif test[-1] == len(all_events['random']) - 1:  # Last trial in test is the last one
                test = test[test >= test_min + 2]
            else:  # General case
                test = test[(test <= test_max - 1) & (test >= test_min + 2)]
            # create epoch for test and train separately
            epochs_train = mne.Epochs(all_reord_raws[cond], all_events_reord[cond][train],
                                      event_id=[1, 2, 3, 4],
                                      tmin=tmin_train, tmax=tmax_train,
                                      baseline=None, preload=True)
            epochs_test = mne.Epochs(all_reord_raws[cond], all_events_reord[cond][test],
                                     event_id=[1, 2, 3, 4],
                                     tmin=tmin_test, tmax=tmax_test,
                                     baseline=None, preload=True)
            # Check for sample overlap between epochs_test and epochs_train
            train_samples = epochs_train.events[:, 0]
            test_samples = epochs_test.events[:, 0]
            assert check_intersection(train_samples, test_samples)
            epochs_test_sp = mne.Epochs(all_reord_raws[cond], all_events_reord_sp[cond][test],
                                        event_id=[1, 2, 3, 4],
                                        tmin=tmin_test, tmax=tmax_test,
                                        baseline=None, preload=True)
            # keep only MEG channels
            epochs_train.pick_types(meg=True, eog=False, ecg=False,
                                    ias=False, stim=False, syst=False)
            epochs_test.pick_types(meg=True, eog=False, ecg=False,
                                   ias=False, stim=False, syst=False)
            epochs_test_sp.pick_types(meg=True, eog=False, ecg=False,
                                      ias=False, stim=False, syst=False)
            # fit and score under normal hypothesis (trying to decode the true label at time 0)
            clf.fit(epochs_train.get_data(), epochs_train.events[:, 2])
            score = clf.score(epochs_test.get_data(), epochs_test.events[:, 2])
            scores.append(score)
            # score under single prediction hypothesis (trying to decode the most probable label at time 0 based on the previous label)
            score_sp = clf.score(epochs_test_sp.get_data(), epochs_test_sp.events[:, 2])
            scores_sp.append(score_sp)
        scores = np.mean(scores, axis=0)
        scores_sp = np.mean(scores_sp, axis=0)
        # save the scores
        np.save(op.join(results_folder, 'newcv_rd_to_reord_%s_scores.npy' % condcode), scores)
        np.save(op.join(results_folder, 'newcv_rd_to_reord_%s_sp_scores.npy' % condcode), scores_sp)
