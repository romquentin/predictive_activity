import numpy as np
import pandas as pd
import seaborn as sns
import mne

# fixed ordering of subject ids, for convenience
canon_subj_order = [
    '19830114RFTM', '19921111BRHC', '19930630MNSU', '19930118IMSH',
    '19861231ANSR', '19851130EIFI', '19930524GNTA', '19871229CRBC',
    '19940403HIEC', '19900606KTAD', '19880328AGSG', '19950326IIQI',
    '19750430PNRK', '19950212BTKC', '19930423EEHB', '19960418GBSH',
    '11920812IONP', '19950604LLZM', '19800616MRGU', '19950905MCSH',
    '19891222GBHL', '19940930SZDT', '19960304SBPE', '19821223KRHR',
    '19920804CRLE', '19810726GDZN', '19960708HLHY', '19810918SLBR',
    '19940601IGSH', '19961118BRSH', '19901026KRKE', '19930621ATLI',
    '19910823SSLD'
]
corresp = dict(zip(canon_subj_order, np.arange(33)))
events_omission = [10, 20, 30, 40]
events_sound = [1, 2, 3, 4]

# define transition matrices for different entropy conditions
Mor = np.array([
    [0.25, 0.75, 0.00, 0.00],
    [0.00, 0.25, 0.75, 0.00],
    [0.00, 0.00, 0.25, 0.75],
    [0.75, 0.00, 0.00, 0.25]
])
Mmp = np.array([
    [0.25, 0.60, 0.15, 0.00],
    [0.00, 0.25, 0.60, 0.15],
    [0.15, 0.00, 0.25, 0.60],
    [0.60, 0.15, 0.00, 0.25]
])
Mmm = np.array([
    [0.25, 0.38, 0.37, 0.00],
    [0.00, 0.25, 0.38, 0.37],
    [0.37, 0.00, 0.25, 0.38],
    [0.38, 0.37, 0.00, 0.25]
])
Mt2M = {'or': Mor, 'mp': Mmp, 'mm': Mmm}


def events_simple_pred(events, Mtype):
    """
    Generate simple prediction events based on the transition matrix.

    Parameters:
    events -- an MNE-produced events array
    Mtype -- string argument, a key in Mt2M dict or 'rd', determines the matrix type

    Returns:
    Transformed events array with simple predictions.
    """
    assert events.ndim == 2
    r = np.zeros_like(events)

    assert np.min(events[:, 2]) == 1
    assert np.max(events[:, 2]) == 4

    if Mtype == 'rd':  # No simple prediction for random condition
        return events.copy()

    M = Mt2M[Mtype]
    r[:, :2] = events[:, :2]
    r[0, 2] = events[0, 2]

    for i in range(1, events.shape[0]):
        prev_stim = events[i - 1, 2]
        pred_stim = M[:, prev_stim - 1].argmax()  # Most probable event index
        r[i, 2] = pred_stim + 1  # Adjust for 1-based indexing in events_sound

    return r


def reorder(random_events, events, raw_rd, dur=200, nsamples=33):
    """
    Reorder random events to match the given sequence.

    Parameters:
    random_events -- array of random events
    events -- target sequence of events
    raw_rd -- raw data corresponding to random events
    dur -- duration (in samples) of pre-task and post-task data
    nsamples -- window duration in samples

    Returns:
    Reordered raw data, original random event indices, and reordered events.
    """
    events = list(events)
    orig_nums = []
    events_reord = []
    raw_Xrd = raw_rd.get_data()
    raw_reord = []
    new_sample = 0
    first_samp = raw_rd.first_samp

    # Start reordered random with the first seconds of random raw
    raw_reord.append(raw_Xrd[:, :dur])
    new_sample += dur

    random_events_numbers = np.arange(len(random_events))
    for event in events:
        if event[2] in random_events[:, 2]:
            index = random_events[:, 2].tolist().index(event[2])
            orig_nums.append(random_events_numbers[index])
            samp = random_events[index, 0] - first_samp
            raw_reord.append(raw_Xrd[:, samp:samp + nsamples])
            random_events = np.delete(random_events, index, axis=0)
            random_events_numbers = np.delete(random_events_numbers, index, axis=0)
            events_reord.append([new_sample, 0, event[2]])
            new_sample += nsamples
        else:
            break

    # End reordered random with the last seconds of random raw
    raw_reord.append(raw_Xrd[:, -dur:])
    orig_nums_reord = np.array(orig_nums)
    events_reord = np.array(events_reord)
    raw_reord = np.concatenate(raw_reord, axis=1)
    raw_reord = mne.io.RawArray(raw_reord, raw_rd.info)

    return raw_reord, orig_nums_reord, events_reord


def check_intersection(train_samples, test_samples):
    if max(test_samples) < min(train_samples):  # Test indices are all below train indices
        return all(t + 66 <= min(train_samples) for t in test_samples)
    elif min(test_samples) > max(train_samples):  # Test indices are all above train indices
        return all(t - 99 >= max(train_samples) for t in test_samples)
    else:  # Test indices are inside train indices
        train_below = [t for t in train_samples if t < min(test_samples)]
        train_above = [t for t in train_samples if t > max(test_samples)]
        return (max(test_samples) + 66 <= min(train_above) if train_above else True) and \
               (min(test_samples) - 99 >= max(train_below) if train_below else True)