import numpy as np
import pandas as pd
import seaborn as sns
import mne

# fixed ordering os subject ids, for convenience
canon_subj_order =  ['19830114RFTM', '19921111BRHC', '19930630MNSU', '19930118IMSH',
       '19861231ANSR', '19851130EIFI', '19930524GNTA', '19871229CRBC',
       '19940403HIEC', '19900606KTAD', '19880328AGSG', '19950326IIQI',
       '19750430PNRK', '19950212BTKC', '19930423EEHB', '19960418GBSH',
       '11920812IONP', '19950604LLZM', '19800616MRGU', '19950905MCSH',
       '19891222GBHL', '19940930SZDT', '19960304SBPE', '19821223KRHR',
       '19920804CRLE', '19810726GDZN', '19960708HLHY', '19810918SLBR',
       '19940601IGSH', '19961118BRSH', '19901026KRKE', '19930621ATLI',
       '19910823SSLD']
corresp = dict( zip(canon_subj_order, np.arange(33) ) )
events_omission = [10,20,30,40]
events_sound = [ 1,2,3,4]

cond2code = dict(zip(['random','midminus','midplus','ordered'],['rd','mm','mp','or']))

colors_ordered = dict(zip(np.arange(4),  ['blue','cyan','yellow','red'] ) )
###### define transition matricies for different entropy conditions
# ordered
Mor = np.array([[0.25, 0.75, 0.  , 0.  ],
                [0.  , 0.25, 0.75, 0.  ],
                [0.  , 0.  , 0.25, 0.75],
                [0.75, 0.  , 0.  , 0.25]])

# meidum-plus
Mmp = np.array([[ 25, 60, 15, 0],
                [0, 25, 60, 15],
                [15, 0, 25, 60],
                [60, 15, 0, 25]])/100.

# meidum-minus
Mmm = np.array( [[25, 38, 37, 0  ],
                [ 0 , 25, 38, 37 ], 
                [ 37, 0 , 25, 38 ], 
                [ 38, 37, 0 , 25 ]] )/100.

Mt2M = {}
Mt2M['or'] = Mor
Mt2M['mp'] = Mmp
Mt2M['mm'] = Mmm

def events_simple_pred(events, Mtype):
    '''
    eventgs -- an MNE-produced events array
    Mtype -- string argument, a key in Mt2M dict or 'rd', it determins for what kind of matrix we generate the 'simple prediction'
    '''
    assert events.ndim == 2
    # this will be the resulting transformed events array
    r = np.zeros_like(events)

    assert np.min(events[:,2]) == 1
    assert np.max(events[:,2]) == 4

    # we are not supposed to have any bias in completely random condition, so no 'simple prediction' is possible
    if Mtype == 'rd':
        return events.copy()
    M = Mt2M[Mtype]

    # this is fair copying of data, not just playing with pointers, so I can safely modify r later within affecting events
    r[:,:2] = events[:,:2] 
    r[0,2]  = events[0,2]
    for i in range(1, events.shape[0]):
        prev_stim = events[i-1,2]
        # set the most probably event index to be the predicted one
        pred_stim = M[:,prev_stim-1].argmax() 
        # +1 is needed buecuse even types in events_sound are starting from 1
        # whereas python indexing starts from 0
        r[i,2] = pred_stim + 1  
    return r


def reorder(random_events, events, raw_rd, del_processed = True,
        cut_fl = 0, dur=200, nsamples = 33, tmin=-0.7, tmax=0.7):
    events = list(events)
    '''
    random events,
    target events 
    raw_rd: raw object for random sequence, sfreq = 100 Hz
    cut_fl: whether we cut out first and last events from the final result
    dur : duration (in samples) of pre-task and post-task data 
    nsamples: trial duration in samples
    tmin, tmax -- time offsets with respect to event onset, used to construct resulting epochs
    del_processed: basically determines version of the algorithm. =False is supposed to be better but it does not work well, so is supposed to be never used

    returns epochs_reord, orig_nums_reord
    '''

    assert abs(raw_rd.info['sfreq'] - 100. ) < 1e-10, raw_rd.info['sfreq']

    events_reord = list()  # the reordered events we will construct 
    # prepare raw data
    raw_Xrd = raw_rd.get_data()
    raw_reord = list()  # the reordered X (based on yor), first contains data extracted from raws
    new_sample = 0  # keep track of the current sample to create the reordered events
    # DQ: why 200?
    raw_reord.append(raw_Xrd[:, :dur])  # start the reorderd random with the 2 first seconds of the random raw
    first_samp = raw_rd.first_samp
    # keep the original trial numbers in the random (for correct cross-validation and also comparison with the same not-reordered random trials)
    #random_events_processed = []
    orig_inds = list() # (roughly) permutation of the original random event indices
    new_sample += dur # sample where we will put the random event

    # Romain's version
    if del_processed:
        random_events_indices = np.arange(len(random_events)) # indices of random events
        #for event in tqdm(events):
        for event in events:
            # event[2] is the event code
            # note that random_events changes on every iteration potentially
            # random events is actually _yet unprocessed_ random events
            if event[2] in random_events[:, 2]:
                # take the index of the ordered event as it is present in random events (need to delete it later)
                # index of first not processed with save code
                index = random_events[:, 2].tolist().index(event[2])

                # save the index of the original (not reordered) random event 
                orig_inds.append(random_events_indices[index])
                # correctly offsetted sample
                samp = random_events[index, 0] - first_samp
                raw_reord.append(raw_Xrd[:, samp:samp+nsamples])

                random_events         = np.delete(random_events,         index, axis=0)
                random_events_indices = np.delete(random_events_indices, index, axis=0)

                # putting target event to the new sample
                events_reord.append([new_sample, 0, event[2]])
                new_sample += nsamples
            else:
                pass
    else:
        raise ValueError('need further debugging before using')
        random_events_aug = np.concatenate([  random_events, 
            np.arange(len(random_events))[:,None]], axis=1 )
        was_processed = np.zeros(len(random_events), dtype=bool )
        for event in tqdm(events):
            random_events_aug_sub = random_events_aug[~was_processed]
            inds = np.where( random_events_aug_sub[:,2] == event[2])[0]
            #inds2 = np.where(~was_processed[inds] )[0]
            if len(inds) == 0:
                continue
            else:
                evt = random_events_aug_sub[inds[0]]
                index = evt[3]  # index of random event in orig array

                orig_inds.append(index)
                samp = evt[0] - first_samp
                raw_reord.append(raw_Xrd[:, samp:samp+nsamples])

                was_processed[index] = True

                # simple artificial sample indices
                events_reord.append([new_sample, 0, event[2]])
                new_sample+=nsamples

    raw_reord.append(raw_Xrd[:, -dur:])  # end the reorderd random with the 2 last seconds of the random raw

    # will be used to define epochs_rd
    orig_inds_reord = np.array(orig_inds)  
    events_reord = np.array(events_reord)  
    # removing the first and last trials
    if cut_fl:
        orig_inds_reord = orig_inds_reord[1:-1]
        events_reord    = events_reord[1:-1]
    raw_reord = np.concatenate(raw_reord, axis=1)
    raw_reord = mne.io.RawArray(raw_reord, raw_rd.info)

    epochs_reord = mne.Epochs(raw_reord, events_reord,
         event_id=events_sound,
         tmin=tmin, tmax=tmax, baseline=None, preload=True)

    assert len(set(orig_inds_reord)) > 4
    return epochs_reord, orig_inds_reord

# for filter patterns extraction
def getFiltPat(genclf):
    from mne.decoding import get_coef
    filters_ = [get_coef(genclf.estimators_[i],'filters_') for i in range(len(genclf.estimators_))]
    filters_ = np.array(filters_)  # times x channels x classes

    patterns_ = [get_coef(genclf.estimators_[i],'patterns_') for i in range(len(genclf.estimators_))]
    patterns_ = np.array(patterns_)  # times x channels x classes

    return filters_, patterns_

# not used 
def runlines(fnf,line_range_start,line_range_end):
    # range is inclusive
    with open(fnf,'r') as f:
        lines = f.readlines()

    if line_range_end >= 0:
        endind = line_range_end+1
    else:
        endind = line_range_end
    sublines = lines[line_range_start:endind]
    
    # remove indent as of the first line
    l0 = sublines[0]
    n = -1
    for i in range(len(l0)):
        if not l0[i].isspace():
            n = i
            break
    sublines = [s[n:] for s in sublines]
    code = ''.join(sublines)
    
    del lines, sublines, n, l0
    exec(code)
    return locals()

def dadd(d,k,v):
    '''
    helper function, check if the key is in the dict, if not -- creates it, otherwise add to it to the list

    d is a dict with values having type = list
    k is a key (usually string)
    v is a value
    '''
    if k in d:
        d[k] += [v]
    else:
        d[k] = [v]

def addEpindChan(epochs, inds):
    assert inds.ndim == 1
    inds = inds.reshape(-1,1)
    preeps = np.repeat(inds[:,:,None], epochs._data.shape[2], axis=2)

    # at every time point we put the epoch number
    # use 'mag' as channel type so that no intemediate function kill this channel later
    info3 = mne.create_info(epochs.info.ch_names + ['EPNUM'], epochs.info['sfreq'], 
                            ch_types=epochs.get_channel_types() + ['mag'], verbose=None)
    preeps_full = np.concatenate([epochs._data, preeps], axis=1)
    eps_full = mne.EpochsArray(preeps_full, info3, tmin=0)
    eps_full.events = epochs.events

    return eps_full

def addSampleindChan(raw):
    index_info = mne.create_info(ch_names=['index_chan'], sfreq=raw.info['sfreq'], ch_types=['stim'])
    index_data = np.arange(raw._data.shape[1]).reshape(1,-1)
    index_raw = mne.io.RawArray(index_data, index_info)
    raw.add_channels([index_raw], force_update_info=True)
    return raw