import pyxdf
import bisect
import numpy as np
import os

def locate_pos(available, targets):
    """Function to get indices 
    
    Parameters
    ----------
    available: 
    targets: 
    Returns
    ----------
    pos
    """
    pos = bisect.bisect_right(available, targets)
    if pos == 0:
        return 0
    if pos == len(available):
        return len(available)-1
    if abs(available[pos]-targets) < abs(available[pos-1]-targets):
        return pos
    else:
        return pos-1  

def epoching_indices(task_time_series, seeg_time_series, indices, win_start=0, win_length=1, sr=1024):
    """ Get start and end indices of epochs
    
    Parameters
    ----------
    task_time_series: array (samples)
        time stamps of task events
    seeg_time_series: array (samples)
        time stamps of EEG data
    indices: list 
        indices of time point at wich event of interest occurred 
    win_start: int 
        offset from each event of interest in indices in seconds (determines start of epoch)
    win_length: int
        length of epoch in seconds
    sr: int
        sampling rate

    Returns
    ----------
    epoch : array (samples, channels, bands)
        filtered EEG time series   
    """
    epoch_start = np.array([locate_pos(seeg_time_series, x) for x in task_time_series[indices]]) + int(win_start*sr)
    epoch_end = np.array([locate_pos(seeg_time_series, x) for x in seeg_time_series[epoch_start]]) + int(win_length*sr)

    return epoch_start, epoch_end

if __name__=='__main__':
    PPs = ['kh25'] #'kh21', 'kh22', 'kh23', 'kh24', 
    bands = dict(zip(['delta', 'theta', 'alpha', 'beta', 'highGamma'], [(1,3), (4,7), (8,12), (13,30), (70,120)]))
    reref = 'ESR'
    sr=1024

    for pp in PPs:
        data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/RawData/'
        envelope_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Envelope/{}/'.format(pp)
        out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Epoching/{}/'.format(pp)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    
        # Load data, seeg and timestamps, sampling rate
        envelopes = np.load(envelope_path + '{}_seeg_{}_envelopes.npy'.format(pp, reref)) # np.array samples*channels*bands
        data, _ = pyxdf.load_xdf(data_path + '{}_test.xdf'.format(pp))
        if '25' in pp:
            seeg_ts = data[0]['time_stamps']
            task_ts = data[1]['time_stamps']
            markers = data[1]['time_series']
            sr = int(float(data[0]['info']['nominal_srate'][0]))

        else:
            seeg_ts = data[1]['time_stamps']
            task_ts = data[0]['time_stamps']
            markers = data[0]['time_series']
            sr = int(float(data[1]['info']['nominal_srate'][0]))

            
        # Define epochs indexes for each window type start and end points
        epochs = {
        'feedback':{'indices':[i for i, x in enumerate(markers) if 'start Fb' in x[0]], 'start':0, 'length':1},
        'baseline':{'indices':[i for i, x in enumerate(markers) if 'Start Cross' in x[0]], 'start':0, 'length':0.5},
        'long_FB':{'indices':[i for i, x in enumerate(markers) if 'Start Cross' in x[0]], 'start':0, 'length':2},
        'stimulus':{'indices':[i for i, x in enumerate(markers) if 'Start Stim' in x[0]], 'start':0, 'length':1},
        'long_stim':{'indices':[i for i, x in enumerate(markers) if 'Start Trial' in x[0]], 'start':0, 'length':2},
        'response':{'indices':[i for i, x in enumerate(markers) if ('Press' in x[0] and 'wrong' not in x[0])], 'start':-0.5, 'length':1}
        }

        # Get start and end indices for each epoch
        for key, val in epochs.items():
            val['start_ind'], val['end_ind'] = epoching_indices(task_ts, seeg_ts, val['indices'], win_start=val['start'], win_length=val['length'], sr=sr)
            n_trials = len(val['indices'])
            w_len = val['end_ind'][0] - val['start_ind'][0]

        # Epoching and Store envelope data into trial x samples x channel x band array for each epoch
            envelope_epoched = np.zeros((n_trials, w_len, envelopes.shape[1], envelopes.shape[2]))
            for trial, start in enumerate(val['start_ind']):
                end = int(val['end_ind'][trial])
                envelope_epoched[trial, :, :, :] = envelopes[start:end, :, :]
            
        # Save data
            np.save(out_path + '{}_{}_envelope_epoched_{}'.format(pp,reref,key), envelope_epoched)