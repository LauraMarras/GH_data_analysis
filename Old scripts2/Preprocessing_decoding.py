import pyxdf
import bisect
import numpy as np
import scipy as sy
import mne
import os

# Define function to get indices
def locate_pos(available, targets):
    pos = bisect.bisect_right(available, targets)
    if pos == 0:
        return 0
    if pos == len(available):
        return len(available)-1
    if abs(available[pos]-targets) < abs(available[pos-1]-targets):
        return pos
    else:
        return pos-1  

# Define function to filter data
def filterdata(data, filter, sr):
    sEEG_filtered = data.astype(float)
    if filter == 'gamma':
        sEEG_filtered = mne.filter.filter_data(sEEG_filtered.T, sr, 70, 120, method='iir').T
        filter_used = 'gamma'
    elif filter == 'theta':
        sEEG_filtered = mne.filter.filter_data(sEEG_filtered.T, sr, 4, 7, method='iir').T
        filter_used = 'theta'
    elif filter == 'delta':
        sEEG_filtered = mne.filter.filter_data(sEEG_filtered.T, sr, 1, 3, method='iir').T
        filter_used = 'delta'
    elif filter == 'alpha':
        sEEG_filtered = mne.filter.filter_data(sEEG_filtered.T, sr, 8, 12, method='iir').T
        filter_used = 'alpha'
    elif filter == 'beta':
        sEEG_filtered = mne.filter.filter_data(sEEG_filtered.T, sr, 13, 30, method='iir').T
        filter_used = 'beta'
    elif filter == 'lowpass':
        sEEG_filtered = mne.filter.filter_data(sEEG_filtered.T, sr, 0.5, 30, method='iir').T
        filter_used = 'lowpass'
    elif filter == 'superlowpass':
        sEEG_filtered = mne.filter.filter_data(sEEG_filtered.T, sr, 0.5, 7, method='iir').T
        filter_used = 'superlowpass'
    return sEEG_filtered, filter_used

# Define function to implement laplacian rereferencing
def laplacianR(data, channels, SIZE=2):
    """Apply a laplacian re-reference to the data
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    channels: array (electrodes, label)
        Channel names
    SIZE: int
        Size of the laplacian (amount of channels to include in average surrounding the channel)
    
    Returns
    ----------
    data: array (samples, channels)
        Laplacian re-referenced EEG time series   
    channels_des: array (electrodes, label)
        Channel name description (CHAN - CHAN±SIZE)
    """
    data_LPR = np.zeros((data.shape[0], data.shape[1]))
    #get shaft information
    shafts = {}
    for i,chan in enumerate(channels):
        if chan.rstrip('0123456789') not in shafts:
            shafts[chan.rstrip('0123456789')] = {'start': i, 'size': 1}
        else:
            shafts[chan.rstrip('0123456789')]['size'] += 1
    #create laplacian signals (i - (((i-1)+(i+1))/2) and names
    channels_LPR_des = [] #channel description 
    index = 0
    for s in shafts:
        ref_neg = [] #reference channels below
        ref_pos = [] #reference channels above
        for ch in range(shafts[s]['size']):
            ref_neg = [data[:,(shafts[s]['start']+(ch-neg))] for neg in range(1,SIZE+1) if (ch-neg) >= 0]
            ref_pos = [data[:,(shafts[s]['start']+(ch+pos))] for pos in range(1,SIZE+1) if (ch+pos) < shafts[s]['size']]
            data_LPR[:,index] = data[:,(shafts[s]['start']+ch)] - np.mean(ref_neg + ref_pos, axis=0)
            channels_LPR_des.append([channels[(shafts[s]['start']+ch)] + '-' + channels[(shafts[s]['start']+ch)] + '±' + str(SIZE)]) #description
            index += 1
    return data_LPR, np.array(channels_LPR_des)

# Define function to implement common average rereferencing
def commonAverageR(data):
    """Apply a common-average re-reference to the data
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    
    Returns
    ----------
    data: array (samples, channels)
        CAR re-referenced EEG time series   
    """
    data_CAR = np.zeros((data.shape[0], data.shape[1]))
    average = np.average(data, axis=1)
    for i in range(data.shape[1]):
        data_CAR[:,i] = data[:,i] - average
    return data_CAR

# Define function to implement common average rereferencing
def elecShaftR(data, channels):
    """Apply an electrode-shaft re-reference to the data
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    channels: array (electrodes, label)
        Channel names
    
    Returns
    ----------
    data: array (samples, channels)
        ESR re-referenced EEG time series   
    """
    data_ESR = np.zeros((data.shape[0], data.shape[1]))
    #get shaft information
    shafts = {}
    for i,chan in enumerate(channels):
        if chan[0].rstrip('0123456789') not in shafts:
            shafts[chan[0].rstrip('0123456789')] = {'start': i, 'size': 1}
        else:
            shafts[chan[0].rstrip('0123456789')]['size'] += 1
    #get average signal per shaft
    for shaft in shafts:
        shafts[shaft]['average'] = np.average(data[:,shafts[shaft]['start']:(shafts[shaft]['start']+shafts[shaft]['size'])], axis=1)
    #subtract the shaft average from each respective channel   
    for i in range(data.shape[1]):
        data_ESR[:,i] = data[:,i] - shafts[channels[i][0].rstrip('0123456789')]['average']
    return data_ESR

# Define function to preprocess data and save relevant .np files
def preprocess_data(reref='elecShaftR', participants=[], bands={'delta':[1, 2, 3], 'theta':[4,5,6,7], 'alpha':[8,9,10,11,12], 'beta':np.arange(13,31).tolist(), 'gamma':np.arange(70,121).tolist()}, window_of_i=['feedback', 'stimulus', 'baseline']):
    data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/RawData/'
    out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/'
    if not os.path.exists(out_path):
            os.makedirs(out_path)
    bands = bands
    participants = participants
    overlap = 0.05
    win_len = 0.1

    for participant in participants:
    # Load data
        data, _ = pyxdf.load_xdf(data_path + '{}_test.xdf'.format(participant))
        if '25' in participant:
            n = 0
            r = 1
        else:
            n = 1
            r = 0
        markers = data[r]['time_series']
        task_ts = data[r]['time_stamps']
        seeg_raw = data[n]['time_series']
        seeg_ts = data[n]['time_stamps']
        sr = int(float(data[n]['info']['nominal_srate'][0]))
       
    # Obtain all possible labels and store them in an array
        trials=[]
        for x in markers:
            if 'Sum' in x[0]:
                trials.append(((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')[0:5])
        
    # Identify no-answer trials and remove them from trials list 
        no_answer_trials = [int(x[0]) for x in trials if x[-1]=='No']
        if window_of_i[0] not in ['stimulus', 'long_stim']:
            for x in no_answer_trials:
                trials.pop(x-1)

    # Translate streams into numbers coding and store all labels in an array
        header_labels_array = ['trial number', 'stimulus number', 'repetition', 'stimulus valence (1=w; 0=l)', 'decision(1=w; 0=l)', 'accuracy']
        labels_array = np.zeros((len(trials),len(header_labels_array)), dtype=int)
        c=0
        for x in trials:
            labels_array[c,0] = int(x[0])
            labels_array[c,1] = int(x[1])
            labels_array[c,2] = int(x[2])
            if x[-1] == 'Correct':
                labels_array[c,-1] = 1
            elif x[-1] == 'Incorrect':
                labels_array[c,-1] = 0
            else:
                labels_array[c,-1] = 2
            if x[-2] == 'w':
                labels_array[c,-2] = 1                     
            elif x[-2] == 'l':
                labels_array[c,-2] = 0
            if labels_array[c,-1] == 1:
                labels_array[c,3] = labels_array[c,-2]
            elif labels_array[c,-1] == 0:
                labels_array[c,3] = 1 - labels_array[c,-2]
            c+=1
    
    # Save labels_array name file
        #np.save(out_path + participant + '_labels_array.npy', labels_array)

    # Define labels vectors
        labels = {'accuracy_labels': labels_array[:,-1], 'stimvalence_labels': labels_array[:,3], 'decision_labels': labels_array[:,-2]}
        
        future_labels = np.zeros(len(trials))
        for num, x in enumerate(labels_array):
            if x[2] == 1: 
                for t in labels_array[list(np.where(labels_array[:,2] == 2)[0])]:
                    if x[1] == t[1]:
                        future_labels[num] = t[-1]
            elif x[2] == 2:
                for t in labels_array[list(np.where(labels_array[:,2] == 3)[0])]:
                    if x[1] == t[1]:
                        future_labels[num] = t[-1]
            elif x[2] == 3:
                future_labels[num] = 3
            
        labels['future_labels']=future_labels

    # Store index trials depending on repetition
        repetitions = {
            'rep_all':list(np.arange(0,len(trials))),
            'rep_1':list(np.where(labels_array[:,2] == 1)[0]),
            'rep_2':list(np.where(labels_array[:,2] == 2)[0]),
            'rep_3':list(np.where(labels_array[:,2] == 3)[0]),
            'rep_2_3':sorted(list(np.where(labels_array[:,2] == 2)[0]) + list(np.where(labels_array[:,2] == 3)[0]))
        }

    # Save labels name files
        for key1, val1 in labels.items():
            for key2, val2 in repetitions.items():
                folder = out_path + key1 + '/'
                if not os.path.exists(folder):
                    os.makedirs(folder)
                #mydata = val1[val2]
                np.save(folder + participant + '_' + key1 + '_' + key2, val1[val2])
        
    # Define epochs indexes for each window type start and end points
        indices = {
            'feedback':[[x for x in range(len(markers)) if 'start Fb' in markers[x][0]], [0,1]],
            'baseline':[[x for x in range(len(markers)) if 'Start Cross' in markers[x][0]], [0, 0.5]],
            'long_FB':[[x for x in range(len(markers)) if 'Start Cross' in markers[x][0]], [0, 2]],
            'stimulus':[[x for x in range(len(markers)) if 'Start Stim' in markers[x][0]], [0, 1]],
            'long_stim':[[x for x in range(len(markers)) if 'Start Trial' in markers[x][0]], [0, 2]]
            #'decision':[x for x in range(len(markers)) if 'Press' in markers[x][0] and 'wrong' not in markers[x][0]]
        }
        
    # Get channel names, remove useless channels
        tot_channels = int(data[n]['info']['channel_count'][0])
        channels = [x['label'][0] for x in data[n]['info']['desc'][0]['channels'][0]['channel']]
        bad_channels = [c for c, x in enumerate(channels) if '+' in x or 'el' in x]
        tot_channels -= len(bad_channels)
        seeg_channels_rej = np.delete(seeg_raw, bad_channels, axis=1)
        for x in reversed(bad_channels):
            del channels[x]
    
    # Rereferencing
        if reref == 'laplacian':
            seeg_reref, chann_reref = laplacianR(seeg_channels_rej, channels)
        elif reref =='CAR':
            seeg_reref = commonAverageR(seeg_channels_rej)
        elif reref =='elecShaftR':
            seeg_reref = elecShaftR(seeg_channels_rej, channels)
        elif reref == 'none':
            seeg_reref = seeg_channels_rej

    # Save channels name file
        np.save(out_path + participant + '_channels.npy', channels)
        if reref == 'laplacian':
            np.save(out_path + participant + '_channels_des.npy', chann_reref)

    # Filter data into bands of interest
        sEEG_envelopes = {}
        for c, filter in enumerate(bands.keys()):
            sEEG_filtered, _ = filterdata(seeg_reref, filter, sr)

    # Estimate envelope for each band and save in dictionary
            hilbert3 = lambda x: sy.signal.hilbert(x, sy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]
            sEEG_envelopes[filter] = np.abs(hilbert3(sEEG_filtered))

    # Epoching and Store sEEG data into trial x time x channel arrays for each window
        for window in window_of_i:
            epoch_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[indices[window][0]]]) + int(indices[window][1][0]*sr)
            epoch_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[epoch_start]]) + int(indices[window][1][1]*sr)
            wl = int((indices[window][1][1] - indices[window][1][0]) * sr)

            if 'long' not in window:
                bands_envelope_tot = np.zeros((len(trials), len(bands), tot_channels))
                for c, filter in enumerate(bands.keys()):
                    sEEG_epoched_f = np.zeros((len(trials), wl, tot_channels)) #sEEG_epoched_f epoched and filtered
                    for count, x in enumerate(epoch_start):
                        for channel in range(tot_channels):
                            sEEG_epoched_f[count,:,channel] = sEEG_envelopes[filter][x:int(epoch_end[count]), channel]

                # Store envelope averages of bands of interest into trials x band x channel array
                    bands_envelope_tot[:,c,:] = np.mean(sEEG_epoched_f, axis = 1)
                
        # Save envelopes array .np file for each repetition
                for key, val in repetitions.items():
                    folder2 = out_path + window + '/' + reref +'/'
                    if not os.path.exists(folder2):
                            os.makedirs(folder2)
                    file = bands_envelope_tot[val,:,:]
                    np.save(folder2 + reref + '_' + participant + '_' + key + '_bands_envelope.npy', file)
            
            else:
                for c, filter in enumerate(bands.keys()):
                    sEEG_epoched_f = np.zeros((len(trials), wl, tot_channels)) #sEEG_epoched_f epoched and filtered
                    envelope_windowed = np.zeros((len(trials), int(wl/(overlap*sr))-1, tot_channels))
                    for count, x in enumerate(epoch_start):
                        for channel in range(tot_channels):
                            sEEG_epoched_f[count,:,channel] = sEEG_envelopes[filter][x:int(epoch_end[count]), channel]
                            step = 0
                            my_wind = np.zeros(int(win_len*sr))
                            for w in [*range(0, int(wl/(overlap*sr))-1)]:
                                my_wind[:] = sEEG_epoched_f[count,step:step+(int(win_len*sr)),channel]
                                envelope_windowed[count,w,channel] = np.mean(my_wind)
                                step += int(overlap*sr)
                
                    # Save envelopes array .np file for each repetition
                    for key, val in repetitions.items():
                        folder2 = out_path + window + '/' + reref +'/'
                        if not os.path.exists(folder2):
                                os.makedirs(folder2)
                        file = envelope_windowed[val,:,:]
                        file2 = sEEG_epoched_f[val,:,:]
                        np.save(folder2 + reref + '_' + participant + '_' + filter + '_' + key + '_envelope_windowed.npy', file)
                        np.save(folder2 + reref + '_' + participant + '_' + filter + '_' + key + '_envelope_dontinuous.npy', file2)


if __name__=="__main__":
    reref = 'elecShaftR' #['laplacian', 'CAR', 'elecShaftR', 'none']
    participants=['kh21','kh22','kh23','kh24','kh25'] #['kh21', 'kh22', 'kh23', 'kh24', 'kh25']
    window_of_i=['long_stim']
    #bands={'gamma':np.arange(70,120).tolist()}


    preprocess_data(reref=reref, participants=participants, window_of_i=window_of_i)