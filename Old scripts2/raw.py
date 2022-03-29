from numpy.core.numeric import indices
import pyxdf
import bisect
import numpy as np
import scipy as sy
import mne
import os
import csv
import matplotlib.pyplot as plt

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

# Define function to preprocess data and save relevant .np files
def raw_trials(reref='laplacian', feature='raw', envelope = 0, epoching = 0, classify='accuracy', preprocess_lowpass=False, participants=[]):
# Preprocessing
    data_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Data_Analysis/Data/RawData/'
    out_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Thesis/Preprocessing plots/kh23_'
    bands = {'delta':[1, 2, 3], 'theta':[4,5,6,7], 'gamma':np.arange(70,121).tolist()} #, 'alpha':[8,9,10,11,12], 'beta':np.arange(13,31).tolist()}
    participants = participants
    
    for participant in participants:
    # Load data
        data, _ = pyxdf.load_xdf(data_path + '{}_test.xdf'.format(participant))
        if 'kh' in participant:
            n = 1
        elif 'us' in participant:
            n = 3
            good_channels = list(np.arange(0,40))
        markers = data[0]['time_series']
        task_ts = data[0]['time_stamps']
        seeg_raw = data[n]['time_series']
        seeg_ts = data[n]['time_stamps']
        sr = int(float(data[n]['info']['nominal_srate'][0]))
        
    # Get channel names, remove useless channels
        tot_channels = int(data[n]['info']['channel_count'][0])
        channels = [x['label'][0] for x in data[n]['info']['desc'][0]['channels'][0]['channel']]
        if 'us' in participant:
            bad_channels = [c for c, _ in enumerate(channels) if c not in good_channels]
        else:
            bad_channels = [c for c, x in enumerate(channels) if '+' in x or 'el' in x]
        tot_channels -= len(bad_channels)
        seeg_channels_rej = np.delete(seeg_raw, bad_channels, axis=1)
        for x in reversed(bad_channels):
            del channels[x]

    # Store channel names in dictionary
        chan_dict = {'L':{}, 'R':{}}
        for channel in channels:
            if channel[1] not in chan_dict[channel[0]].keys():
                chan_dict[channel[0]][channel[1]] = [channel[2:]]
            else:
                chan_dict[channel[0]][channel[1]].append(channel[2:])
    
    # Rereferencing
        if reref == 'laplacian':
            seeg_reref, chann_reref = laplacianR(seeg_channels_rej, channels)
        elif reref =='CAR':
            seeg_reref = commonAverageR(seeg_channels_rej)

    # Obtain labels depending on function argument
        labels = []
        labels_array = np.zeros((90,6), dtype=int)
        c=0
        for x in markers:
            if 'Sum' in x[0]:
                summary = ((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                labels_array[c,0] = int(summary[0])
                labels_array[c,1] = int(summary[1])
                labels_array[c,2] = int(summary[2])
                if summary[-1] == 'Correct':
                    labels_array[c,-1] = 1
                elif summary[-1] == 'Incorrect':
                    labels_array[c,-1] = 0
                if summary[-2] == 'w':
                    labels_array[c,-2] = 1                     
                elif summary[-2] == 'l':
                    labels_array[c,-2] = 0
                if labels_array[c,-1] == 1:
                    labels_array[c,3] = labels_array[c,-2]
                elif labels_array[c,-1] == 0:
                    labels_array[c,3] = 1 - labels_array[c,-2]
                c+=1

        # Distinguish correct from incorrect trials and create labels vector
        if classify == 'accuracy':
            labels = labels_array[:,-1]
            colors = []
            for l in labels:
                if l==0:
                    colors.append('r')
                else:
                    colors.append('g')
        # Distinguish winning from losing stimuli trials and create labels vector
        elif classify == 'stim_valence':
            labels = labels_array[:,3]
        # Distinguish winning from losing decisions and create labels vector
        elif classify == 'decision':
            labels = labels_array[:,-2]
        # Distinguish learning from non-learning trials and create labels vector
        elif classify == 'learning':
            labels = np.zeros((90), dtype=int)
            for c, x in enumerate(labels_array):
                if x[2] == 1:
                    for m in labels_array[c:,:]:
                        if m[1] == x[1] and m[2]==2:
                            labels[c]= m[-1]
                elif x[2] == 2:
                    for m in labels_array[c:,:]:
                        if m[1] == x[1] and m[2]==3:
                            labels[c]= m[-1]
                elif x[2] == 3:
                    labels[c]= 3
          
        tot_trials = len(labels)
        zerotrials = np.count_nonzero(labels == 3)
        labels = np.array(labels)
    
    # Define epochs indexes for each trial
        indices = [x for x in range(len(markers)) if 'Start Trial' in markers[x][0]]
        indices_end = [x for x in range(len(markers)) if 'start Fb' in markers[x][0]]
        indices_resp = [x for x in range(len(markers)) if 'Press' in markers[x][0] and 'wrong' not in markers[x][0]]
        indices_FB = [x for x in range(len(markers)) if 'start Fb' in markers[x][0]]
        indices_stim = [x for x in range(len(markers)) if 'Start Stim' in markers[x][0]]
        epoch_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[indices]])
        epoch_stim = np.array([locate_pos(seeg_ts, x) for x in task_ts[indices_stim]])
        epoch_stim_end = np.array([locate_pos(seeg_ts, x) for x in task_ts[indices_stim]]) + sr
        epoch_resp = np.array([locate_pos(seeg_ts, x) for x in task_ts[indices_resp]])
        epoch_FB = np.array([locate_pos(seeg_ts, x) for x in task_ts[indices_FB]])
        epoch_FB_end = np.array([locate_pos(seeg_ts, x) for x in task_ts[indices_FB]]) + sr
        epoch_end = np.array([locate_pos(seeg_ts, x) for x in task_ts[indices_end]]) + sr
        rt = epoch_resp - epoch_start
        fb = epoch_FB - epoch_start
        fb_end = epoch_FB_end - epoch_start
        wl = epoch_end - epoch_start
        st = epoch_stim - epoch_start
        st_end = epoch_stim_end - epoch_start 
    
    ### Raw
        if feature == 'raw':
            # lowpass if required
            if preprocess_lowpass:
                seeg_lowpassed, _ = filterdata(seeg_reref, 'lowpass', sr)
                signal = seeg_lowpassed[epoch_start[0]:int(epoch_end[2]), channels.index('LK13')] # long epoch including first three trials
            else:
                signal = seeg_reref[epoch_start[0]:int(epoch_end[2]), channels.index('LK13')]
            # Plot
            window_length = epoch_end[2] - epoch_start[0]
            t = np.arange(window_length)
            plt.figure(figsize=(9.5,2), tight_layout=True)
            plt.plot(t, signal)
            plt.margins(x=0)
            plt.xlabel('Time [ms]')
            plt.ylabel('Voltage [µV]')
            plt.title('Re-referenced continuous signal')
            time_lab = np.array([int(x) for x in np.arange(0, window_length/sr*1000, 1000)])
            plt.xticks(ticks=np.arange(0,window_length, sr), labels=time_lab)
            plt.savefig(out_path + 'Re-referenced_cs')
    
    ### Filtered
        elif feature == 'filter':
            for c, filter in enumerate(bands.keys()): #['delta', 'theta', 'alpha', 'beta', 'gamma']
                seeg_filtered, _ = filterdata(seeg_reref, filter, sr)
                if envelope != 1:
                    signal = seeg_filtered[epoch_start[0]:int(epoch_end[2]), channels.index('LK13')]
                    # Plot
                    window_length = epoch_end[2] - epoch_start[0]
                    t = np.arange(window_length)
                    plt.figure(figsize=(9.5,2), tight_layout=True)
                    plt.plot(t, signal)
                    plt.margins(x=0)
                    plt.xlabel('Time [ms]')
                    plt.ylabel('Voltage [µV]')
                    plt.title(filter + ' filtered continuous signal')
                    time_lab = np.array([int(x) for x in np.arange(0, window_length/sr*1000, 1000)])
                    plt.xticks(ticks=np.arange(0,window_length, sr), labels=time_lab)
                    plt.savefig(out_path + filter + '_filtered_cs')
                elif envelope == 1:
                    hilbert3 = lambda x: sy.signal.hilbert(x, sy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]
                    seeg_envelope = np.abs(hilbert3(seeg_filtered))
                    if epoching != 1:
                        signal = seeg_envelope[epoch_start[0]:int(epoch_end[2]), channels.index('LK13')]
                        # Plot
                        window_length = epoch_end[2] - epoch_start[0]
                        t = np.arange(window_length)
                        plt.figure(figsize=(9.5,2), tight_layout=True)
                        plt.plot(t, signal)
                        plt.margins(x=0)
                        plt.xlabel('Time [ms]')
                        plt.ylabel('Voltage [µV]')
                        plt.title(filter + ' envelope continuous signal')
                        time_lab = np.array([int(x) for x in np.arange(0, window_length/sr*1000, 1000)])
                        plt.xticks(ticks=np.arange(0,window_length, sr), labels=time_lab)
                        plt.savefig(out_path + filter + '_envelope_cs')
                    elif epoching == 1:
                        # Plot
                        fig, axs = plt.subplots(3)
                        fig.suptitle('LK13' + ' Raw Trials')
                        for c, x in enumerate(epoch_start[:3]):
                            epoch = seeg_envelope[x:int(epoch_end[c]), channels.index('LK13')]
                            ymin = min(epoch)
                            ymax = max(epoch)
                            t = np.arange(wl[c])
                            axs[c].plot(t, epoch) 
                            axs[c].axvline(x=rt[c], color='k', label='response')
                            axs[c].fill_betweenx((ymin, ymax), st[c], st_end[c], color='blue', alpha = 0.2, label='stimulus')
                            axs[c].fill_betweenx((ymin, ymax), rt[c]-512, rt[c], color='grey', alpha = 0.2, label = 'choice')
                            axs[c].fill_betweenx((ymin, ymax), rt[c], fb[c], color='y', alpha = 0.2, label = 'baseline')
                            #axs[c].axvline(x=st[c], color='m')
                            #axs[c].axvline(x=fb[c], color=colors[c])
                            axs[c].fill_betweenx((ymin, ymax), fb[c], fb_end[c], color=colors[c], alpha = 0.2, label='feedback')                
                            axs[c].set_xticks(np.arange(0, wl[c], int(sr/2)))
                            axs[c].margins(x=0, y=0)
                            time_lab = np.array([int(x) for x in np.arange(0, wl[c]/sr*1000, 500)])
                            # time2 = np.linspace((-rt[c])/sr*1000, (fb_end[c] - rt[c])/sr*1000, int(wl[c]/sr*1000))
                            # time2_arr= np.array([int(x) for x in time2])
                            # time_lab1 = -1*np.flip(np.arange(0, -time2_arr[0], 500))
                            # time_lab2 = np.arange(0, time2_arr[-1], 500)
                            # time_lab3 = np.concatenate([time_lab1, time_lab2[1:]])
                            axs[c].set_xticklabels(time_lab)
                        #for ax in axs:
                            #ax.set_xticks(np.arange(0, wl[0], int(sr/2)))
                            #time_lab = np.array([int(x) for x in np.arange(0, wl[0]/sr*1000, 500)])
                            #ax.set_xticklabels(time_lab)
                            #ax.label_outer()
                        plt.savefig(out_path + '4 - ' + filter + '_envelope_epoched')

if __name__=="__main__":
    raw_trials(reref='laplacian', feature='filter', envelope=1, epoching=0, classify='accuracy', preprocess_lowpass=False, participants=['kh23'])
    '''
    reref = ['laplacian', 'CAR']
    feature = ['filter', 'raw']
    envelope = [0, 1]
    epoching = [0, 1]
    classify = ['accuracy', 'stim_valence', 'decision', 'learning']
    preprocess_lowpass= False or True
    participants = ['kh21', 'kh22', 'kh23', 'kh24']
    '''