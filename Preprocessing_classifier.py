from numpy.core.numeric import indices
import pyxdf
import bisect
import numpy as np
import scipy as sy
import mne
import os
import csv

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
def preprocess_data(reref='laplacian', feature='envelope', window='feedback', window_length=1, delay=0, subtract_delay=False, classify='accuracy', preprocess_lowpass=False):
# Preprocessing
    data_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/Data/RawData/'
    out_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/Data/PreprocessedData_rereferencing/{}/'.format(window + '-' + str(window_length) + '_' + classify)
    bands = {'delta':[1, 2, 3], 'theta':[4,5,6,7], 'alpha':[8,9,10,11,12], 'beta':np.arange(13,31).tolist(), 'gamma':np.arange(70,121).tolist()}
    participants = ['kh23']
    
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

    # Save channels name file
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            np.save(out_path + '/{}_channels.npy'.format(participant), channels)
            np.save(out_path + '/{}_channels_des.npy'.format(participant), chann_reref)
        else:
            if not os.path.exists(out_path + '/{}_channels.npy'.format(participant)):
                np.save(out_path + '/{}_channels.npy'.format(participant), channels)
            else:
                print('{}_channels.npy'.format(participant) + ' File already existing in folder')
            if reref == 'laplacian':
                if not os.path.exists(out_path + '/{}_channels_des.npy'.format(participant)):
                    np.save(out_path + '/{}_channels_des.npy'.format(participant), chann_reref)
                else:
                    print('{}_channels.npy'.format(participant) + ' File already existing in folder')

    # Obtain labels depending on function argument
        labels = []
        # Distinguish correct from incorrect trials and create labels vector
        if classify == 'accuracy':
            for x in markers:
                if 'Sum' in x[0]:
                    summary = ((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                    if summary[-1] == 'Correct':
                        labels.append(1)                        
                    elif summary[-1] == 'Incorrect':
                        labels.append(0)
        # Distinguish winning from losing stimuli trials and create labels vector
        elif classify == 'stim_valence':
            for x in markers:
                if 'Sum' in x[0]:
                    summary = ((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                    if summary[-1] == 'Correct':
                        if summary[-2] == 'w':
                            labels.append(1)
                        elif summary[-2] == 'l':
                            labels.append(0)                        
                    elif summary[-1] == 'Incorrect':
                        if summary[-2] == 'w':
                            labels.append(0)
                        elif summary[-2] == 'l':
                            labels.append(1) 
        # Distinguish winning from losing decisions and create labels vector
        elif classify == 'decision':
            for x in markers:
                if 'Sum' in x[0]:
                    summary = ((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                    if summary[-2] == 'w':
                        labels.append(1)                     
                    elif summary[-2] == 'l':
                        labels.append(0)
        # Distinguish learning from non-learning trials and create labels vector
        elif classify == 'learning':
            thirds = []
            for x in markers:
                if 'Sum' in x[0]:
                    summary = ((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                    stim = summary[1]
                    if summary[2] == '1':
                        for y in markers:
                            if 'Sum' in y[0]:
                                summary2 = ((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                                if summary2[1] == stim:
                                    if summary2[2] == '2':
                                        if summary2[-1] == 'Correct':
                                            labels.append(1)
                                        elif summary2[-1] == 'Incorrect':
                                            labels.append(0)
                    elif summary[2] == '2':
                        stim = summary[1]
                        for y in markers:
                            if 'Sum' in y[0]:
                                summary2 = ((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                                if summary2[1] == stim:
                                    if summary2[2] == '3':
                                        if summary2[-1] == 'Correct':
                                            labels.append(1)
                                        elif summary2[-1] == 'Incorrect':
                                            labels.append(0)
                    elif summary[-3] == '3':
                        thirds.append(summary[0])                        
        tot_trials = len(labels)
        labels = np.array(labels)
    
    # Save labels name file
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            np.save(out_path + '/{}_labels.npy'.format(participant + '_' + classify), labels)
        else:
            if not os.path.exists(out_path + '/{}_labels.npy'.format(participant + '_' + classify)):
                np.save(out_path + '/{}_labels.npy'.format(participant + '_' + classify), labels)
            else:
                print('{}_labels.npy'.format(participant + '_' + classify) + ' File already existing in folder')

    # Define epochs indexes based on function arguments
        if window == 'feedback':
            indices = [x for x in range(len(markers)) if 'start Fb' in markers[x][0]]
        elif window == 'baseline':
            indices = [x for x in range(len(markers)) if 'Start Cross' in markers[x][0]]
        elif window == 'stimulus' or 'decision':
            indices = [x for x in range(len(markers)) if 'Start Stim' in markers[x][0]]
        epoch_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[indices]]) + int(delay*sr)
        if subtract_delay:
            epoch_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[epoch_start]]) + int((window_length * sr)-delay*sr)
            wl = int((window_length * sr)-delay*sr)
        else:
            epoch_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[epoch_start]]) + int(window_length * sr)
            wl = int(window_length * sr)

    ### Preprocess only
        if feature == 'preprocess_only':
        # lowpass if required
            if preprocess_lowpass:
                seeg_lowpassed, _ = filterdata(seeg_reref, 'lowpass', sr)
                string = '_low_pass'
            else:
                seeg_lowpassed = seeg_reref
                string = ''
        # Epoching and Store sEEG data into trial x time x channel arrays
            sEEG_epoched_nf = np.zeros((tot_trials, wl, tot_channels)) #sEEG_epoched_f epoched and not filtered
            for c, x in enumerate(epoch_start):
                for channel in range(tot_channels):
                    sEEG_epoched_nf[c,:,channel] = seeg_lowpassed[x:int(epoch_end[c]), channel] 
        # Save spectra array .np file            
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                np.save(out_path + '/{}_preprocessed_sEEG.npy'.format(participant + string), sEEG_epoched_nf)
            else:
                if not os.path.exists(out_path + '/{}_preprocessed_sEEG.npy'.format(participant + string)):
                    np.save(out_path + '/{}_preprocessed_sEEG.npy'.format(participant + string), sEEG_epoched_nf)
                else:
                    print('{}_preprocessed_sEEG'.format(participant + string) + ' File already existing in folder')
        
    ### Spectrum
        if feature == 'spectra':
        # Epoching and Store sEEG data into trial x time x channel arrays
            sEEG_epoched_nf = np.zeros((tot_trials, wl, tot_channels)) #sEEG_epoched_f epoched and not filtered
            for c, x in enumerate(epoch_start):
                for channel in range(tot_channels):
                    sEEG_epoched_nf[c,:,channel] = seeg_reref[x:int(epoch_end[c]), channel]
        # Extract Spectrum
            freqs = np.fft.rfftfreq(sr, d=1/sr) 
            spectra = np.zeros((tot_trials, freqs.shape[0], tot_channels))
            for c in range(tot_channels):
                for trial in range(tot_trials):
                    x = sEEG_epoched_nf[trial,:,c]
                    xf = np.fft.rfft(x - x.mean())                        
                    spectra[trial,:,c] = np.real(2 * 1/sr ** 2 / sr * (xf * np.conj(xf)))
        # Save spectra averages of bands of interest into trials x band x channel array
            bands_spectra = np.zeros((tot_trials, len(bands), tot_channels))
            for c, values in enumerate(bands.values()):
                spec = np.mean(spectra[:,values,:], axis=1)
                bands_spectra[:,c,:] = spec
        # Save spectra array .np file            
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                np.save(out_path + '/{}_bands_spectra.npy'.format(participant), bands_spectra)
            else:
                if not os.path.exists(out_path + '/{}_bands_spectra.npy'.format(participant)):
                    np.save(out_path + '/{}_bands_spectra.npy'.format(participant), bands_spectra)
                else:
                    print('{}_bands_spectra'.format(participant) + ' File already existing in folder')
        
    ### Envelope
        elif feature == 'envelope':
            bands_envelope = np.zeros((tot_trials, len(bands), tot_channels))
        # Filter data into bands of interest and for each band:
            for c, filter in enumerate(bands.keys()): #['delta', 'theta', 'alpha', 'beta', 'gamma']
                sEEG_filtered, _ = filterdata(seeg_reref, filter, sr)
            # Estimate envelope
                hilbert3 = lambda x: sy.signal.hilbert(x, sy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]
                sEEG_filtered = np.abs(hilbert3(sEEG_filtered))
            # Epoching and Store sEEG data into trial x time x channel arrays
                sEEG_epoched_f = np.zeros((tot_trials, wl, tot_channels)) #sEEG_epoched_f epoched and filtered
                for count, x in enumerate(epoch_start):
                    for channel in range(tot_channels):
                        sEEG_epoched_f[count,:,channel] = sEEG_filtered[x:int(epoch_end[count]), channel]
            # Save envelope averages of bands of interest into trials x band x channel array
                bands_envelope[:,c,:] = np.mean(sEEG_epoched_f, axis = 1)
        
        # Save envelopes array .np file
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                np.save(out_path + '/{}_bands_envelope.npy'.format(participant), bands_envelope)
            else:
                if not os.path.exists(out_path + '/{}_bands_envelope.npy'.format(participant)):
                    np.save(out_path + '/{}_bands_envelope.npy'.format(participant), bands_envelope)
                else:
                    print('/{}_bands_envelope.npy'.format(participant) + ' File already existing in folder')


if __name__=="__main__":
    preprocess_data(reref='laplacian', feature='preprocess_only', window='baseline', window_length=1.5, preprocess_lowpass=True)
    '''
    reref = ['laplacian', 'CAR']
    feature = ['envelope', 'spectra', 'preprocess_only']
    window = ['feedback', 'baseline', 'stimulus', 'decision']
    window_length = int
    delay = int
    subtract_delay = False or True
    classify = ['accuracy', 'stim_valence', 'decision', 'learning']
    preprocess_lowpass= False or True
    '''