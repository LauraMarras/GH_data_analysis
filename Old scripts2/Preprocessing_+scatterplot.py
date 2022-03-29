import pyxdf
import bisect
import numpy as np
import scipy as sy
import mne
import os
import matplotlib.pyplot as plt

data_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/StreamFiles/exp001/'
directory = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/PreprocessedData/'
bands = {'delta':[1, 2, 3], 'theta':[4,5,6,7], 'alpha':[8,9,10,11,12], 'beta':np.arange(13,31).tolist(), 'gamma':np.arange(70,121).tolist()}

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
def filterdata(data, filter):
    seeg = data.astype(float)
    if filter == 'gamma':
        seeg = mne.filter.filter_data(seeg.T, sr, 70, 120, method='iir').T
        filter_used = 'gamma'
    elif filter == 'theta':
        seeg = mne.filter.filter_data(seeg.T, sr, 4, 7, method='iir').T
        filter_used = 'theta'
    elif filter == 'delta':
        seeg = mne.filter.filter_data(seeg.T, sr, 1, 4, method='iir').T
        filter_used = 'delta'
    elif filter == 'alpha':
        seeg = mne.filter.filter_data(seeg.T, sr, 7, 12, method='iir').T
        filter_used = 'alpha'
    elif filter == 'beta':
        seeg = mne.filter.filter_data(seeg.T, sr, 12, 30, method='iir').T
        filter_used = 'beta'
    return seeg, filter_used

if __name__=="__main__":
    pp_labels = {}
    pp_spectra = {}
    pp_envs = {}
    for participant in ['01','03']:
        data, _ = pyxdf.load_xdf(data_path + 'Epilpp{}_test.xdf'.format(participant))

        if participant == '01' or '03':
            n = 1
        elif participant == '02':
            n = 3
    
        markers = data[0]['time_series']
        task_ts = data[0]['time_stamps']
        seeg_notfiltered = data[n]['time_series']
        seeg_ts = data[n]['time_stamps']

        sr = float(data[n]['info']['nominal_srate'][0])
        sr = int(sr)
        
        # Get channel names
        tot_channels = int(data[n]['info']['channel_count'][0])
        channels = [x['label'][0] for x in data[n]['info']['desc'][0]['channels'][0]['channel']]
        bad_channels = [c for c, x in enumerate(channels) if '+' in x]
        tot_channels -= len(bad_channels)
        for x in reversed(bad_channels):
            del channels[x]
        #np.save(directory + participant + '/channels.npy', channels)

        # Distinguish correct from incorrect trials
        correct_trials = []
        incorrect_trials = []
        labels = []
        for x in markers:
                if 'Sum' in x[0]:
                    summary = ((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                    if summary[-1] == 'Correct':
                        correct_trials.append(int(summary[0]))
                        labels.append(1)                        
                    elif summary[-1] == 'Incorrect':
                        incorrect_trials.append(int(summary[0]))
                        labels.append(0)
        tot_trials = len(labels)
        labels = np.array(labels)
        pp_labels[participant] = labels
        

        # Extract feedback epochs
        FB_indices = [x for x in range(len(markers)) if 'start Fb' in markers[x][0]]
        FB_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[FB_indices]])
        FB_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[FB_start]]) + int(sr)

        # Remove bad channels
        seeg_channels_rej = np.delete(seeg_notfiltered, bad_channels, axis=1)

        # Store sEEG data into trial x time x channel arrays
        X = np.zeros((tot_trials, int(sr), tot_channels))
        for c, x in enumerate(FB_start):
            for channel in range(tot_channels):
                X[c,:,channel] = seeg_channels_rej[x:int(FB_end[c]), channel]
        
        # Extract Spectrum
        freqs = np.fft.rfftfreq(sr, d=1/sr) 
        spectra = np.zeros((tot_trials, freqs.shape[0], tot_channels))
        for c in range(tot_channels):
            for trial in range(tot_trials):
                x = X[trial,:,c]
                #freqPower=np.log10(np.abs(np.fft.rfft(x))**2)
                #spectra[trial,:,c]=freqPower
                xf = np.fft.rfft(x - x.mean())                        
                spectra[trial,:,c] = np.real(2 * 1/sr ** 2 / sr * (xf * np.conj(xf)))
        bands_spectra = {}
        for k, v in bands.items():
            spec = np.mean(spectra[:,v,:], axis=1)
            bands_spectra[k] = spec
        
        pp_spectra[participant] = bands_spectra    
        #np.save(directory + participant + '/bands_spectra.npy', bands_spectra)    

        # Filter data
        bands_envelopes = {}
        counter = 0
        be = np.zeros((tot_trials, len(bands), tot_channels))
        for k, v in bands.items():
            seeg, filter_used = filterdata(seeg_channels_rej, k)

            # Estimate envelope
            hilbert3 = lambda x: sy.signal.hilbert(x, sy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]
            seeg = np.abs(hilbert3(seeg))            
            
            # Store sEEG data into trial x time x channel arrays
            sEEG = np.zeros((tot_trials, int(sr), tot_channels))
            for c, x in enumerate(FB_start):
                for channel in range(tot_channels):
                    sEEG[c,:,channel] = seeg[x:int(FB_end[c]), channel]
                    
            bands_envelopes[k] = np.mean(sEEG, axis = 1)
            be[:,counter,:] = np.mean(sEEG, axis = 1)
            
            counter+=1
        pp_envs[participant] = bands_envelopes
        #np.save(directory + participant + '/be.npy', be)
