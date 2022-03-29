from numpy.core.function_base import linspace
import pyxdf
import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy as sy
from statsmodels.stats.multitest import fdrcorrection
import mne
import os

data_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/StreamFiles/exp001/'
output_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/Results/'

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
    elif filter == 'delta':
        seeg = mne.filter.filter_data(seeg.T, sr, 0, 4, method='iir').T
        filter_used = 'delta'
    elif filter == 'theta':
        seeg = mne.filter.filter_data(seeg.T, sr, 4, 7, method='iir').T
        filter_used = 'theta'
    elif filter == 'alpha':
        seeg = mne.filter.filter_data(seeg.T, sr, 7, 12, method='iir').T
        filter_used = 'alpha'
    elif filter == 'beta':
        seeg = mne.filter.filter_data(seeg.T, sr, 12.5, 30, method='iir').T
        filter_used = 'beta'
    elif filter == 'lowpass':
        seeg = mne.filter.filter_data(seeg.T, sr, 0, 40, method='iir').T
        filter_used = 'lowpass'
    elif filter == 'superlowpass10':
        seeg = mne.filter.filter_data(seeg.T, sr, 0, 10, method='iir').T
        filter_used = 'superlowpass10'
    elif filter == 'pline':
        seeg = mne.filter.filter_data(seeg.T, sr, 61, 59, method='iir').T
        seeg = mne.filter.filter_data(seeg.T, sr, 121, 119, method='iir').T
        seeg = mne.filter.filter_data(seeg.T, sr, 181, 179, method='iir').T
        seeg = mne.filter.filter_data(seeg.T, sr, 51, 49, method='iir').T  
        seeg = mne.filter.filter_data(seeg.T, sr, 101, 99, method='iir').T     
        seeg = mne.filter.filter_data(seeg.T, sr, 151, 149, method='iir').T
        filter_used = 'pline'
    else:
        filter_used = 'nf'
        seeg = mne.filter.filter_data(seeg.T, sr, 0, 511, method='iir').T
    return seeg, filter_used

if __name__=="__main__":
    for participant in ['01', '03']:
        data, _ = pyxdf.load_xdf(data_path + 'Epilpp{}_test.xdf'.format(participant))

        if participant == '01' or '03':
            n = 1
        elif participant == '02':
            n = 3
   
        markers = data[0]['time_series']
        task_ts = data[0]['time_stamps']
        seeg = data[n]['time_series']
        seeg_ts = data[n]['time_stamps']

        sr = data[n]['info']['nominal_srate'][0]
        sr = float(sr)
        sr =int(sr)

        # Get channel names
        tot_channels = int(data[n]['info']['channel_count'][0])
        channels = [x['label'][0] for x in data[n]['info']['desc'][0]['channels'][0]['channel']]

        # Distinguish correct from incorrect trials
        correct_trials = []
        incorrect_trials = []
        for x in markers:
                if 'Sum' in x[0]:
                    summary = ((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                    if summary[-1] == 'Correct':
                        correct_trials.append(int(summary[0]))
                    elif summary[-1] == 'Incorrect':
                        incorrect_trials.append(int(summary[0]))
        tot_trials = len(correct_trials) + len(incorrect_trials)

        # Extract fixation + feedback epochs
        FB_indices = [x for x in range(len(markers)) if 'Start Cross' in markers[x][0]]
        FB_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[FB_indices]])
        FB_start = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[FB_start]]) - int(0.5*sr)
        FB_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[FB_start]]) + int(2.5*sr)

        BL_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[FB_indices]])
        BL_start = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[FB_start]]) + int(0.3*sr)
        BL_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[FB_start]]) + int(0.5*sr)

        # Store sEEG data into (trial x time x channel) array
        sEEG = np.zeros((tot_trials, int(2.5*sr), tot_channels))
        t = np.zeros((tot_trials, int(2.5*sr)))
        for c, x in enumerate(FB_start):
            for channel in range(tot_channels):
                sEEG[c,:,channel] = seeg[x:int(FB_end[c]), channel]
                t[c,:] = seeg_ts[x:int(FB_end[c])]

        corrects = [x - 1 for x in correct_trials]
        incorrects = [x - 1 for x in incorrect_trials]

        sEEG_i = sEEG[incorrects,:,:]
        sEEG_c = sEEG[corrects,:,:]


        # Baseline correction
        sEEG_BL_c = np.zeros((len(correct_trials), int(sr*0.2)+1, tot_channels))
        sEEG_BL_i = np.zeros((len(incorrect_trials), int(sr*0.2)+1, tot_channels))
        c_count = 0
        i_count = 0
        for c, x in enumerate(BL_start):
            if c+1 in correct_trials:
                for channel in range(tot_channels):
                    sEEG_BL_c[c_count,:,channel] = seeg[x:int(BL_end[c]), channel]
                c_count += 1
            elif c+1 in incorrect_trials:
                for channel in range(tot_channels):
                    sEEG_BL_i[i_count,:,channel] = seeg[x:int(BL_end[c]), channel]
                i_count += 1

        BL_i = np.mean(sEEG_BL_i, axis=(0,1))
        BL_c = np.mean(sEEG_BL_c, axis=(0,1))

        # Estimate ERPs
        i_ERP = np.mean(sEEG_i - BL_i, axis=0)
        c_ERP = np.mean(sEEG_c - BL_c, axis=0)

        i_ERP, filter_used = filterdata(i_ERP, 'theta')
        c_ERP, filter_used = filterdata(c_ERP, 'theta')

        for c, channel in enumerate(channels):
            corr = c_ERP[:,c]
            incorr = i_ERP[:,c]
            f, tspec, sg_i_mean = sy.signal.spectrogram(incorr, fs=sr, nperseg=int(sr), noverlap=int(sr*0.75))
            f, tspec, sg_c_mean = sy.signal.spectrogram(corr, fs=sr, nperseg=int(sr), noverlap=int(sr*0.75))

            #Plot spectrogram x channel
            plt.rcParams['figure.figsize']=(11,4)               # Change the default figure size
            time = 34
            t = linspace(-500, 1000, sr + int(sr/2), endpoint = False)
                
            fig, (ax_c, ax_i) = plt.subplots(1,2)
            pcc = ax_c.pcolormesh(tspec, f, (sg_c_mean), cmap='jet')
            pci = ax_i.pcolormesh(tspec, f, (sg_i_mean), cmap='jet')
            fig.colorbar(pcc, ax = ax_c)
            fig.colorbar(pci, ax = ax_i)
            fig.suptitle('Spectrogram of channel: ' + str(channel))
            ax_c.set_title('CORRECT')
            ax_i.set_title('INCORRECT')
            
            ax_c.set_xlabel('Time [s]')        # ... and label the axes
            ax_i.set_xlabel('Time [s]')
            ax_c.set_ylabel('Frequency [Hz]')
            #ax_i.set_ylabel('Frequency [Hz]')
            #plt.show()
            ax_c.set_yticks(np.arange(11))
            ax_i.set_yticks(np.arange(11))

            ax_c.set_ylim(0, 10)
            ax_i.set_ylim(0, 10)
            ax_c.axvline(x=1, color='k')
            ax_i.axvline(x=1, color='k')
            
            ax_c.axhline(y=4, color='c')
            ax_i.axhline(y=7, color='c')
            ax_i.axhline(y=4, color='c')
            ax_c.axhline(y=7, color='c')
        
            plt.show()

        