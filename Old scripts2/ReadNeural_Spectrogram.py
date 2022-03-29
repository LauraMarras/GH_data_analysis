from numpy.core.function_base import linspace
import pyxdf
import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scystats
import scipy.signal as sy
from statsmodels.stats.multitest import fdrcorrection
import os
from matplotlib.collections import LineCollection

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

        # Extract feedback epochs
        FB_indices = [x for x in range(len(markers)) if 'Start Cross' in markers[x][0]]
        FB_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[FB_indices]])
        FB_start = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[FB_start]]) - int(0.5*sr)
        FB_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[FB_start]]) + int(2.5*sr)

        ddd = FB_end[0] - FB_start[0]
        
        # Store sEEG data into (trial x time x channel) array
        sEEG = np.zeros((tot_trials, int(2.5*sr), tot_channels))
        t = np.zeros((tot_trials, int(2.5*sr)))
        for c, x in enumerate(FB_start):
            for channel in range(tot_channels):
                sEEG[c,:,channel] = seeg[x:int(FB_end[c]), channel]
                t[c,:] = seeg_ts[x:int(FB_end[c])]

        corrects = [x - 1 for x in correct_trials]
        incorrects = [x - 1 for x in incorrect_trials]
        
        # Estimate spectrogram for each trial and channel
        for c, channel in enumerate(channels):
            my_signal = sEEG[:,:,c]
            f, tspec, spectrogram = sy.spectrogram(my_signal, fs=sr, nperseg=int(sr), noverlap=int(sr*0.75))     

            # Estimate mean spectra per condition and channel
            sg_c_mean = spectrogram[corrects, :, :].mean(axis = 0)
            sg_i_mean = spectrogram[incorrects, :, :].mean(axis = 0)

            
            # Statistics: 2samples t-test
            _, p_values = scystats.ttest_ind(spectrogram[corrects, :, :], spectrogram[incorrects, :, :], axis=0)
            significant_uncorrected = p_values < 0.05

            # FDR correction
            for t in range(p_values.shape[1]):
                _, p_values[:,t] = fdrcorrection(p_values[:,t], alpha=0.05, method='indep', is_sorted=False)
            significant_corrected = p_values < 0.05

            uncorrected = np.sum(significant_uncorrected)
            corrected = np.sum(significant_corrected)

            theta = np.arange(4,8)
            delta = np.arange (0,4)
            both = np.arange(0,8)

            #Plot spectrogram x channel
            plt.rcParams['figure.figsize']=(11,4)               # Change the default figure size
            time = 34
            t = linspace(-500, 1000, sr + int(sr/2), endpoint = False)
                
            fig, (ax_c, ax_i) = plt.subplots(1,2)
            pcc = ax_c.pcolormesh(tspec, f, 10 * np.log10(sg_c_mean), cmap='jet')
            pci = ax_i.pcolormesh(tspec, f, 10 * np.log10(sg_i_mean), cmap='jet')
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

            #highlight = np.ma.masked_where(significant_uncorrected, 1).T
            #ax_c.pcolormesh(f, tspec, highlight, facecolor = 'None', edgecolors = 'w')

            # Save Images
            if True in significant_corrected[both, :]:
                directory = (output_path + participant + '/TFR_10_2/significant/')
            else:
                directory = (output_path + participant + '/TFR_10_2/not_s/')

            
            if not os.path.exists(directory):
                os.makedirs(directory)
                plt.savefig(directory + str(c+1) + '_' + str(channel))
            else:
                plt.savefig(directory + str(c+1) + '_' + str(channel))