from numpy.core.function_base import linspace
import pyxdf
import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scystats
import scipy.signal as sy
from statsmodels.stats.multitest import fdrcorrection
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


if __name__=="__main__":
    for participant in ['03']:
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
        FB_indices = [x for x in range(len(markers)) if 'start Fb' in markers[x][0]]
        FB_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[FB_indices]])
        FB_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[FB_start]]) + int(sr)
        
        # Store sEEG data into (trial x time x channel) array
        sEEG = np.zeros((tot_trials, int(sr), tot_channels))
        t = np.zeros((tot_trials, int(sr)))
        for c, x in enumerate(FB_start):
            for channel in range(tot_channels):
                sEEG[c,:,channel] = seeg[x:int(FB_end[c]), channel]
                t[c,:] = seeg_ts[x:int(FB_end[c])]
                t[c,:] = [x - t[c,0] for x in t[c,:]]
        
        # Estimate spectrum for each trial and channel
        freqs = np.fft.rfftfreq(sr, d=1/sr) 
        # Choose method
        #method = 'Christian'
        method = 'notebooks'
        if method == 'Christian':
            spectra = np.zeros((tot_trials, freqs.shape[0], tot_channels))
            for c in range(tot_channels):
                for trial in range(tot_trials):
                    x = sEEG [trial,:,c]
                    freqPower=np.log10(np.abs(np.fft.rfft(x))**2)
                    spectra[trial,:,c]=freqPower
                
        elif method == 'notebooks':
            spectra = np.zeros((tot_trials, freqs.shape[0], tot_channels))
            for c in range(tot_channels):
                for trial in range(tot_trials):
                    dt = t[trial,2] - t[trial,1]  # Define the sampling interval
                    T = t[trial,-1] - t[trial,0]         # ... and duration of data
                    Fs = 1 / dt
                    x = sEEG [trial,:,c]

                # Compute the Fourier transform of x and the spectrum
                    xf = np.fft.rfft(x - x.mean())                        
                    #sx = np.real(2 * dt ** 2 / T * (xf * np.conj(xf)))
                    spectra[trial, :, c] = np.real(2 * dt ** 2 / T * (xf * np.conj(xf)))

        # Estimate mean spectra per condition and channel
        corrects = [x - 1 for x in correct_trials]
        incorrects = [x - 1 for x in incorrect_trials]
        spectrum_c_mean = spectra[corrects, :, :].mean(axis = 0)
        spectrum_i_mean = spectra[incorrects, :, :].mean(axis = 0)
        
        # Statistics: 2samples t-test
        _, p_values = scystats.ttest_ind(spectra[corrects, :, :], spectra[incorrects, :, :], axis=0)

        # FDR correction
        for c in range(tot_channels):
            _, p_values[:,c] = fdrcorrection(p_values[:,c], alpha=0.05, method='indep', is_sorted=False)
        
        # Estimate significant intervals
        significant = p_values < 0.05

        # Plot spectra x channel
        df = 1 / T                                     # Define the frequency resolution.
        fNQ = 1 / dt / 2                               # Define the Nyquist frequency.
        
        def plotting(method, fr_range, log=False):
            plt.rcParams['figure.figsize']=(11,4) # Change the default figure size
            method = method
            for c, channel in enumerate(channels):
                y_c = spectrum_c_mean[:, c]
                y_i = spectrum_i_mean[:, c]
                faxis = freqs
                ymax = max([max(y_c), max(y_i)]) 
            
                plt.figure()
                if method == 'notebooks' and log == False:
                    scale = 'normal'
                    plt.plot(faxis, y_c, 'g', label='correct')
                    plt.plot(faxis, y_i, 'r', label='incorrect')
                    plt.ylabel('Power [$\mu V^2$/Hz]')
                    plt.xlim(fr_range)
                    plt.xticks(range(0, fr_range[1]+1, int(df)*int(fr_range[1]/10)))

                elif method == 'Christian':
                    scale = 'log_Ch'
                    plt.plot(faxis, y_c, 'g', label='correct')
                    plt.plot(faxis, y_i, 'r', label='incorrect')
                    plt.ylabel('Power [dB]')
                    plt.xlim(fr_range)
                    plt.xticks(range(0, fr_range[1]+1, int(df)*int(fr_range[1]/10)))

                else:
                    scale = 'logarithmic'
                    plt.semilogx(faxis[1:], 10*np.log10(y_c[1:]), 'g', label='correct')
                    plt.semilogx(faxis[1:], 10*np.log10(y_i[1:]), 'r', label='incorrect')
                    plt.ylabel('Power [dB]')
                    plt.xlim(fr_range)
                    plt.xticks(range(0, fr_range[1]+1, int(df)*int(fr_range[1]/10)))
           
                plt.grid(axis = 'x', alpha = 0.5)
                plt.margins(x=0, y=0)
                plt.fill_between(faxis[1:], 0, ymax, where=(significant[1:,c]), color = 'r', alpha = 1)
                plt.xlabel('Frequency (Hz)')                       
                plt.title('Power spectrum of channel: ' + str(channel))        
                plt.legend(loc="upper left")   

                if True in significant[1:fr_range[1],c]:
                    directory = output_path + participant + '/Spectrum/{}/significant/'.format(scale + '0 - ' + str(fr_range[1]))
                else:
                    directory = output_path + participant + '/Spectrum/{}/not_s/'.format(scale + '0 - ' + str(fr_range[1]))

                if not os.path.exists(directory):
                    os.makedirs(directory)
                    plt.savefig(directory + str(c+1) + '_' + str(channel))
                else:
                    plt.savefig(directory + str(c+1) + '_' + str(channel)) 

        plotting('Christian', [0, 60])
        plotting('notebooks', [0, 60], log=True)
        plotting('notebooks', [0, 60])