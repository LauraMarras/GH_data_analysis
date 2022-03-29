from numpy.core.function_base import linspace
import pyxdf
import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scystats
import scipy.signal as sy
from statsmodels.stats.multitest import fdrcorrection
import os
from ipywidgets import interact, widgets,fixed, interact_manual


data_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/StreamFiles/exp001'
output_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/Results/spectral_analysis/mean_spectrogram/'

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
    data, _ = pyxdf.load_xdf(data_path + '/Epilpp01_test.xdf')
   
    markers = data[0]['time_series']
    task_ts = data[0]['time_stamps']
    seeg = data[1]['time_series']
    seeg_ts = data[1]['time_stamps']

    sr = data[1]['info']['nominal_srate'][0]
    sr = float(sr)
    sr = int(sr)
    
    
    # Get channel names
    tot_channels = int(data[1]['info']['channel_count'][0])
    channels = [x['label'][0] for x in data[1]['info']['desc'][0]['channels'][0]['channel']]
    
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
    FB_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[FB_start]]) + int(1.5*sr)
    
    # Store sEEG data into (trial x time x channel) array
    sEEG = np.zeros((tot_trials, int(1.5*sr), tot_channels))
    t = np.zeros((tot_trials, int(1.5*sr)))
    for c, x in enumerate(FB_start):
        for channel in range(tot_channels):
            sEEG[c,:,channel] = seeg[x:int(FB_end[c]), channel]
            t[c,:] = seeg_ts[x:int(FB_end[c])]

    
    # This function plots the raw data and the corresponding spectrogram
    def plotRawSpec(data,sr,start=0,duration=1,winLength=0.05):
        e=start+duration
        x=np.arange(start*sr,sr*e)/sr-start
        #t = np.arange(-0.5*sr, 1*sr)/sr
        fig, ax = plt.subplots(2,figsize=[20,4])
        dat = data[sr*start:int(sr*e)]
        # Plotting the raw data
        ax[0].set_title('Raw')
        ax[0].plot(x,dat,label='Raw')
        # Adding the lines for the end and start of each window in which spectra are calculated
        for i in np.arange(np.min(x),np.max(x),winLength):
            ax[0].axvline(x=i,color='r')
        ax[0].set_xlim(0,x[-1])
        numWindows=int(np.floor((dat.shape[0])/(winLength*sr)))
        numSpecs=int(np.floor(winLength*sr / 2 + 1))
        spec=np.zeros((numWindows,numSpecs))
        freqs = np.fft.rfftfreq(int(winLength*sr), d=1/sr)
        # Calculate spectrum for each window
        for w in range(numWindows):
            s=int(w*winLength*sr)
            e=int(s+winLength*sr)
            spec[w,:]=np.abs(np.fft.rfft(dat[s:e]))**2  
        ax[1].set_title('Spectrogram')
        spec=np.log10(spec)
        spec = spec[:, :int(numSpecs*60/(sr/2))]
        ax[1].imshow(np.flipud(spec.T), aspect='auto', cmap='viridis')
        ax[1].set_xlabel('Spectrogram Number')
        ax[1].set_ylabel(str(len(freqs)) + ' Freqs')
        #ax[1].set_ylim([0,40])
        ax[1].set_yticks(range(0,spec.shape[1]))
        #ax[1].set_yticklabels([str(sr/2),str(0),])
        for axs in ax:
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)
            axs.spines['bottom'].set_visible(False)
        plt.show()


    plotRawSpec(data=(sEEG[0,:,0]),sr=(sr),duration=(1.5),winLength=0.2)
