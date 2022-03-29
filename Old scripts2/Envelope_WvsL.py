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
        seeg_notfiltered = data[n]['time_series']
        seeg_ts = data[n]['time_stamps']

        sr = data[n]['info']['nominal_srate'][0]
        sr = float(sr)
        sr =int(sr)

        # Get channel names
        tot_channels = int(data[n]['info']['channel_count'][0])
        channels = [x['label'][0] for x in data[n]['info']['desc'][0]['channels'][0]['channel']]

        # Distinguish winning from losing trials
        winning_trials = []
        losing_trials = []
        for x in markers:
                if 'Sum' in x[0]:
                    summary = ((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                    if summary[-1] == 'Correct':
                        if summary[-2] == 'w':
                            winning_trials.append(int(summary[0]))
                        elif summary[-2] == 'l':
                            losing_trials.append(int(summary[0]))
                    elif summary[-1] == 'Incorrect':
                        if summary[-2] == 'l':
                            winning_trials.append(int(summary[0]))
                        elif summary[-2] == 'w':
                            losing_trials.append(int(summary[0]))
        
        tot_trials = len(winning_trials) + len(losing_trials)

        # Extract fixation + feedback epochs
        BL_indices = [x for x in range(len(markers)) if 'Start Cross' in markers[x][0]]
        FB_indices = [x for x in range(len(markers)) if 'Start Trial' in markers[x][0]]
        FB_indices.append([x for x in range(len(markers)) if 'End Experiment' in markers[x][0]][0])
        FB_indices = FB_indices[1:]

        BL_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[BL_indices]])
        BL_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[BL_start]]) + sr/2
        FB_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[BL_start]]) + int(1.5 * sr)
        
        # Filter data
        for filter in ['theta', 'alpha', 'beta', 'gamma']:
            seeg, filter_used = filterdata(seeg_notfiltered, filter)          
            
            # Estimate envelope
            hilbert3 = lambda x: sy.signal.hilbert(x, sy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]
            seeg = np.abs(hilbert3(seeg))

            # Store sEEG data into correct vs incorrect (trial x time x channel) arrays
            sEEG_w = np.zeros((len(winning_trials), int(1.5 * sr), tot_channels))
            sEEG_l = np.zeros((len(losing_trials), int(1.5 * sr), tot_channels))
            w_count = 0
            l_count = 0
            for c, x in enumerate(BL_start):
                if c+1 in winning_trials:
                    for channel in range(tot_channels):
                        sEEG_w[w_count,:,channel] = seeg[x:int(FB_end[c]), channel]
                    w_count += 1
                elif c+1 in losing_trials:
                    for channel in range(tot_channels):
                        sEEG_l[l_count,:,channel] = seeg[x:int(FB_end[c]), channel]
                    l_count += 1
            
            # Baseline correction
            sEEG_BL_w = np.zeros((len(winning_trials), int(sr/2), tot_channels))
            sEEG_BL_l = np.zeros((len(losing_trials), int(sr/2), tot_channels))
            w_count = 0
            l_count = 0
            for c, x in enumerate(BL_start):
                if c+1 in winning_trials:
                    for channel in range(tot_channels):
                        sEEG_BL_w[w_count,:,channel] = seeg[x:int(BL_end[c]), channel]
                    w_count += 1
                elif c+1 in losing_trials:
                    for channel in range(tot_channels):
                        sEEG_BL_l[l_count,:,channel] = seeg[x:int(BL_end[c]), channel]
                    l_count += 1

            BL_l = np.mean(sEEG_BL_l, axis=(0,1))
            BL_w = np.mean(sEEG_BL_w, axis=(0,1))

            # Estimate ERPs
            l_ERP = np.mean(sEEG_l - BL_l, axis=0)
            w_ERP = np.mean(sEEG_w - BL_w, axis=0)

            # Estimate std of the mean (standard error)
            l_sdm = (np.std(sEEG_l - BL_l, axis=0))/len(losing_trials)
            w_sdm = (np.std(sEEG_w - BL_w, axis=0))/len(winning_trials)

            # Statistics: 2samples t-test
            _, p_values = sy.stats.ttest_ind(sEEG_w - BL_w, sEEG_l - BL_l, axis=0)
            
            # Bonferroni correction
            #p_values = p_values*(p_values.shape[0]*p_values.shape[1])

            # FDR correction
            for c in range(tot_channels):
                _, p_values[:,c] = fdrcorrection(p_values[:,c], alpha=0.05, method='indep', is_sorted=False)
            
            # Estimate significant intervals
            significant = p_values < 0.05
            threshold = 25 #is it ok? How do I choose it?
            sign_th = np.zeros((int(1.5 * sr),tot_channels), dtype=bool)
            for x in range(tot_channels):
                for c, t in enumerate(significant[:,x]):
                    if not t:
                        continue
                    elif t and False not in significant[c:c+threshold,x]:
                        sign_th[c:c+threshold, x] = True
            
            
            # Plot ERPs with confidence interval
            t = linspace(-500, 1000, sr + int(sr/2), endpoint = False)
            for c, channel in enumerate(channels):
                xw = w_ERP[:,c]
                xl = l_ERP[:,c]
                u_w = xw + 2 * w_sdm[:,c]
                b_w = xw - 2 * w_sdm[:,c]
                u_l = xl + 2 * l_sdm[:,c]
                b_l = xl - 2 * l_sdm[:,c]
                # Plot ERP + confidence interval
                plt.figure()
                plt.plot(t, xw, 'b', lw=3, label='winning')
                plt.fill_between(t, u_w, b_w, color = 'b', alpha = 0.3)
                plt.plot(t, xl, 'y', lw=3, label='losing')
                plt.fill_between(t, u_l, b_l, color = 'y', alpha = 0.3)
                
                # Shade where significant
                ymax = max([max(u_w), max(u_l)])
                ymin = min([min(b_w), min(b_l)])
                lim = max([ymax, abs(ymin)])
                ymin = -(lim + 10*(lim)/100)
                ymax = (lim + 10*(lim)/100)
                plt.ylim([ymin, ymax])
                plt.fill_between(t, ymin, ymax, where=(sign_th[:,c]), color = 'k', alpha = 0.1)
                plt.fill_between(t, ymin, ymax, where=(t==0), color = 'k', alpha=1)

                # Remove white space and set title and legend
                plt.margins(x=0)
                plt.autoscale(tight=True)
                plt.xlabel('Time [ms]')
                plt.ylabel('Envelope')
                plt.title('channel ' + str(channel))
                plt.legend(loc="upper left")    

                # Save Images
                if True in sign_th[:,c]:
                    directory = (output_path + participant + '/Envelope_WvsL/{}/significant/'.format(filter_used))
                else:
                    directory = (output_path + participant + '/Envelope_WvsL/{}/not_s/'.format(filter_used))

                if not os.path.exists(directory):
                    os.makedirs(directory)
                    plt.savefig(directory + str(c+1) + '_' + str(channel))
                else:
                    plt.savefig(directory + str(c+1) + '_' + str(channel))
