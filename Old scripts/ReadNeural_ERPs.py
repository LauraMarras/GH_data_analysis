import pyxdf
import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy as sy
from statsmodels.stats.multitest import fdrcorrection
import mne

data_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/StreamFiles/exp001'
output_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/Results'

def locate_pos(available, targets):
    pos = bisect.bisect_right(available, targets)
    #print('pos is: ' + str(pos))
    if pos == 0:
        return 0
    if pos == len(available):
        return len(available)-1
    if abs(available[pos]-targets) < abs(available[pos-1]-targets):
        return pos
    else:
        return pos-1  

if __name__=="__main__":
    data, _ = pyxdf.load_xdf(data_path + '/Epilpp_test.xdf')
   
    markers = data[0]['time_series']
    task_ts = data[0]['time_stamps']
    seeg_data = data[1]['time_series']
    seeg_ts = data[1]['time_stamps']

    sr = int(data[1]['info']['nominal_srate'][0])

    
    # Filter data
    def filterdata(data, filter):
        seeg = data.astype(float)
        if filter == 'gamma':
            seeg = mne.filter.filter_data(seeg.T, sr, 70, 120, method='iir').T
            filter_used = 'gamma'
        elif filter == 'theta':
            seeg = mne.filter.filter_data(seeg.T, sr, 4, 7, method='iir').T
            filter_used = 'theta'
        elif filter == 'beta':
            seeg = mne.filter.filter_data(seeg.T, sr, 12.5, 30, method='iir').T
            filter_used = 'beta'
        else:
            filter_used = 'nf' #no filter
            seeg = mne.filter.filter_data(seeg.T, sr, 0, 511, method='iir').T
        return seeg, filter_used
    
    seeg, filter_used = filterdata(seeg_data, 'beta')
     
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

    # Extract Feedback epochs
    FB_indices = [x for x in range(len(markers)) if markers[x] == ['start Fb']]
    FB_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[FB_indices]])
    FB_end = FB_start + int(sr/2) # instead of: FB_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[FB_start]]) + int(sr/2)
    
    # Store sEEG data into correct vs incorrect (trial x time x channel) arrays
    sEEG_c = np.zeros((len(correct_trials), int(sr/2), tot_channels))
    sEEG_i = np.zeros((len(incorrect_trials), int(sr/2), tot_channels))
    c_count = 0
    i_count = 0
    for c, x in enumerate(FB_start):
        if c+1 in correct_trials:
            for channel in range(tot_channels):
                sEEG_c[c_count,:,channel] = seeg[x:int(FB_end[c]), channel]
            c_count += 1
        elif c+1 in incorrect_trials:
            for channel in range(tot_channels):
                sEEG_i[i_count,:,channel] = seeg[x:int(FB_end[c]), channel]
            i_count += 1
    
    # Extract Baseline epochs (during fixation before FB presentation) and store them in (trial x time x channel) array 
    BL_indices = [x for x in range(len(markers)) if 'Start Cross' in markers[x][0]]
    BL_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[BL_indices]])
    BL_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[BL_start]]) + sr/2

    sEEG_BL = np.zeros((tot_trials, int(sr/2), tot_channels))
    for c, x in enumerate(BL_start):
        for channel in range(tot_channels):
            sEEG_BL[c,:,channel] = seeg[x:int(BL_end[c]), channel]

    # Estimate baseline mean for each channel --> mean across trials
    BL_ERP = np.mean(sEEG_BL, axis=(0,1))

    # Estimate ERPs on corrected trials (baseline substraction)
    i_ERP = np.mean(sEEG_i - BL_ERP, axis=0)
    c_ERP = np.mean(sEEG_c - BL_ERP, axis=0)

    # Estimate std of the mean (standard error)
    i_sdm = (np.std(sEEG_i - BL_ERP, axis=0))/len(incorrect_trials)
    c_sdm = (np.std(sEEG_c - BL_ERP, axis=0))/len(correct_trials)

    # Statistics: 2samples t-test
    _, p_values = sy.stats.ttest_ind((sEEG_c - BL_ERP), (sEEG_i - BL_ERP), axis=0)
    
    # Bonferroni correction
    #p_values = p_values*(p_values.shape[0]*p_values.shape[1])

    # FDR correction
    for c in range(130):
        _, p_values[:,c] = fdrcorrection(p_values[:,c], alpha=0.05, method='indep', is_sorted=False)
    
    # Estimate significant intervals
    significant = p_values < 0.05
    threshold = 25 #is it ok? How do I choose it?
    sign_th = np.zeros((int(sr/2),tot_channels), dtype=bool)
    for x in range(tot_channels):
        for c, t in enumerate(significant[:,x]):
            if not t:
                continue
            elif t and False not in significant[c:c+threshold,x]:
                sign_th[c:c+threshold, x] = True
    
    # Plot ERPs with confidence interval
    t = np.array(range(int(sr/2)))
    for c, channel in enumerate(channels):
        # Plot ERP + confidence interval
        plt.figure()
        plt.plot(t, c_ERP[:,c], 'g', lw=3, label='correct')
        plt.fill_between(t, c_ERP[:,c] + 2 * c_sdm[:,c], c_ERP[:,c] - 2 * c_sdm[:,c], color = 'g', alpha = 0.3)
        plt.plot(t, i_ERP[:,c], 'r', lw=3, label='incorrect')
        plt.fill_between(t, i_ERP[:,c] + 2 * i_sdm[:,c], i_ERP[:,c] - 2 * i_sdm[:,c], color = 'r', alpha = 0.3)

        # Shade where significant
        ymax = max([max(c_ERP[:,c]), max(i_ERP[:,c])]) #+ ((20*max([max(c_ERP[:,c]), max(i_ERP[:,c])]))/100)
        ymin = min([min(c_ERP[:,c]), min(i_ERP[:,c])]) #- ((20*min([min(c_ERP[:,c]), min(i_ERP[:,c])]))/100)
        lim = max([ymax, abs(ymin)])
        ymin = 0 - (lim + 10*(lim)/100)
        ymax = (lim + 10*(lim)/100)
        plt.ylim([ymin, ymax])
        plt.fill_between(t, ymin, ymax, where=(sign_th[:,c]), color = 'k', alpha = 0.1)

        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage')
        plt.title('channel ' + str(channel))
        plt.legend(loc="upper left")    
        #plt.show()

        if True in sign_th[:,c]:
            plt.savefig(output_path + '/ERPs/{}/significant/'.format(filter_used) + str(c+1) + '_' + str(channel))
        else:
            plt.savefig(output_path + '/ERPs/{}/not_s/'.format(filter_used) + str(c+1) + '_' + str(channel))
       
    # Save epoched data
    np.save(output_path + '/FBc_sEEG_{}.npy'.format(filter_used), sEEG_c)
    np.save(output_path + '/FBi_sEEG_{}.npy'.format(filter_used), sEEG_i)
    np.save(output_path + '/BL_sEEG_{}.npy'.format(filter_used), sEEG_BL)
        