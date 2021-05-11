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
    seeg = data[1]['time_series']
    seeg_ts = data[1]['time_stamps']

    sr = int(data[1]['info']['nominal_srate'][0])

    
    # Filter data
    filter_used = ''
    def filterdata(filter):
        seeg = seeg.astype(float)
        if filter == 'gamma':
            seeg = mne.filter.filter_data(seeg.T, sr, 120, 70, method='iir').T
            filter_used = 'gamma'
        elif filter == 'theta':
            seeg = mne.filter.filter_data(seeg.T, sr, 7, 4, method='iir').T
            filter_used = 'theta'
        else:
            print('Filter not defined, returned unfiltered data')
        return seeg
    
     
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

    # Compute envelope
    hilbert3 = lambda x: sy.signal.hilbert(x, sy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]
    c_env = np.abs(hilbert3(sEEG_c - BL_ERP)) #is baseline correction needed here?
    i_env = np.abs(hilbert3(sEEG_i - BL_ERP))

    # Estimate ERPs on corrected trials (baseline substraction)
    #i_ERP = np.mean(sEEG_i - BL_ERP, axis=0)
    #c_ERP = np.mean(sEEG_c - BL_ERP, axis=0)

    i_ERP = np.mean(i_env, axis=0)
    c_ERP = np.mean(c_env, axis=0)
    
    # Estimate std of the mean (standard error)
    #i_sdm = (np.std(sEEG_i - BL_ERP, axis=0))/len(incorrect_trials)
    #c_sdm = (np.std(sEEG_c - BL_ERP, axis=0))/len(correct_trials)

    i_sdm = (np.std(i_env, axis=0))/len(incorrect_trials)
    c_sdm = (np.std(c_env, axis=0))/len(correct_trials)

    # Statistics: 2samples t-test
    #_, p_values = sy.stats.ttest_ind((sEEG_c - BL_ERP), (sEEG_i - BL_ERP), axis=0)
    _, p_values = sy.stats.ttest_ind(c_env, i_env, axis=0)

    #Bonferroni correction
    #p_values = p_values*(p_values.shape[0]*p_values.shape[1])

    #FDR correction
    for c in range(130):
        _, p_values[:,c] = fdrcorrection(p_values[:,c], alpha=0.05, method='indep', is_sorted=False)
    
    #Plot ERPs with confidence interval
    t = np.array(range(int(sr/2)))
    significant = p_values < 0.05
    for c, channel in enumerate(channels):
        plt.figure()
        plt.plot(t, c_ERP[:,c], 'g', lw=3, label='correct')
        plt.fill_between(t, c_ERP[:,c] + 2 * c_sdm[:,c], c_ERP[:,c] - 2 * c_sdm[:,c], color = 'g', alpha = 0.3)
        plt.plot(t, i_ERP[:,c], 'r', lw=3, label='incorrect')
        plt.fill_between(t, i_ERP[:,c] + 2 * i_sdm[:,c], i_ERP[:,c] - 2 * i_sdm[:,c], color = 'r', alpha = 0.3)

        #plt.fill_between(t, -200, 200, where=(significant[:,c]), color = 'k', alpha = 0.1)

        """ if (max(c_ERP[:,c]) or max(i_ERP[:,c])) > 150 or (min(i_ERP[:,c]) or min(c_ERP[:,c])) < -150:
            ymax = 200
            ymin = -200
            elif (max(c_ERP[:,c]) or max(i_ERP[:,c])) > 100 or (min(i_ERP[:,c]) or min(c_ERP[:,c])) < -100:
                ymax = 150
                ymin = -150
            elif (max(c_ERP[:,c]) or max(i_ERP[:,c])) > 50 or (min(i_ERP[:,c]) or min(c_ERP[:,c])) < -50:
                ymax = 100
                ymin = -100
            else:
                ymax = 50
                ymin = -50 """
        ymax = max([max(c_ERP[:,c]), max(i_ERP[:,c])]) + ((20*max([max(c_ERP[:,c]), max(i_ERP[:,c])]))/100)
        ymin = min([min(c_ERP[:,c]), min(i_ERP[:,c])]) - ((20*min([min(c_ERP[:,c]), min(i_ERP[:,c])]))/100)
        plt.ylim([ymin - ((10*ymin)/100), ymax + ((10*ymax)/100)])
        plt.fill_between(t, -ymin - ((10*ymin)/100), ymax + ((10*ymax)/100), where=(significant[:,c]), color = 'k', alpha = 0.1)

        plt.xlabel('Time [ms]')
        #plt.ylabel('Voltage')
        plt.ylabel('Energy') #?
        plt.title('channel ' + str(channel))
        plt.legend(loc="upper left")    
        #plt.show()

        if True in significant[:,c]:
            plt.savefig('./ERPs/FDR_filtered_gamma_envelope/' + str(c+1) + '_' + str(channel))
        else:
            plt.savefig('./ERPs/FDR_filtered_gamma_envelope/not_s/' + str(c+1) + '_' + str(channel))
       
    # Save epoched data
    np.save('output_path/FBc_sEEG_{}.npy'.format(filter_used), sEEG_c)
    np.save('output_path/FBi_sEEG{}.npy'.format(filter_used), sEEG_i)
    np.save('output_path/BL_sEEG_{}.npy'.format(filter_used), sEEG_BL)
        