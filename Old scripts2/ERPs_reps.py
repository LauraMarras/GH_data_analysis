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

        # Distinguish reps
        first = []
        second = []
        third = []
        for x in markers:
                if 'Sum' in x[0]:
                    summary = ((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                    if summary[2] == '1':
                        first.append(int(summary[0]))
                    elif summary[2] == '2':
                        second.append(int(summary[0]))
                    elif summary[2] == '3':
                        third.append(int(summary[0]))

        # Extract fixation + feedback epochs
        BL_indices = [x for x in range(len(markers)) if 'Start Cross' in markers[x][0]]
        FB_indices = [x for x in range(len(markers)) if 'Start Trial' in markers[x][0]]
        FB_indices.append([x for x in range(len(markers)) if 'End Experiment' in markers[x][0]][0])
        FB_indices = FB_indices[1:]

        BL_start = np.array([locate_pos(seeg_ts, x) for x in task_ts[BL_indices]])
        BL_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[BL_start]]) + sr/2
        FB_end = np.array([locate_pos(seeg_ts, x) for x in seeg_ts[BL_start]]) + int(1.5 * sr)
        
        # Filter data
        for filter in ['lowpass']:
            seeg, filter_used = filterdata(seeg_notfiltered, filter)
            
            # Store sEEG data into correct vs incorrect (trial x time x channel) arrays
            sEEG = np.zeros((tot_trials, int(1.5 * sr), tot_channels))
            for c, x in enumerate(BL_start):
                for channel in range(tot_channels):
                    sEEG[c,:,channel] = seeg[x:int(FB_end[c]), channel]

            # Baseline correction
            sEEG_BL = np.zeros((tot_trials, int(sr/2), tot_channels))
            for c, x in enumerate(BL_start):
                for channel in range(tot_channels):
                    sEEG_BL[c,:,channel] = seeg[x:int(BL_end[c]), channel]
            BL = np.mean(sEEG_BL, axis=1)

            for channel in range(tot_channels):
                for trial in range(tot_trials):
                    sEEG[trial,:,channel] = sEEG[trial,:,channel] - BL[trial, channel]
            
            # Store into 1st vs 2nd vs 3rd rep
            firsts_i = [x - 1 for x in first if x in incorrect_trials]
            seconds_i = [x - 1 for x in second if x in incorrect_trials]
            thirds_i = [x - 1 for x in third if x in incorrect_trials]
            firsts_c = [x - 1 for x in first if x in correct_trials]
            seconds_c = [x - 1 for x in second if x in correct_trials]
            thirds_c = [x - 1 for x in third if x in correct_trials]

            # Estimate ERPs
            ERP_i_1 = sEEG[firsts_i, :, :].mean(axis = 0)
            ERP_i_2 = sEEG[seconds_i, :, :].mean(axis = 0)
            ERP_i_3 = sEEG[thirds_i, :, :].mean(axis = 0)
            ERP_c_1 = sEEG[firsts_c, :, :].mean(axis = 0)
            ERP_c_2 = sEEG[seconds_c, :, :].mean(axis = 0)
            ERP_c_3 = sEEG[thirds_c, :, :].mean(axis = 0)

            # Estimate std of the mean (standard error)
            i_1_sdm = (np.std(sEEG[firsts_i, :, :], axis=0))/len(sEEG[firsts_i, :, :])
            c_1_sdm = (np.std(sEEG[firsts_c, :, :], axis=0))/len(sEEG[firsts_c, :, :])
            i_2_sdm = (np.std(sEEG[seconds_i, :, :], axis=0))/len(sEEG[seconds_i, :, :])
            c_2_sdm = (np.std(sEEG[seconds_c, :, :], axis=0))/len(sEEG[seconds_c, :, :])
            i_3_sdm = (np.std(sEEG[thirds_i, :, :], axis=0))/len(sEEG[thirds_i, :, :])
            c_3_sdm = (np.std(sEEG[thirds_c, :, :], axis=0))/len(sEEG[thirds_c, :, :])

            # Statistics: 2samples t-test
            _, p_values_c1_2 = sy.stats.ttest_ind(sEEG[firsts_c, :, :], sEEG[seconds_c, :, :], axis=0)
            _, p_values_i1_2 = sy.stats.ttest_ind(sEEG[firsts_i, :, :], sEEG[seconds_i, :, :], axis=0)
            _, p_values_c1_3 = sy.stats.ttest_ind(sEEG[firsts_c, :, :], sEEG[thirds_c, :, :], axis=0)
            _, p_values_i1_3 = sy.stats.ttest_ind(sEEG[firsts_i, :, :], sEEG[thirds_i, :, :], axis=0)

            
            # Bonferroni correction
            #p_values = p_values*(p_values.shape[0]*p_values.shape[1])

            # FDR correction
            for c in range(tot_channels):
                _, p_values_c1_2[:,c] = fdrcorrection(p_values_c1_2[:,c], alpha=0.05, method='indep', is_sorted=False)
                _, p_values_i1_2[:,c] = fdrcorrection(p_values_i1_2[:,c], alpha=0.05, method='indep', is_sorted=False)
                _, p_values_c1_3[:,c] = fdrcorrection(p_values_c1_3[:,c], alpha=0.05, method='indep', is_sorted=False)
                _, p_values_c1_3[:,c] = fdrcorrection(p_values_c1_3[:,c], alpha=0.05, method='indep', is_sorted=False)
            
            # Estimate significant intervals
            significant_c1_2 = p_values_c1_2 < 0.05
            threshold = 25
            sign_th_c1_2 = np.zeros((int(1.5 * sr),tot_channels), dtype=bool)
            for x in range(tot_channels):
                for c, t in enumerate(significant_c1_2[:,x]):
                    if not t:
                        continue
                    elif t and False not in significant_c1_2[c:c+threshold,x]:
                        sign_th_c1_2[c:c+threshold, x] = True

            significant_i1_2 = p_values_i1_2 < 0.05
            threshold = 25
            sign_th_i1_2 = np.zeros((int(1.5 * sr),tot_channels), dtype=bool)
            for x in range(tot_channels):
                for c, t in enumerate(significant_i1_2[:,x]):
                    if not t:
                        continue
                    elif t and False not in significant_i1_2[c:c+threshold,x]:
                        sign_th_i1_2[c:c+threshold, x] = True
            
            significant_c1_3 = p_values_c1_3 < 0.05
            sign_th_c1_3 = np.zeros((int(1.5 * sr),tot_channels), dtype=bool)
            for x in range(tot_channels):
                for c, t in enumerate(significant_c1_3[:,x]):
                    if not t:
                        continue
                    elif t and False not in significant_c1_3[c:c+threshold,x]:
                        sign_th_c1_3[c:c+threshold, x] = True

            significant_i1_3 = p_values_i1_3 < 0.05
            sign_th_i1_3 = np.zeros((int(1.5 * sr),tot_channels), dtype=bool)
            for x in range(tot_channels):
                for c, t in enumerate(significant_i1_3[:,x]):
                    if not t:
                        continue
                    elif t and False not in significant_i1_3[c:c+threshold,x]:
                        sign_th_i1_3[c:c+threshold, x] = True
            
            
            # Plot ERPs with confidence interval
            t = linspace(-500, 1000, sr + int(sr/2), endpoint = False)
            for c, channel in enumerate(channels):
                xc1 = ERP_c_1[:,c]
                xi1 = ERP_i_1[:,c]
                xc2 = ERP_c_2[:,c]
                xi2 = ERP_i_2[:,c]
                xc3 = ERP_c_3[:,c]
                xi3 = ERP_i_3[:,c]
                u_c1 = xc1 + 2 * c_1_sdm[:,c]
                b_c1 = xc1 - 2 * c_1_sdm[:,c]
                u_i1 = xi1 + 2 * i_1_sdm[:,c]
                b_i1 = xi1 - 2 * i_1_sdm[:,c]
                u_c2 = xc2 + 2 * c_2_sdm[:,c]
                b_c2 = xc2 - 2 * c_2_sdm[:,c]
                u_i2 = xi2 + 2 * i_2_sdm[:,c]
                b_i2 = xi2 - 2 * i_2_sdm[:,c]
                u_c3 = xc3 + 2 * c_3_sdm[:,c]
                b_c3 = xc3 - 2 * c_3_sdm[:,c]
                u_i3 = xi3 + 2 * i_3_sdm[:,c]
                b_i3 = xi3 - 2 * i_3_sdm[:,c]
                # Plot ERP + confidence interval
                fig, (ax1, ax2) = plt.subplots(2)
                ax1.plot(t, xc1, color='yellowgreen', lw=2, label='correct first rep')
                ax1.plot(t, xc2, color='limegreen', lw=2, label='correct second rep')
                ax1.plot(t, xc3, color='green', lw=2, label='correct third rep')
                ax1.fill_between(t, u_c1, b_c1, color='g', alpha = 0.1)
                ax1.fill_between(t, u_c2, b_c2, color ='g', alpha = 0.1)
                ax1.fill_between(t, u_c3, b_c3, color = 'g', alpha = 0.1)
                ax2.plot(t, xi1, color='darkorange', lw=2, label='incorrect first rep')
                ax2.plot(t, xi2, color='tomato', lw=2, label='incorrect second rep')
                ax2.plot(t, xi3, color='red', lw=2, label='incorrect third rep')
                ax2.fill_between(t, u_i1, b_i1, color = 'r', alpha = 0.1)
                ax2.fill_between(t, u_i2, b_i2, color = 'r', alpha = 0.1)
                ax2.fill_between(t, u_i3, b_i3, color = 'r', alpha = 0.1)
                
                # Shade where significant
                ymax = max([max(u_c1), max(u_i1), max(u_c2), max(u_i2), max(u_c3), max(u_i3)])
                ymin = min([min(b_c1), min(b_i1), min(b_c2), min(b_i2), min(b_c3), min(b_i3)])
                lim = max([ymax, abs(ymin)])
                ymin = -(lim + 10*(lim)/100)
                ymax = (lim + 10*(lim)/100)
                ax1.set_ylim([ymin, ymax])
                ax2.set_ylim([ymin, ymax])
                ax1.fill_between(t, ymin, ymax, where=(sign_th_c1_2[:,c]), color = 'b', alpha = 0.2, label = 'first vs second')
                ax1.fill_between(t, ymin, ymax, where=(sign_th_c1_3[:,c]), color = 'y', alpha = 0.2, label = 'first vs third')
                ax2.fill_between(t, ymin, ymax, where=(sign_th_i1_2[:,c]), color = 'b', alpha = 0.2, label = 'first vs second')
                ax2.fill_between(t, ymin, ymax, where=(sign_th_i1_3[:,c]), color = 'y', alpha = 0.2, label = 'first vs third')
                ax1.fill_between(t, ymin, ymax, where=(t==0), color = 'k', alpha=1)
                ax2.fill_between(t, ymin, ymax, where=(t==0), color = 'k', alpha=1)

                # Remove white space and set title and legend
                ax1.margins(x=0)
                ax2.margins(x=0)
                ax2.set_xlabel('Time [ms]')
                ax1.set_ylabel('Voltage')
                ax2.set_ylabel('Voltage')
                fig.suptitle('channel ' + str(channel))
                ax1.legend(loc="best", fontsize='xx-small', ncol=2)    
                ax2.legend(loc="best", fontsize='xx-small', ncol=2) 

                # Save Images
                if True in sign_th_c1_2[:,c] or True in sign_th_i1_2[:,c] or True in sign_th_c1_3[:,c] or True in sign_th_i1_3[:,c]:
                    directory = (output_path + participant + '/ERP_reps_subs/{}/significant/'.format(filter_used))
                else:
                    directory = (output_path + participant + '/ERP_reps_subs/{}/not_s/'.format(filter_used))

                if not os.path.exists(directory):
                    os.makedirs(directory)
                    plt.savefig(directory + str(c+1) + '_' + str(channel))
                else:
                    plt.savefig(directory + str(c+1) + '_' + str(channel))
