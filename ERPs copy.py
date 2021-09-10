from numpy.core.function_base import linspace
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
import os
import Preprocessing_classifier

def ERP(classify='accuracy', lowpass='', participants = []):
    data_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Data_Analysis/Data/PreprocessedData_rereferencing/baseline-1.5_{}/'.format(classify)
    out_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Data_Analysis/Results/ERPs_rereference/{}/'.format(classify)
    participants = participants
    sampling_rates = [1024, 1024, 1024, 1024]#, 1024, 1200]

    for pNr, participant in enumerate(participants):
    # Load data   
        labels = np.load(data_path + participant + '_' + classify + '_labels.npy')
        channels = np.load(data_path + participant + '_channels.npy')
        sr = sampling_rates[pNr]

        if lowpass == 'low':
            seeg = np.load(data_path + participant + '_low_pass_preprocessed_sEEG.npy')
            string='_low_pass'
        elif lowpass == 'superlow':
            seeg = np.load(data_path + participant + '_superlow_pass_preprocessed_sEEG.npy')
            string='_superlow_pass'
        else:
            seeg = np.load(data_path + participant + '_preprocessed_sEEG.npy')
            string=''

    # Baseline correction
        seeg_BL = np.mean(seeg[:,int(0.3*sr):int(0.5*sr),:], axis=(0,1))
        seeg_corrected = seeg - seeg_BL

    # Distinguish correct from incorrect trials
        corr = seeg_corrected[labels>0,:,:]
        incorr = seeg_corrected[labels<1,:,:]
        
    # Estimate ERPs and stds
        i_ERP = np.mean(incorr, axis=0)
        c_ERP = np.mean(corr, axis=0)
        i_sdm = (np.std(incorr, axis=0))/incorr.shape[0]
        c_sdm = (np.std(corr, axis=0))/corr.shape[0]

    # Statistics: 2samples t-test
        _, p_values = ttest_ind(corr, incorr, axis=0)
          
    # FDR correction
        for cNr, channel in enumerate(channels):
            _, p_values[:,cNr] = fdrcorrection(p_values[:,cNr], alpha=0.05, method='indep', is_sorted=False)
    
    # Estimate significant intervals (only intervals longer than 50 ms)
            significant = p_values < 0.05
            threshold = 50
            sign_th = np.zeros((int(1.5 * sr),len(channels)), dtype=bool)
            for x in range(len(channels)):
                for ch, t in enumerate(significant[:,x]):
                    if not t:
                        continue
                    elif t and False not in significant[ch:ch+threshold,x]:
                        sign_th[ch:ch+threshold, x] = True
            
    # Plot ERPs with confidence interval x channel, x subject
        t = linspace(-500, 1000, sr + int(sr/2), endpoint = False) # x axis
        for c, channel in enumerate(channels):
        # define ERP, and confidence intervals
            xc = c_ERP[:,c]
            xi = i_ERP[:,c]
            u_c = xc + 2 * c_sdm[:,c]
            b_c = xc - 2 * c_sdm[:,c]
            u_i = xi + 2 * i_sdm[:,c]
            b_i = xi - 2 * i_sdm[:,c]
        # Define line colors depending on classifier condition
            if classify == 'accuracy':
                c_color = 'g'
                c_label = 'correct'
                i_color = 'r'
                i_label = 'incorrect'
            elif classify == 'stim_valence':
                c_color = 'b'
                c_label = 'winning stimulus'
                i_color = 'y'
                i_label = 'losing stimulus'
            elif classify == 'decision':
                c_color = 'm'
                c_label = 'choice: W'
                i_color = 'c'
                i_label = 'choice: L'
        # Plot ERPs + confidence intervals
            plt.figure(tight_layout=True)
            plt.plot(t, xc, c_color, lw=3, label=c_label)
            plt.fill_between(t, u_c, b_c, color=c_color, alpha = 0.3)
            plt.plot(t, xi, i_color, lw=3, label=i_label)
            plt.fill_between(t, u_i, b_i, color=i_color, alpha = 0.3)
        # Set y axis limits and shade significant intervals
            ymax = max([max(u_c), max(u_i)])
            ymin = min([min(b_c), min(b_i)])
            lim = max([ymax, abs(ymin)])
            ymin = -(lim + 10*(lim)/100)
            ymax = (lim + 10*(lim)/100)
            plt.ylim([ymin, ymax])
            plt.fill_between(t, ymin, ymax, where=(sign_th[:,c]), color = 'k', alpha = 0.1)
        #Add lines at point 0
            plt.axvline(x=0, color='k')
            plt.axhline(y=0,color='k', linestyle=':')
        # Remove white space and set title and legend
            plt.margins(x=0)
            plt.xlabel('Time [ms]')
            plt.ylabel('Voltage')
            plt.title('channel ' + str(channel))
            plt.legend(loc="upper left")    
        # Save Images
            if True in sign_th[:,c]:
                directory = (out_path + participant + string + '/significant/')
            else:
                directory = (out_path + participant + string + '/not_s/')

            if not os.path.exists(directory):
                os.makedirs(directory)
                plt.savefig(directory + str(c+1) + '_' + str(channel) + '_' + classify)
            else:
                plt.savefig(directory + str(c+1) + '_' + str(channel) + '_' + classify)


if __name__=="__main__":
    #Preprocessing_classifier.preprocess_data(reref='laplacian', feature='preprocess_only', window='baseline', window_length=1.5, preprocess_lowpass='superlow', participants=['kh21', 'kh22', 'kh23', 'kh24'])
    ERP(lowpass='superlow', participants=['kh21', 'kh22', 'kh23', 'kh24'])