import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import permutation_test_score

def decoder_singlePP(reref='elecShaftR', window='long_FB', classify='accuracy', participants=[], repetitions=['rep_1', 'rep_2_3', 'rep_all'], band='gamma', n_permutations=1000, plotting=True):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/'
   label_path = data_path + classify + '_labels' + '/' 
   feature_path = data_path + window + '/' + reref + '/' + reref + '_'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/{}/'.format(window + '_' + classify)
   participants = participants
   if window == 'long_FB':
      ww = 'baseline and feedback '
   if band == 'gamma':
      band_string = ' High gamma (70-120 Hz)'
   elif band == 'beta':
      band_string = ' Beta (13-30 Hz)'
   elif band == 'theta':
      band_string = ' Theta (4-7 Hz)'
   elif band == 'delta':
      band_string = ' Delta (1-3 Hz)'
   elif band == 'alpha':
      band_string = ' Alfa (8-12 Hz)'
   else: 
      band_string = band

#### Classifier based on all bands power from one channel
   PPs_means = {}
   PPs_perms = {}
   PPs_pvals = {}
   for pNr, participant in enumerate(participants):
      if plotting:
         fig, axs = plt.subplots(3, tight_layout=True)
         fig.suptitle('P0' + str(pNr+1) + ' ROC AUC scores over time\n Classification of ' + classify + ' during ' + ww + '\n based on ' + reref + band_string)
      for r, rep in enumerate(repetitions):
         if '1' in rep:
            rep_string = 'First repetition'
         elif '2' in rep:
            rep_string = 'Second and Third repetitions'
         elif 'all' in rep:
            rep_string = 'All repetitions together'

      # Load data   
         features = np.load(feature_path + participant + '_' + band + '_' + rep + '_envelope_windowed.npy')
         labels = np.load(label_path + participant + '_' + classify + '_labels_' + rep + '.npy')
         channels = np.load(data_path + participant + '_channels.npy')

         windows = [*range(0, features.shape[1])]


      # Define Classifier, run Crossvalidation and permutation and store scores and stds for each window
         score_means = np.zeros((len(windows)))
         score_perms = np.zeros((len(windows),n_permutations))
         p_vals = np.zeros((len(windows)))

         for win in windows:
            print('participant nr.' + str(pNr) + ' - ' + rep + ' windows nr.' + str(win))
            X = features[:,win,:]
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            kf = StratifiedKFold(n_splits=5)
            #CV_scores_env = cross_val_score(clf, X, labels, cv=kf, scoring='roc_auc')
            CV_scores_env, perm_scores_env, pvalue_env = permutation_test_score(clf, X, labels, scoring="roc_auc", cv=kf, n_permutations=n_permutations)
            
            score_means[win] = CV_scores_env
            p_vals[win] = pvalue_env
            score_perms[win,:] = perm_scores_env

         PPs_means[rep +'_'+ str(pNr)]=score_means
         PPs_perms[rep +'_'+ str(pNr)]=score_perms
         PPs_pvals[rep +'_'+ str(pNr)]=p_vals

   # Plot scores
         if plotting:
            x1 = [x - 500 for x in [x + 25 for x in [*range(0, 1989, 51)]]]
            y1 = score_means
            yupper = np.ma.masked_where(p_vals > 0.05, y1)

            axs[r].plot(x1,y1, color='b')
            axs[r].plot(x1,yupper, 'r*', label='p<0.05')

            axs[r].axhline(y=0.5,color='r', linestyle=':', lw=1)
            axs[r].axvline(x=0,color='k', lw=2)
            axs[r].axvline(x=1000,color='k', lw=2)
            #axs[r].fill_between(x1[:], 0.7, 1, color='r', alpha = 0.1)
            axs[r].set_title(rep_string)
            axs[r].set_ylabel('ROC AUC')
            axs[r].set_xlabel('time (ms)')
            y_tick = [x/100 for x in [*range(0, 101, 25)]]
            axs[r].set_yticks(y_tick) 
            axs[r].set_ylim(0,1)      
            axs[r].set_xlim(x1[0],x1[-1])
      
      if plotting:
         for ax in fig.get_axes():
            ax.label_outer()
            ax.margins(x=0)
         plt.legend(loc='best')

         # Save Figure
         if not os.path.exists(out_path):
            os.makedirs(out_path)   
         plt.savefig(out_path + participant + '/' + participant + '_' + reref + '_' + band + '_significance', dpi=300)

# Save dataset
   if not os.path.exists(out_path):
      os.makedirs(out_path)   
   np.save(out_path + 'PPs_perms_' + reref + '_' + band, PPs_perms)
   np.save(out_path + 'PPs_means_' + reref + '_' + band, PPs_means)
   np.save(out_path + 'PPs_pvals_' + reref + '_' + band, PPs_pvals)




### Average across PPs
def decoder_allPPs(reref='elecShaftR', window='long_FB', classify='accuracy', participants=[], repetitions=['rep_1', 'rep_2_3', 'rep_all'], band='gamma', n_permutations=1000):
   out_path = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/{}/'.format(window + '_' + classify)
   
   # Load data
   PPs_means = np.load(out_path + 'PPs_means_' + reref + '_' + band + '.npy', allow_pickle=True).item()
   PPs_perms = np.load(out_path + 'PPs_perms_' + reref + '_' + band + '.npy', allow_pickle=True).item()
   
   # Average across PPs
   PPs_means_avg = {}
   PPs_perms_avg = {}
   PPs_pvals_avg = {}
   windows = [*range(0, len(PPs_means['rep_1_0']))]
   
   for rep in repetitions:
      PPs_means_arr = np.zeros((len(participants), len(windows)))
      for key, val in PPs_means.items():
         if rep in key:
            PPs_means_arr[int(key[-1]),:]=val

      PPs_means_avg[rep] = np.mean(PPs_means_arr, axis=0)

      PPs_perms_arr = np.zeros((len(participants), len(windows), n_permutations))
      for key, val in PPs_perms.items():
         if rep in key:
            PPs_perms_arr[int(key[-1]),:]=val

      PPs_perms_avg[rep] = np.mean(PPs_perms_arr, axis=0)

      PPs_pvals_avg[rep] = np.zeros(len(windows))
      for w, win in enumerate(PPs_perms_avg[rep]):
         p_val = sum(p > PPs_means_avg[rep][w] for p in win) / len(win)
         PPs_pvals_avg[rep][w] = p_val

      # Save dataset
   if not os.path.exists(out_path):
      os.makedirs(out_path)   
   np.save(out_path + 'PPs_perms_avg_' + reref + '_' + band, PPs_perms_avg)
   np.save(out_path + 'PPs_means_avg_' + reref + '_' + band, PPs_means_avg)
   np.save(out_path + 'PPs_pvals_avg_' + reref + '_' + band, PPs_pvals_avg)

           
if __name__=="__main__":
   '''
   method = ['single_band-all_channels', 'single_channel-all_bands']
   window = ['feedback', 'baseline', 'stimulus']
   classify = ['accuracy', 'stim_valence', 'decision']
   '''
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'long_FB' #'feedback' # 'baseline', 'stimulus', 'decision'
   classify = 'accuracy' #'decision', 'stim_valence', 'accuracy', 'learning'
   participants = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = ['rep_1', 'rep_2_3', 'rep_all']
   band = 'gamma' #'delta', 'theta', 'alpha', 'beta'
   n_permutations = 1000
   plotting = True
   
   
   #decoder_singlePP(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, band=band, n_permutations=n_permutations, plotting=plotting)
   decoder_allPPs(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, band=band, n_permutations=n_permutations)

