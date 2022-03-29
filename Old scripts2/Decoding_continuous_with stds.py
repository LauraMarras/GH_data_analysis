import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import cross_val_score
from Plot_decoder_results import plot_decoding_results_average_PPs, plot_decoding_results_single_PP
from Plot_decoder_results import plot_decoding_results_single_PP
from Plot_decoder_results import plot_decoding_results_average_PPs_allbands

def decoder_singlePP(reref='elecShaftR', window='long_FB', classify='accuracy', participants=[], repetitions=['rep_1', 'rep_2_3', 'rep_all'], band='gamma', n_permutations=1000):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/'
   label_path = data_path + classify + '_labels' + '/' 
   feature_path = data_path + window + '/' + reref + '/' + reref + '_'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/standarderrors/{}/'.format(window + '_' + classify)

# Classifier
   PPs_means = {}
   PPs_perms = {}
   PPs_pvals = {}
   PPs_err = {}

   for pNr, participant in enumerate(participants):
      for rep in repetitions:

      # Load data   
         features = np.load(feature_path + participant + '_' + band + '_' + rep + '_envelope_windowed.npy')
         labels = np.load(label_path + participant + '_' + classify + '_labels_' + rep + '.npy')

         windows = [*range(0, features.shape[1])]


      # Define Classifier, run Crossvalidation and permutation and store scores each window
         score_means = np.zeros((len(windows)))
         errors = np.zeros(len(windows))
         score_perms = np.zeros((len(windows),n_permutations))
         error_perms = np.zeros((len(windows),n_permutations))
         p_vals = np.zeros((len(windows)))

         for win in windows:
            print('participant nr.' + str(pNr) + ' - ' + rep + ' windows nr.' + str(win))
            X = features[:,win,:]
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            kf = StratifiedKFold(n_splits=5)
            
            CV_scores_env = cross_val_score(clf, X, labels, scoring="roc_auc", cv=kf)
            score_means[win] = CV_scores_env.mean()
            errors[win] = CV_scores_env.std()

            # for perm in range(n_permutations):
            #    permuted_labels = np.random.permutation(labels)
            #    perm_scores_env = cross_val_score(clf, X, permuted_labels, scoring="roc_auc", cv=kf)
            
            #    score_perms[win,perm] = perm_scores_env.mean()
            #    error_perms[win,perm] = perm_scores_env.std()
            
         
            # p_vals[win] = np.count_nonzero(score_perms[win] > score_means[win]) / n_permutations

         PPs_means[rep +'_'+ str(pNr)]=score_means
         # PPs_perms[rep +'_'+ str(pNr)]=score_perms
         # PPs_pvals[rep +'_'+ str(pNr)]=p_vals
         PPs_err[rep +'_'+ str(pNr)]=errors

# Save dataset
   if not os.path.exists(out_path):
      os.makedirs(out_path)     
   # np.save(out_path + 'PPs_perms_' +  reref + '_' + band, PPs_perms)
   np.save(out_path + 'PPs_means_' +  reref + '_' + band, PPs_means)
   # np.save(out_path + 'PPs_pvals_' + reref + '_' + band, PPs_pvals)
   np.save(out_path + 'PPs_err_' + reref + '_' + band, PPs_err)


def decoder_allPPs(reref='elecShaftR', window='long_FB', classify='accuracy', participants=[], repetitions=['rep_1', 'rep_2_3', 'rep_all'], band='gamma', n_permutations=1000):
   out_path = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/{}/'.format(window + '_' + classify)
   out_path_err = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/standarderrors/{}/'.format(window + '_' + classify)

   # Load data
   PPs_means = np.load(out_path + 'PPs_means_' + reref + '_' + band + '.npy', allow_pickle=True).item()
   PPs_perms = np.load(out_path + 'PPs_perms_' + reref + '_' + band + '.npy', allow_pickle=True).item()
   PPs_errors = np.load(out_path_err + 'PPs_err_' + reref + '_' + band + '.npy', allow_pickle=True).item()

   # Average across PPs
   PPs_means_avg = {}
   PPs_perms_avg = {}
   PPs_pvals_avg = {}
   PPs_errors_avg = {}
   PPs_errors_avg1 = {}

   windows = [*range(0, len(PPs_means['rep_1_0']))]
   
   for rep in repetitions:
      PPs_means_arr = np.zeros((len(participants), len(windows)))
      for key, val in PPs_means.items():
         if rep in key:
            PPs_means_arr[int(key[-1]),:]=val

      PPs_means_avg[rep] = np.mean(PPs_means_arr, axis=0)

      PPs_errors_avg[rep] = np.std(PPs_means_arr, axis=0)

      PPs_errors_arr = np.zeros((len(participants), len(windows)))
      for key, val in PPs_errors.items():
         if rep in key:
            PPs_errors_arr[int(key[-1]),:]=val

      PPs_errors_avg1[rep] = np.mean(PPs_errors_arr, axis=0)

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
   np.save(out_path + 'PPs_errors_avg_' + reref + '_' + band, PPs_errors_avg)
   np.save(out_path + 'PPs_errors_avg_1' + reref + '_' + band, PPs_errors_avg1)
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
   participants = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25']
   repetitions = ['rep_1', 'rep_2_3', 'rep_all']
   bands = ['gamma', 'theta', 'delta']#,'alpha', 'beta'] #'gamma', 'delta', 'theta', 'alpha', 'beta'
   n_permutations = 1000
   action = 'save' #'show'
   
   for band in bands:
      #decoder_singlePP(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, band=band, n_permutations=n_permutations)
      decoder_allPPs(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, band=band, n_permutations=n_permutations)
      
      #plot_decoding_results_single_PP(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, band=band, action=action)
      #plot_decoding_results_average_PPs(reref=reref, window=window, classify=classify, repetitions=repetitions, band=band, action=action)
      #plot_decoding_results_average_PPs_allbands(reref=reref, window=window, classify=classify, repetitions=repetitions, bands=bands, action=action)
   