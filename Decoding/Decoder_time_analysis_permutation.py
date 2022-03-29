import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

def decoder_singlePP(reref='elecShaftR', window='long_FB', classify='accuracy', participants=[], repetitions=['rep_1', 'rep_2_3', 'rep_all'], band='gamma', n_permutations=1000):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/' + band + '/'
   
   for participant in participants:
      for rep in repetitions:
      # Load data   
         features = np.load(data_path + window + '/' + reref + '/' + reref + '_' + participant + '_' + band + '_' + rep + '_envelope_windowed.npy')
         labels = np.load(data_path + classify + '_labels' + '/' + participant + '_' + classify + '_labels_' + rep + '.npy')

      # Classifier + permutation test
         windows = [*range(0, features.shape[1])]
         score_means = np.zeros((len(windows)))
         errors = np.zeros(len(windows))
         score_perms = np.zeros((n_permutations))
         error_perms = np.zeros((n_permutations))
         p_vals = np.zeros((len(windows)))

         # Run permutation test
         for perm in range(n_permutations):
            permuted_labels = np.random.permutation(labels)
            random_window = np.random.choice(len(windows))
      
         # Define Classifier, run Crossvalidation for permuted labels dataset
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            kf = StratifiedKFold(n_splits=5)
            perm_scores_env = cross_val_score(clf, features[:,random_window,:], permuted_labels, cv=kf, scoring="roc_auc")
         
         # Store permuted scores, stds, threshold
            score_perms[perm] = perm_scores_env.mean()
            error_perms[perm] = perm_scores_env.std()

         # Run Classifier for each window
         for win in windows:
            X = features[:,win,:]
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            kf = StratifiedKFold(n_splits=5)
            CV_scores_env = cross_val_score(clf, X, labels, scoring="roc_auc", cv=kf)
            score_means[win] = CV_scores_env.mean()
            errors[win] = CV_scores_env.std()

         # Estimate p-value of each window
            p_vals[win] = np.count_nonzero(score_perms > score_means[win]) / n_permutations

         # Estimate significant threshold and significant windows
         score_perms_sorted = np.sort(score_perms)
         threshold = np.percentile(score_perms_sorted, q=[0.1, 1, 5, 95, 99, 99.9])
         significant_windows = [win for win in windows if score_means[win] > threshold[3]]

      # Save dataset
         if not os.path.exists(out_path + participant + '/'):
            os.makedirs(out_path + participant + '/')     
         np.savez(out_path  + participant + '/' + participant + '_decoder_' + band + '_' + rep, 
            score_means=score_means,
            p_vals_2=p_vals,
            errors=errors,
            score_perms_2=score_perms,
            error_perms_2=error_perms,
            threshold=threshold,
            significant_windows=significant_windows)

def grand_average(window='long_FB', classify='accuracy', participants=[], repetitions=['rep_1', 'rep_2_3', 'rep_all'], band='gamma'):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/' + band + '/'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/' + band + '/'
   
   # Load data
   for rep in repetitions:
      for pNr, participant in enumerate(participants):
         data = np.load(data_path + participant + '_decoder_' + band + '_' + rep + '.npz')
         
      # Store into means and perms arrays
         if pNr==0:
            score_means_pp = data['score_means']
            score_perms_pp = data['score_perms_2']
            errors_pp = data['errors']
         else:
            score_means_pp = np.c_[score_means_pp, data['score_means']]
            score_perms_pp = np.c_[score_perms_pp, data['score_perms_2']]
            errors_pp = np.c_[errors_pp, data['errors']]
         
      windows = [*range(0, score_means_pp.shape[0])]
      n_permutations = score_perms_pp.shape[0]
     
   # Estimate average scores across PPs and errors
      score_means_avg = np.mean(score_means_pp, axis=1)
      errors_avg = np.std(score_means_pp, axis=1) #old method
      errors_avg2 = np.square(np.sum(errors_pp**2, axis=1)/errors_pp.shape[1]) #new method using square root of mean of variances
      errors_avg3 = np.mean(errors_pp, axis=1) # last method: average of stds across PPs

   # Estimate p-value of each window
      p_vals = np.zeros(len(windows))
      for win in windows: 
         p_vals[win] = np.count_nonzero(score_perms_pp > score_means_avg[win]) / (n_permutations * score_perms_pp.shape[1])

   # Estimate significant threshold and significant windows
      score_perms_sorted = np.sort(score_perms_pp, axis=None)
      threshold = np.percentile(score_perms_sorted, q=[0.1, 1, 5, 95, 99, 99.9])
      significant_windows = [win for win in windows if score_means_avg[win] > threshold[3]]

   # Save dataset
      if not os.path.exists(out_path):
         os.makedirs(out_path)   
      np.savez(out_path + 'PPs_decoder_' + band + '_' + rep, 
         score_means_avg=score_means_avg,
         p_vals_avg=p_vals,
         errors_avg=errors_avg,
         errors_avg2=errors_avg2,
         errors_avg3=errors_avg3,
         threshold_avg=threshold,
         significant_windows_avg=significant_windows)
            
if __name__=="__main__":
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'long_FB' #'feedback' # 'baseline', 'stimulus', 'decision'
   classify = 'accuracy' #'decision', 'stim_valence', 'accuracy', 'learning'
   participants = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25']
   repetitions = ['rep_1', 'rep_2_3', 'rep_all']
   bands = ['gamma'] #'gamma', 'delta', 'theta', 'alpha', 'beta'
   n_permutations = 1000
   
   for band in bands:
      #decoder_singlePP(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, band=band, n_permutations=n_permutations)
      grand_average(window=window, classify=classify, participants=participants, repetitions=repetitions, band=band)