import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

#### Classifier based on single band power from all channels
def decoding_single_bands(reref='elecShaftR', window='feedback', classify='accuracy', participants=[], repetitions='all', bands=['delta', 'theta', 'alpha', 'beta', 'gamma'], n_permutations=1000):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/'
   label_path = data_path + classify + '_labels' + '/' 
   feature_path = data_path + window + '/' + reref + '/' + reref + '_'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/' 
   
   for participant in participants:
   # Load data   
      features = np.load(feature_path + participant + '_' + repetitions + '_bands_envelope.npy')
      labels = np.load(label_path + participant + '_' + classify + '_labels_' + repetitions + '.npy')
      
   # Classifier + permutation test
      score_means = np.zeros((len(bands)))
      errors = np.zeros((len(bands)))
      score_perms = np.zeros((n_permutations))
      error_perms = np.zeros((n_permutations))
      p_vals = np.zeros((len(bands)))
      
      # Run permutation test 
      for perm in range(n_permutations):
         permuted_labels = np.random.permutation(labels)
         random_band = np.random.choice(len(bands))
         
         # Define Classifier, run Crossvalidation for permuted labels dataset
         clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
         kf = StratifiedKFold(n_splits=5)
         X = features[:,random_band,:]
         perm_scores_env = cross_val_score(clf, X, permuted_labels, cv=kf, scoring="roc_auc", error_score='raise')
         
         # Store permuted scores, stds
         score_perms[perm] = perm_scores_env.mean()
         error_perms[perm] = perm_scores_env.std()
            
      # Run Classifier for each band
      for b in range(len(bands)):     
         X = features[:,b,:]
         clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
         kf = StratifiedKFold(n_splits=5)
         CV_scores = cross_val_score(clf, X, labels, cv=kf, scoring='roc_auc')
         score_means[b] = CV_scores.mean()
         errors[b] = CV_scores.std()
      
      # Estimate p-value of each channel
         p_vals[b] = np.count_nonzero(score_perms > score_means[b]) / n_permutations

      # Estimate significant threshold and significant channels
      score_perms_sorted = np.sort(score_perms)
      threshold = np.percentile(score_perms_sorted, q=[0.1, 1, 5, 95, 99, 99.9])
      significant_bands = [x for b, x in enumerate(bands) if score_means[b] > threshold[3]]

   # Save decoding results
      file_fold = out_path + participant + '/'
      file_name = participant + '_decoder_single_bands'
      
      if not os.path.exists(file_fold):
         os.makedirs(file_fold)     
      
      np.savez(file_fold + file_name, 
         score_means=score_means,
         p_vals=p_vals,
         errors=errors,
         score_perms=score_perms,
         error_perms=error_perms,
         threshold=threshold,
         significant_channels=significant_bands)

   
if __name__=="__main__":
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'feedback' #'feedback' # 'baseline', 'stimulus', 'decision'
   classify = 'accuracy' #'decision', 'stim_valence', 'accuracy', 'learning'
   participants = ['kh21','kh22','kh23','kh24', 'kh25'] #['kh21','kh22','kh23','kh24'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = 'rep_all'
   n_permutations = 1000
   bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

   decoding_single_bands(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, bands=bands, n_permutations=n_permutations)