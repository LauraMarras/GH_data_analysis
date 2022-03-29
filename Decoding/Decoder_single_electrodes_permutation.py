import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

#### Classifier based on all bands power from one channel
def decoding_single_channel(reref='elecShaftR', window='feedback', classify='accuracy', participants=[], repetitions='rep_all', n_permutations=1000, band=None):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/'
   label_path = data_path + classify + '_labels' + '/' 
   feature_path = data_path + window + '/' + reref + '/' + reref + '_'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/' 
   
   for participant in participants:
   # Load data   
      features = np.load(feature_path + participant + '_' + repetitions + '_bands_envelope.npy')
      labels = np.load(label_path + participant + '_' + classify + '_labels_' + repetitions + '.npy')
      channels = np.load(data_path + participant + '_channels.npy')
      
   # Classifier + permutation test
      score_means = np.zeros((len(channels)))
      errors = np.zeros((len(channels)))
      score_perms = np.zeros((n_permutations))
      error_perms = np.zeros((n_permutations))
      p_vals = np.zeros((len(channels)))
      
      # Run permutation test
      for perm in range(n_permutations):
         permuted_labels = np.random.permutation(labels)
         random_channel = np.random.choice(len(channels))
         
         # Define Classifier, run Crossvalidation for permuted labels dataset
         clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
         kf = StratifiedKFold(n_splits=5)
         if band is None:
            X = features[:,:,random_channel]
         else:
            X = features[:,band,random_channel].reshape(-1,1)
         perm_scores_env = cross_val_score(clf, X, permuted_labels, cv=kf, scoring="roc_auc", error_score='raise')
         
         # Store permuted scores, stds
         score_perms[perm] = perm_scores_env.mean()
         error_perms[perm] = perm_scores_env.std()
         
      # Run Classifier for each channel
      for c in range(len(channels)):
         if band is None:
            X = features[:,:,c]
         else:
            X = features[:,band,c].reshape(-1,1)
         clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
         kf = StratifiedKFold(n_splits=5)
         CV_scores_env = cross_val_score(clf, X, labels, cv=kf, scoring='roc_auc')
         score_means[c] = CV_scores_env.mean()
         errors[c] = CV_scores_env.std()

      # Estimate p-value of each channel
         p_vals[c] = np.count_nonzero(score_perms > score_means[c]) / n_permutations

      # Estimate significant threshold and significant channels
      score_perms_sorted = np.sort(score_perms)
      threshold = np.percentile(score_perms_sorted, q=[0.1, 1, 5, 95, 99, 99.9])
      significant_channels = [x for c,x in enumerate(channels) if score_means[c] > threshold[3]]

   # Save decoding results
      bands_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
      if band is None:
         file_fold = out_path + participant + '/'
         file_name = participant + '_decoder_single_electrodes'
      else:
         file_fold = out_path + participant + '/'
         file_name = participant + '_decoder_single_electrodes_' + bands_names[band]
      
      if not os.path.exists(file_fold):
         os.makedirs(file_fold)     
      
      np.savez(file_fold + file_name, 
         score_means=score_means,
         p_vals=p_vals,
         errors=errors,
         score_perms=score_perms,
         error_perms=error_perms,
         read_channels=channels,
         threshold=threshold,
         significant_channels=significant_channels)

if __name__=="__main__":
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'feedback' #'feedback' # 'baseline', 'stimulus', 'decision'
   classify = 'accuracy' #'decision', 'stim_valence', 'accuracy', 'learning'
   participants = ['kh21','kh22','kh23','kh24', 'kh25'] #['kh21','kh22','kh23','kh24'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = 'rep_all'
   n_permutations = 1000
   
   decoding_single_channel(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, n_permutations=n_permutations, band=None)

 















