import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

#### Classifier based on all bands power from one channel
def decoding_single_channel(reref='elecShaftR', window='feedback', classify='accuracy', participants=[], repetitions='rep_all', n_permutations=1000):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/'
   label_path = data_path + classify + '_labels' + '/' 
   feature_path = data_path + window + '/' + reref + '/' + reref + '_'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/'
   
   for participant in participants:
   # Load data   
      features = np.load(feature_path + participant + '_' + repetitions + '_bands_envelope.npy')
      labels = np.load(label_path + participant + '_' + classify + '_labels_' + repetitions + '.npy')
      channels = np.load(data_path + participant + '_channels.npy')
      
   # Define Classifier, run Crossvalidation and store scores and stds
      score_means = np.zeros((len(channels)))
      errors = np.zeros((len(channels)))
      score_perms = np.zeros((len(channels), n_permutations))
      error_perms = np.zeros((len(channels), n_permutations))
      p_vals = np.zeros((len(channels)))

      for c in range(len(channels)):
         X = features[:,:,c]
         clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
         kf = StratifiedKFold(n_splits=5)
         CV_scores_env = cross_val_score(clf, X, labels, cv=kf, scoring='roc_auc')
         score_means[c] = CV_scores_env.mean()
         errors[c] = CV_scores_env.std()

         for perm in range(n_permutations):
            permuted_labels = np.random.permutation(labels)
            perm_scores_env = cross_val_score(clf, X, permuted_labels, cv=kf, scoring="roc_auc")
         
            score_perms[c,perm] = perm_scores_env.mean()
            error_perms[c,perm] = perm_scores_env.std()
      
         p_vals[c] = np.count_nonzero(score_perms[c] > score_means[c]) / n_permutations


   # Save decoding results
      folder = out_path + participant + '/' 
      if not os.path.exists(folder):
         os.makedirs(folder)     
      np.save(folder + participant + '_score_means.npy', score_means)
      np.save(folder + participant + '_p_vals.npy', p_vals)
      np.save(folder + participant + '_errors.npy', errors)
      np.save(folder + participant + '_score_perms.npy', score_perms)
      np.save(folder + participant + '_error_perms.npy', error_perms)
      np.save(folder + participant + '_read_channels.npy', channels)


if __name__=="__main__":
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'feedback' #'feedback' # 'baseline', 'stimulus', 'decision'
   classify = 'accuracy' #'decision', 'stim_valence', 'accuracy', 'learning'
   participants = ['kh25'] #['kh21','kh22','kh23','kh24'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = 'rep_all'
   n_permutations = 1000
   
   decoding_single_channel(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, n_permutations=n_permutations)