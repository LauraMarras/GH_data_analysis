import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score


#### Classifier based on single band power from all channels
def decoding_single_bands(reref='elecShaftR', window='feedback', classify='accuracy', participants=[], repetitions='all', bands=['delta', 'theta', 'alpha', 'beta', 'gamma'], n_permutations=1000, cv_method='KFold'):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/'
   label_path = data_path + classify + '_labels' + '/' 
   feature_path = data_path + window + '/' + reref + '/' + reref + '_'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/' 
   
   for participant in participants:
   # Load data   
      features = np.load(feature_path + participant + '_' + repetitions + '_bands_envelope.npy')
      labels = np.load(label_path + participant + '_' + classify + '_labels_' + repetitions + '.npy')
      
   # Classifier
      score_means = np.zeros((len(bands)))
      errors = np.zeros((len(bands)))
      
   # Run Classifier for each band
      for b in range(len(bands)):     
         X = features[:,b,:]
         clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      # Define cross validation method to use
         if cv_method == 'LeaveOneOut':
            cv = LeaveOneOut()
            CV_pred = cross_val_predict(clf, X, labels, cv=cv)
            score_means[b] = roc_auc_score(labels, CV_pred)
         
         elif cv_method == 'KFold':
            cv = StratifiedKFold(n_splits=5)
            CV_scores = cross_val_score(clf, X, labels, cv=cv, scoring='roc_auc')

            score_means[b] = CV_scores.mean()
            errors[b] = CV_scores.std()
      
   # Run permutation test 
      if n_permutations!=0:
         score_perms = np.zeros((n_permutations))
         error_perms = np.zeros((n_permutations))
         p_vals = np.zeros((len(bands)))

         for perm in range(n_permutations):
            permuted_labels = np.random.permutation(labels)
            random_band = np.random.choice(len(bands))
            
            # Define Classifier, run Crossvalidation for permuted labels dataset
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            cv = StratifiedKFold(n_splits=5)
            X = features[:,random_band,:]
            perm_scores_env = cross_val_score(clf, X, permuted_labels, cv=cv, scoring="roc_auc", error_score='raise')
            
            # Store permuted scores, stds
            score_perms[perm] = perm_scores_env.mean()
            error_perms[perm] = perm_scores_env.std()
      
      # Estimate p-value of each channel
         for b in range(len(bands)):
            p_vals[b] = np.count_nonzero(score_perms > score_means[b]) / n_permutations

      # Estimate significant threshold and significant channels
         score_perms_sorted = np.sort(score_perms)
         threshold = np.percentile(score_perms_sorted, q=[0.1, 1, 5, 95, 99, 99.9])
         significant_bands = [x for b, x in enumerate(bands) if score_means[b] > threshold[3]]

   # Save decoding results
      file_fold = out_path + participant + '/'
      file_name = '{}_decoder_single_bands_{}'.format(participant, cv_method)
      
      if not os.path.exists(file_fold):
         os.makedirs(file_fold)     
      
      if n_permutations!=0:
         np.savez(file_fold + file_name + '_permTest', 
            score_means=score_means,
            errors=errors,
            p_vals=p_vals,
            score_perms=score_perms,
            error_perms=error_perms,
            threshold=threshold,
            significant_bands=significant_bands)
      else:
         np.savez(file_fold + file_name, 
         score_means=score_means,
         errors=errors)

   
if __name__=="__main__":
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'stimulus' #'feedback' # 'baseline', 'stimulus', 'decision'
   classify = 'stimvalence' #'decision', 'stimvalence', 'accuracy', 'learning'
   participants = ['kh21','kh22','kh23','kh24','kh25'] #[] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = 'rep_2_3'
   n_permutations = 1000 #1000
   bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
   cv_method = 'KFold' #'KFold', 'LeaveOneOut'

   decoding_single_bands(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, bands=bands, n_permutations=n_permutations, cv_method=cv_method)