import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

#### Classifier based on all bands power from one channel
def decoding_single_channel(reref='elecShaftR', window='feedback', classify='accuracy', participants=[], repetitions='rep_all', n_permutations=1000, cv_method='KFold', band=None):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/'
   label_path = data_path + classify + '_labels' + '/' 
   feature_path = data_path + window + '/' + reref + '/' + reref + '_'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/' 
   
   for participant in participants:
   # Load data   
      features = np.load(feature_path + participant + '_' + repetitions + '_bands_envelope.npy')
      labels = np.load(label_path + participant + '_' + classify + '_labels_' + repetitions + '.npy')
      channels = np.load(data_path + participant + '_channels.npy')
      
   # Classifier
      score_means = np.zeros((len(channels)))
      errors = np.zeros((len(channels)))
      
      # Run Classifier for each channel
      for c in range(len(channels)):
         if band is None:
            X = features[:,:,c]
         else:
            X = features[:,band,c].reshape(-1,1)
         clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
         
         # Define cross validation method to use
         if cv_method == 'LeaveOneOut':
            cv = LeaveOneOut()
            CV_pred = cross_val_predict(clf, X, labels, cv=cv)
            score_means[c] = roc_auc_score(labels, CV_pred)
         
         elif cv_method == 'KFold':
            cv = StratifiedKFold(n_splits=5)
            CV_scores_env = cross_val_score(clf, X, labels, cv=cv, scoring='roc_auc')

            score_means[c] = CV_scores_env.mean()
            errors[c] = CV_scores_env.std()

   # Run permutation test
      if n_permutations!=0:
         score_perms = np.zeros((n_permutations))
         error_perms = np.zeros((n_permutations))
         p_vals = np.zeros((len(channels)))
      
      
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
         
      

      # Estimate p-value of each channel
         for c in range(len(channels)):
            p_vals[c] = np.count_nonzero(score_perms > score_means[c]) / n_permutations

      # Estimate significant threshold and significant channels
         score_perms_sorted = np.sort(score_perms)
         threshold = np.percentile(score_perms_sorted, q=[0.1, 1, 5, 95, 99, 99.9])
         significant_channels = [x for c,x in enumerate(channels) if score_means[c] > threshold[3]]

   # Save decoding results
      bands_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
      if band is None:
         file_fold = out_path + participant + '/'
         file_name = '{}_decoder_single_electrodes_{}'.format(participant, cv_method)
      else:
         file_fold = out_path + participant + '/'
         file_name = '{}_decoder_single_electrodes_{}_{}'.format(participant, cv_method, bands_names[band])
      
      if not os.path.exists(file_fold):
         os.makedirs(file_fold)     
      
      if n_permutations!=0:
         np.savez(file_fold + file_name + '_permTest', 
            score_means=score_means,
            p_vals=p_vals,
            errors=errors,
            score_perms=score_perms,
            error_perms=error_perms,
            read_channels=channels,
            threshold=threshold,
            significant_channels=significant_channels)
      else:
         np.savez(file_fold + file_name, 
            score_means=score_means,
            errors=errors,
            read_channels=channels)

if __name__=="__main__":
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'stimulus' #'feedback' # 'baseline', 'stimulus', 'decision'
   classify = 'stimvalence' #'decision', 'stim_valence', 'accuracy', 'learning'
   participants = ['kh21','kh22','kh23','kh24', 'kh25'] #['kh21','kh22','kh23','kh24'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = 'rep_2_3'
   n_permutations = 0 #1000
   cv_method = 'LeaveOneOut' #'KFold', 'LeaveOneOut'
   
   decoding_single_channel(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, n_permutations=n_permutations, cv_method=cv_method, band=None)

 















