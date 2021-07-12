import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import Preprocessing_classifier

def my_classifier(method='single_channel-all_bands', feature='envelope', window='feedback', window_length=1, classify='accuracy'):
   
   data_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/Data/PreprocessedData_rereferencing/{}/'.format(window + '-' + str(window_length) + '_' + classify)
   out_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/Results/Classification_rereferencing/{}/'.format(window + '-' + str(window_length) + '_' + classify)
   bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
   participants = ['kh23']

#### Classifier based on all bands power from one channel
   if method == 'single_channel-all_bands':
      for participant in participants:
      # Load data   
         features = np.load(data_path + participant + '_bands_{}.npy'.format(feature))
         labels = np.load(data_path + participant + '_' + classify + '_labels.npy')
         channels = np.load(data_path + participant + '_channels.npy')
      
      # Define Classifier, run Crossvalidation and store scores and stds
         score_means = np.zeros((len(channels)))
         errors = np.zeros((len(channels)))
         for c in range(len(channels)):
            X = features[:,:,c]
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            kf = KFold(n_splits=5)
            CV_scores_env = cross_val_score(clf, X, labels, cv=kf, scoring='roc_auc')
            score_means[c] = CV_scores_env.mean()
            errors[c] = CV_scores_env.std()
      
      # Plot scores
         x = np.arange(len(channels))  # the label locations
         y = np.arange(11)/10
         col = []
         colors_labels = {'>0.5':'blue', '0.5 - 0.6':'green', '0.6 - 0.7':'yellow', '0.7 - 0.8':'orange', '0.8 - 0.9':'pink', '0.9 - 1':'red'}  
         for val in score_means:
            if val < 0.5:
               col.append('blue')
            elif val >= 0.5 and val <0.6:
               col.append('green')
            elif val >= 0.6 and val <0.7:
               col.append('yellow')
            elif val >= 0.7 and val <0.8:
               col.append('orange')
            elif val >= 0.8 and val <0.9:
               col.append('pink')
            else:
               col.append('red')
         
         plt.figure(figsize=[9.5,4.5], tight_layout=True)
         plt.bar(x, score_means, color=col, tick_label=channels, alpha=0.8, yerr=errors, ecolor='k', error_kw=dict(lw=0.4, capsize=0.5))
         plt.axhline(y=0.5,color='r', linestyle=':', lw=1)
         for l in [0.6, 0.7, 0.8, 0.9]:
            plt.axhline(y=l, color='grey', linestyle='--', alpha=0.2, lw=1)
         plt.title(participant + ' ROC AUC scores by channel ({})'.format(feature))
         plt.ylabel('ROC AUC')
         plt.xlabel('Channels')
         plt.xticks(x, rotation='vertical', fontsize=5)
         plt.yticks(y, fontsize=8) 
         plt.ylim(0,1)      
         labels = list(colors_labels.keys())
         handles = [plt.Rectangle((0,0),1,1, color=colors_labels[label]) for label in labels]
         plt.legend(handles, labels, loc='upper left', ncol=3, fontsize='xx-small')
         plt.margins(x=0)
      # Save Figure
         if not os.path.exists(out_path):
               os.makedirs(out_path)
               plt.savefig(out_path + participant + '_' + method + '_' + feature, dpi=300)
         else:
            if not os.path.exists(out_path + participant + '_' + method + '_' + feature):
               plt.savefig(out_path + participant + '_' + method + '_' + feature, dpi=300)
            else:
               print('/' + participant + '_' + method + '_' + feature + ' File already existing in folder')
   
#### Classifier based on single band power from all channels
   if method == 'single_band-all_channels':
      score_means = np.zeros((len(bands), len(participants)))
      errors = np.zeros((len(bands), len(participants)))
      for pp, participant in enumerate(participants):
      # Load data   
         features = np.load(data_path + participant + '_bands_{}.npy'.format(feature))
         labels = np.load(data_path + participant + '_' + classify + '_labels.npy')
         channels = np.load(data_path + participant + '_channels.npy')
      
      # Define Classifier, run Crossvalidation and store scores and stds
         for ff, _ in enumerate(bands):
            X = features[:,ff,:]
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            kf = KFold(n_splits=5)
            CV_scores = cross_val_score(clf, X, labels, cv=kf, scoring='roc_auc')
            score_means[ff,pp] = CV_scores.mean()
            errors[ff,pp] = CV_scores.std()
      
   # Plot Scores
      bands = ['delta\n1-4 Hz', 'theta\n4-7 Hz', 'alpha\n7-12 Hz', 'beta\n12-30 Hz', 'high gamma\n70-120 Hz']
      x = np.arange(len(bands))  # the label locations
      width = 0.35  # the width of the bars
      fig, ax = plt.subplots()
      rects1 = ax.bar(x - width/2, score_means[:,0], width, yerr=errors[:,0], align='center', alpha=0.5, ecolor='gray', capsize=5, label='kh21')
      rects2 = ax.bar(x + width/2, score_means[:,1], width, yerr=errors[:,0], align='center', alpha=0.5, ecolor='gray', capsize=5, label='kh22')
      ax.axhline(y=0.5,color='r', linestyle=':')
      for b in rects1:
         height = b.get_height()
         ax.annotate('{}'.format(round(height, 2)),
            xy=(b.get_x() + b.get_width() / 2, height),
            xytext=(-2, 1),
            textcoords="offset points",
            ha='right', va='bottom', size=7)
      for b in rects2:
         height = b.get_height()
         ax.annotate('{}'.format(round(height, 2)),
            xy=(b.get_x() + b.get_width() / 2, height),
            xytext=(2, 1),
            textcoords="offset points",
            ha='left', va='bottom', size=7)
      ax.set_ylabel('ROC AUC')
      ax.set_title('ROC AUC scores by frequency band and participant ({})'.format(feature))
      ax.set_xticks(x)
      ax.set_xticklabels(bands)
      ax.set_ylim(0,1)
      ax.legend()
      fig.tight_layout()
   # Save Figures
      if not os.path.exists(out_path):
         os.makedirs(out_path)
         plt.savefig(out_path + method + '_' + feature, dpi=300)
      else:
         if not os.path.exists(out_path + method + '_' + feature):
            plt.savefig(out_path + method + '_' + feature, dpi=300)
         else:
            print('/' + method + '_' + feature + ' File already existing in folder')


if __name__=="__main__":
   '''
   ###method = ['single_band-all_channels', 'single_channel-all_bands']
   feature = ['envelope', 'spectra']
   window = ['feedback', 'baseline', 'stimulus', 'decision']
   window_length = int
   delay = int
   subtract_delay = False or True
   classify = ['accuracy', 'stim_valence', 'decision']
   '''
   reref = 'laplacian'
   feature = 'envelope' # 'spectra'
   window = 'feedback' #'feedback' # 'baseline', 'stimulus', 'decision'
   window_length = 1
   delay = 0
   subtract_delay = False #or True
   classify = 'accuracy' #'decision', 'stim_valence', 'accuracy'

   Preprocessing_classifier.preprocess_data(reref=reref, feature=feature, window=window, window_length=window_length, delay=delay, subtract_delay=subtract_delay, classify=classify)
   for method in ['single_channel-all_bands']:
      my_classifier(method=method, feature=feature, window=window, window_length=window_length, classify=classify)