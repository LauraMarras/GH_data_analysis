import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd

def my_classifier(method='single_channel-all_bands', reref='elecShaftR', window='feedback', classify='accuracy', participants=[], repetitions='all'):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/'
   chan_locs_path = 'C:/Users/laura/Documents/Data_Analysis/Labelling/'
   label_path = data_path + classify + '_labels' + '/' 
   feature_path = data_path + window + '/' + reref + '/' + reref + '_'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/Final Plots/{}/'.format(window + '_' + classify)
   bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
   participants = participants 

#### Classifier based on all bands power from one channel
   if method == 'single_channel-all_bands':
      for pNr, participant in enumerate(participants):
      # Load data   
         features = np.load(feature_path + participant + '_' + repetitions + '_bands_envelope.npy')
         labels = np.load(label_path + participant + '_' + classify + '_labels_' + repetitions + '.npy')
         channels = np.load(data_path + participant + '_channels.npy')
         chan_locs = pd.read_excel(chan_locs_path + participant + '/locs.xlsx')
         chan_locs = chan_locs.set_index('Label')
         tick_labels = [chan_locs.loc[chan, 'Location'] + '  ' + chan for chan in channels]
         
      # Define Classifier, run Crossvalidation and store scores and stds
         score_means = np.zeros((len(channels)))
         errors = np.zeros((len(channels)))
         for c in range(len(channels)):
            X = features[:,:,c]
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            kf = StratifiedKFold(n_splits=5)
            CV_scores_env = cross_val_score(clf, X, labels, cv=kf, scoring='roc_auc')
            score_means[c] = CV_scores_env.mean()

            errors[c] = CV_scores_env.std()

      # Save decoding results
         np.save(chan_locs_path + participant + '/score_means.npy', score_means)
         np.save(chan_locs_path + participant + '/red_channels.npy', channels)


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
         plt.bar(x, score_means, color=col, tick_label=tick_labels, alpha=0.8, yerr=errors, ecolor='k', error_kw=dict(lw=0.4, capsize=0.5))
         plt.axhline(y=0.5,color='r', linestyle=':', lw=1)
         for l in [0.6, 0.7, 0.8, 0.9]:
            plt.axhline(y=l, color='grey', linestyle='--', alpha=0.2, lw=1)
         plt.title('P0' + str(pNr+1) + ' ROC AUC scores by channel')#\n Classification of ' + classify + ' during ' + window + ' (' + repetitions + ' ' + reref + ')')
         plt.ylabel('ROC AUC')
         plt.xlabel('Channels')
         plt.xticks(x, rotation='vertical', fontsize=5)
         plt.yticks(y, fontsize=8) 
         plt.ylim(0,1)      
         labels = list(colors_labels.keys())
         handles = [plt.Rectangle((0,0),1,1, color=colors_labels[label]) for label in labels]
         plt.legend(handles, labels, loc='upper left', ncol=3, fontsize='xx-small')
         plt.margins(x=0)
         plt.gca().spines['right'].set_visible(False)
         plt.gca().spines['top'].set_visible(False)
      # Save Figure
         folder1 = out_path + participant + '/'
         if not os.path.exists(folder1):
            os.makedirs(folder1)   
         plt.savefig(folder1 + participant + '_' + reref + '_' + repetitions, dpi=300)
           
#### Classifier based on single band power from all channels
   elif method == 'single_band-all_channels':
      score_means = np.zeros((len(bands), len(participants)))
      errors = np.zeros((len(bands), len(participants)))
      for pp, participant in enumerate(participants):
      # Load data   
         features = np.load(feature_path + participant + '_' + repetitions + '_bands_envelope.npy')
         labels = np.load(label_path + participant + '_' + classify + '_labels_' + repetitions + '.npy')
         channels = np.load(data_path + participant + '_channels.npy')
         

      # Define Classifier, run Crossvalidation and store scores and stds
         for ff, _ in enumerate(bands):
            X = features[:,ff,:]
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            kf = StratifiedKFold(n_splits=5)
            CV_scores = cross_val_score(clf, X, labels, cv=kf, scoring='roc_auc')
            score_means[ff,pp] = CV_scores.mean()
            errors[ff,pp] = CV_scores.std()
      
      # Add average across pps score
      score_band_avgs = np.zeros((1,5))
      err_avg = np.zeros((1,5))
      for ff, _ in enumerate(bands):
         avg = np.mean(score_means[ff,:])
         score_band_avgs[:,ff]=avg
         err_avg[:,ff] = score_means[ff,:].std()
      score_means = np.concatenate((score_means, score_band_avgs.T), axis=1)
      errors = np.concatenate((errors, err_avg.T), axis=1)

      participants_plus_avg = participants.copy()
      participants_plus_avg.append('average')

   # Plot Scores
      bands = ['delta\n1-4 Hz', 'theta\n4-7 Hz', 'alpha\n7-12 Hz', 'beta\n12-30 Hz', 'high gamma\n70-120 Hz']
      x = np.arange(len(bands))  # the label locations
      width = 0.15  # the width of the bars
      fig, ax = plt.subplots()
      for pNr, participant in enumerate(participants_plus_avg):
         if pNr == 0:
            rects = ax.bar(x - 2.5*width, score_means[:,pNr], width, yerr=errors[:,pNr], align='center', alpha=0.3, ecolor='gray', error_kw=dict(lw=0.8, capsize=3), label='P0' + str(pNr+1))
            pos = 'right'
            xy_text = (-1, 0.5)
         elif pNr == 1:
            rects = ax.bar(x - 1.5*width, score_means[:,pNr], width, yerr=errors[:,pNr], align='center', alpha=0.3, ecolor='gray', error_kw=dict(lw=0.8, capsize=3), label='P0' + str(pNr+1))
            pos = 'right'
            xy_text = (-1, 0.5)
         elif pNr == 2:
            rects = ax.bar(x - 0.5*width, score_means[:,pNr], width, yerr=errors[:,pNr], align='center', alpha=0.3, ecolor='gray', error_kw=dict(lw=0.8, capsize=3), label='P0' + str(pNr+1))
            pos = 'left'
            xy_text = (1, 0.5)
         elif pNr == 3:
            rects = ax.bar(x + 0.5*width, score_means[:,pNr], width, yerr=errors[:,pNr], align='center', alpha=0.3, ecolor='gray', error_kw=dict(lw=0.8, capsize=3), label='P0' + str(pNr+1))
            pos = 'left'
            xy_text = (1, 0.5)
         elif pNr == 4:
            rects = ax.bar(x + 1.5*width, score_means[:,pNr], width, yerr=errors[:,pNr], align='center', alpha=0.3, ecolor='gray', error_kw=dict(lw=0.8, capsize=3), label='P0' + str(pNr+1))
            pos = 'left'
            xy_text = (1, 0.5)
         elif pNr == 5:
            rects = ax.bar(x + 2.5*width, score_means[:,pNr], width, align='center', alpha=0.6, ecolor='gray', label='average')
            pos = 'center'
            xy_text = (0, 0.5)
         for b in rects:
            height = b.get_height()
            ax.annotate('{}'.format(round(height, 2)),
               xy=(b.get_x() + b.get_width() / 2, height),
               xytext= xy_text,
               textcoords="offset points",
               ha=pos, va='bottom', size=4)
      ax.axhline(y=0.5,color='r', linestyle=':')
      ax.set_ylabel('ROC AUC')
      ax.set_title('ROC AUC scores by frequency band and participant')#\n\nClassification of ' + classify + ' during ' + window + ' (' + repetitions + ' ' + reref + ')')
      ax.set_xticks(x)
      ax.set_xticklabels(bands)
      ax.set_ymargin(0.1)
      ax.legend()
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
   # Save Figures
      if not os.path.exists(out_path):
         os.makedirs(out_path)
      plt.savefig(out_path +'_' + reref + '_' + repetitions, dpi=300)

if __name__=="__main__":
   '''
   method = ['single_band-all_channels', 'single_channel-all_bands']
   window = ['feedback', 'baseline', 'stimulus']
   classify = ['accuracy', 'stim_valence', 'decision']
   '''
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'feedback' #'feedback' # 'baseline', 'stimulus', 'decision'
   classify = 'accuracy' #'decision', 'stim_valence', 'accuracy', 'learning'
   participants = ['kh21','kh22','kh23','kh24']#'kh21', 'kh22', 'kh23', 'kh24', 'kh25'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = ['rep_all']#, 'rep_2']
   
   for rep in repetitions:
      for method in ['single_channel-all_bands']: #'single_channel-all_bands', 'single_band-all_channels', 
         my_classifier(reref=reref, method=method, window=window, classify=classify, participants=participants, repetitions=rep)