import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

directory = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/PreprocessedData/'
output = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/Results/Classification/'
bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
classifier = 2

if __name__=="__main__":
   if classifier == 1: #### Classifier based on single band power from all channels
      score_means = np.zeros((5, 2))
      errors = np.zeros((5, 2))
      for c1, filter in enumerate(bands):
         for c2, participant in enumerate(['01', '03']):
            if participant == '01':
               pp='kh21'
            elif participant == '03':
               pp='kh22'
            # Load data
            labels = np.load(directory + participant + '/labels.npy')
            channels = np.load(directory + participant + '/channels.npy')
            tot_channels = len(channels)
            sEEG = np.load(directory + participant + '/sEEG_{}.npy'.format(filter))
            powerspectra = np.load(directory + participant + '/powerspectra2_{}.npy'.format(filter))

            # Extract feature vectors
            sEEG = np.mean(sEEG, axis = 1)

            # Split test and train data
            #X_train, X_test, y_train, y_test = train_test_split(sEEG, labels, test_size=0.4, random_state=0)

            # Classifier
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            model = clf.fit(sEEG, labels)
            mc = model.coef_
            #clf.fit(sEEG,labels)
            #score = clf.score(X_test, y_test)

            # CrossValidation
            #ss = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
            kf = KFold(n_splits=5)
            CV_scores = cross_val_score(clf, sEEG, labels, cv=kf, scoring='roc_auc')
            score_means[c1,c2] = CV_scores.mean()
            errors[c1,c2] = CV_scores.std()
         
            # Plot Coefficients
            x = np.arange(len(channels))
            plt.figure(figsize=[9.5,4.5], tight_layout=True)
            plt.bar(x, mc[0], tick_label=channels, alpha=0.8)
            plt.ylabel('Coefficients')
            plt.xlabel('Channels')
            plt.title(pp + ' Coefficient scores by channel - ' + filter)
            plt.xticks(x, rotation='vertical', fontsize=5)  
            plt.grid(alpha=0.2)
            plt.margins(x=0)
            ##plt.show()
            #plt.savefig(output + pp + 'Classifier_coefficients_' + filter, dpi=300)

      # Plot Scores
      bands = ['delta\n1-4 Hz', 'theta\n4-7 Hz', 'alpha\n7-12 Hz', 'beta\n12-30 Hz', 'high gamma\n70-120 Hz']
      x = np.arange(len(bands))  # the label locations
      width = 0.35  # the width of the bars

      fig, ax = plt.subplots()
      rects1 = ax.bar(x - width/2, score_means[:,0], width, yerr=errors[:,0], align='center', alpha=0.5, ecolor='gray', capsize=5, label='kh21')
      rects2 = ax.bar(x + width/2, score_means[:,1], width, yerr=errors[:,0], align='center', alpha=0.5, ecolor='gray', capsize=5, label='kh22')

      # Add some text for labels, title and custom x-axis tick labels, etc.
      ax.set_ylabel('ROC AUC')
      ax.set_title('ROC AUC scores by frequency band and participant')
      ax.set_xticks(x)
      ax.set_xticklabels(bands)
      ax.legend()
      ax.axhline(y=0.5,color='r', linestyle=':')
      ax.set_ylim(0,1)

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
      fig.tight_layout()
      #plt.savefig(output + 'Classifier_envelope2')


   elif classifier==2: #### Classifier based on all bands power from one channel
      for c2, participant in enumerate(['01', '03']):
         if participant == '01':
            pp='kh21'
         elif participant == '03':
            pp='kh22'
         # Load data
         envelopes = np.load(directory + participant + '/be.npy')
         spectra = np.load(directory + participant + '/bs.npy')
         labels = np.load(directory + participant + '/labels.npy')
         channels = np.load(directory + participant + '/channels.npy')

         score_means = np.zeros((len(channels)))
         errors = np.zeros((len(channels)))
         # Classifier
         for c, channel in enumerate(channels):
            Xenv = envelopes[:,:,c]
            Xsp = spectra[:,:,c]
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            model = clf.fit(Xenv, labels)
            mc = model.coef_
         # CrossValidation
            kf = KFold(n_splits=5)
            CV_scores_env = cross_val_score(clf, Xenv, labels, cv=kf, scoring='roc_auc')
            score_means[c] = CV_scores_env.mean()
            errors[c] = CV_scores_env.std()
         # Plot Coefficients
            bands = ['delta\n1-4 Hz', 'theta\n4-7 Hz', 'alpha\n7-12 Hz', 'beta\n12-30 Hz', 'high gamma\n70-120 Hz']
            x = np.arange(len(bands))
            plt.figure(tight_layout=True)
            plt.bar(x, mc[0], tick_label=bands, alpha=0.8)
            plt.ylabel('Coefficients')
            plt.title(pp + ' Coefficient scores by band - ' + channel)
            plt.xticks(x)
            plt.axhline(y=0,color='k', linestyle='-', lw=0.5)  
            #plt.grid(alpha=0.2)
            #plt.margins(x=0)
            #plt.show()
            #plt.savefig(output + 'Coefficients/' + pp + '_' + channel, dpi=300)

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
         error_kw=dict(lw=0.4, capsize=0.5)
         plt.bar(x, score_means, color=col, tick_label=channels, alpha=0.8, yerr=errors, ecolor='k', error_kw=error_kw)
         plt.ylabel('ROC AUC')
         plt.xlabel('Channels')
         plt.title(pp + 'ROC AUC scores by channel' + '(envelope)')
         plt.xticks(x, rotation='vertical', fontsize=5)
         plt.yticks(y, fontsize=8)       
         labels = list(colors_labels.keys())
         handles = [plt.Rectangle((0,0),1,1, color=colors_labels[label]) for label in labels]
         plt.legend(handles, labels, loc='upper left', ncol=3, fontsize='xx-small')
         plt.axhline(y=0.5,color='r', linestyle=':', lw=1)
         for l in [0.6, 0.7, 0.8, 0.9]:
            plt.axhline(y=l, color='grey', linestyle='--', alpha=0.2, lw=1)
         plt.ylim(0,1)
         plt.margins(x=0)
         #plt.show()
         #plt.savefig(output + pp + 'Classifier_by_channels_envelope', dpi=300)
   
   