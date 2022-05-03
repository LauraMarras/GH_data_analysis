import numpy as np
import os
import matplotlib.pyplot as plt

def plot_decoding_single_band(reref='elecShaftR', window='feedback', classify='accuracy', participants=[], repetitions=['rep_all'], action='show', bands=['delta', 'theta', 'alpha', 'beta', 'gamma'], n_permutations=1000, cv_method='KFold'):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Plots/' + window + '_' + classify + '/'
   
   score_means = np.zeros((len(participants), len(bands)))
   errors = np.zeros((len(participants), len(bands)))
   
   score_perms = np.zeros((len(participants), n_permutations))
   p_vals = np.zeros((len(participants), len(bands)))
   threshold = np.zeros((len(participants)))

# Load data
   for pNr, participant in enumerate(participants):
      data = np.load(data_path +'{}/{}_decoder_single_bands_{}_permTest.npz'.format(participant,participant,cv_method))
      score_means[pNr] = data ['score_means']
      p_vals[pNr] = data ['p_vals']
      score_perms[pNr] = data ['score_perms']
      threshold[pNr] = data['threshold'][3]

      if cv_method=='KFold':
         errors[pNr] = data ['errors']

# Add average score across participants per band
   score_means_avg = np.mean(score_means, axis=0)
   errors_avg = np.std(score_means, axis=0)

# Estimate average p-values
   p_vals_avg = np.zeros((len(bands)))
   for b in range(len(bands)):
      p_vals_avg[b] = np.count_nonzero(score_perms > score_means_avg[b]) / (n_permutations * score_perms.shape[0])

# Estimate significant threshold
   score_perms_sorted = np.sort(score_perms, axis=None)
   threshold_avg = np.percentile(score_perms_sorted, q=95)

# Add average to arrays
   participants.append('average')
   scores = np.vstack((score_means, score_means_avg))
   errs = np.vstack((errors, errors_avg))
   thresh = np.hstack((threshold, threshold_avg))

# Plot Scores
   fig, ax = plt.subplots()

   # Define height of bars to stack
   bottom_bars = np.zeros((len(participants), len(bands)))
   up_bars = np.zeros((len(participants), len(bands)))
   for pp in range(len(participants)):
      for band in range(len(bands)):
         min = np.minimum(scores[pp,band], thresh[pp])
         bottom_bars[pp,band] = min
         up_bars[pp,band] = scores[pp,band] - thresh[pp]
      up_bars[pp][bottom_bars[pp,:]!= thresh[pp]] = 0
   
   # Define colors for each PP and text for X ticks
   #colors = ['#50ffb1', '#058c42', '#56cfe1', '#004e98', '#6f2dbd', '#d81159']
   colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
   
   x = np.arange(len(bands))  # the label locations
   width = 0.10  # the width of the bars
   
   # Plot each PPs bars
   diff = - 3
   for pNr, participant in enumerate(participants[:-1]):
      if cv_method=='KFold':
         ax.bar(x + diff*width, up_bars[pNr,:], width, bottom=bottom_bars[pNr,:], color=colors[pNr], yerr=errs[pNr,:], align='center', alpha=0.7, error_kw=dict(ecolor='k', lw=0.4, capsize=0.5, alpha=0.7), label='P0' + str(pNr+1))
      elif cv_method=='LeaveOneOut':
         ax.bar(x + diff*width, up_bars[pNr,:], width, bottom=bottom_bars[pNr,:], color=colors[pNr], alpha=0.7, label='P0' + str(pNr+1))
      
      ax.bar(x + diff*width, bottom_bars[pNr,:], width, color=colors[pNr], alpha=0.3)
      diff += 1
   
   # Plot average bars
   if cv_method=='KFold':
      ax.bar(x + (1+diff)*width, up_bars[-1,:], width, bottom=bottom_bars[-1,:], color=colors[-1], alpha=0.7, label='Average', yerr=errs[-1,:], align='center', error_kw=dict(ecolor='k', lw=0.4, capsize=0.5, alpha=0.7))
   elif cv_method=='LeaveOneOut':   
      ax.bar(x + (1+diff)*width, up_bars[-1,:], width, bottom=bottom_bars[-1,:], color=colors[-1], alpha=0.7, label='Average')
   
   ax.bar(x + (1+diff)*width, bottom_bars[-1,:], width, color=colors[-1], alpha=0.3)
      
   # Draw horizontal lines
   for l in [0.5, 0.6, 0.7, 0.8, 0.9]:
      ax.axhline(y=l, color='grey', linestyle='--', alpha=0.2, lw=0.3)
      

   # Set title, legend, axes labels and ticks
   ax.set_title('ROC AUC scores by frequency band and participant', fontdict={'fontsize':7})#\n\nClassification of ' + classify + ' during ' + window + ' (' + repetitions + ' ' + reref + ')')
   
   ax.legend(loc='best', ncol=2, fontsize=7, frameon=True)#, fontdict={'fontsize':7})

   ax.set_ylabel('ROC AUC', fontdict={'fontsize':7})
   ax.set_yticks([x/10 for x in[*range(0, 11, 1)]])
   ax.set_yticklabels([x/10 for x in[*range(0, 11, 1)]],fontdict={'fontsize':7})
   
   band_ticks = ['delta\n1-4 Hz', 'theta\n4-7 Hz', 'alpha\n7-12 Hz', 'beta\n12-30 Hz', 'high gamma\n70-120 Hz']
   ax.set_xticks(x)
   ax.set_xticklabels(band_ticks, fontdict={'fontsize':7})

   ax.set_ymargin(0)
   ax.spines['right'].set_visible(False)
   ax.spines['top'].set_visible(False)

# Save Figures
   if action == 'save':
      if not os.path.exists(out_path):
         os.makedirs(out_path)   
      plt.savefig(out_path + 'decoder_singleBands_barplot_colors2_{}'.format(cv_method), dpi=300)

# or just show it
   elif action == 'show':
      plt.show()


if __name__=="__main__":
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'stimulus'
   classify = 'stimvalence' #'decision', 'stim_valence', 'accuracy', 'learning'
   participants = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = ['rep_2_3']
   action = 'save' #'show'
   bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
   cv_method= 'KFold' #KFold

   plot_decoding_single_band(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, action=action, bands=bands, cv_method=cv_method)






   # for b in bars:
   #       height = b.get_height()
   #       ax.annotate('{}'.format(round(height, 2)),
   #          xy=(b.get_x() + b.get_width() / 2, height),
   #          #xytext= xy_text,
   #          textcoords="offset points",
   #          #ha=pos,
   #          va='bottom', size=4)