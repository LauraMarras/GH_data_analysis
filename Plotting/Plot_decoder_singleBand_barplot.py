import numpy as np
import os
import matplotlib.pyplot as plt

def groupLevel_results(reref='ESR', epoch='feedback', target='accuracy', pps=[], repetitions=[1,2,3], cv_method='KFold'):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/{}/{}/{}/'.format(reref, epoch, target)
   
   rep_name = ','.join([str(x) for x in repetitions])

   score_means = []
   score_perms = []
   errors = []
   p_vals = []
   threshold = []

   # Load data
   for pp in pps:
      data = np.load(data_path +'{}/{}/{}_decoder_singleBand_{}_{}.npz'.format(pp, 'band', pp, rep_name, cv_method))
      score_means.append(data['score_means'])
      if cv_method=='KFold':
         errors.append(data['errors'])

      data = np.load(data_path +'{}/{}/{}_decoder_singleBand_{}_{}_permTest.npz'.format(pp, 'band', pp, rep_name, cv_method))
      
      p_vals.append(data['p_vals'])
      score_perms.append(data['score_perms'])
      threshold.append(data['threshold'][3])
   
   # Convert into arrays
   score_means = np.stack(score_means)
   score_perms = np.stack(score_perms)
   p_vals = np.stack(p_vals)
   threshold = np.stack(threshold)
   if cv_method=='KFold':
      errors = np.stack(errors)

   # Add average score across pps per band
   score_means_avg = np.mean(score_means, axis=0)
   if cv_method=='KFold':
      errors_avg = np.std(score_means, axis=0)

   # Estimate average p-values
   p_vals_avg = []
   for b in range(score_means.shape[1]):
      p_vals_avg.append(np.count_nonzero(score_perms > score_means_avg[b]) / (score_perms.shape[1] * score_perms.shape[0]))

   # Estimate significant threshold
   score_perms_sorted = np.sort(score_perms, axis=None)
   threshold_avg = np.percentile(score_perms_sorted, q=95)

   # Add average to arrays
   group_scores = np.vstack((score_means, score_means_avg))
   group_threshold = np.hstack((threshold, threshold_avg))

   if cv_method=='KFold':
      group_errors = np.vstack((errors, errors_avg))
      return group_scores, group_errors, group_threshold
   
   elif cv_method=='LeaveOneOut':
      return group_scores, group_threshold

def plot_decoding_single_band(reref='ESR', epoch='feedback', target='accuracy', pps=[], repetitions=[1,2,3], action='show', cv_method='KFold'):
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Plots/{}/{}/{}/'.format(reref, epoch, target)
   
   rep_name = ' and '.join([str(x) for x in repetitions])
   r_str = ','.join([str(x) for x in repetitions])

# Load data and group results
   if cv_method=='KFold':
      group_scores, group_errors, group_threshold = groupLevel_results(reref, epoch, target, pps, repetitions, cv_method)
   elif cv_method=='LeaveOneOut':
      group_scores, group_threshold = groupLevel_results(reref, epoch, target, pps, repetitions, cv_method)


# Plot Scores
   fig, ax = plt.subplots()

   # Define height of bars to stack
   bottom_bars = np.zeros((group_scores.shape[0], group_scores.shape[1]))
   up_bars = np.zeros((group_scores.shape[0], group_scores.shape[1]))

   for pp in range(group_scores.shape[0]):
      for band in range(group_scores.shape[1]):
         min = np.minimum(group_scores[pp,band], group_threshold[pp])
         bottom_bars[pp,band] = min
         up_bars[pp,band] = group_scores[pp,band] - group_threshold[pp]
      up_bars[pp][bottom_bars[pp,:]!= group_threshold[pp]] = 0
   
   # Define colors for each PP and text for X ticks
   colors = ['#50ffb1', '#058c42', '#56cfe1', '#004e98', '#6f2dbd', '#d81159']
   #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
   
   x = np.arange(group_scores.shape[1])  # the label locations
   width = 0.10  # the width of the bars
   
   # Plot each PPs bars
   diff = - 3
   for pNr in range(group_scores.shape[0]-1):
      if cv_method=='KFold':
         ax.bar(x + diff*width, up_bars[pNr,:], width, bottom=bottom_bars[pNr,:], color=colors[pNr], yerr=group_errors[pNr], align='center', alpha=0.7, error_kw=dict(ecolor='k', lw=0.4, capsize=0.5, alpha=0.7), label='P0' + str(pNr+1))
      elif cv_method=='LeaveOneOut':
         ax.bar(x + diff*width, up_bars[pNr,:], width, bottom=bottom_bars[pNr,:], color=colors[pNr], alpha=0.7, label='P0' + str(pNr+1))
      
      ax.bar(x + diff*width, bottom_bars[pNr,:], width, color=colors[pNr], alpha=0.3)
      diff += 1
   
   # Plot average bars
   if cv_method=='KFold':
      ax.bar(x + (1+diff)*width, up_bars[-1,:], width, bottom=bottom_bars[-1,:], color=colors[-1], alpha=0.7, label='Average', yerr=group_errors[-1,:], align='center', error_kw=dict(ecolor='k', lw=0.4, capsize=0.5, alpha=0.7))
   elif cv_method=='LeaveOneOut':   
      ax.bar(x + (1+diff)*width, up_bars[-1,:], width, bottom=bottom_bars[-1,:], color=colors[-1], alpha=0.7, label='Average')
   
   ax.bar(x + (1+diff)*width, bottom_bars[-1,:], width, color=colors[-1], alpha=0.3)
      
   # Draw horizontal lines
   for l in [0.5, 0.6, 0.7, 0.8, 0.9]:
      ax.axhline(y=l, color='grey', linestyle='--', alpha=0.2, lw=0.3)

   # Set title, legend, axes labels and ticks
   ax.set_title('Decoding {} during {}'.format(target, epoch), fontdict={'fontsize':10})#\n\nClassification of ' + classify + ' during ' + window + ' (' + repetitions + ' ' + reref + ')')
   plt.suptitle('Data from {} repetitions, cross-validation method: {}'.format(rep_name, cv_method), x = 0.98, ha='right',fontsize=5)
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
      plt.savefig(out_path + 'decoder_singleBands_{}_barplot_{}'.format(r_str, cv_method), dpi=300)

# or just show it
   elif action == 'show':
      plt.show()


if __name__=="__main__":
   reref = 'ESR' #, 'laplacian', 'CAR', 'none']
   epoch = 'stimulus'
   target = 'decision' #'decision', 'stim_valence', 'accuracy', 'learning', stimulus_category
   pps = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = [1,2,3]
   action = 'save' #'show'
   bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
   cv_method= 'KFold' #'LeaveOneOut' #KFold

   plot_decoding_single_band(reref, epoch, target, pps, repetitions, action, cv_method)
