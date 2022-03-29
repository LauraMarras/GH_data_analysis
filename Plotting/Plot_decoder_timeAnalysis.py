import matplotlib.cm as mcm
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

def plot_decoding_results_single_PP(reref='elecShaftR', window='long_FB', classify='accuracy', participants=[], repetitions=['rep_1', 'rep_2_3', 'rep_all'], band='gamma', action='show'):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Plots/' + window + '_' + classify + '/'

# Set strings for picture title
   band_strings = {
      'gamma':' High Gamma (70-120 Hz)',
      'beta':' Beta (13-30 Hz)',
      'theta':' Theta (4-7 Hz)',
      'delta':' Delta (1-3 Hz)',
      'alpha':' Alfa (8-12 Hz)'
   }
   rep_strings = {
      'rep_1':'1st repetition',
      'rep_2_3':'2nd and 3rd repetitions',
      'rep_all':'All repetitions',
      'rep_2':'2nd repetition',
      'rep_3':'3rd repetition'
   }
   reref_strings = {
      'elecShaftR':' electrode shaft-rereferenced',
      'laplacian':' laplacian-rereferenced',
      'CAR':' common average-rereferenced',
      'none':''
   }
   window_strings = {
      'long_FB': ' during baseline and feedback based on ',
      'long_stim':' during baseline and stimulus presentation based on ' 
   }

   for pNr, participant in enumerate(participants):
   # Create figure with 3 subplots
      fig, axs = plt.subplots(len(repetitions))
      
      for r, rep in enumerate(repetitions):
      # Load data   
         score_means = np.load(data_path + band + '/' + participant + '_decoder_' + band + '_' + rep + '.npz')['score_means']
         errors = np.load(data_path + band + '/' + participant + '_decoder_' + band + '_' + rep + '.npz')['errors']
         threshold = np.load(data_path + band + '/' + participant + '_decoder_' + band + '_' + rep + '.npz')['threshold']
         
      # Define x, y and confidence intervals
         x = [x - 500 for x in [x + 25 for x in [*range(0, 1989, 51)]]]
         y = score_means
         yu = y + errors
         yl = y - errors

      # Interpolate x and y
         x_interp = np.linspace(x[0], x[-1], num=(len(x)*2)-1, endpoint=True)
         yu_interp = interp1d(x, yu)(x_interp)
         yl_interp = interp1d(x, yl)(x_interp)
         
      # Create ColorMap
         # Define color palette
         palette = ['#adb5bd']
         colors = ['#12b2e2', '#153ae0', '#850ad6','#f61379']
         num_colors = 10-int(threshold[3]*10)
         palette.extend(colors[-(num_colors):])

         # Define bounds, norm and create colormap
         limits = np.linspace(0,1, num=11, endpoint=True)
         bounds = np.insert(limits[limits>threshold[3]], 0, [0, threshold[3]])
         norm = mcolors.BoundaryNorm(bounds, len(bounds)-1)
         cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', palette, len(bounds)-1)
         
      # Plot scores
         axs[r].plot(x,y, color='midnightblue', lw=1)

      # Add confidence interval color coding significant windows
         count = 0.5
         for i in range(len(x_interp)-1):
            axs[r].fill_between([x_interp[i],x_interp[i+1]], [yu_interp[i], yu_interp[i+1]], [yl_interp[i], yl_interp[i+1]], color=cmap(norm(score_means[int(count)])), norm=norm, alpha=0.2, lw=0)
            count+=0.5
      
      # Plot horizontal and vertical lines
         axs[r].axhline(y=0.5,color='k', linestyle=':', lw=0.5, alpha=0.5)
         axs[r].axvline(x=0,color='k', lw=1)
         axs[r].axvline(x=1000,color='k', lw=1)
         
      # Set labels, subtitles, ax limits and ticks
         axs[r].set_title(rep_strings[rep], fontsize=8, pad=0.6)
         
         axs[r].set_ylabel('ROC AUC', fontsize=7)
         axs[r].set_xlabel('Time (ms)', fontsize=7)
         
         y_tick = [x/100 for x in [*range(0, 101, 25)]]
         axs[r].set_yticks(y_tick)
         for tick in axs[r].yaxis.get_major_ticks():
            tick.label.set_fontsize(6)

         x_tick_lab = ['-500', '-250',    '0\n\nFeedback onset',  '250',  '500',  '750', '1000\n\nFeedback offset', '1250', '1500']
         axs[r].set_xticklabels(x_tick_lab)
         for tick in axs[r].xaxis.get_major_ticks():
            tick.label.set_fontsize(7)
         
         axs[r].set_ylim(0,1.1)      
         axs[r].set_xlim(x[0],x[-1])
      
      # Set colorbar
         plt.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), 
         ax=axs[r], use_gridspec=True,
         fraction=0.01, shrink=0.8, 
         alpha=0.3,
         ticks = bounds, label='AUC score', format='%.2f')

   # Remove spines
      for ax in fig.get_axes():
         ax.label_outer()
         ax.spines['right'].set_visible(False)
         ax.spines['top'].set_visible(False)

   # Set Title and figure size
      title = 'P0' + str(pNr+1) + ' ROC AUC scores over time ' + band_strings[band] #'\n\n Classification of ' + classify + window_strings[window]
      fig.suptitle(title, fontsize=10, y=0.98, ha='center')
      figure = plt.gcf()
      figure.set_size_inches(10,7)

# Save Figure   
      if action == 'save':
         if not os.path.exists(out_path + participant + '/'):
            os.makedirs(out_path + participant + '/')   
         plt.savefig(out_path + participant + '/' + participant + '_decoder_' + band + '_perm2', dpi=300)

# or just show it
      if action == 'show':
         plt.show()


def plot_decoding_results_average_PPs(reref='elecShaftR', window='long_FB', classify='accuracy', repetitions=['rep_1', 'rep_2_3', 'rep_all'], band='gamma', action='show'):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Plots/' + window + '_' + classify + '/'

# Set strings for picture title
   band_strings = {
      'gamma':' High Gamma (70-120 Hz)',
      'beta':' Beta (13-30 Hz)',
      'theta':' Theta (4-7 Hz)',
      'delta':' Delta (1-3 Hz)',
      'alpha':' Alfa (8-12 Hz)'
   }
   rep_strings = {
      'rep_1':'1st repetition',
      'rep_2_3':'2nd and 3rd repetitions',
      'rep_all':'All repetitions',
      'rep_2':'2nd repetition',
      'rep_3':'3rd repetition'
   }
   reref_strings = {
      'elecShaftR':' electrode shaft-rereferenced',
      'laplacian':' laplacian-rereferenced',
      'CAR':' common average-rereferenced',
      'none':''
   }
   window_strings = {
      'long_FB': ' during baseline and feedback based on ',
      'long_stim':' during baseline and stimulus presentation based on ' 
   }

# Create figure with 3 subplots
   fig, axs = plt.subplots(len(repetitions))

   for r, rep in enumerate(repetitions):
   # Load data   
      score_means = np.load(data_path + band + '/PPs_decoder_' + band + '_' + rep + '.npz')['score_means_avg']
      errors = np.load(data_path + band + '/PPs_decoder_' + band + '_' + rep + '.npz')['errors_avg']
      threshold = np.load(data_path + band + '/PPs_decoder_' + band + '_' + rep + '.npz')['threshold_avg']
      errors2 = np.load(data_path + band + '/PPs_decoder_' + band + '_' + rep + '.npz')['errors_avg2']
      errors3 = np.load(data_path + band + '/PPs_decoder_' + band + '_' + rep + '.npz')['errors_avg3']
         
   # Define x, y and confidence intervals
      x = [x - 500 for x in [x + 25 for x in [*range(0, 1989, 51)]]]
      y = score_means
      yu = y + errors
      yl = y - errors

  # Interpolate x and y
      x_interp = np.linspace(x[0], x[-1], num=(len(x)*2)-1, endpoint=True)
      yu_interp = interp1d(x, yu)(x_interp)
      yl_interp = interp1d(x, yl)(x_interp)
         
   # Create ColorMap
      # Define color palette
      palette = ['#adb5bd']
      colors = ['#12b2e2', '#153ae0', '#850ad6','#f61379']
      num_colors = 10-int(threshold[3]*10)
      palette.extend(colors[-(num_colors):])

      # Define bounds, norm and create colormap
      limits = np.linspace(0,1, num=11, endpoint=True)
      bounds = np.insert(limits[limits>threshold[3]], 0, [0, threshold[3]])
      norm = mcolors.BoundaryNorm(bounds, len(bounds)-1)
      cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', palette, len(bounds)-1)
         
   # Plot scores
      axs[r].plot(x,y, color='midnightblue', lw=1)
    
   # Add confidence interval color coding significant windows
      count = 0.5
      for i in range(len(x_interp)-1):
         axs[r].fill_between([x_interp[i],x_interp[i+1]], [yu_interp[i], yu_interp[i+1]], [yl_interp[i], yl_interp[i+1]], color=cmap(norm(score_means[int(count)])), norm=norm, alpha=0.2, lw=0)
         count+=0.5
      
   # Plot horizontal and vertical lines
      axs[r].axhline(y=0.5,color='k', linestyle=':', lw=0.5, alpha=0.5)
      axs[r].axvline(x=0,color='k', lw=1)
      axs[r].axvline(x=1000,color='k', lw=1)
         
   # Set labels, subtitles, ax limits and ticks
      axs[r].set_title(rep_strings[rep], fontsize=8, pad=0.6)
      
      axs[r].set_ylabel('ROC AUC', fontsize=7)
      axs[r].set_xlabel('Time (ms)', fontsize=7)
      
      y_tick = [x/100 for x in [*range(0, 101, 25)]]
      axs[r].set_yticks(y_tick)
      for tick in axs[r].yaxis.get_major_ticks():
         tick.label.set_fontsize(6)

      x_tick_lab = ['-500', '-250',    '0\n\nFeedback onset',  '250',  '500',  '750', '1000\n\nFeedback offset', '1250', '1500']
      axs[r].set_xticklabels(x_tick_lab)
      for tick in axs[r].xaxis.get_major_ticks():
         tick.label.set_fontsize(7)
         
      axs[r].set_ylim(0,1.1)      
      axs[r].set_xlim(x[0],x[-1])
   
   # Set colorbar
      plt.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), 
      ax=axs[r], use_gridspec=True,
      fraction=0.01, shrink=0.8, 
      alpha=0.3,
      ticks = bounds, label='AUC score', format='%.2f')
   
# Remove spines
   for ax in fig.get_axes():
      ax.label_outer()                       
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)

# Set Title and figure size
   title = 'Average across PPs of ROC AUC scores over time ' + band_strings[band] #\n\nClassification of ' + classify + window_strings[window] + reref_strings[reref]
   fig.suptitle(title, fontsize=12, y=0.98, ha='center')
   figure = plt.gcf()
   figure.set_size_inches(10,7)
 
# Save Figure   
   if action == 'save':
      if not os.path.exists(out_path):
         os.makedirs(out_path)   
      plt.savefig(out_path + 'PPs_decoder_' + band + '_perm2', dpi=300)

# or just show it
   elif action == 'show':
      plt.show()


if __name__=="__main__":
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'long_FB'
   classify = 'accuracy' #'decision', 'stim_valence', 'accuracy', 'learning'
   participants = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = ['rep_1', 'rep_2_3', 'rep_all']
   bands = ['gamma', 'theta', 'delta', 'alpha', 'beta'] #'theta', 'alpha', 'beta'
   action = 'show' #'save'
   
   for band in bands:
      #plot_decoding_results_single_PP(reref=reref, classify=classify, participants=participants, repetitions=repetitions, band=band, action=action)
      plot_decoding_results_average_PPs(reref=reref, window=window, classify=classify, repetitions=repetitions, band=band, action=action)
      