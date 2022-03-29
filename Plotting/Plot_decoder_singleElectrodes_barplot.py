import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm 
import pandas as pd

def plot_decoding_single_channel(reref='elecShaftR', window='feedback', classify='accuracy', participants=[], repetitions=['rep_all'], action='show', band=None):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/' + window + '_' + classify + '/'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Plots/' + window + '_' + classify + '/'
   chan_locs_path = 'C:/Users/laura/Documents/Data_Analysis/Data/Labelling/'
   
   for pNr, participant in enumerate(participants):
      # Load data
      if band is None:
         data = np.load(data_path + participant + '/' + participant + '_decoder_single_electrodes.npz')
      else:
         data = np.load(data_path + participant + '/' + participant + '_decoder_single_electrodes{}.npz'.format('_' + band))
      score_means = data ['score_means']
      p_vals = data ['p_vals']
      errors = data ['errors']
      channels = data ['read_channels']
      threshold = data['threshold']
      chan_locs = pd.read_excel(chan_locs_path + '{}/locs.xlsx'.format(participant))

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
      color_coding = cmap(norm(score_means))


      # set TickLabels
      chan_locs = chan_locs.set_index('Label')
      #tick_labels = [chan_locs.loc[chan, 'Location'] + '  ' + chan for chan in channels]
      tick_description = [chan_locs.loc[chan, 'Description'] + '  ' + chan for chan in channels]
      
      # Plot scores
      x = np.arange(len(channels)) 
      y = np.arange(11)/10 

      plt.figure(figsize=[9.5,4.5], tight_layout=True)
      plt.bar(x, score_means, color=color_coding, alpha=0.5, tick_label=tick_description, yerr=errors, ecolor='k', error_kw=dict(lw=0.4, capsize=0.5))
      plt.axhline(y=0.5,color='r', linestyle=':', lw=1)
      for l in [0.6, 0.7, 0.8, 0.9]:
         plt.axhline(y=l, color='grey', linestyle='--', alpha=0.2, lw=1)
      
      if band is None:
         plt.title('P0{} ROC AUC scores by channel'.format(str(pNr+1)))
      else:
         plt.title('P0{} ROC AUC scores by channel based on {}'.format(str(pNr+1), band))
      
      plt.ylabel('ROC AUC')
      plt.xlabel('Channels')
      plt.xticks(x, rotation='vertical', fontsize=5)
      plt.yticks(y, fontsize=8) 
      plt.ylim(0,1)      
      plt.margins(x=0)
      plt.gca().spines['right'].set_visible(False)
      plt.gca().spines['top'].set_visible(False)
   
      # Set colorbar
      cb = plt.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), use_gridspec=True,
      fraction=0.01, shrink=0.8, alpha=0.5,
      ticks=bounds, label='AUC score', format='%.2f')

      # Save Figure
      if action == 'save':
         if band is None:
            folder = out_path
            filename = participant + '_decoder_singleElectrodes_barPlot'
         else:
            folder = out_path + band + '/'
            filename = participant + '_decoder_singleElectrodes_barPlot_' + band
         
         if not os.path.exists(folder):
            os.makedirs(folder)   
         plt.savefig(folder + filename, dpi=300)
      else:
         plt.show()


if __name__=="__main__":
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'feedback'
   classify = 'accuracy' #'decision', 'stim_valence', 'accuracy', 'learning'
   participants = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = ['rep_all']
   action = 'save' #'show'
   bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

   plot_decoding_single_channel(reref=reref, window=window, classify=classify, participants=participants, repetitions=repetitions, action=action, band=None)