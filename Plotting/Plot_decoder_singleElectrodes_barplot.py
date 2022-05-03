import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm 
import pandas as pd

def plot_decoding_single_channel(reref='ESR', epoch='feedback', target='accuracy', pps=[], repetitions=[1,2,3], action='show', cv_method='KFold'):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/{}/{}/{}/'.format(reref, epoch, target)
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Plots/{}/{}/{}/'.format(reref, epoch, target)
   if not os.path.exists(out_path):
      os.makedirs(out_path)
   chan_locs_path = 'C:/Users/laura/Documents/Data_Analysis/Data/Labelling/'
   chan_name_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Rereferencing/'

   rep_name = ' and '.join([str(x) for x in repetitions])
   r_str = ','.join([str(x) for x in repetitions])
   
   for pNr, pp in enumerate(pps):
      # Load data
      data = np.load(data_path +'{}/{}/{}_decoder_singleChannel_{}_{}.npz'.format(pp, 'channel', pp, r_str, cv_method))
      score_means = data['score_means']
      if cv_method == 'KFold':
         errors = data['errors']
      
      threshold = np.load(data_path +'{}/{}/{}_decoder_singleChannel_{}_{}_permTest.npz'.format(pp, 'channel', pp, r_str, cv_method))['threshold'][3]

      chan_names = np.load(chan_name_path + '{}/{}_cleanChannels.npy'.format(pp,pp))
      chan_locs = pd.read_excel(chan_locs_path + '{}/locs.xlsx'.format(pp))

      # Define color palette
      palette = ['#adb5bd']
      colors = ['#12b2e2', '#153ae0', '#850ad6','#f61379']
      num_colors = 10-int(threshold*10)
      palette.extend(colors[-(num_colors):])

      # Define bounds, norm and create colormap
      limits = np.linspace(0,1, num=11, endpoint=True)
      bounds = np.insert(limits[limits>threshold], 0, [0, threshold])
      norm = mcolors.BoundaryNorm(bounds, len(bounds)-1)
      cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', palette, len(bounds)-1)
      color_coding = cmap(norm(score_means))

      # set TickLabels
      chan_locs = chan_locs.set_index('Label')
      #tick_labels = [chan_locs.loc[chan, 'Location'] + '  ' + chan for chan in chan_names]
      tick_description = [chan_locs.loc[chan, 'Description'] + '  ' + chan for chan in chan_names]
      
      # Plot scores
      x = np.arange(len(chan_names)) 
      y = np.arange(11)/10 

      plt.figure(figsize=[9.5,4.5], tight_layout=True)
      if cv_method == 'KFold':
         plt.bar(x, score_means, color=color_coding, alpha=0.5, tick_label=tick_description, yerr=errors, ecolor='k', error_kw=dict(lw=0.4, capsize=0.5))
      else:
         plt.bar(x, score_means, color=color_coding, alpha=0.5, tick_label=tick_description)
      
      plt.axhline(y=0.5,color='r', linestyle=':', lw=1)
      
      for l in [0.6, 0.7, 0.8, 0.9]:
         plt.axhline(y=l, color='grey', linestyle='--', alpha=0.2, lw=1)
      
      plt.title('Decoding {} during {} in P0{}'.format(target, epoch, str(pNr+1)), fontdict={'fontsize':10})#\n\nClassification of ' + classify + ' during ' + window + ' (' + repetitions + ' ' + reref + ')')
      plt.suptitle('Data from {} repetitions, cross-validation method: {}'.format(rep_name, cv_method), x = 0.98, ha='right',fontsize=5)
   
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
         plt.savefig(out_path + '{}_decoder_singleElectrodes_{}_barPlot_{}'.format(pp, r_str, cv_method), dpi=300)
      else:
         plt.show()


if __name__=="__main__":

   reref = 'ESR' #, 'laplacian', 'CAR', 'none']
   epoch = 'feedback'
   target = 'accuracy' #'decision', 'stim_valence', 'accuracy', 'learning', stimulus_category
   pps = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = [1,2,3]
   action = 'save' #'show'
   bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
   cv_method= 'LeaveOneOut' #'LeaveOneOut' #KFold

   plot_decoding_single_channel(reref, epoch, target, pps, repetitions, action, cv_method)
