import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.linalg.linalg import norm

def interpolate_list(array):
   interp_arr = []
   for i,x in enumerate(array[:len(array)-1]):
      list = [x, array[i+1]]
      my_val = (max(list) - min(list))/2
      if x == max(list):
         interp_arr.append(x - my_val)
      elif x == min(list):
         interp_arr.append(x + my_val)

   arr_l = array.tolist()
   c=0
   for x, _ in enumerate (array[:len(array)-1]):
      arr_l.insert(x+1+c, interp_arr[x])
      c+=1
   arr_l.append(array.tolist()[-1])
   
   return arr_l, interp_arr

def make_space_above(axes, topmargin=1):
    """ increase figure size to make topmargin (in inches) space for 
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, flip=False):
   '''
   https://stackoverflow.com/a/18926541
   '''
   if isinstance(cmap, str):
      cmap = plt.get_cmap(cmap)
   if flip:
      new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
         cmap(np.flip(np.linspace(minval, maxval, n))))
   else:
      new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
   
   return new_cmap

def plot_decoding_results_single_PP(reref='elecShaftR', window='long_FB', classify='accuracy', participants=[], repetitions=['rep_1', 'rep_2_3', 'rep_all'], band='gamma', action='show'):
   out_path = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/{}/'.format(window + '_' + classify)
   out_path_errors = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/standarderrors/{}/'.format(window + '_' + classify)
   results_path = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/Final Plots/{}/'.format(window + '_' + classify)
   
# Load data   
   PPs_means = np.load(out_path + 'PPs_means_' + reref + '_' + band + '.npy', allow_pickle=True).item()
   PPs_pvals = np.load(out_path + 'PPs_pvals_' + reref + '_' + band + '.npy', allow_pickle=True).item()
   PPs_errors = np.load(out_path_errors + 'PPs_err_' + reref + '_' + band + '.npy', allow_pickle=True).item()

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

# Plot
   for pNr, participant in enumerate(participants):
      fig, axs = plt.subplots(len(repetitions))
   
      for r, rep in enumerate(repetitions):
         x = [x - 500 for x in [x + 25 for x in [*range(0, 1989, 51)]]]
         y = PPs_means[rep + '_' + str(pNr)]
         yu = y + 1*PPs_errors[rep + '_' + str(pNr)]
         yl = y - 1*PPs_errors[rep + '_' + str(pNr)]
         pvals = PPs_pvals[rep + '_' + str(pNr)]
         #y_05 = np.ma.masked_where(p_vals > 0.05, y)
         #y_01 = np.ma.masked_where(p_vals > 0.01, y)
   
      # Set ColorMap
         cmap = truncate_colormap('brg', minval=0, maxval=0.6, n=100, flip=False)
         #cmap = truncate_colormap('PiYG', minval=0, maxval=1, n=100)
         bounds = np.array([0, 0.001, 0.01, 0.05, 1])
         #bounds = np.array([0, 0.05, 1])
         normbound = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

         axs[r].plot(x,y, color='midnightblue', lw=1)
         for i in range(len(x) - 1):
            axs[r].fill_between([x[i],x[i+1]], [yu[i], yu[i+1]], [yl[i], yl[i+1]], color=cmap(normbound(pvals[i])), norm=normbound, alpha=0.3, lw=0)
            #axs[r].fill_between([x[i],x[i+1]], 0, 1, color=cmap(normbound(pvals[i])), norm=normbound, alpha=0.2, lw=0)

      
         #axs[r].fill_between(x, yu, yl, facecolor='none', edgecolor='blue', alpha = 0.2)
         #axs[r].plot(x,y_05, '*', markersize=4, color='indigo', label='p<0.05')
         #axs[r].plot(x,y_01, '*', markersize=4, color='mediumslateblue', label='p<0.01')
         #axs[r].fill_betweenx([x/100 for x in [*range(0, 101, 25)]], 0, 1000, color='lavender', alpha = 0.3)
         
      # Plot horizontal and vertical lines
         axs[r].axhline(y=0.5,color='k', linestyle=':', lw=0.5, alpha=0.5)
         axs[r].axvline(x=0,color='k', lw=1)
         axs[r].axvline(x=1000,color='k', lw=1)
         
      # Set labels, titles, ax limits and ticks
         axs[r].set_title(rep_strings[rep], fontsize=8)
         axs[r].set_ylabel('ROC AUC', fontsize=7)
         axs[r].set_xlabel('Time (ms)', fontsize=7)
         y_tick = [x/100 for x in [*range(0, 101, 25)]]
         axs[r].set_yticks(y_tick)
         x_tick_lab = ['-500', '-250',    '0\n\nFeedback onset',  '250',  '500',  '750', '1000\n\nFeedback offset', '1250', '1500']
         axs[r].set_xticklabels(x_tick_lab)
         axs[r].set_ylim(0,1.1)      
         axs[r].set_xlim(x[0],x[-1])
         for tick in axs[r].yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
         for tick in axs[r].xaxis.get_major_ticks():
            tick.label.set_fontsize(7)
      
      for ax in fig.get_axes():
         ax.label_outer()
         ax.spines['right'].set_visible(False)
         ax.spines['top'].set_visible(False)
         #ax.margins(x=0)
      
      # Set colorbar
      plt.colorbar(matplotlib.cm.ScalarMappable(norm=normbound, cmap=cmap), 
      ax=axs.ravel().tolist(), use_gridspec=True,
      fraction=0.01, shrink=0.8, alpha=0.3,
      ticks=[0, 0.001, 0.01, 0.05, 1], label='p value')
      #ticks=[0, 0.05, 1], label='p value')

      # Set legend
      #p05 = mlines.Line2D([], [], ls='', marker='*', markersize=5, color='indigo', label='p<0.05')
      #p01 = mlines.Line2D([], [], ls='', marker='*', markersize=5, color='mediumslateblue', label='p<0.01')
      #fig.legend(handles=[p05, p01], fontsize=7, loc='upper right')

      # Set Title and figure size
      title = 'P0' + str(pNr+1) + ' ROC AUC scores over time\n\n Classification of ' + classify + window_strings[window] + reref_strings[reref] + band_strings[band]
      fig.suptitle(title, fontsize=10, y=0.98, ha='center')
      figure = plt.gcf()
      figure.set_size_inches(10,7)

# Save Figure   
      if action == 'save':
         if not os.path.exists(results_path + participant + '/'):
            os.makedirs(results_path + participant + '/')   
         plt.savefig(results_path + participant + '/' + participant + '_' + reref + '_' + band + '_significance_final', dpi=300)

# or just show it
      elif action == 'show':
         plt.show()


def plot_decoding_results_average_PPs(reref='elecShaftR', window='long_FB', classify='accuracy', repetitions=['rep_1', 'rep_2_3', 'rep_all'], band='gamma', action='show'):
   out_path = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/long_FB_{}/'.format(classify)
   results_path = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/Final Plots/{}/'.format(window + '_' + classify)

# Load data
   PPs_means_avg = np.load(out_path + 'PPs_means_avg_' + reref + '_' + band + '.npy', allow_pickle=True).item()
   PPs_pvals_avg = np.load(out_path + 'PPs_pvals_avg_' + reref + '_' + band + '.npy', allow_pickle=True).item()
   PPs_errors_avg = np.load(out_path + 'PPs_errors_avg_' + reref + '_' + band + '.npy', allow_pickle=True).item()
   
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

# Plot
   fig, axs = plt.subplots(len(repetitions))
   
   for r, rep in enumerate(repetitions):
      x = [x - 500 for x in [x + 25 for x in [*range(0, 1989, 51)]]]
      y = PPs_means_avg[rep]
      pvals = PPs_pvals_avg[rep]
      yu = y + 1*PPs_errors_avg[rep]
      yl = y - 1*PPs_errors_avg[rep]
      #y_05 = np.ma.masked_where(PPs_pvals_avg[rep] > 0.05, y)
      #y_01 = np.ma.masked_where(PPs_pvals_avg[rep] > 0.01, y)

   # Set ColorMap
      cmap = truncate_colormap('brg', minval=0, maxval=0.6, n=100, flip=False)
      #cmap = truncate_colormap('PiYG', minval=0, maxval=1, n=100)
      bounds = np.array([0, 0.001, 0.01, 0.05, 1])
      #bounds = np.array([0, 0.05, 1])
      normbound = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

   # Plot Auc, Confidence Intervals with p-value coded colors
      axs[r].plot(x,y, color='midnightblue', lw=1)
      for i in range(len(x) - 1):
         axs[r].fill_between([x[i],x[i+1]], [yu[i], yu[i+1]], [yl[i], yl[i+1]], color=cmap(normbound(pvals[i])), norm=normbound, alpha=0.3, lw=0)
      #axs[r].fill_between(x, yu, yl, where = (PPs_pvals_avg[rep] <= 0.05) & (PPs_pvals_avg[rep] > 0.01), facecolor='lightskyblue', interpolate=False, alpha = 0.3)
      #axs[r].fill_between(x, yu, yl, where = PPs_pvals_avg[rep] <= 0.01, facecolor='dodgerblue', interpolate=False, alpha = 0.3)
      #axs[r].plot(x,y_05, '*', markersize=4, color='indigo', label='p<0.05')
      #axs[r].plot(x,y_01, '*', markersize=4, color='mediumslateblue', label='p<0.01')
      #axs[r].fill_betweenx([x/100 for x in [*range(0, 101, 25)]], 0, 1000, color='lavender', alpha = 0.3)
   
   # Plot horizontal and vertical lines
      axs[r].axhline(y=0.5,color='k', linestyle=':', lw=0.5, alpha=0.5)
      axs[r].axvline(x=0,color='k', lw=1)
      axs[r].axvline(x=1000,color='k', lw=1)
   
   # Set labels, titles, ax limits and ticks
      axs[r].set_title(rep_strings[rep], fontsize=7)
      axs[r].set_ylabel('ROC AUC', fontsize=7)
      axs[r].set_xlabel('Time (ms)', fontsize=7)
      y_tick = [x/100 for x in [*range(0, 101, 25)]]
      axs[r].set_yticks(y_tick)
      x_tick_lab = ['-500', '-250',    '0\n\nFeedback onset',  '250',  '500',  '750', '1000\n\nFeedback offset', '1250', '1500']
      axs[r].set_xticklabels(x_tick_lab)
      axs[r].set_ylim(0,1)      
      axs[r].set_xlim(x[0],x[-1])
      for tick in axs[r].yaxis.get_major_ticks():
         tick.label.set_fontsize(6)
      for tick in axs[r].xaxis.get_major_ticks():
         tick.label.set_fontsize(7)
   for ax in fig.get_axes():
      ax.label_outer()
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      #ax.margins(x=0)

   # Set colorbar
   plt.colorbar(matplotlib.cm.ScalarMappable(norm=normbound, cmap=cmap), 
   ax=axs.ravel().tolist(), use_gridspec=True,
   fraction=0.01, shrink=0.8, alpha=0.3,
   ticks=[0, 0.001, 0.01, 0.05, 1], label='p value')
   #ticks=[0, 0.05, 1], label='p value')

   # Set legend
   #p05 = mlines.Line2D([], [], ls='', marker='s', mew =0, markersize=8, color='lightskyblue', alpha=0.3, label='p<0.05')
   #p01 = mlines.Line2D([], [], ls='', marker='s', mew=0, markersize=8, color='dodgerblue', alpha=0.3, label='p<0.01')
   #fig.legend(handles=[p05, p01], fontsize=7, loc='upper right')

# Set Title and figure size
   title = 'Average across PPs of ROC AUC scores over time\n\nClassification of ' + classify + window_strings[window] + reref_strings[reref] + band_strings[band]
   fig.suptitle(title, fontsize=10, y=0.98, ha='center')
   figure = plt.gcf()
   figure.set_size_inches(10,7)
 
# Save Figure   
   if action == 'save':
      if not os.path.exists(results_path):
         os.makedirs(results_path)   
      plt.savefig(results_path + 'grandAVG_' + reref + '_' + band + '_significance_final', dpi=300)

# or just show it
   elif action == 'show':
      plt.show()
   

def plot_decoding_results_average_PPs_allbands(reref='elecShaftR', window='long_FB', classify='accuracy', repetitions=['rep_1', 'rep_2_3', 'rep_all'], bands=['gamma', 'theta', 'delta'], action='show'):
   out_path = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/long_FB_{}/'.format(classify)

# Set strings for picture title
   band_strings = {
      'gamma':'High Gamma (70-120 Hz)',
      'beta':'Beta (13-30 Hz)',
      'theta':'Theta (4-7 Hz)',
      'delta':'Delta (1-3 Hz)',
      'alpha':'Alfa (8-12 Hz)'
   }
   band_colors = {
      'gamma':'mediumseagreen', 
      'beta':'dodgerblue',
      'theta':'mediumslateblue',
      'delta':'coral',
      'alpha':'black'
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
      'long_FB': ' during baseline and feedback\n based on ',
      'long_stim':' during baseline and stimulus presentation\n based on ' 
   }
   
# Create figure and title
   fig, axs = plt.subplots(len(repetitions), tight_layout=True)
   title = 'Average across PPs of ROC AUC scores over time\n Classification of ' + classify +  window_strings[window] + reref_strings[reref]
   fig.suptitle(title, fontsize=10, x = 0.35)
   
   for band in bands:
# Load data
      PPs_means_avg = np.load(out_path + 'PPs_means_avg_' + reref + '_' + band + '.npy', allow_pickle=True).item()
      PPs_pvals_avg = np.load(out_path + 'PPs_pvals_avg_' + reref + '_' + band + '.npy', allow_pickle=True).item()
      
# Plot
      for r, rep in enumerate(repetitions):
         x = [x - 500 for x in [x + 25 for x in [*range(0, 1989, 51)]]]
         y = PPs_means_avg[rep]   
         y_05 = np.ma.masked_where(PPs_pvals_avg[rep] > 0.05, y)
         y_01 = np.ma.masked_where(PPs_pvals_avg[rep] > 0.01, y)

         axs[r].plot(x,y, color=band_colors[band], lw=1)
         #axs[r].plot(x,y_05, '.', markersize=4, color=band_colors[band])
         axs[r].plot(x,y_01, '.', markersize=5, color=band_colors[band])
         axs[r].axhline(y=0.5,color='k', linestyle=':', lw=0.5, alpha=0.5)
         axs[r].axvline(x=0,color='k', lw=1)
         axs[r].axvline(x=1000,color='k', lw=1)
         axs[r].fill_betweenx([x/100 for x in [*range(0, 101, 25)]], 0, 1000, color='lavender', alpha = 0.1)
         axs[r].set_title(rep_strings[rep], fontsize=8)
         axs[r].set_ylabel('ROC AUC', fontsize=7)
         axs[r].set_xlabel('Time (ms)', fontsize=7)
         y_tick = [x/100 for x in [*range(0, 101, 25)]]
         axs[r].set_yticks(y_tick)
         axs[r].set_ylim(0,1)      
         axs[r].set_xlim(x[0],x[-1])
         for tick in axs[r].yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
         for tick in axs[r].xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
      
   for ax in fig.get_axes():
      ax.label_outer()
      ax.margins(x=0)
   
   #p05 = mlines.Line2D([], [], ls='', marker='.', markersize=5, color='k', label='p<0.05')
   p01 = mlines.Line2D([], [], ls='', marker='.', markersize=5, color='k', label='p<0.01')
   
   handles=[p01]
   for band in bands:
      label = mlines.Line2D([], [], color=band_colors[band], lw=1, label=band_strings[band])
      handles.append(label)
   
   fig.legend(handles=handles, fontsize=7, loc='upper right')
   
# Save Figure   
   if action == 'save':
      if not os.path.exists(out_path):
         os.makedirs(out_path)   
      plt.savefig(out_path + 'grandAVG_' + reref + '_allbands_significance', dpi=300)

# or just show it
   elif action == 'show':
      plt.show()




if __name__=="__main__":
   reref = 'elecShaftR' #, 'laplacian', 'CAR', 'none']
   window = 'long_FB'
   classify = 'accuracy' #'decision', 'stim_valence', 'accuracy', 'learning'
   participants = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   repetitions = ['rep_1', 'rep_2_3', 'rep_all']
   bands = ['gamma','theta','delta']#,'alpha', 'beta'] #'theta', 'alpha', 'beta'
   action = 'show' #'save'
   
   for band in bands:
      plot_decoding_results_single_PP(reref=reref, classify=classify, participants=participants, repetitions=repetitions, band=band, action=action)
      #plot_decoding_results_average_PPs(reref=reref, window=window, classify=classify, repetitions=repetitions, band=band, action=action)
      