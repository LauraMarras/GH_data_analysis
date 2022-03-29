import numpy as np
import os
import matplotlib.pyplot as plt

def plot_accuracy(participants=[]):
   data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/BehaviouralResults/'
   out_path = 'C:/Users/laura/Documents/Data_Analysis/Plots/Behavioural/'
   
# Load data
   data = np.load(data_path + 'behavioural_accuracyScores_pVals.npz')
   scores = data['pp_scores'][:,:-1]
   pvals = data['pp_pvals']

   # Plot Scores
   fig, ax = plt.subplots()
   
   # Define colors for each PP and text for X ticks
   colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']#, '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
   #colors = ['#50ffb1', '#058c42', '#56cfe1', '#004e98', '#6f2dbd']#, '#d81159']
   
   x = np.arange(scores.shape[1])  # the label locations
   width = 0.10  # the width of the bars
   
   # Plot each PPs bars
   diff = - 3
   for pNr, participant in enumerate(participants):
      bars = ax.bar(x + diff*width, scores[pNr,:], width, color=colors[pNr], alpha=0.7, label='P0' + str(pNr+1))
      diff += 1.5

      p_ast=[]
      for p in pvals[pNr,:-1]:
         
         if p<=0.05:
            p_ast.append('*')
         else:
            p_ast.append('')

      count=0
      for b in bars:
         if count%3!=0:
            if pvals[pNr, count-1] <= 0.001:
               text = '*\n*\n*'
            elif pvals[pNr, count-1] <= 0.01:
               text = '*\n*'
            elif pvals[pNr, count-1] <= 0.05:
               text='*'
            else:
               text = ''
            
            height = b.get_height()
            ax.annotate(text,
               xy=(b.get_x() + b.get_width() / 2, height),
               #xytext= xy_text,
               textcoords="offset points",
               ha='center',
               va='bottom', size=15)

         count+=1
      
   # Draw horizontal lines
   for l in [0.5, 0.6, 0.7, 0.8, 0.9]:
      ax.axhline(y=l, color='grey', linestyle='--', alpha=0.2, lw=0.3)
      

   # Set title, legend, axes labels and ticks
   ax.set_title('Behavioural accuracy', fontdict={'fontsize':7})

   ax.legend(loc='upper left', ncol=2, fontsize=7, frameon=True)

   ax.set_ylabel('Accuracy (correct trials/total trials)', fontdict={'fontsize':7})
   ax.set_yticks([x/10 for x in[*range(0, 11, 1)]])
   ax.set_yticklabels([x/10 for x in[*range(0, 11, 1)]],fontdict={'fontsize':7})
   
   rep_ticks = ['1st\nPresentation', '2nd\nPresentation', '3rd\nPresentation', 'Total'][:-1]
   ax.set_xticks(x)
   ax.set_xticklabels(rep_ticks, fontdict={'fontsize':7})

   #ax.set_ymargin(0)
   ax.spines['right'].set_visible(False)
   ax.spines['top'].set_visible(False)

# Save Figures
   if action == 'save':
      if not os.path.exists(out_path):
         os.makedirs(out_path)   
      plt.savefig(out_path + 'behavioural_accuracy_sign_colors2', dpi=300)

# or just show it
   elif action == 'show':
      plt.show()


if __name__=="__main__":
   participants = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25'] # 'kh21', 'kh22', 'kh23', 'kh24', 'kh25'
   action = 'save' #'show'

   plot_accuracy(participants=participants)