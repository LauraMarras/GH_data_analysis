import numpy as np
import matplotlib.pyplot as plt
import os

directory = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/PreprocessedData/'
output = 'C:/Users/laura/OneDrive/Documenti/Internship/Python/Results/'
bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

if __name__=="__main__":
   for c2, participant in enumerate(['01', '03']):
      outpath = output + participant + '/ScatterPlots/'
      # Load data
      bands_envelopes = np.load(directory + participant + '/be.npy')
      bands_spectra = np.load(directory + participant + '/powerspectra2array.npy')
      labels = np.load(directory + participant + '/labels.npy')
      labels_bool = labels>0
      channels = np.load(directory + participant + '/channels.npy')
      tot_channels = len(channels)

      # Plot scatter
      for channel in range(tot_channels):
            theta_env = bands_envelopes[:,1,channel] #array with len=90(trials)
            delta_env = bands_envelopes[:,0,channel] #array with len=90(trials)
            theta_sp = bands_spectra[:,1,channel] #array with len=90(trials)
            delta_sp = bands_spectra[:,0,channel] #array with len=90(trials)

            X = (theta_env - theta_env.mean())/(theta_env.std())
            Y = (delta_env - delta_env.mean())/(delta_env.std())

            plt.figure()
            scatterc = plt.scatter(X[labels_bool], Y[labels_bool], c='g', alpha=0.5, label='Correct')
            scatteri = plt.scatter(X[~labels_bool], Y[~labels_bool], c='r', alpha=0.5, label='Incorrect')
            plt.plot(list(plt.xlim()), list(plt.ylim()), 'k--')
            plt.legend(loc="best")
            plt.margins(x=0)
            plt.xlabel('Normalized Theta Envelope')
            plt.ylabel('Normalized Delta Envelope')
            plt.title('Envelope - Channel ' + str(channels[channel]))         
            #plt.show()

            # Save Images
            if not os.path.exists(outpath):
               os.makedirs(outpath)
               plt.savefig(outpath + 'Envelope_' + str(channel+1) + '_' + str(channels[channel]))
            else:
               plt.savefig(outpath + 'Envelope_' + str(channel+1) + '_' + str(channels[channel]))
