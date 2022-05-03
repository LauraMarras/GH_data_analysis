import numpy as np
import os

if __name__=='__main__':
    PPs = ['kh25'] #'kh21', 'kh22', 'kh23', 'kh24', 
    bands = dict(zip(['delta', 'theta', 'alpha', 'beta', 'highGamma'], [(1,3), (4,7), (8,12), (13,30), (70,120)]))
    reref = 'ESR'
    sr=1024

    for pp in PPs:
        data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Epoching/{}/'.format(pp)
        out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Features/{}/'.format(pp)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        short_epochs = ['feedback', 'stimulus', 'response', 'baseline']
        long_epochs = ['long_FB', 'long_stim']

        # Load epoched envelopes array
        for epoch in short_epochs:
            envelope_epoched = np.load(data_path + '{}_{}_envelope_epoched_{}.npy'.format(pp, reref, epoch)) # np.array samples*channels*bands
        # estimate mean along samples axis
            features = envelope_epoched.mean(axis=1)
        
        # Save data
            np.save(out_path + '{}_{}_features_{}'.format(pp,reref,epoch), features)

        for epoch in long_epochs:
            envelope_epoched = np.load(data_path + '{}_{}_envelope_epoched_{}.npy'.format(pp, reref, epoch)) # np.array samples*channels*bands
            
            epoch_length = envelope_epoched.shape[1]
            win_len = int(0.1*sr)
            overlap = int(0.05*sr)
            n_win = int(epoch_length/overlap)-1

            start = np.arange(0, envelope_epoched.shape[1], overlap)
            stop = start+win_len


            envelope_windowed = np.zeros((envelope_epoched.shape[0], n_win, win_len, envelope_epoched.shape[2], envelope_epoched.shape[3]))
            
            for w in range(n_win):
                envelope_windowed[:, w, :, :, :] = envelope_epoched[:, start[w]:stop[w], :, :]

            features = envelope_windowed.mean(axis=2)
            
            # Save data
            np.save(out_path + '{}_{}_features_{}'.format(pp,reref,epoch), features) 