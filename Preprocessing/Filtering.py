import numpy as np
import mne.filter 
import os
import scipy.signal 

def filtering(data, bands, sr=1024):
    """Apply one or more filters to the data
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    bands: dict (keys: str (band names); values: tuple (start and stop of frequency range))
        frequency bands and relative range limits to filter the data
    sr: int
        sampling rate

    Returns
    ----------
    filtered data: array (samples, channels, bands)
        filtered EEG time series   
    """
    seeg_filtered = np.zeros((data.shape[0], data.shape[1], len(bands)))
    for i, range in enumerate(bands.values()):
        start = range[0]
        stop = range[1]

        data = data.astype(float)
        seeg_filtered[:,:,i] = mne.filter.filter_data(data.T, sr, start, stop, method='iir').T
    return seeg_filtered

def extract_envelope(data, bands):
    """Apply Hilber transform to extract envelope from filtered data
    
    Parameters
    ----------
    data: array (samples, channels, bands)
        filtered EEG time series in each band
    bands: dict (keys: str (band names); values: tuple (start and stop of frequency range))
        frequency bands and relative range limits to filter the data
    
    Returns
    ----------
    envelope: array (samples, channels, bands)
        envelope of EEG time series   
    """
    hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]
    seeg_envelope = np.zeros((data.shape[0], data.shape[1], len(bands)))

    for i, _ in enumerate(bands.keys()):
        seeg_envelope[:,:,i] = np.abs(hilbert3(data[:,:,i]))
    return seeg_envelope


if __name__=='__main__':
    PPs = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25'] #'kh21', 'kh22', 'kh23', 'kh24', 
    reref = ['ESR'] #'LapR', 'CAR', 'BPR'
    bands = dict(zip(['delta', 'theta', 'alpha', 'beta', 'highGamma'], [(1,3), (4,7), (8,12), (13,30), (70,120)]))
    sr=1024

# Filtering
    for pp in PPs:
        data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Rereferencing/{}/'.format(pp)
        out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Filtering/{}/'.format(pp)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    
        for r in reref:
            # load data
            seeg = np.load(data_path + '{}_seeg_{}.npy'.format(pp,r))
            
            # filter data returns samples * channels * band array
            seeg_filtered = filtering(seeg, bands, sr)
            
            # save data fileterd
            np.save(out_path + '{}_seeg_{}_filtered'.format(pp,r), seeg_filtered)

# Extract envelope
    for pp in PPs:
        data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Filtering/{}/'.format(pp)
        out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Envelope/{}/'.format(pp)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for r in reref:
            # load data
            seeg = np.load(data_path + '{}_seeg_{}_filtered.npy'.format(pp,r))

            # extract envelopes
            seeg_envelope = extract_envelope(seeg, bands)

            # save data fileterd
            np.save(out_path + '{}_seeg_{}_envelopes'.format(pp,r), seeg_envelope)
