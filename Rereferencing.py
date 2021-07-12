import numpy as np 
import numpy.matlib as matlib
from mne.filter import filter_data
import scipy.io.wavfile as wav
import scipy
from scipy import fftpack
from scipy.stats import mode
import MelFilterBank as mel
from scipy.signal import decimate, hilbert

# Helper function to drastically speed up the hilbert transform of larger data
hilbert3 = lambda x: hilbert(x, fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def cleanData(data, channels):
    """Clean data by removing irrelevant channels
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    channels: array (electrodes, label)
        Labels of the electrodes
    
    Returns
    ----------
    data: array (samples, channels)
        Cleaned EEG time series
    channels: array (electrodes, label)
        Cleaned labels of the electrodes      
    """
    clean_data = []
    clean_channels = []
    for i in range(channels.shape[0]):
        if '+' in channels[i][0]: #EKG/MRK/etc channels
            continue
        elif channels[i][0][0] == 'E': #Empty channels
            continue
        else:
            clean_channels.append(channels[i])
            clean_data.append(data[:,i])
    return np.transpose(np.array(clean_data,dtype="float64")), np.array(clean_channels)

def commonAverageR(data):
    """Apply a common-average re-reference to the data
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    
    Returns
    ----------
    data: array (samples, channels)
        CAR re-referenced EEG time series   
    """
    data_CAR = np.zeros((data.shape[0], data.shape[1]))
    average = np.average(data, axis=1)
    for i in range(data.shape[1]):
        data_CAR[:,i] = data[:,i] - average
    return data_CAR

def elecShaftR(data, channels):
    """Apply an electrode-shaft re-reference to the data
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    channels: array (electrodes, label)
        Channel names
    
    Returns
    ----------
    data: array (samples, channels)
        ESR re-referenced EEG time series   
    """
    data_ESR = np.zeros((data.shape[0], data.shape[1]))
    #get shaft information
    shafts = {}
    for i,chan in enumerate(channels):
        if chan[0].rstrip('0123456789') not in shafts:
            shafts[chan[0].rstrip('0123456789')] = {'start': i, 'size': 1}
        else:
            shafts[chan[0].rstrip('0123456789')]['size'] += 1
    #get average signal per shaft
    for shaft in shafts:
        shafts[shaft]['average'] = np.average(data[:,shafts[shaft]['start']:(shafts[shaft]['start']+shafts[shaft]['size'])], axis=1)
    #subtract the shaft average from each respective channel   
    for i in range(data.shape[1]):
        data_ESR[:,i] = data[:,i] - shafts[channels[i][0].rstrip('0123456789')]['average']
    return data_ESR

def bipolarR(data, channels):
    """Apply a bipolar re-reference to the data (to the nearest electrode in the same shaft)
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    channels: array (electrodes, label)
        Channel names
    
    Returns
    ----------
    data: array (samples, channels)
        Bipolar re-referenced EEG time series   
    channels: array (electrodes, label)
        Channel names reduced (- last in shaft)
    channels_des: array (electrodes, label)
        Channel name description (PAIR1 - PAIR2)
    """
    #get shaft information
    shafts = {}
    for i,chan in enumerate(channels):
        if chan[0].rstrip('0123456789') not in shafts:
            shafts[chan[0].rstrip('0123456789')] = {'start': i, 'size': 1}
        else:
            shafts[chan[0].rstrip('0123456789')]['size'] += 1
    #create bipolar signals (i - (i+1)) and names
    data_BPR = np.zeros((data.shape[0], (len(channels)-len(shafts)))) #N pairs = all - 1*shaft
    channels_BPR = [] #new feat names (since size changed)
    channels_BPR_des = [] #channel description
    index = 0
    for s in shafts:
        for ch in range(shafts[s]['size']-1):
            data_BPR[:,index] = data[:,(shafts[s]['start']+ch)] - data[:,(shafts[s]['start']+(ch+1))] #bipolar signal
            channels_BPR.append([channels[(shafts[s]['start']+ch)][0]]) #channel name
            channels_BPR_des.append([channels[(shafts[s]['start']+ch)][0] + '-' + channels[(shafts[s]['start']+(ch+1))][0]]) #channel description
            index += 1
    return data_BPR, np.array(channels_BPR), np.array(channels_BPR_des)
        
def laplacianR(data, channels, SIZE=1):
    """Apply a laplacian re-reference to the data
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    channels: array (electrodes, label)
        Channel names
    SIZE: int
        Size of the laplacian (amount of channels to include in average surrounding the channel)
    
    Returns
    ----------
    data: array (samples, channels)
        Laplacian re-referenced EEG time series   
    channels_des: array (electrodes, label)
        Channel name description (CHAN - CHAN±SIZE)
    """
    data_LPR = np.zeros((data.shape[0], data.shape[1]))
    #get shaft information
    shafts = {}
    for i,chan in enumerate(channels):
        if chan[0].rstrip('0123456789') not in shafts:
            shafts[chan[0].rstrip('0123456789')] = {'start': i, 'size': 1}
        else:
            shafts[chan[0].rstrip('0123456789')]['size'] += 1
    #create laplacian signals (i - (((i-1)+(i+1))/2) and names
    channels_LPR_des = [] #channel description 
    index = 0
    for s in shafts:
        ref_neg = [] #reference channels below
        ref_pos = [] #reference channels above
        for ch in range(shafts[s]['size']):
            ref_neg = [data[:,(shafts[s]['start']+(ch-neg))] for neg in range(1,SIZE+1) if (ch-neg) >= 0]
            ref_pos = [data[:,(shafts[s]['start']+(ch+pos))] for pos in range(1,SIZE+1) if (ch+pos) < shafts[s]['size']]
            data_LPR[:,index] = data[:,(shafts[s]['start']+ch)] - np.mean(ref_neg + ref_pos, axis=0)
            channels_LPR_des.append([channels[(shafts[s]['start']+ch)][0] + '-' + channels[(shafts[s]['start']+ch)][0] + '±' + str(SIZE)]) #description
            index += 1
    return data_LPR, np.array(channels_LPR_des)   

def extractFB(data, sr, band, windowLength=0.05,frameshift=0.01):
    """Window data and extract frequency-band envelope using the hilbert transform
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat, array shape (windows, channels)
        Frequency-band feature matrix
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    numWindows=int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    if band == 'beta':
        # Band-pass for beta (between 12 and 35 Hx)
        data = filter_data(data.T, sr, 12,35,method='iir').T 
    elif band == 'high-gamma':
        # Band-pass for high-gamma (between 70 and 170 Hz)
        data = filter_data(data.T, sr, 70,170,method='iir').T 
        # Band-stop filter for first two harmonics of 50 Hz line noise
        data = filter_data(data.T, sr, 102, 98,method='iir').T 
        data = filter_data(data.T, sr, 152, 148,method='iir').T
    else:
        print('Not a valid frequency band, should be "beta" or "high-gamma".')
    data = np.abs(hilbert3(data))
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return feat

def stackFeatures(features, modelOrder=4, stepSize=5):
    """Add temporal context to each window by stacking neighboring feature vectors
    
    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    Returns
    ----------
    featStacked, array shape (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    featStacked=np.zeros((features.shape[0]-(2*modelOrder*stepSize),(2*modelOrder+1)*features.shape[1]))
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        featStacked[fNum,:]=ef.flatten() # Add 'F' if stacked the same as matlab
    return featStacked

def downsampleLabels(labels, sr, windowLength=0.05, frameshift=0.01):
    """Downsamples non-numerical data by using the mode
    
    Parameters
    ----------
    labels: array of str
        Label time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which mode will be used
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    newLabels array of str
        downsampled labels
    """
    numWindows=int(np.floor((labels.shape[0]-windowLength*sr)/(frameshift*sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        newLabels[w]=mode(labels[start:stop])[0][0].encode("ascii", errors="ignore").decode()
    return newLabels

def windowAudio(audio, sr, windowLength=0.05, frameshift=0.01):
    """Window Audio data
    
    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) for which raw audio will be extracted
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    winAudio array (windows, audiosamples)
        Windowed audio
    """
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    winAudio = np.zeros((numWindows, int(windowLength*sr )))
    for w in range(numWindows):
        startAudio = int(np.floor((w*frameshift)*sr))
        stopAudio = int(np.floor(startAudio+windowLength*sr))    
        winAudio[w,:] = audio[startAudio:stopAudio]
    return winAudio

def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01,numFilter=23):
    """Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms
    
    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    numFilter: int
        Number of triangular filters in the mel filterbank
    Returns
    ----------
    spectrogram, array shape (numWindows, numFilter)
        Logarithmic mel scaled spectrogram
    """
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    win = scipy.hanning(np.floor(windowLength*sr + 1))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        startAudio = int(np.floor((w*frameshift)*sr))
        stopAudio = int(np.floor(startAudio+windowLength*sr))
        a = audio[startAudio:stopAudio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], numFilter, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram

def nameVector(elecs,modelOrder=4):
    """Creates list of electrode names.
    
    Parameters
    ----------
    elecs: array of strings 
        Original electrode names
    modelOrder: int
        Temporal context stacked prior and after current window
        Will be added as T-modelOrder, T-(modelOrder+1), ...,  T0, ..., T+modelOrder
        to the elctrode names

    Returns
    ----------
    names: array of strings 
        List of electrodes including contexts, will have size elecs.shape[0]*(2*modelOrder+1)
    """
    names = matlib.repmat(elecs.astype(np.dtype(('U', 10))),1,2 * modelOrder +1).T
    for i, off in enumerate(range(-modelOrder,modelOrder+1)):
        names[i,:] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  # Add 'F' if stacked the same as matlab


if __name__=="__main__":
    winL = 0.05 # 0.01
    frameshift = 0.01 #0.01
    modelOrder=4
    stepSize=5
    ref = 'LPR' #options: raw, CAR, ESR, BPR, LPR
    band = 'beta' #or: 'high-gamma'
    path = r'C:\Users\p70074461\Documents\Subjects\\'
    outPath = r'C:\Users\p70074461\Documents\Subjects\\'
    pts = ['kh1', 'kh2', 'kh3', 'kh4', 'kh6', 'kh7', 'kh9', 'kh10', 'kh12', 'kh13']
    sessions = [1]
    
    for pNr, p in enumerate(pts):
        for ses in range(1,sessions[0]+1):
            dat = np.load(path + p + '/preprocessing/' + p + '_' + str(ses)  + '_sEEG.npy')
            sr=1024
            
            # Clean up irrelevant channels
            elecs = np.load(path + p + '/preprocessing/' + p + '_' + str(ses)  + '_channelNames.npy')
            data, channels = cleanData(dat,elecs)
            data = data.astype('float64') #in case of kh9
            
            # Re-referencing
            if ref == 'CAR': #common-average
                data = commonAverageR(data) 
            elif ref == 'ESR': #electrode-shaft
                data = elecShaftR(data, channels) 
            elif ref == 'BPR': #bipolar
                data, channels, channels_des = bipolarR(data, channels) 
            elif ref == 'LPR': #laplacian
                data, channels_des = laplacianR(data, channels, 2) 
                
            # Extract frequency-band features
            feat = extractFB(dat,sr,band,windowLength=winL,frameshift=frameshift)
            
            # Extract labels
            words=np.load(path + p + '/preprocessing/' + p + '_' + str(ses) + '_words.npy')
            words=downsampleLabels(words,sr,windowLength=winL,frameshift=frameshift)
            words=words[modelOrder*stepSize:words.shape[0]-modelOrder*stepSize]

            # Load audio
            audio = np.load(path + p + '/preprocessing/' + p + '_' + str(ses)  + '_audio.npy')
            audioSamplingRate = 48000
            # Downsample Audio to 16kHz
            targetSR = 16000
            audio = decimate(audio,int(audioSamplingRate / targetSR))
            audioSamplingRate = targetSR
            # Write wav file of the audio
            scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
            wav.write(outPath + p + '/preprocessing/' + band + '/' + ref + '/' + p + '_' + str(ses)  + '_orig_audio.wav',audioSamplingRate,scaled)   
            # Extact log mel-scaled spectrograms
            melSpec = extractMelSpecs(scaled,audioSamplingRate,windowLength=winL,frameshift=frameshift,numFilter=23)
            # Raw audio aligned to each window (for unit selection)
            winAudio = windowAudio(scaled, audioSamplingRate,windowLength=winL,frameshift=frameshift)
            
            # Comment if you don't want to use feature stacking
            # Stack features
            feat = stackFeatures(feat,modelOrder=modelOrder,stepSize=stepSize)
            # Align to EEEG features
            melSpec = melSpec[modelOrder*stepSize:melSpec.shape[0]-modelOrder*stepSize,:]
            winAudio = winAudio[modelOrder*stepSize:winAudio.shape[0]-modelOrder*stepSize,:]
            if melSpec.shape[0]!=feat.shape[0]:
                print('Possible Problem with ECoG/Audio alignment for %s session %d.' % (p,ses))
                print('Diff is %d' % (np.abs(feat.shape[0]-melSpec.shape[0])))
                tLen = np.min([melSpec.shape[0],feat.shape[0]])
                melSpec = melSpec[:tLen,:]
                winAudio = winAudio[:tLen,:]
                feat = feat[:tLen,:]
            
            # Save everything
            np.save(outPath + p + '/preprocessing/' + band + '/' + ref + '/' + p + '_' + str(ses)  + '_feat.npy', feat)
            np.save(outPath + p + '/preprocessing/' + band + '/' + ref + '/' + p + '_' + str(ses)  + '_procWords.npy', words)
            np.save(outPath + p + '/preprocessing/' + band + '/' + ref + '/' + p + '_' + str(ses)  + '_spec.npy', melSpec)
            np.save(outPath + p + '/preprocessing/' + band + '/' + ref + '/' + p + '_' + str(ses)  + '_winAudio.npy', winAudio)   
            #Add context - channel names
            try: 
                np.save(outPath + p + '/preprocessing/' + band + '/' + ref + '/' + p + '_' + str(ses)  + '_channel_names.npy', channels_des)
            except NameError:
                np.save(outPath + p + '/preprocessing/' + band + '/' + ref + '/' + p + '_' + str(ses)  + '_channel_names.npy', channels)
            #Add context - feature names
            channels = nameVector(channels, modelOrder=modelOrder) #comment if no stacking
            np.save(outPath + p + '/preprocessing/' + band + '/' + ref + '/' + p + '_' + str(ses)  + '_feat_names.npy', channels)
           