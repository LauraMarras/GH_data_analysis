import numpy as np 
import os
import pyxdf

def cleanData(channels, data):
    """Remove useless channels form channel list and seeg data
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    channels: list of strings
        channel names

    Returns
    ----------
    data: array (samples, channels)
        cleaned EEG time series   
    channels: list of strings
        channels names
    """
    bad_channels_i = [c for c, x in enumerate(channels) if '+' in x or 'el' in x] # define useless channels
    bad_channels = [c for c in channels if '+' in x or 'el' in x] # define useless channels
    
    channels = [c for c in channels if c not in bad_channels]
    data = np.delete(data, bad_channels_i, axis=1) # remove from sEEG data
    
    return channels, data 

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


if __name__=='__main__':
    PPs = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25']

    for pp in PPs:
        data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/RawData/'
        out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Rereferencing/{}/'.format(pp)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    # Load data
        data, _ = pyxdf.load_xdf(data_path + '{}_test.xdf'.format(pp))
        if '25' in pp:
            n = 0
        else:
            n = 1
        
        seeg_raw = data[n]['time_series']
    
    # Get channels names, remove useless channels and clean data
        channels = [x['label'][0] for x in data[n]['info']['desc'][0]['channels'][0]['channel']] # get channels names
        channels, seeg = cleanData(channels, seeg_raw)
  
        np.save(out_path + '{}_cleanChannels'.format(pp), channels)
        
    # Rereferencing
        # LaplacianR
        seeg_LapR, chann_LapR = laplacianR(seeg, channels)
        np.save(out_path + '{}_seeg_LapR'.format(pp), seeg_LapR)
        np.save(out_path + '{}_chann_LapR'.format(pp), chann_LapR)
        
        # CAR
        seeg_CAR = commonAverageR(seeg)
        np.save(out_path + '{}_seeg_CAR'.format(pp), seeg_CAR)

        # elecShaftR
        seeg_ESR = elecShaftR(seeg, channels)
        np.save(out_path + '{}_seeg_ESR'.format(pp), seeg_ESR) 

        # Bipolar
        seeg_BPR, chann_BPR, chann_BPRdes = bipolarR(seeg, channels)
        np.save(out_path + '{}_seeg_BPR'.format(pp), seeg_BPR)
        np.save(out_path + '{}_chann_BPR'.format(pp), chann_BPR)
        np.save(out_path + '{}_chann_des_BPR'.format(pp), chann_BPRdes)   