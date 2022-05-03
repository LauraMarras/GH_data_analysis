import pyxdf
import numpy as np
import os
import pandas as pd

# Define function to extract labels
def extract_labels(participants=[]):
    data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/RawData/'
    out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Labels/'
    if not os.path.exists(out_path):
            os.makedirs(out_path)
    
    
    for pNr, participant in enumerate(participants):
    # Load data
        data, _ = pyxdf.load_xdf(data_path + '{}_test.xdf'.format(participant))
        if '25' in participant:
            markers = data[1]['time_series']
        else:
            markers = data[0]['time_series']

    # Extract info for each trial and store in list
        trials=[(((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')[0:5])+[pNr] for x in markers if 'Sum' in x[0]]
        header_labels = ['trial_nr', 'stimulus_nr', 'repetition', 'decision', 'accuracy', 'pNr']
    
    # Create dataframe containing all info about each trial: all possible labels
        labelsDF = pd.DataFrame(trials, columns=header_labels)
        labelsDF = labelsDF.replace(['Correct', 'Incorrect', 'No', 'w', 'l', 'None'], [1, 0, np.nan, 1, 0, np.nan])

    # Add label about stimulus category
        labelsDF['stimulus_category'] = labelsDF.decision[labelsDF.accuracy == 1]
        labelsDF.stimulus_category[labelsDF.accuracy == 0] = 1-labelsDF.decision

        labelsDF['trial_ind'] = (labelsDF['trial_nr'].astype(int))-1
        labelsDF.set_index('trial_ind', inplace=True)
        labelsDF = labelsDF.astype(np.float).astype('Int64')
 
    # Save dataframe
        labelsDF.to_pickle(out_path + '{}_labels.pkl'.format(participant))
    
if __name__=="__main__":
    participants=['kh21', 'kh22', 'kh23', 'kh24', 'kh25'] #['kh21', 'kh22', 'kh23', 'kh24', 'kh25']
    
    labelsDF = pd.read_pickle('C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Labels/kh25_labels.pkl')


    print('d')
    #extract_labels(participants=participants)