import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import bisect
import statsmodels.stats.anova as sm
import pandas as pd

data_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Data_Analysis/Data/RawData/'
output_path = 'C:/Users/laura/OneDrive/Documenti/Internship/Data_Analysis/Results/'
participants = ['kh21', 'kh22', 'kh23', 'kh24','kh25'] #'kh21', 'kh22', 'kh23', 'kh24', 

for pNr, participant in enumerate(participants):
    data, header = pyxdf.load_xdf(data_path + participant + '_test.xdf')
    streams = {}
    initial_time = None
    trial_n = 0
    Stim_n = 0

    allTrials = {}

    for stream in data:
        markers = stream['time_series']

        if isinstance(markers, list):
            # list of strings
            all_streams = zip(stream['time_stamps'], markers)
            for timestamp, marker in all_streams:
                if initial_time is None:
                    initial_time = timestamp

                #print(f'{marker[0]} @ {(timestamp-initial_time):.3f} s')
                streams[(round((timestamp-initial_time), 3))] = marker[0]

            for value in streams.values():
                if 'Sum' in value:
                    string2 = value.replace(',', '')
                    string3 = string2.replace('Sum Trail: ', '')
                    my_list = string3.split(' ')
                    allTrials[my_list[0]] = my_list[1:]

        elif isinstance(markers, np.ndarray): #numeric data
            continue
        else:
            raise RuntimeError('Unknown stream format')
        
    #print(streams)
    #print(allTrials)

    # Analyze data per stimulus repetition
    repitions={}

    for trial in allTrials.values():
        if trial[1] in repitions:
            repitions[trial[1]]+=[trial[3]]
        else:
            repitions[trial[1]]=[trial[3]]

    #print(repitions)

    results={}
    total_c=0
    total_i=0
    total_n=0
    for key in repitions:
        correct = repitions[key].count('Correct')
        incorrect = repitions[key].count('Incorrect')
        no = repitions[key].count('No')
        results[key]=[correct,incorrect,no]
        total_c+=correct
        total_i+=incorrect
        total_n+=no


    accuracy_total = [total_c, total_i, total_n]
    print(results)
    print(accuracy_total)