import pyxdf
import numpy as np

data, header = pyxdf.load_xdf('C:/Users/laura/OneDrive/Documenti/Internship/Python/Data/RawData/kh23_test.xdf')
streams = {}
initial_time = None
trial_n = 0
Stim_n = 0

allTrials = {}
all_FB = {}
streams_originaltime = {}


for stream in data:
    markers = stream['time_series']
    time_points = stream['time_stamps']

    if isinstance(markers, list):
        # list of strings
        all_streams = zip(time_points, markers)
        for timestamp, marker in all_streams:
            if initial_time is None:
                initial_time = timestamp

            #print(f'{marker[0]} @ {(timestamp-initial_time):.3f} s')
            streams[(round((timestamp-initial_time), 3))] = marker[0]
            #streams_originaltime[timestamp] = marker[0] 

        for value in streams.values():
            if 'Sum' in value:
                my_list = ((value.replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                #string3 = string2.replace('Sum Trail: ', '')
                #my_list = string2.split(' ')
                allTrials[my_list[0]] = my_list[1:]


        # trialn = 0
        # for key, value in streams_originaltime.items():
        #     if 'Start Trial' in value:
        #         trialn += 1
        #     elif 'start Fb' in value:
        #         all_FB[trialn] = [key]
        #     elif 'Sum' in value:
        #         string2 = value.replace(',', '')
        #         string3 = string2.replace('Sum Trail: ', '')
        #         my_list = string3.split(' ')
        #         all_FB[trialn].append(my_list[2:])


    elif isinstance(markers, np.ndarray):
        #numeric data
        continue

    else:
        raise RuntimeError('Unknown stream format')

#print(streams)
#print(allTrials)
#print(all_FB)

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





