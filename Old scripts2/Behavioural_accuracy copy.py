import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import bisect
import statsmodels.stats.anova as sm
import os

# Define function to get indices
def locate_pos(available, targets):
    pos = bisect.bisect_right(available, targets)
    if pos == 0:
        return 0
    if pos == len(available):
        return len(available)-1
    if abs(available[pos]-targets) < abs(available[pos-1]-targets):
        return pos
    else:
        return pos-1  

# Define function for significant bars
def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black', alpha= 0.3)

def behavioural(participants=['kh21', 'kh22', 'kh23', 'kh24','kh25']):
    data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/RawData/'
    out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/BehaviouralResults/'


    # Load data
    for pNr, participant in enumerate(participants):
        data, _ = pyxdf.load_xdf(data_path + '{}_test.xdf'.format(participant))
        streams = {}
        initial_time = None
        trial_n = 0
        Stim_n = 0

        allTrials = {}
        all_FB = {}
        streams_originaltime = {}
        bad_trials=[]


    for stream in data:
        markers = stream['time_series']
        time_points = stream['time_stamps']

        if isinstance(markers, list): # list of strings
            all_streams = zip(time_points, markers)
            for timestamp, marker in all_streams:
                if initial_time is None:
                    initial_time = timestamp
                streams[(round((timestamp-initial_time), 3))] = marker[0]
            for value in streams.values():
                if 'Sum' in value:
                    my_list = ((value.replace(',', '')).replace('Sum Trail: ', '')).split(' ')
                    allTrials[my_list[0]] = my_list[1:]
        elif isinstance(markers, np.ndarray): #numeric data
            continue
        else:
            raise RuntimeError('Unknown stream format')

    for key, trial in allTrials.items():
        if trial[-1] == 'Correct':
            allTrials[key][-1] = 1
        elif trial[-1] == 'Incorrect':
            allTrials[key][-1] = 0
        else:
            allTrials[key] = allTrials[key][0:4]
            allTrials[key][-1] = 2
            bad_trials.append(int(key)-1)


    # Analyze x stimulus
    stimuli={}
    for trial in allTrials.values():
        if trial[0] in stimuli:
            stimuli[trial[0]]+=[trial[3]]
        else:
            stimuli[trial[0]]=[trial[3]]
    
    repetitions = {1:[], 2:[], 3:[]}
    for stim in stimuli.values():
        repetitions[1].append(stim[0])
        repetitions[2].append(stim[1])
        repetitions[3].append(stim[2])

    # Statistics: paired samples t-test
    _, p1_2 = ss.ttest_rel(repetitions[1], repetitions[2])
    _, p1_3 = ss.ttest_rel(repetitions[1], repetitions[3])
    _, p2_3 = ss.ttest_rel(repetitions[2], repetitions[3])
    pp_sigh[participant] = [p1_2, p1_3, p2_3]
    
    # Analyze data per stimulus repetition
    accuracy={1:0, 2:0, 3:0, 'tot':0}
    for trial in stimuli.values():
        for rep, acc in enumerate(trial):
            accuracy[rep+1]+= acc
            accuracy['tot']+= acc
    
    # Load data
    data, _ = pyxdf.load_xdf(data_path + '{}_test.xdf'.format(participant))
    if 'kh' in participant and '25' not in participant:
        n = 1
        r = 0
    elif 'us' in participant:
        n = 3
        r = 0
        good_channels = list(np.arange(0,40))
    else:
        n = 0
        r = 1
    markers = data[r]['time_series']
    task_ts = data[r]['time_stamps']
    seeg_ts = data[n]['time_stamps']
    sr = int(float(data[n]['info']['nominal_srate'][0]))
    # Reaction times
    indices_cue = [x for x in range(len(markers)) if 'Start Cue' in markers[x][0]]
    indices_response = [x for x in range(len(markers)) if 'Press' in markers[x][0] and 'wrong' not in markers[x][0]]
    indices_noresponse = [x for x in range(len(markers)) if 'Start Warning' in markers[x][0]]
    for bt in bad_trials:
        ind = indices_noresponse.pop(0)
        indices_response.insert(bt, ind)
    epoch_cue = np.array([locate_pos(seeg_ts, x) for x in task_ts[indices_cue]])
    epoch_response = np.array([locate_pos(seeg_ts, x) for x in task_ts[indices_response]])
    RTs =(epoch_response - epoch_cue)/sr
    repetition_labels = []
    for M in markers:
        if 'Sum' in M[0]:
            repetition_labels.append(int(((M[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')[2]))
    repetition_labels = np.array(repetition_labels)
    RTs1 = RTs[repetition_labels<2]
    RTs2 = RTs[repetition_labels==2]
    RTs3 = RTs[repetition_labels>2]
    
    # Statistics: paired samples t-test
    _, pRT1_2 = ss.ttest_rel(RTs1, RTs2)
    _, pRT1_3 = ss.ttest_rel(RTs1, RTs3)
    _, pRT2_3 = ss.ttest_rel(RTs2, RTs3)
    pp_sighRT[participant] = [pRT1_2, pRT1_3, pRT2_3]


# # RTs plots
#     T = [np.mean(RTs1)*1000, np.mean(RTs2)*1000, np.mean(RTs3)*1000]
#     RTsax.plot(X,T, marker='.', linestyle=':', label= 'P0' + str(pNr+1))   

# Histograms
    y = [v for v in accuracy.values()]
    y[:3] = [v/30 for v in y[:3]]
    y[3] = y[3]/90

    if pNr == 0:
        rects = ax.bar(x - 2*width, y, width, align='center', alpha=0.5, label='P0' + str(pNr+1))
    elif pNr == 1:
        rects = ax.bar(x - width, y, width, align='center', alpha=0.5, label='P0' + str(pNr+1))
    elif pNr == 2:
        rects = ax.bar(x, y, width, align='center', alpha=0.5, label='P0' + str(pNr+1))
    elif pNr == 3:
        rects = ax.bar(x + 1*width, y, width, align='center', alpha=0.5, label='P0' + str(pNr+1))
    elif pNr == 4:
        rects = ax.bar(x + 2*width, y, width, align='center', alpha=0.5, label='P0' + str(pNr+1))
    for b in rects:
        height = b.get_height()
        ax.annotate('{}'.format(round(height, 2)),
            xy=(b.get_x() + b.get_width() / 2, height),
            xytext= (0, 0.5),
            textcoords="offset points",
            ha='center', va='bottom', size=7)
    ax.axhline(y=0.5, color='grey', alpha=0.3, linestyle='--')
ax.set_ylabel('Accuracy (correct/total)')
ax.set_title('Accuracy scores per participant and repetition')
ax.set_xticks(x)
ax.set_xticklabels(['1st\npresentation', '2nd\npresentation', '3rd\npresentation', 'total'])
ax.set_ymargin(0.1)
ax.legend(loc="upper left")


# RTsax.set_ylabel('Reaction times (ms)')
# RTsax.set_title('Reaction times per participant and repetition')
# RTsax.set_xticks(X)
# RTsax.set_xticklabels(['1st\npresentation', '2nd\npresentation', '3rd\npresentation'])
# RTsax.set_ymargin(0.1)
# RTsax.legend(loc="best")

if not os.path.exists(out_path):
    os.makedirs(out_path)   
      
plt.savefig(out_path + 'accuracy_behavioural_RTs', dpi=300)

for part, pp in pp_sigh.items():
    pp_sigh[part] = np.array(pp) < 0.05
for part, pp in pp_sighRT.items():
    pp_sighRT[part] = np.array(pp) < 0.05

print(pp_sigh)
print(pp_sighRT)

