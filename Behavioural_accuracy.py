from cmath import nan
import pyxdf
import numpy as np
from scipy.stats import ttest_rel
import pandas as pd
import os
from statsmodels.stats.multitest import (fdrcorrection, multipletests)
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
import scikit_posthocs as sp
import pingouin as pg


def behavioural_accuracy(participants=['kh21','kh22', 'kh23', 'kh24','kh25']):
    data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/RawData/'
    out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/BehaviouralResults/'

    pp_scores = np.zeros((len(participants), 4))
    pp_pvals = np.zeros((len(participants), 3))

    ppDF = pd.DataFrame(columns = ['pNr', 'trial nr.', 'stimulus nr.', 'repetition', 'decision', 'accuracy'])
        

    for pNr, participant in enumerate(participants):
    # Load data
        data, _ = pyxdf.load_xdf(data_path + '{}_test.xdf'.format(participant))
        if '25' in participant:
            markers = data[1]['time_series']
        else:
            markers = data[0]['time_series']
        

    # Create dataframe with all info for each trial
        trials=[(((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')[0:5])+[pNr] for x in markers if 'Sum' in x[0]]
        header_labels = ['trial nr.', 'stimulus nr.', 'repetition', 'decision', 'accuracy', 'pNr']
        trialsDF = pd.DataFrame(trials, columns=header_labels)
        trialsDF = trialsDF.replace(['Correct', 'Incorrect', 'No', 'w', 'l', 'None'], [1, 0, nan, 1, 0, nan])
    # Add each PP dataframe to general DF
        ppDF = ppDF.append(trialsDF)
    
    # Extract accuracy info for each repetition + total
        rep1 = trialsDF['accuracy'][trialsDF['repetition']=='1'].to_numpy()
        rep2 = trialsDF['accuracy'][trialsDF['repetition']=='2'].to_numpy()
        rep3 = trialsDF['accuracy'][trialsDF['repetition']=='3'].to_numpy()
        total = trialsDF['accuracy'].to_numpy()

    # Estimate and store scores for each repetition + total
        pp_scores[pNr,:] = [np.nanmean(rep1), np.nanmean(rep2), np.nanmean(rep3), np.nanmean(total)]

    # Single participant level statistics: paired samples t-test for each rep combination
        _, p1_2 = ttest_rel(rep1, rep2, nan_policy='omit')
        _, p1_3 = ttest_rel(rep1, rep3, nan_policy='omit')
        _, p2_3 = ttest_rel(rep2, rep3, nan_policy='omit')
        
    # Store p_vals
        pp_pvals[pNr,:] = [p1_2, p1_3, p2_3]
        #_, pcorrected, _, _ = multipletests(pp_pvals[pNr,:], method='bonferroni', is_sorted=False)


    # Group level statistics: AnovaRM
    ppDF = ppDF.dropna().astype(int) 
    ppMeans = ppDF[['pNr', 'repetition', 'accuracy']].groupby(['pNr', 'repetition']).mean().reset_index()

    anovaRes = AnovaRM(data=ppMeans, depvar='accuracy', subject='pNr', within=['repetition']).fit()
    anovaResDF = anovaRes.anova_table
    print(anovaRes)

    anovaRes2 = pg.rm_anova(dv='accuracy', within='repetition', subject='pNr', data=ppMeans, detailed=True)
    print(anovaRes2)

    # multiple comparisons TukeyHSD test
    res = pairwise_tukeyhsd(ppDF['accuracy'], ppDF['repetition'])
    np.set_printoptions(suppress=True)
    print(res)

    res2 = pairwise_tukeyhsd(ppMeans['accuracy'], ppMeans['repetition'])
    np.set_printoptions(suppress=True)
    print(res2)

    res3 = pg.pairwise_ttests(dv='accuracy', within='repetition', subject='pNr', padjust='fdr_bh', data=ppMeans)
    print(res3)

    sp.posthoc_ttest(ppDF, val_col='accuracy', group_col='repetition', p_adjust=None, pool_sd=True)

    #mod = MultiComparison(ppDF[ppDF['pNr']==3]['accuracy'], ppDF[ppDF['pNr']==3]['repetition'])
   
    # Save results
    if not os.path.exists(out_path):
        os.makedirs(out_path)     
    
    #np.savez(out_path + 'behavioural_accuracyScores_pVals', 
        #pp_scores=pp_scores,
        #pp_pvals=pp_pvals)

if __name__=="__main__":
    behavioural_accuracy()
