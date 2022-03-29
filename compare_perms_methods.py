import numpy as np

participants = ['kh21', 'kh22', 'kh23', 'kh24', 'kh25']
repetitions = ['rep_1', 'rep_2_3', 'rep_all']
bands = ['gamma', 'theta', 'delta', 'alpha', 'beta'] #'gamma', 'delta', 'theta', 'alpha', 'beta'
  
dp = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/'
dp2 = 'C:/Users/laura/Documents/Data_Analysis/Labelling/'
data_path1 = 'C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/long_FB_accuracy/'
data_path = 'C:/Users/laura/Documents/Data_Analysis/DecodingResults/long_FB_accuracy/'

PPs_pvals = np.load(data_path + 'PPs_pvals_elecShaftR_gamma.npy', allow_pickle=True).item()
pvals = PPs_pvals['rep_all_0']
p_vals = np.load(data_path1 + 'gamma/kh21_decoder_gamma_rep_all.npz')['p_vals_2']

# for band in bands:
#     for participant in participants:
#         for rep in repetitions:
#             threshold = np.load(data_path1 + band + '/' + participant + '_decoder_' + band + '_' + rep + '.npz')['threshold'][3]
#             print(band + ' ' + participant + ' ' + rep + ': ' + str(threshold))


# for band in bands:
#     for rep in repetitions:
#         threshold = np.load(data_path1 + band + '/PPs_decoder_' + band + '_' + rep + '.npz')['threshold_avg'][3]
#         print(band + ' ' + ' ' + rep + ': ' + str(threshold))

for pp in participants:
    chans = np.load(dp + pp + '_channels.npy')
    chans2 = np.load(dp2 + pp + '/recorded_channels.npy')

    print('d')