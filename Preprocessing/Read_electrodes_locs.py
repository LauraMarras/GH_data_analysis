import scipy.io
import pyxdf
import pandas as pd

def shaft_electrodes(elec_list):
    elec_dict = {}
    for e in elec_list:
        lab = e[0:2]
        if len(e)>3:
            num = e[-2:]
        else:
            num = e[-1]
        if lab not in elec_dict.keys():
            elec_dict[lab] = [num]
        else:
            elec_dict[lab].append(num)
    return elec_dict

def extract_FreeSurf_label(label):
   path = 'C:/Users/laura/Documents/Data_Analysis/Labelling/'
   df = pd.read_excel(path + 'FreesurferLabels.xlsx', header=0)
   df = df.set_index('label')
   df['radlex_label']=df['radlex_label'].fillna(df.index.to_series())
   if label == 'Unknown':
       name = 'Unknown'
   else:
      name = df.loc[label]['radlex_label']
   return name

def read_mat_elecs(path, pp):
    mat_file = scipy.io.loadmat(path + pp + '/elecs_all.mat')
    elecs_label = [x[0][0]for x in mat_file['anatomy']]
    elecs_loc = [x[-1][0] for x in mat_file['anatomy']]

    locs = pd.DataFrame({'Label': elecs_label, 'Location':elecs_loc}).set_index('Label')
    locs['Description'] = locs['Location'].apply(extract_FreeSurf_label)

    locs.to_excel(excel_writer = path + pp + '/locs.xlsx')
    return locs

def read_csv_elecs(path,pp):
    csv_file = pd.read_csv(path + pp + '/electrode_locations.csv', header=0)
    locs_1 = csv_file[['electrode_name_1', 'location']]
    locs_2 = locs_1.rename(columns={'electrode_name_1':'Label', 'location':'Location'})
    locs = locs_2.set_index('Label')
    locs.to_excel(excel_writer=path + pp + '/locs.xlsx')
    return locs

def read_raw_file(path, pp, n):
    raw_data, _ = pyxdf.load_xdf(path + pp + '_test.xdf')
    raw_data_channels = [x['label'][0] for x in raw_data[n]['info']['desc'][0]['channels'][0]['channel']]
    return raw_data_channels


if __name__=="__main__":
    raw_path = 'C:/Users/laura/Documents/Data_Analysis/Data/RawData/'
    path = 'C:/Users/laura/Documents/Data_Analysis/Data/Labelling/'
    PPs = ['kh25'] #'kh21', 'kh22', 'kh23', 'kh24'
    for pp in PPs:
        read_mat_elecs(path, pp)
