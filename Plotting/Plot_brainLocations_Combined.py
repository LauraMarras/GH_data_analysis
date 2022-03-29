from mayavi import mlab
import scipy.io as sio
import numpy as np

from ctmr_brain_plot import ctmr_gauss_plot,el_add

pathToParticipant='C:/Users/laura/Documents/Data_Analysis/Data/Labelling/kh21-25_warped/cvs35inMNI152 - Meshes'

lh = sio.loadmat(pathToParticipant + '/lh_pial_trivert.mat')

lhTri=lh['tri']
lhVert=lh['vert']
# # Load right hemisphere
rh = sio.loadmat(pathToParticipant + '/rh_pial_trivert.mat')
rhTri=rh['tri']
rhVert=rh['vert']

cortexVert = np.concatenate([lhVert,rhVert])
cortexTri = np.concatenate([lhTri,rhTri+lhVert.shape[0]])
mesh = ctmr_gauss_plot(cortexTri,cortexVert,opacity = 0.1)
pathToElectrodes = 'C:/Users/laura/Documents/Data_Analysis/Data/Labelling/kh21-25_warped/warped coordinates'
defaultColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
rgbc = [tuple(int(c.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) for c in defaultColors]

for i, sub in enumerate(['kh21','kh22','kh23','kh24','kh25']):
    elecs = sio.loadmat(pathToElectrodes + '\\' + sub + '_elecs_all_warped.mat')['elecmatrix']
    el_add(elecs,color=rgbc[i])
