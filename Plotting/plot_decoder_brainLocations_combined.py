from mayavi import mlab
from ctmr_brain_plot import el_add, ctmr_gauss_plot
import matplotlib.colors as mcolors
import os
import scipy.io as sio
import numpy as np

pathToParticipant='C:/Users/laura/Documents/Data_Analysis/Data/Labelling/kh21-25_warped/cvs35inMNI152 - Meshes'

# Load left hemisphere
lh = sio.loadmat(pathToParticipant + '/lh_pial_trivert.mat')
lhTri=lh['tri']
lhVert=lh['vert']

# Load right hemisphere
rh = sio.loadmat(pathToParticipant + '/rh_pial_trivert.mat')
rhTri=rh['tri']
rhVert=rh['vert']

# Combine to cortex
cortexVert = np.concatenate([lhVert,rhVert])
cortexTri = np.concatenate([lhTri,rhTri+lhVert.shape[0]])

# Plot brain
mesh, mlab_mesh = ctmr_gauss_plot(cortexTri,cortexVert,opacity = 0.1)

# Loading electrodes
pathToElectrodes = 'C:/Users/laura/Documents/Data_Analysis/Data/Labelling/kh21-25_warped/warped coordinates'

for pNr, sub in enumerate(['kh21','kh22','kh23','kh24','kh25']):
    elecs = sio.loadmat(pathToElectrodes + '\\' + sub + '_elecs_all_warped.mat')
    elecLocs = elecs['elecmatrix']
    anatomy=elecs['anatomy'][:,3]
    anatomy = np.array([i[0] for i in anatomy])
    elecNames = elecs['anatomy'][:,0]
    elecNames = np.array([i[0] for i in elecNames])


# Load recorded channels and check for not labelled channels that weren't recorded
    recorded_channels = np.load('C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/{}_channels.npy'.format(sub))
            
    not_recorded = []
    if len(elecNames) > len(recorded_channels):
        print('labelled electrodes are ' + str(len(elecNames)-len(recorded_channels)) + ' more than recorded electrodes')
        for el in elecNames:
            if el not in recorded_channels:
                not_recorded.append(el)
    elif len(elecNames) < len(recorded_channels):
        print('labelled electrodes are ' + str(len(recorded_channels)-len(elecNames)) + ' less than recorded electrodes')
        for el in recorded_channels:
            if el not in elecNames:
                not_recorded.append(el)
    else:
        print('labelled electrodes are equal number than recorded electrodes')

    index_not_recorded = [np.where(elecNames == x)[0][0] for x in not_recorded]

# Remove not registered electrodes from elecLocs
    recElecLocs = np.delete(elecLocs, index_not_recorded, axis=0)


# Load p vals and AUC scores 
    scoresFile = np.load('C:/Users/laura/Documents/Data_Analysis/Data/DecodingResults/feedback_accuracy/{}/{}_decoder_single_electrodes.npz'.format(sub,sub))

    score_means = scoresFile['score_means']
    p_vals = scoresFile['p_vals']
    threshold = scoresFile['threshold']

# Select only significant channels (location and AUC scores)
    index_not_signif = np.where(p_vals > 0.05)
    signElecLocs = np.delete(recElecLocs, index_not_signif, axis=0)
    signScores = np.delete(score_means, index_not_signif, axis=0)

# Select not significant channels
    index_signif = np.where(p_vals <= 0.05)
    not_signElecLocs = np.delete(recElecLocs, index_signif, axis=0)
    not_signScores = np.delete(score_means, index_signif, axis=0)

# Assign Weights to electrodes: AUC scores
    electrodeWeights = score_means

# Create ColorMap
    # Define color palette
    palette = ['#6C757D']
    colors = ['#12b2e2', '#153ae0', '#850ad6','#f61379']
    num_colors = 10-int(threshold[3]*10)
    palette.extend(colors[-(num_colors):])

    # Define bounds, norm and create colormap
    limits = np.linspace(0,1, num=11, endpoint=True)
    bounds = np.insert(limits[limits>threshold[3]], 0, [0, threshold[3]])
    norm = mcolors.BoundaryNorm(bounds, len(bounds)-1)
    cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', palette, len(bounds)-1)

    # Extract color RGB values from colormap
    colors = cmap(norm(electrodeWeights))[:,:3]

# Plot Electrodes with color coding
    el_add(recElecLocs, colors)

# Add colorbar
mesh.module_manager.scalar_lut_manager.lut.table = (cmap(np.linspace(0, 1, 255)) * 255).astype('int')
mlab.colorbar(title='AUC score', orientation='vertical', nb_labels = 0)


@mlab.animate
def anim():
    f = mlab.gcf()
    while 1:
        f.scene.camera.azimuth(10)
        f.scene.render()
        yield

a = anim() # Starts the animation.

# Save fig
out_path = 'C:/Users/laura/Documents/Data_Analysis/Plots/brain_locations/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
mlab.savefig(out_path + 'Combined_Accuracy_{}.png'.format(pNr, str(1)))