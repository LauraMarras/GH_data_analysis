from plotly.subplots import make_subplots
#import plotly.io as pio
#pio.kaleido.scope.chromium_args = tuple([arg for arg in pio.kaleido.scope.chromium_args if arg != "--disable-dev-shm-usage"])
import plotly.graph_objects as go
import scipy.io as sio
import numpy as np
import open3d as o3d
import matplotlib.colors as mcolors
import matplotlib.cm as mcm 
import matplotlib.pyplot as plt

PPs = ['21']#, '22', '23', '24']


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, flip=False):
   '''
   https://stackoverflow.com/a/18926541
   '''
   if isinstance(cmap, str):
      cmap = plt.get_cmap(cmap)
   if flip:
      new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
         cmap(np.flip(np.linspace(minval, maxval, n))))
   else:
      new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
   
   return new_cmap


def compare_lists(labelled,decoded):
    surplus = []
    if len(labelled) > len(decoded):
        print('labelled electrodes are ' + str(len(labelled)-len(decoded)) + ' more than decoded electrodes')
        for el in labelled:
            if el not in decoded:
                surplus.append(el)
    elif len(labelled) < len(decoded):
        print('labelled electrodes are ' + str(len(decoded)-len(labelled)) + ' less than decoded electrodes')
        for el in decoded:
            if el not in labelled:
                surplus.append(el)
    else:
        print('labelled electrodes are equal number than decoded electrodes')
    
    return surplus

#Define general plotting layout
#
layout={'showlegend':False,'width':1000,'height':800,
       'scene':{'xaxis': {
    'showgrid': False, # thin lines in the background
    'zeroline': False, # thick line at x=0
    'visible': False,  # numbers below
},
        'yaxis': {
    'showgrid': False, # thin lines in the background
    'zeroline': False, # thick line at x=0
    'visible': False,  # numbers below
},       
       'zaxis': {
    'showgrid': False, # thin lines in the background
    'zeroline': False, # thick line at x=0
    'visible': False,  # numbers below
}, 
               }
       }


# Simplify the cortex mesh to speed up plotting
# If installing open3d is difficult, this can be skipped
def reduceMesh(vertices,triangles,numTriangles=20000):
    tm = o3d.geometry.TriangleMesh()
    tm.vertices= o3d.utility.Vector3dVector(vertices)
    tm.triangles=o3d.utility.Vector3iVector(triangles)
    tm2=tm.simplify_quadric_decimation(numTriangles)
    smallTri=np.asarray(tm2.triangles)
    smallVert=np.asarray(tm2.vertices)
    return smallVert, smallTri

# Load the meshes extracted by img_pipe/freesurfer
for pNr in PPs:    
    name = 'patient_' + pNr
    pathToParticipant = r'C:/Users/laura/Documents/Data_Analysis/Labelling/kh{}/'.format(pNr)
    # Load left hemisphere
    lh = sio.loadmat(pathToParticipant + name +  '_lh_pial.mat')
    lhTri = lh['cortex']['tri'][0][0]-1
    lhVert = lh['cortex']['vert'][0][0]
    lhVert, lhTri = reduceMesh(lhVert,lhTri,numTriangles=5000)
    # Load right hemisphere
    rh = sio.loadmat(pathToParticipant + name +  '_rh_pial.mat')
    rhTri=rh['cortex']['tri'][0][0]-1
    rhVert=rh['cortex']['vert'][0][0]
    rhVert, rhTri = reduceMesh(rhVert,rhTri,numTriangles=5000)
    #Combine to cortex
    cortexVert = np.concatenate([lhVert,rhVert])
    cortexTri = np.concatenate([lhTri,rhTri+lhVert.shape[0]])
    #Loading electrodes
    elecs = sio.loadmat(pathToParticipant + 'elecs_all.mat')
    elecLocs = elecs['elecmatrix']
    anatomy = elecs['anatomy'][:,3]
    anatomy = np.array([i[0] for i in anatomy])
    elecNames = elecs['anatomy'][:,0]
    elecNames = np.array([i[0] for i in elecNames])

    # Load decoded channels and check for differences
    decoded_channels = np.load(pathToParticipant + 'recorded_channels.npy')
    surplus = compare_lists(elecNames,decoded_channels)
    index_surplus = [np.where(elecNames == x)[0][0] for x in surplus]

    #Assign Weights to electrodes (for example decoding accuracy)
    electrodeWeights = np.zeros(elecNames.shape[0])
    
    # Load decoding accuracy scores and add -1 for not  decoded electrodes
    scores = np.load(pathToParticipant + 'score_means.npy')
    for x in index_surplus:
        scores = np.insert(scores, x, 0)
    
    # Load p_values and add 2 for not  decoded electrodes
    p_vals = np.load(pathToParticipant + 'p_vals.npy')
    for x in index_surplus:
        p_vals = np.insert(p_vals, x, 1)

    electrodeWeights = scores
    
    # Create colorscale
    # Create colorMap
    cmap = truncate_colormap('autumn', minval=0, maxval=0.8, n=100, flip=False)
    bounds = np.array([[0.5, 0.6, 0.7, 0.8, 0.9, 1]])
    normbound = mcolors.BoundaryNorm(bounds[0], len(bounds[0])-1)

    bounds = np.array([[0.5, 0.6, 0.7, 0.8, 0.9, 1]])
    my_cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', ['#111a6f', '#8e008b', '#ef2e78', '#ffc01c', '#fff835'], len(bounds[0])-1) 
    norm = mcolors.BoundaryNorm(bounds[0], len(bounds[0])-1)
    
    color_coding = cmap(normbound(scores))

    cols=[]
    for p in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        cols.append([p, 'rgba' + str(cmap(normbound(scores)))])
    
    # Plotting Cortex
    gold='rgb(0.9,0.8,0.5)'
    grey='rgb(0.7,0.7,0.7)'
    #opacity=0.4, representation='surface',color=(0.9,0.8,0.5)

    opacity=0.2 #0.2
    lightning=dict(specular=1.0,) #diffuse=0.9,roughness=0.9specular=2.0,
    lightposition=dict(x=0,
                        y=80,
                        z=0)

    cortex = go.Mesh3d(x=cortexVert[:,0],y=cortexVert[:,1],z=cortexVert[:,2],
                        i=cortexTri[:,0],j=cortexTri[:,1],k=cortexTri[:,2],
                    color=grey,opacity=opacity,name='cortex',hoverinfo='skip',lighting=lightning,lightposition=lightposition)

    #Plotting the electrodes,
    #Colors can be used to visualize electrode weights, selected electrodes, ...
    #Alternatively, all can be set to the same color

    elecMarker = go.Scatter3d(x=elecLocs[:,0],y=elecLocs[:,1],z=elecLocs[:,2],mode='markers',hovertext=anatomy,
                        marker = dict(
            size = 5,
            color = electrodeWeights,#'rgba(152, 0, 0, .8)',
            #color= np.ones((electrodeWeights.shape)),
            #color_discrete_map=cols,
            colorscale=cols,
            colorbar=dict(len=0.5, title='AUC score')
        )
    )



    fig=go.Figure(data=[cortex,elecMarker],layout=layout) #hipp,thalamus,
    fig['layout'].update(autosize=False)
    #Tight layout
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=1.25)
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    fig.show()
