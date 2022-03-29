import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm 

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



# Create colorMap
cmap = truncate_colormap('brg', minval=0, maxval=0.8, n=100, flip=False)
bounds = np.array([[0.5, 0.6, 0.7, 0.8, 0.9, 1]])
normbound = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)


#cmapp = mcm.get_cmap('RdBu_r')

#color_coding = cmap(normbound(p_vals))

# create colormap2
my_cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', ['#560bad', '#b388eb', '#f72585', '#fdc500', '#484848']) #palette1: ['#8ecae6', '#219ebc', '#023047', '#ffb703', '#fb8500'], palette2: ['#560bad', '#7a3bdf', '#f72585', '#fdc500', '#484848']
norm = mcolors.BoundaryNorm(bounds[0], 256) #len(bounds[0])-1

colorss = my_cmap(norm([1,0,0.1,0.005]))

print('d')
#color_coding = my_cmap(norm(p_vals))