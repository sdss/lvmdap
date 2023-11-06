# 
#
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.path import Path
from   scipy import interp

'''
This function is hardcoded to return the 90, 80 and 60 % contours.
This function is hardcoded to return the index of the elements within the 80% contour.
'''

def sel2dhist_xy( x, y, nbins = 20): 

  x_plt, y_plt = x, y

  # Creates the 2d histogram, normalized
  counts, xbins, ybins = np.histogram2d(x_plt, y_plt, bins=nbins,
                                        normed=True,
                                        range=[[np.nanmin(x_plt),np.nanmax(x_plt)],[np.nanmin(y_plt),np.nanmax(y_plt)]])
  counts /= counts.max()

  # Creates the contour at the specified levels 
  mylevels1 = [0.05, 0.10, 0.20, 0.50, 0.70, 0.80, 0.90] # 90, 50, 30, 20, and 10% 
  cont_prev = plt.contour(counts.transpose(), mylevels1, 
            extent=(np.nanmin(x_plt),np.nanmax(x_plt),np.nanmin(y_plt),np.nanmax(y_plt)),linestyles='dashed', colors = 'k')

  # Checking if the contours do enclose the expected fractions
  fracs = []
  for (icollection, collection) in enumerate(cont_prev.collections):

      path = collection.get_paths()[0]    
      pathxy = path.vertices
      x_test = pathxy[:,0]
      y_test = pathxy[:,1]
      pathxy = np.vstack((x_test, y_test)).T

      # frac_inside_poly
      xy   = np.vstack([x_plt,y_plt]).transpose()
      frac = float(sum(Path(pathxy).contains_points(xy)))/len(x_plt)
      fracs.append(frac)

# Where are the true fractions....
  fractions= [0.90, 0.80, 0.40]
  levs     = cont_prev.levels
  fracs_r  = np.array(fracs)
  sortinds = np.argsort(fracs_r)
  levs     = levs[sortinds]
  fracs_r  = fracs_r[sortinds]
  levels   = interp(fractions, fracs_r, levs)
  print('Levels for 90, 80, 60%:', levels)

# Selection 80 percent of the sample
  cont_final = plt.contour(counts.transpose(), levels, 
                           extent=(np.nanmin(x_plt),np.nanmax(x_plt),np.nanmin(y_plt),np.nanmax(y_plt)))
  path_80    = cont_final.collections[1].get_paths()[0]
  pathxy     = path_80.vertices
  x_test     = pathxy[:,0]
  y_test     = pathxy[:,1]
  pathxy     = np.vstack((x_test, y_test)).T

  xy   = np.vstack([x_plt,y_plt]).T
  mask = Path(pathxy).contains_points(xy)
  selected = xy[mask]
  x1_tmp = selected[:,0] 
  y1_tmp = selected[:,1]
  print('Number of points plotted:', x_plt.size)
  print('----------')
  print('Fraction enclosed by the 80% contour:',x1_tmp.size / x_plt.size)
  print('----------')

  path_90 = cont_final.collections[0].get_paths()[0]
  path_80 = cont_final.collections[1].get_paths()[0]
  path_60 = cont_final.collections[2].get_paths()[0]

  cont_90, cont_80, cont_60 = path_90.vertices, path_80.vertices, path_60.vertices

  conts = [cont_90, cont_80, cont_60]

  plt.clf()

  return(x1_tmp, y1_tmp, mask, conts)

