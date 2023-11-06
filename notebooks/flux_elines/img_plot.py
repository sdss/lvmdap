#!/usr/bin/python3

import numpy as np
from pylab import *
import matplotlib
import scipy.ndimage as spimage
import matplotlib as mpl
#from numpy import std as biweight_midvariance
import matplotlib.cm as cm

from matplotlib import colors
import scipy.ndimage as ndimage
from matplotlib.legend import Legend
import matplotlib.patches as patches
from matplotlib import pyplot
#
#
#

# 3D plots!
#
import numpy as np
import matplotlib.pyplot as plt

#biweight_midvariance


import warnings
warnings.simplefilter("ignore")

#
# my_utils
#from my_utils import *

#
#
#
#
# Carlos Color map
#from cmaps_CLC import vel_map

from matplotlib import rcParams as rc
rc.update({'font.size': 20,\
           'font.weight': 900,\
           'text.usetex': True,\
           'path.simplify'           :   True,\
           'xtick.labelsize' : 20,\
           'ytick.labelsize' : 20,\
#           'xtick.major.size' : 3.5,\
#           'ytick.major.size' : 3.5,\
           'axes.linewidth'  : 2.0,\
               # Increase the tick-mark lengths (defaults are 4 and 2)
           'xtick.major.size'        :   6,\
           'ytick.major.size'        :   6,\
           'xtick.minor.size'        :   3,\
           'ytick.minor.size'        :   3,\
           'xtick.major.width'       :   1,\
           'ytick.major.width'       :   1,\
           'lines.markeredgewidth'   :   1,\
           'legend.numpoints'        :   1,\
           'xtick.minor.width'       :   1,\
           'ytick.minor.width'       :   1,\
           'legend.frameon'          :   False,\
           'legend.handletextpad'    :   0.3,\
           'font.family'    :   'serif',\
           'mathtext.fontset'        :   'stix',\
           'axes.facecolor' : "w",\
           
          })

def img_rgb_plot(name='NGC5947',dir='images',out_dir='fig',pixel_size=0.369,h_size=74/2,format='png',offx=-5,offy=0):
    SDSS_jpg_file=dir+'/'+name+'_SDSS.rgb.png'
    img_v22=dir+'/'+name+'.rgb.png'
    img_v23=dir+'/'+name+'.RGB.png'
    image_in = plt.imread(SDSS_jpg_file)
    image_v22 = plt.imread(img_v22)
    image_v23 = plt.imread(img_v23)

    image = np.flipud(image_in)
    image_shape=image.shape
    nx=image_shape[1]
    ny=image_shape[0]
    print(nx,ny)    
    size=210
   # offx=-5
   # offy=0
    nx0=int(nx/2)+offx-int(size/2)
    nx1=int(nx/2)+offx+int(size/2)
    ny0=int(ny/2)+offy-int(0.95*size/2)
    ny1=int(ny/2)+offy+int(0.95*size/2)
    print(nx0,nx1,ny0,ny1)
#        nx1=int(nx/2)+int(ny/2)
#        image=image[:,nx0:nx1,:]
#    else:
#        if (nx<ny):
#            ny0=int(ny/2)-int(nx/2)
#            ny1=int(ny/2)+int(nx/2)    
#    if (nx>ny):
#        nx0=int(nx/2)-int(ny/2)
#        nx1=int(nx/2)+int(ny/2)
#        image=image[:,nx0:nx1,:]
#    else:
#        if (nx<ny):
#            ny0=int(ny/2)-int(nx/2)
#            ny1=int(ny/2)+int(nx/2)
    image=image[ny0:ny1,nx0:nx1,:]
    
    image_shape=image.shape
    nx=image_shape[1]*pixel_size
    ny=image_shape[0]*pixel_size
    print(nx,ny)
    print(image_shape)
    
    image_v22 = np.flipud(image_v22)
    nx_v22=image_v22.shape[1]
    ny_v22=image_v22.shape[0]
    
    image_v23 = np.flipud(image_v23)
    nx_v23=image_v23.shape[1]*0.5
    ny_v23=image_v23.shape[0]*0.5
    
    
#    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 7))
    fig = plt.figure(figsize=(14, 4))
    xsize=0.24
    ysize=0.85
    bottom=0.15
    left=0.075
    ax0 = fig.add_axes([left+0*xsize,bottom,xsize,ysize])
    ax1 = fig.add_axes([2*left+1*xsize,bottom,xsize,ysize])
    ax2 = fig.add_axes([3*left+2*xsize,bottom,xsize,ysize])
    ax=[ax0,ax1,ax2]
#    fig = plt.subplots(nrows=1, ncols=3, figsize=(16, 7), sharey=True)

    
    
    im = ax[0].imshow(image, extent=[-nx/2, nx/2, -ny/2, ny/2], origin='lower',aspect=1,interpolation='none')
    #h_size=h_size/pixel_size
    ax[0].add_patch(
        patches.RegularPolygon(
            (-0.25, -0.25),     # (x,y)
            6,              # number of vertices
            h_size,            # radius
            3.1416/2,
            fill=False,
            edgecolor="#eeeeee",
            linewidth=2
        )
    )
    ax[0].patch.set_facecolor('black')
    ax[0].set_xlabel(r'$\Delta$ RA (arcsec)')
    ax[0].set_ylabel(r'$\Delta$ DEC (arcsec)')
    
#    im = ax[1].imshow(image_v22, extent=[-nx_v22/2, nx_v22/2, -ny_v22/2, ny_v22/2], origin='lower')
    im = ax[2].imshow(image_v22, extent=[-nx_v22/2, nx_v22/2, -ny_v22/2, ny_v22/2], origin='lower', aspect=1,interpolation='none')
    ax[2].patch.set_facecolor('black')
    ax[2].set_xlabel(r'$\Delta$ RA (arcsec)')
    ax[2].set_ylabel(r'$\Delta$ DEC (arcsec)')
    
    im = ax[1].imshow(image_v23, extent=[-nx_v23/2, nx_v23/2, -ny_v23/2, ny_v23/2], origin='lower', aspect=1,interpolation='none')
    ax[1].patch.set_facecolor('black')
    ax[1].set_xlabel(r'$\Delta$ RA (arcsec)')
    ax[1].set_ylabel(r'$\Delta$ DEC (arcsec)')
    
    for ax_now in ax:
        ax_now.set_aspect('equal', 'box')

    ax[0].set_xticks([-30,-15,0,15,30])
    ax[1].set_xticks([-30,-15,0,15,30])
    ax[2].set_xticks([-30,-15,0,15,30])
    
    ax[0].set_yticks([-30,-15,0,15,30])
    ax[1].set_yticks([-30,-15,0,15,30])
    ax[2].set_yticks([-30,-15,0,15,30])
    
    ax[0].text(-37,30,'SDSS',fontsize=22,color='white')
    ax[1].text(-37,30,'v2.3',fontsize=22,color='white')
    ax[2].text(-37,30,'v2.2',fontsize=22,color='white')
               
               
    plt.tight_layout()
    out_fig=out_dir+'/'+name+'.rgb_comp.'+format
    fig.savefig(out_fig, transparent=False, facecolor='white', edgecolor='white')#.pdf")

        #    plt.setp(ax[1], ylim=ax[0].get_ylim())   
#    plt.setp(ax[1], xlim=ax[0].get_xlim())   
        
    

format='png'
offx=-5
offy=0
nargs=len(sys.argv)
if (nargs>3):
    name=sys.argv[1]
    dir=sys.argv[2]
    out_dir=sys.argv[3]
    if (nargs==7):
        format=sys.argv[4]
        offx=int(sys.argv[5])
        offy=int(sys.argv[6])
else:
    print('USE: img_plot.py name dir outdir [0] [0]')
    quit()

img_rgb_plot(name=name,dir=dir,out_dir=out_dir,offx=offx,offy=offy,format=format)
    
