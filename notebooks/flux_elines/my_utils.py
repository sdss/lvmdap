import numpy as np
import pandas as pd
from pylab import *
import matplotlib
import plplot
from scipy import stats
#from io import StringIO
print(pd.__version__)
#AttributeError: 'Series' object has no attribute 'to_numpy'
import re

import math
import astropy as astro
import scipy.ndimage as spimage
import scipy.ndimage as ndimage
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
import matplotlib as mpl
from numpy import std as biweight_midvariance
import matplotlib.cm as cm

from scipy import optimize
from scipy.stats import gaussian_kde
from matplotlib import colors

from collections import Counter

from scipy.stats import binned_statistic_2d
from scipy.ndimage.filters import gaussian_filter

import math

def list_columns(obj, cols=4, columnwise=True, gap=4):
    """
    Print the given list in evenly-spaced columns.

    Parameters
    ----------
    obj : list
        The list to be printed.
    cols : int
        The number of columns in which the list should be printed.
    columnwise : bool, default=True
        If True, the items in the list will be printed column-wise.
        If False the items in the list will be printed row-wise.
    gap : int
        The number of spaces that should separate the longest column
        item/s from the next column. This is the effective spacing
        between columns based on the maximum len() of the list items.
    """

    sobj = [str(item) for item in obj]
    if cols > len(sobj): cols = len(sobj)
    max_len = max([len(item) for item in sobj])
    if columnwise: cols = int(math.ceil(float(len(sobj)) / float(cols)))
    plist = [sobj[i: i+cols] for i in range(0, len(sobj), cols)]
    if columnwise:
        if not len(plist[-1]) == cols:
            plist[-1].extend(['']*(len(sobj) - len(plist[-1])))
        plist = zip(*plist)
    printer = '\n'.join([
        ''.join([c.ljust(max_len + gap) for c in p])
        for p in plist])
    print(printer)
    
def make_colourmap(ind, red, green, blue, name):
    newInd = range(0, 256)
    r = np.interp(newInd, ind, red, left=None, right=None)
    g = np.interp(newInd, ind, green, left=None, right=None)
    b = np.interp(newInd, ind, blue, left=None, right=None)
    colours = np.transpose(np.asarray((r, g, b)))
    fctab= colours/255.0
    cmap = colors.ListedColormap(fctab, name=name,N=None)
    return cmap

def get_califa_velocity_cmap():
    ind = [1., 35., 90.,125.,160.,220.,255.]
    red = [148., 0., 0., 55.,221.,255.,255.]
    green = [ 0., 0.,191., 55.,160., 0.,165.]
    blue = [211.,128.,255., 55.,221., 0., 0.]
    return make_colourmap(ind, red, green, blue, 'califa_vel')


def get_califa_velocity_cmap_2():
    ind = [0., 1., 35., 90.,125.,160.,220.,255.]
    red = [ 0.,148., 0., 0., 55.,221.,255.,255.]
    green = [ 0., 0., 0.,191., 55.,160., 0.,165.]
    blue = [ 0.,211.,128.,255., 55.,221., 0., 0.]
    return make_colourmap(ind, red, green, blue, 'califa_vel')

def get_califa_intensity_cmap_2():
    ind = [ 0., 1., 50.,100.,150.,200.,255.]
    red = [ 0., 0., 0.,255.,255., 55.,221.]
    green =	[ 0., 0.,191., 0.,165., 55.,160.]
    blue = [ 0.,128.,255., 0., 0., 55.,221.]
    return make_colourmap(ind, red, green, blue, 'califa_int')

def get_califa_intensity_cmap():
    ind = [ 1., 50.,100.,150.,200.,255.]
    red = [ 0., 0.,255.,255., 55.,221.]
    green =	[ 0.,191., 0.,165., 55.,160.]
    blue = [ 128.,255., 0., 0., 55.,221.]
    return make_colourmap(ind, red, green, blue, 'califa_int')

def get_califa_velocity_cmap_r():
    ind = [0., 1., 35., 90.,125.,160.,220.,255.]
    blue = [ 0.,148., 0., 0., 55.,221.,255.,255.]
    green = [ 0., 0., 0.,191., 55.,160., 0.,165.]
    red = [ 0.,211.,128.,255., 55.,221., 0., 0.]
    return make_colourmap(ind, red[::-1], green[::-1], blue[::-1], 'califa_vel_r')
#    red = [148., 0., 0., 55.,221.,255.,255.]
#    green = [ 0., 0.,191., 55.,160., 0.,165.]
#    blue = [211.,128.,255., 55.,221., 0., 0.]


califa_vel = get_califa_velocity_cmap()
califa_vel_r = get_califa_velocity_cmap_r()
califa_int = get_califa_intensity_cmap()

#califa_int=cm.Spectral
#califa_vel=cm.Spectral

def fit_leastsq_pure(p0, datax, datay, function):

    errfunc = lambda p, x, y: function(x,p) - y

    pfit, pcov, infodict, errmsg, success = \
        optimize.leastsq(errfunc, p0, args=(datax, datay), \
                          full_output=1, epsfcn=0.0001)

    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    return pfit_leastsq, perr_leastsq 

def fit_leastsq(p0, datax, datay, function):

    errfunc = lambda p, x, y: function(x,p) - y

    pfit, pcov, infodict, errmsg, success = \
        optimize.leastsq(errfunc, p0, args=(datax, datay), \
                          full_output=1, epsfcn=0.01)
# epsfcn=0.0001)


    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    return pfit_leastsq, pcov 

#
# Binning!
#

def binning_OH(M_OK, OH_Ref_OK, bin1 , min1 , max1 ):
    
    M_bin=[]
    OH_bin=[]
    D_OH_bin=[]
    
    OH_binM    = np.arange(min1,max1,bin1) 
    OH_binM = OH_binM-bin1*0.5
    m_range    = np.zeros(OH_binM.size)
    OH_binD    = np.zeros(OH_binM.size)

    for i, val  in enumerate(OH_binM):
        tmp = (OH_Ref_OK >= val) & (OH_Ref_OK <= val+bin1)
        m_sub=M_OK[tmp]
        n_vals=m_sub.size
        m_range[i]   = np.median(M_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])
        if (n_vals > 10):
            M_bin.append(m_range[i])
            OH_bin.append(OH_binM[i])
            D_OH_bin.append(OH_binD[i])
    m_range=np.array(M_bin)
    OH_binM=np.array(OH_bin)
    OH_binD=np.array(D_OH_bin)
        
    return(m_range, OH_binM, OH_binD)

def binning(M_OK, OH_Ref_OK, bin1 , min1 , max1 ):
    
    
    m_range = np.arange(min1,max1,bin1)
    M_binM    = np.zeros(m_range.size)
    M_binV    = np.zeros(m_range.size)
    OH_binM    = np.zeros(m_range.size)
    OH_binD    = np.zeros(m_range.size)
    n_vals    = np.zeros(m_range.size)

    for i, val  in enumerate(m_range):
        tmp = (M_OK >= val) & (M_OK <= val+bin1)
        OH_binM[i]   = np.median(OH_Ref_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])+0.02
#        tmp = (OH_Ref_OK >= OH_binM[i]-0.125*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+0.125*OH_binD[i]) & (M_OK >= val-4*bin1) & (M_OK <= val+5*bin1)       
        tmp = (OH_Ref_OK >= OH_binM[i]-0.1*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+0.1*OH_binD[i]) & (M_OK >= val-3*bin1) & (M_OK <= val+3*bin1)       
        m_sub=M_OK[tmp]
        n_vals[i]=m_sub.size
#        print('n_val',n_vals,', vals = ',M_OK[tmp])
#        if (n_vals > 2):
        M_binM[i]   = np.median(M_OK[tmp])
        M_binV[i] = val+0.5*bin1        
        if ((np.isnan(M_binM[i])) or (np.isinf(M_binM[i]))):
            M_binM[i]=M_binV[i]
    M_bin_out=0.5*(M_binM+M_binV)
    #print '',M_binM,M_binV,M_bin_out
    mask_val= n_vals>5
    
    return(M_bin_out[mask_val], OH_binM[mask_val], OH_binD[mask_val])

def binning_M(M_OK, OH_Ref_OK, bin1 , min1 , max1 , Nmax, delta_y=0.1, delta_x=3.0):
    
    
    m_range = np.arange(min1,max1,bin1)
    M_binM    = np.zeros(m_range.size)
    M_binV    = np.zeros(m_range.size)
    OH_binM    = np.zeros(m_range.size)
    OH_binD    = np.zeros(m_range.size)
    n_vals    = np.zeros(m_range.size)

    for i, val  in enumerate(m_range):
        tmp = (M_OK >= val) & (M_OK <= val+bin1)
        OH_binM[i]   = np.median(OH_Ref_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])+0.02
        #print('Y_vals =',OH_binM[i],OH_binD[i],val,val+bin1)
        #print('vector =',OH_Ref_OK[tmp])
        tmp = (OH_Ref_OK >= OH_binM[i]-delta_y*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+delta_y*OH_binD[i]) & (M_OK >= val-delta_x*bin1) & (M_OK <= val+delta_x*bin1)       
        m_sub=M_OK[tmp]
        n_vals[i]=m_sub.size
#        print('n_val',n_vals,', vals = ',M_OK[tmp])
#        if (n_vals > 2):
        M_binM[i]   = np.median(M_OK[tmp])
        M_binV[i] = val+0.5*bin1        
        if ((np.isnan(M_binM[i])) or (np.isinf(M_binM[i]))):
            M_binM[i]=M_binV[i]
    M_bin_out=0.5*(M_binM+M_binV)
    #print '',M_binM,M_binV,M_bin_out
    #print('# ',n_vals,Nmax)
    mask_val= n_vals>Nmax
    
    return(M_bin_out[mask_val], OH_binM[mask_val], OH_binD[mask_val])


def binning_M2(M_OK, OH_Ref_OK, bin1 , min1 , max1 , Nmax, delta_y=0.1, delta_x=3.0):
    
    
    m_range = np.arange(min1,max1,bin1)
    M_binM    = np.zeros(m_range.size)
    M_binV    = np.zeros(m_range.size)
    OH_binM    = np.zeros(m_range.size)
    OH_binD    = np.zeros(m_range.size)
    n_vals    = np.zeros(m_range.size)

    for i, val  in enumerate(m_range):
        tmp = (M_OK >= val) & (M_OK <= val+bin1)
        OH_binM[i]   = np.median(OH_Ref_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])+0.02
        #print('Y_vals =',OH_binM[i],OH_binD[i],val,val+bin1)
        #print('vector =',OH_Ref_OK[tmp])
        tmp = (OH_Ref_OK >= OH_binM[i]-delta_y*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+delta_y*OH_binD[i]) & (M_OK >= val-delta_x*bin1) & (M_OK <= val+delta_x*bin1)       
        m_sub=M_OK[tmp]
        n_vals[i]=m_sub.size
        
        print('n_val',i,', vals = ',M_OK[tmp], OH_Ref_OK[tmp])
#        if (n_vals > 2):
        M_binM[i]   = np.median(M_OK[tmp])
        M_binV[i] = val+0.5*bin1        
        if ((np.isnan(M_binM[i])) or (np.isinf(M_binM[i]))):
            M_binM[i]=M_binV[i]
    M_bin_out=0.5*(M_binM+M_binV)
    #print '',M_binM,M_binV,M_bin_out
    #print('# ',n_vals,Nmax)
    mask_val= n_vals>Nmax
    
    return(M_bin_out[mask_val], OH_binM[mask_val], OH_binD[mask_val])



def binning2(M_OK, OH_Ref_OK, bin1 , min1 , max1 ):
    
    
    m_range = np.arange(min1,max1,bin1)
    M_binM    = np.zeros(m_range.size)
    M_binV    = np.zeros(m_range.size)
    OH_binM    = np.zeros(m_range.size)
    OH_binD    = np.zeros(m_range.size)

    for i, val  in enumerate(m_range):
        tmp = (M_OK >= val) & (M_OK <= val+bin1)
        OH_binM[i]   = np.median(OH_Ref_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])+0.02
#        tmp = (OH_Ref_OK >= OH_binM[i]-0.125*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+0.125*OH_binD[i]) & (M_OK >= val-4*bin1) & (M_OK <= val+5*bin1)       
        tmp = (OH_Ref_OK >= OH_binM[i]-0.1*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+0.1*OH_binD[i]) & (M_OK >= val-3*bin1) & (M_OK <= val+3*bin1)       

        M_binM[i]   = np.median(M_OK[tmp])
        M_binV[i] = val+0.5*bin1        
        if ((np.isnan(M_binM[i])) or (np.isinf(M_binM[i]))):
            M_binM[i]=M_binV[i]
    M_bin_out=0.5*(M_binM+M_binV)
    #print '',M_binM,M_binV,M_bin_out
    mask = np.logical_not(np.isnan(M_bin_out)) & np.logical_not(np.isnan(OH_binM)) & np.logical_not(np.isnan(OH_binD))
    M_bin_out = M_bin_out[mask]
    OH_binM = OH_binM[mask]
    OH_binD = OH_binD[mask]
    return(M_bin_out, OH_binM, OH_binD)

def binning_old(M_OK, OH_Ref_OK, bin1 , min1 , max1 ):
    
    
    m_range = np.arange(min1,max1,bin1)
    OH_binM    = np.zeros(m_range.size)
    OH_binD    = np.zeros(m_range.size)

    for i, val  in enumerate(m_range):
        tmp = (M_OK >= val) & (M_OK <= val+bin1)
        OH_binM[i]   = np.median(OH_Ref_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])
            
    return(m_range, OH_binM, OH_binD)

def make_cont(x ,y, min2s,max2s,min1s,max1s,bin1s,bin2s, frac):

    m1s      = math.floor((max1s-min1s)/bin1s) + 1
    m2s      = math.floor((max2s-min2s)/bin2s) + 1
    
    vals, xedges, yedges = np.histogram2d(x, y, bins=[m1s,m2s])
    
    xbins = 0.5 * (xedges[:-1] + xedges[1:])
    ybins = 0.5 * (yedges[:-1] + yedges[1:])
    
    L = (1-frac)*(np.max(vals) - np.min(vals))+ np.min(vals)
    return(xbins, ybins, vals.T, L)





#
# Pandas reading columns
#
def header_columns_old_pd(filename,column):
    COMMENT_CHAR = '#'
    col_NAME = []
    with open(filename, 'r') as td:
        for line in td:
            if line[0] == COMMENT_CHAR:
                info = re.split(' +', line.rstrip('\n'))
                col_NAME.append(info[column])
    return col_NAME

def header_columns_formatted(filename,column):
    COMMENT_CHAR = '#'
    col_NAME = []
    with open(filename, 'r') as td:
        for line in td:
            if (line[0] == COMMENT_CHAR) and (line.find("COLUMN")>-1):
                start_info = re.split(',+', line.rstrip('\n'))
                info = re.split(' +', start_info[0])
                col_NAME.append(info[column])
    counts = {k:v for k,v in Counter(col_NAME).items() if v > 1}
    col_NAME_NEW = col_NAME[:]
    for i in reversed(range(len(col_NAME))):
        item = col_NAME[i]
        if item in counts and counts[item]:
            if (counts[item]>1):
                col_NAME_NEW[i] += str(counts[item]-1)
            counts[item]-=1                
    return col_NAME_NEW

def header_columns(filename,column):
    COMMENT_CHAR = '#'
    col_NAME = []
    with open(filename, 'r') as td:
        for line in td:
            if line[0] == COMMENT_CHAR:
                info = re.split(' +', line.rstrip('\n'))
                col_NAME.append(info[column])
    counts = {k:v for k,v in Counter(col_NAME).items() if v > 1}
    col_NAME_NEW = col_NAME[:]
    for i in reversed(range(len(col_NAME))):
        item = col_NAME[i]
        if item in counts and counts[item]:
            if (counts[item]>1):
                col_NAME_NEW[i] += str(counts[item]-1)
            counts[item]-=1                
    return col_NAME_NEW

def header_columns_header(filename,column):
    COMMENT_CHAR = '#'
    col_NAME = []
    with open(filename, 'r') as td:
        for line in td:
            if ((line[0] == COMMENT_CHAR) and ('HEADER' not in line)):
                info = re.split(' +', line.rstrip('\n'))
                col_NAME.append(info[column])
    counts = {k:v for k,v in Counter(col_NAME).items() if v > 1}
    col_NAME_NEW = col_NAME[:]
    for i in reversed(range(len(col_NAME))):
        item = col_NAME[i]
        if item in counts and counts[item]:
            if (counts[item]>1):
                col_NAME_NEW[i] += str(counts[item]-1)
            counts[item]-=1                
    return col_NAME_NEW


def header_columns_space(filename,column):
    COMMENT_CHAR = '#'
    col_NAME = []
    with open(filename, 'r') as td:
        for line in td:
            if ((line[0] == COMMENT_CHAR) and (line.find('HEADER')==-1) and (line.find(')')>-1)):
                line.replace("\)","")
                line.replace("\(","")
                line.replace("\/","")
                info = re.split(' +', line.rstrip('\n'))
                info_now=info[column]
                for i in np.arange(column+1,len(info)):
                    if ((info[i]) and not (")" in info[i])):
                        info_now=info_now+'_'+info[i]
                col_NAME.append(info_now)
    counts = {k:v for k,v in Counter(col_NAME).items() if v > 1}
    col_NAME_NEW = col_NAME[:]
    for i in reversed(range(len(col_NAME))):
        item = col_NAME[i]
        if item in counts and counts[item]:
            if (counts[item]>1):
                col_NAME_NEW[i] += str(counts[item]-1)
            counts[item]-=1                
    return col_NAME_NEW

####################################################################
# Plotting routines                                                #
####################################################################

def get_den_map(x, y, z, bins=60, xLims=None, yLims=None, normValue=None, statistic='median',sigma=0.7):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    
    if (xLims==None):
        xLims=[np.nanmin(x),np.nanmax(x)]
    if (yLims==None):
        yLims=[np.nanmin(y),np.nanmax(y)]

    counts, xbins, ybins = np.histogram2d(x, y, bins=bins,
                                              range=[xLims, yLims],
                                          normed=True)
    counts = gaussian_filter(counts, sigma)
    counts /= counts.max()
    mask_d = counts.transpose() == 0
    bin_means = binned_statistic_2d(x, y, z, bins=bins,
                                    range=[xLims, yLims],
                                    statistic=statistic).statistic
#    bin_means= gaussian_filter(bin_means, 0.5)
    if normValue is not None:
        bin_means /= normValue
        bin_means = np.abs(bin_means)
    dens_map = bin_means.T
    dens_map[mask_d] = np.nan
    output={}
    output['den']=counts
    output['xbins']=xbins
    output['ybins']=ybins
    output['val']=dens_map.T
    output['xLims']=xLims
    output['yLims']=yLims
    return output



def my_contour(ax,x_cont,y_cont,x_min,x_max,y_min,y_max,c_color='red', title='', nbins=30, zorder=1, linewidths=2,alpha=0.75, conts=[0.95,0.65,0.40],label=''):
    N_min=2
    Delta=0.3
#    nbins=30
    mask_cont = (x_cont>x_min) & (x_cont<x_max) & (y_cont>y_min) & (y_cont<y_max)  
    x_plt, y_plt = x_cont[mask_cont], y_cont[mask_cont]
    counts, xbins, ybins = np.histogram2d(x_plt, y_plt, bins=nbins,
        normed=True,range=[[x_min,x_max],[y_min,y_max]])
    counts=ndimage.gaussian_filter(counts, sigma=1, order=0)
    counts /= counts.max()
    sum_total=counts.sum()
    vals=[]
    levels=[]
    for idx,cuts in enumerate(np.arange(0.00,1.0,0.01)):
        mask_now= counts>cuts
        levels.append(cuts)
        vals.append(counts[mask_now].sum()/sum_total)
#    vals_cont=np.array([0.95,0.65,0.40])
    vals_cont=np.array(conts)
    levels_cont=np.interp(vals_cont,np.array(levels),np.array(vals))
    counts_rot=np.rot90(counts,3)
    xbins=xbins+0.5*(x_max-x_min)/nbins
    ybins=ybins+0.5*(y_max-y_min)/nbins
    flip_counts_rot=np.fliplr(counts_rot)
    p_cont=ax.contour(xbins[0:nbins],ybins[0:nbins],flip_counts_rot,levels_cont,\
                      colors=c_color,alpha=alpha,linewidths=linewidths,zorder=zorder)
#    labels = ['Div Neg', 'Div Pos', 'Rot Neg', 'Rot Pos']
    if (len(title)>0):
        p_cont.collections[0].set_label(title)
#    print(f'label={label},{len(label)}')
    if (len(label)>0):
        ax.plot([x_min,x_min],[y_min,y_min],linewidth=linewidths,color=c_color,alpha=alpha,label=label)
#        h_legend,l_legend = p_cont.legend_elements()
    #    h_cont,l_cont = p_cont.legend_elements(title)
#    ax.legend(h_cont, l_cont)


def my_contourf(ax,x_cont,y_cont,x_min,x_max,y_min,y_max,c_color='red', title='', nbins=30, zorder=1, linewidths=2,alpha=0.75, conts=[0.95,0.65,0.40],cmap='jet',colors=None):
    N_min=2
    Delta=0.3
#    nbins=30
    mask_cont = (x_cont>x_min) & (x_cont<x_max) & (y_cont>y_min) & (y_cont<y_max)  
    x_plt, y_plt = x_cont[mask_cont], y_cont[mask_cont]
    counts, xbins, ybins = np.histogram2d(x_plt, y_plt, bins=nbins,
        normed=True,range=[[x_min,x_max],[y_min,y_max]])
    counts=ndimage.gaussian_filter(counts, sigma=1, order=0)
    counts /= counts.max()
    sum_total=counts.sum()
    vals=[]
    levels=[]
    for idx,cuts in enumerate(np.arange(0.00,1.0,0.01)):
        mask_now= counts>cuts
        levels.append(cuts)
        vals.append(counts[mask_now].sum()/sum_total)
#    vals_cont=np.array([0.95,0.65,0.40])
    vals_cont=np.array(conts)
    levels_cont=np.interp(vals_cont,np.array(levels),np.array(vals))
    counts_rot=np.rot90(counts,3)
    xbins=xbins+0.5*(x_max-x_min)/nbins
    ybins=ybins+0.5*(y_max-y_min)/nbins
    flip_counts_rot=np.fliplr(counts_rot)
    flip_counts_rot /=  flip_counts_rot.max()
    for indx in arange(len(levels_cont)-1):
        if (levels_cont[indx]==levels_cont[indx+1]):
            levels_cont[indx]=levels_cont[indx]/(1+0.1*indx)
            print(indx,levels_cont[indx])
    if (colors is None):
        p_contf=ax.contourf(xbins[0:nbins],ybins[0:nbins],flip_counts_rot,levels_cont,\
                            alpha=alpha,linewidths=linewidths,zorder=zorder, cmap=cmap)
        p_cont=ax.contour(xbins[0:nbins],ybins[0:nbins],flip_counts_rot,levels_cont,\
                          colors=c_color,alpha=alpha,linewidths=linewidths,zorder=zorder)  
    else:
        p_cont=ax.contourf(xbins[0:nbins],ybins[0:nbins],flip_counts_rot,levels_cont,\
                            alpha=alpha,linewidths=linewidths,zorder=zorder, cmap=cmap)


    
#    labels = ['Div Neg', 'Div Pos', 'Rot Neg', 'Rot Pos']
    if (len(title)>0):
        p_cont.collections[0].set_label(title)

def biweight_midvariance(par):
    val=np.nanstd(par)/np.sqrt(2)
    return val

def my_scatter(ax,x_par,y_par,c_par,x_cont,y_cont,x_min,x_max,y_min,y_max,c_min,c_max,x_label,y_label,xf_min=0.0,yf_min=0.0,den_par_min=0.85,bin_size=0.3,bin_number=2,c_color="black",c_color2="darksalmon",error=0.05,MC=10, Delta=0.15):
    cm = califa_vel_r
    if (xf_min==0.0):
        xf_min=x_min
    if (yf_min==0.0):
        yf_min=y_min
#    MC=5
#    error=0.05
    N_min=2
#    Delta=0.15
#    den_par_min=0.9
    nbins=30
    lEW_cut=0.78 #0.78
    n_obj_org=len(x_par)
#    print("# N.OBJ = ",n_obj_org)
#    mask = (x_par>x_min) & (x_par<x_max) & (y_par>y_min) & (y_par<y_max)  
    mask = x_par>-20
    n_obj=len(x_par[mask])
#    print(x_par[~mask])
#    print("# N.Obj = ",n_obj,x_min,x_max,y_min,y_max)
    mask_SFGs = mask & (c_par>lEW_cut)
    mask_cont = (x_cont>x_min) & (x_cont<x_max) & (y_cont>y_min) & (y_cont<y_max)  
#
# Density to plot!
#
    x_plt, y_plt = x_cont[mask_cont], y_cont[mask_cont]
#    nbins=40
    counts, xbins, ybins = np.histogram2d(x_plt, y_plt, bins=nbins,
        normed=True,range=[[x_min,x_max],[y_min,y_max]])
#        range=[[np.nanmin(x_plt),np.nanmax(x_plt)],[np.nanmin(y_plt),np.nanmax(y_plt)]])
                                            #    counts /= counts.max()

    counts=ndimage.gaussian_filter(counts, sigma=1, order=0)
    counts /= counts.max()
    sum_total=counts.sum()
    vals=[]
    levels=[]
    for idx,cuts in enumerate(np.arange(0.00,1.0,0.01)):
        mask_now= counts>cuts
        levels.append(cuts)
        vals.append(counts[mask_now].sum()/sum_total)
        #print(idx,levels[idx],vals[idx])
    vals_cont=np.array([0.95,0.80,0.40])
    levels_cont=np.interp(vals_cont,np.array(levels),np.array(vals))
    figure=ax.scatter(x_par, y_par, c=c_par, vmin=c_min,vmax=c_max,alpha=0.4,edgecolor='none',\
                      rasterized=True,cmap=cm)
    counts_rot=np.rot90(counts,3)
    xbins=xbins+0.5*(x_max-x_min)/nbins
    ybins=ybins+0.5*(y_max-y_min)/nbins
    flip_counts_rot=np.fliplr(counts_rot)
    p_cont=ax.contour(xbins[0:nbins],ybins[0:nbins],flip_counts_rot,levels_cont,colors=c_color)

    #
    # Density near a point
    #
    x_plt, y_plt = x_par[mask_SFGs], y_par[mask_SFGs]
    n_sf=len(x_par[mask_SFGs])
    #    nbins=40
    counts, xbins, ybins = np.histogram2d(x_plt, y_plt, bins=nbins,
        normed=True,
        range=[[np.nanmin(x_plt),np.nanmax(x_plt)],[np.nanmin(y_plt),np.nanmax(y_plt)]])
                                            #    counts /= counts.max()
#    print(xbins,ybins)
    counts=ndimage.gaussian_filter(counts, sigma=1, order=0)
    counts /= counts.max()
    sum_total=counts.sum()
    vals_new=[]
    levels_new=[]
    for idx,cuts in enumerate(np.arange(0.00,1.0,0.01)):
        mask_now= counts>cuts
        levels_new.append(cuts)
        vals_new.append(counts[mask_now].sum()/sum_total)
        #print(idx,levels[idx],vals[idx])
    vals_cont=np.array([0.95,0.80,0.40])
    levels_cont=np.interp(vals_cont,np.array(levels_new),np.array(vals))
    
#    figure=ax.scatter(x_par, y_par, c=c_par, vmin=c_min,vmax=c_max,alpha=0.4,edgecolor='none',cmap=cm)
    counts_rot=np.rot90(counts,3)
    xbins=xbins+0.5*(x_max-x_min)/nbins
    ybins=ybins+0.5*(y_max-y_min)/nbins
    flip_counts_rot=np.fliplr(counts_rot)
    
  
    den_par=np.zeros(len(x_par))
    for i in range(len(x_par)):
        if ((np.isfinite(x_par[i])) and (np.isfinite(y_par[i]))):
            i_x=np.argmin(np.abs(xbins-x_par[i]))
            i_y=np.argmin(np.abs(ybins-y_par[i]))
            if ((i_x>0) and (i_x<nbins) and (i_y>0) and (i_y<nbins)):
                den_par[i]=np.interp(counts[i_x,i_y],np.array(levels_new),np.array(vals_new))

# Density plot!
#    figure=ax.scatter(x_par, y_par, c=den_par, vmin=0,vmax=1,alpha=0.4,edgecolor='none',cmap=cm)
    p_cont2=ax.contour(xbins[0:nbins],ybins[0:nbins],flip_counts_rot,levels_cont,colors=c_color2)
#
# We mask low density points
#
#    print("# den_par_min=",den_par_min)
    mask_SFGs = mask_SFGs & (den_par<den_par_min) & \
    np.logical_not(np.isnan(x_par)) & np.isfinite(x_par) & \
    np.logical_not(np.isnan(y_par)) & np.isfinite(y_par)  

    x_sf=x_par[mask_SFGs]
    y_sf=y_par[mask_SFGs]
    
#    figure=ax.scatter(x_par, y_par, c=den_par, vmin=0,vmax=1,alpha=0.4,edgecolor='none',cmap=cm)
    
    bin1 , min1 , max1= Delta,x_min,x_max
    m1, sM1, sD1 = binning_M(x_sf, y_sf, bin1 , min1 , max1, N_min, bin_size, bin_number )
    p1 = ax.errorbar(m1, sM1, yerr= sD1, markersize = 9, markerfacecolor =c_color2, markeredgecolor = 'black', fmt = 'o', ecolor='black', elinewidth = 1, label = '', zorder=2, alpha=0.7)


#    x_sf=x_par[mask_SFGs]
#    y_sf=y_par[mask_SFGs]
    bin1 , min1 , max1= Delta,xf_min,x_max
    funct    = lambda x,a,b: a + b * x
    START=[-1,1.0]
    pa1=np.zeros(2)
    ea1=np.zeros(2)
#    print('#vals to fit =',m1,sM1,sD1)
    if m1.size > 2:
        m1, sM1, sD1 = binning_M(x_sf, y_sf, bin1 , min1 , max1, N_min, 0.3,2 )
        mean_sD1=np.nanmean(sD1)
        sD1=sD1+mean_sD1
        np.clip(sD1,0.5*mean_sD1,1.5*mean_sD1)
        pa1, ea1  = optimize.curve_fit(funct, m1, sM1, sigma=np.sqrt(sD1), p0 = START )
        a_pa=np.zeros((MC,2))
        a_ea=np.zeros((MC,2,2))
        a_rc=np.zeros((MC,2,2))
        for iMC in range(MC):
            e_x_sf=np.abs(2*error-error*(x_sf-x_min)/(x_max-x_min))
            e_y_sf=np.abs(2*error-error*(y_sf-y_min)/(y_max-y_min))
            
#            print('e_y=',e_y_sf,len(e_y_sf))
#            print('e-x=',e_x_sf,len(e_x_sf))
            x_sf_now=x_sf+np.random.normal(loc=0.0,scale=e_x_sf,size=len(e_x_sf))
            y_sf_now=y_sf+np.random.normal(loc=0.0,scale=e_y_sf,size=len(e_y_sf))
            mNOW, sMNOW, sDNOW = binning_M(x_sf_now, y_sf_now, bin1 , min1 , max1, N_min, 0.3,2 )
            rcNOW=np.corrcoef(x_sf_now, y_sf_now)
            #for i in range(len(sDNOW)):
            #    i_x=np.argmin(np.abs(xbins-mNOW[i]))
            #    i_y=np.argmin(np.abs(ybins-sMNOW[i]))
            #    if ((i_x>0) and (i_x<nbins) and (i_y>0) and (i_y<nbins)):
            #        sDNOW[i]=0.5*sDNOW[i]+\
            #        0.1*np.interp(counts[i_x,i_y],np.array(levels_new),np.array(vals_new))
#                    if (iMC==0):
#                        print(mNOW[i],sDNOW[i])
            sDNOW=0.001/sDNOW
    
#            np.clip(sDNOW,0.05,0.2)
            paNOW, eaNOW  = optimize.curve_fit(funct, mNOW, sMNOW, sigma=np.sqrt(sDNOW), p0 = START )
#            print('Coeffs lin fit: ',round(paNOW[0],3),round(np.sqrt(np.diag(eaNOW))[0],3),round(paNOW[1],3),round(np.sqrt(np.diag(eaNOW))[1],3),round(rcNOW[0,1],3))    
            a_pa[iMC,:]=paNOW
            a_ea[iMC,:,:]=eaNOW
            a_rc[iMC,:,:]=rcNOW
#        print(pa1,ea1)
#        test=np.mean(a_pa,axis=(0,1))
#        print(test)
        np.mean(a_pa, axis=(0), out=pa1)
        np.mean(a_ea, axis=(0), out=ea1)
        e_pa1=np.std(a_pa, axis=(0))
        rc=np.mean(a_rc, axis=(0))
        ea1[0,0]=ea1[0,0]+e_pa1[0]
        ea1[1,1]=ea1[1,1]+e_pa1[1]
        
        
        #        pa1=a_pa.mean(axis=(1,2))
#        ea1=a_ea.mean(axis=(1,2))
#        ea1=np.mean(a_ea,axis=2)
        
#        pa1, ea1  = optimize.curve_fit(funct, m1, sM1, sigma=sD1, p0 = START )
#round(pa1[0],3),round(np.sqrt(np.diag(ea1))[0],3),round(pa1[1],3),round(np.sqrt(np.diag(ea1))[1],3),round(rc[0,1],3)

        if (xf_min!=x_min):
            yl_plot=np.linspace(y_min,y_max,10)
            xl_plot=xf_min+0.0*yl_plot
            rect=patches.Rectangle((x_min, y_min), xf_min-x_min, y_max-y_min, facecolor="black", alpha=0.2,zorder=3)
            ax.add_patch(rect)
        x_plot=np.linspace(np.min(m1)-1.5*bin1, np.max(m1)+1.5*bin1, 10)
        y_plot=funct(x_plot,pa1[0],pa1[1])
        p_fit=ax.plot(x_plot, y_plot, '-.', linewidth = 3, markerfacecolor ='black'  , color ='black', zorder=3,label='This work')
#        p_fit=ax.plot(xl_plot, yl_plot, '-', linewidth = 2, markerfacecolor ='grey'  , color ='grey', zorder=2)        
#        print("PASO")

    mask_SFGs = mask_SFGs & (den_par<0.85) & \
    np.logical_not(np.isnan(x_par)) & np.isfinite(x_par) & \
    np.logical_not(np.isnan(y_par)) & np.isfinite(y_par)  

    x_sf=x_par[mask_SFGs]
    y_sf=y_par[mask_SFGs]

    mask_x_sf=x_sf>xf_min
    x_sf=x_sf[mask_x_sf]
    y_sf=y_sf[mask_x_sf]
    rc=np.corrcoef(x_sf, y_sf)
    delta_y_par=y_sf-(pa1[0]+pa1[1]*x_sf)
    s_y_par = biweight_midvariance(y_sf[np.isfinite(y_sf)])
    s_dy_par = biweight_midvariance(delta_y_par[np.isfinite(delta_y_par)])

#    s_y_par = np.std(y_sf[np.isfinite(y_sf)])
#    s_dy_par = np.std(delta_y_par[np.isfinite(delta_y_par)])

    print('Mean Coeff.: ',round(pa1[0],3),round(np.sqrt(np.diag(ea1))[0],3),round(pa1[1],3),round(np.sqrt(np.diag(ea1))[1],3),round(rc[0,1],3),round(s_y_par,3),round(s_dy_par,3),n_obj,n_sf)
    #    print('Stddev: ',round(s_y_par,3),round(s_dy_par,3))
        
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.minorticks_on()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both',direction="in")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return pa1,ea1,s_y_par,s_dy_par,rc

def my_hist(ax,x_par,y_par,x_min,x_max,c_x,c_y,x_label,y_label):
    mask_x = (x_par > x_min) & (x_par < x_max) 
    mask_y = (y_par > x_min) & (y_par < x_max) 
    p1=sns.distplot(x_par[mask_x],bins=40,ax=ax,hist=False,color=c_x)
    p2=sns.distplot(y_par[mask_y],bins=40,ax=ax,hist=False,color=c_y)
    ax.set_xlim([x_min,x_max])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return p1,p2



from cmaps_CLC import vel_map

def plot_2Dhist(x_par=None,y_par=None,z_par=None,\
                x_min=-25,x_max=-16,y_min=0,y_max=4.5,z_min=-2,z_max=10,c_main='grey',label='MPL-11',\
                x_label='NSA z-band abs mag',y_label='u-z mag',figname='CMD_diag',alpha=0.75,\
                   size_scatter=5.0,n_zbins=11,labels_zbins=None,color_cm_now='coolwarm_r',z_lim=80,\
               x_loc=0.02,y_loc=0.03,z_label='',reverse=0, xscale='linear', yscale='linear'):
    mask = np.isfinite(x_par) & np.isfinite(y_par) & np.isfinite(z_par)
    x_par = x_par[mask]
    y_par = y_par[mask]
    z_par = z_par[mask]
    fig = plt.figure(figsize=(9,7))
    left, width = 0.1, 0.75
    bottom, height = 0.1, 0.75
    spacing = 0.00
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 1-(bottom+height)]
    rect_histy = [left + width + spacing, bottom, 1-(left+width), height]
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    if (color_cm_now is None):
        color_cm_now = vel_map()
        if (reverse==1):
            color_cm_now=color_cm_now.reversed()
    colormap=cm.get_cmap(color_cm_now)


#    colormap = plt.cm(color_cm_now)
    ax.scatter(x_par, y_par, s=size_scatter*3, c=z_par,\
               vmin=z_min,vmax=z_max,alpha=alpha,edgecolor='none',cmap=color_cm_now,\
               rasterized=True,label=label)
    my_contour(ax,x_par,y_par,x_min,x_max,y_min,y_max,c_color=c_main,\
               nbins=50,title=label,linewidths=3,alpha=alpha)
#
# X-histogram 
#
    
    
    xx = np.linspace(x_min, x_max, 1000)
    kde_xx=stats.gaussian_kde(x_par)
    kde_xx.set_bandwidth(bw_method=kde_xx.factor / 5.)
    max_kde_xx=np.max(kde_xx(xx))
    
    delta_z=(z_max-z_min)/n_zbins
    Ncolors = min(colormap.N,n_zbins)
    mapcolors = [colormap(int(x*colormap.N/Ncolors)) for x in range(Ncolors)]
    if (labels_zbins==None):
        labels_zbins=np.zeros(n_zbins)
    for indx,z_bins in enumerate(np.linspace(z_min,z_max,n_zbins)):
        z_bin_min = z_bins-0.5*delta_z
        z_bin_max = z_bins+0.5*delta_z
        if (labels_zbins[indx]==0):
            labels_zbins[indx]=z_bins
        xx = np.linspace(x_min, x_max, 1000)
        mask_z_par = (z_par>z_bin_min) & (z_par<=z_bin_max)
        if (len(x_par[mask_z_par])>0):
            kde=stats.gaussian_kde(x_par[mask_z_par])
            kde.set_bandwidth(bw_method=kde.factor / 1.)
            if(len(x_par[mask_z_par])>z_lim):
                ax_histx.plot(xx,max_kde_xx*(kde(xx)/np.max(kde(xx))),\
                              color=mapcolors[indx],linewidth=3,alpha=alpha,\
                              label=labels_zbins[indx])
    ax_histx.plot(xx,kde_xx(xx),color=c_main,linewidth=3,alpha=alpha)
    ax_histx.set_ylim(0,1.1*np.max(kde_xx(xx)))    
#
# Y-histogram 
#

    yy = np.linspace(y_min, y_max, 1000)
    kde_yy=stats.gaussian_kde(y_par)
    kde_yy.set_bandwidth(bw_method=kde_yy.factor / 2.)
    max_kde_yy=np.max(kde_yy(yy))
    for indx,z_bins in enumerate(np.linspace(z_min,z_max,n_zbins)):
        z_bin_min = z_bins-0.5*delta_z
        z_bin_max = z_bins+0.5*delta_z
        if (labels_zbins[indx]==0):
            labels_zbins[indx]=z_bins
        mask_z_par = (z_par>z_bin_min) & (z_par<=z_bin_max)
        #print(z_bins,len(x_par[mask_z_par]))
        if (len(y_par[mask_z_par])>0):
            kde=stats.gaussian_kde(y_par[mask_z_par])
            kde.set_bandwidth(bw_method=kde.factor / 1.)
            if(len(y_par[mask_z_par])>z_lim):
                ax_histy.plot(max_kde_yy*(kde(yy)/np.max(kde(yy))),yy,\
                              color=mapcolors[indx],linewidth=3,alpha=alpha,\
                              label=labels_zbins[indx])
    
    
    ax_histy.plot(kde_yy(yy),yy,color=c_main,linewidth=3,alpha=alpha)
    ax_histy.set_xlim(0,1.1*np.max(kde_yy(yy)))

    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['bottom'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.get_xaxis().set_visible(False)
    ax_histx.get_yaxis().set_visible(False)

    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    ax_histy.spines['left'].set_visible(False)
    ax_histy.get_xaxis().set_visible(False)
    ax_histy.get_yaxis().set_visible(False)

    ax.set_xlabel(x_label, fontsize=23)
    ax.set_ylabel(y_label, fontsize=23)

    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)

    handles, labels = ax_histx.get_legend_handles_labels()
    ax.legend(handles, labels,loc=(x_loc,y_loc),frameon=True,\
              handlelength=1.5,ncol=3,columnspacing=0.15,title=z_label)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    fig.tight_layout()
    fig.savefig(figname+".pdf", transparent=False, facecolor='white', edgecolor='white')#.pdf")



####################################################################
# Pipe3D DR17 MaNGA utils!                                         #
####################################################################

#def read_Pipe3D_MaNGA_table(Pipe3D_file="SDSS17Pipe3D_v3_1_1.fits",DIR='')
#Pipe3D_tab_hdu=fits.open(DIR+Pipe3D_file)
#SSP_lib_data=Pipe3D_tab_hdu[2].data
#SSP_lib_hdr=Pipe3D_tab_hdu[2].header
#Pipe3D_tab = Table.read(DIR+Pipe3D_file, hdu=1)
#print(SSP_lib_hdr)

def read_Pipe3D_MaNGA_table(Pipe3D_file="SDSS17Pipe3D_v3_1_1.fits",DIR='',verbose=0):
    Pipe3D_tab_hdu=fits.open(DIR+Pipe3D_file)
    SSP_lib_data=Pipe3D_tab_hdu[2].data
    SSP_lib_hdr=Pipe3D_tab_hdu[2].header
    (ny,nx)=SSP_lib_data.shape
    SSP_ML={}
    SSP_ML_age={}
    SSP_ML_met={}
    a_age=[]
    a_met=[]
    a_age_met=[]
    for i in arange(0,ny):
        key='NORM'+str(i)
        ssp_name='NAME'+str(i)
        ssp_file=SSP_lib_hdr[ssp_name]
        ssp_file=ssp_file.replace('spec_ssp_','')
        ssp_file=ssp_file.replace('.spec','')
        ssp_file=ssp_file.replace('z','')
        (age,met)=ssp_file.split('_')
        met='0.'+met
        age ='{:0.4f}'.format(float(age))
        met ='{:0.4f}'.format(float(met))
        age_met=age+'-'+met
        a_age_met.append(age_met)
        a_age.append(age)
        a_met.append(met)
        norm=(1/SSP_lib_hdr[key])/ 3500#*(1e-16)/3.826e33
        SSP_ML[age_met]=norm
        age_find = np.array(list(SSP_ML_age.keys()))
        if(len(age_find)==0):
            SSP_ML_age[age]=norm
        else:
            if (len(age_find[np.char.find(age_find,str(age))>-1])>0):
                SSP_ML_age[age]=SSP_ML_age[age]+norm
            else:
                SSP_ML_age[age]=norm   
        met_find = np.array(list(SSP_ML_met.keys()))
        if(len(met_find)==0):
            SSP_ML_met[met]=norm
        else:
            if (len(met_find[np.char.find(met_find,str(met))>-1])>0):
                SSP_ML_met[met]=SSP_ML_met[met]+norm
            else:
                SSP_ML_met[met]=norm
        if (verbose==1):
            print(age,met,age_met,SSP_ML[age_met],SSP_ML_age[age],SSP_ML_met[met])

            
    a_age_met=np.unique(a_age_met)
    a_age=np.unique(a_age)
    a_met=np.unique(a_met)
    for age in a_age:
        SSP_ML_age[age]=SSP_ML_age[age]/len(a_met)
    for met in a_met:
        SSP_ML_met[met]=SSP_ML_met[met]/len(a_age)
    if (verbose==1):
        print(len(a_age_met),len(a_age),len(a_met))
        print(len(SSP_ML_age),len(SSP_ML_met))
    Pipe3D_tab = Table.read(DIR+Pipe3D_file, hdu=1)
    return Pipe3D_tab,SSP_ML,SSP_ML_age,SSP_ML_met,SSP_lib_data,SSP_lib_hdr

def read_Pipe3D_MaNGA(name,DIR='',verbose=0):
    dat=name.split("-")
    plate=dat[1]
    ifu=dat[2]
    DIR_plate=DIR+"/"+plate+"/"
    Pipe3D_file=DIR_plate+name+".Pipe3D.cube.fits.gz"
    Pipe3D_hdu=fits.open(Pipe3D_file)


    ######################################################
    # SSP cube                                           #
    ######################################################    
    SELECT_REG_data=Pipe3D_hdu[8].data
    SSP_data=Pipe3D_hdu[1].data
    SSP_hdr=Pipe3D_hdu[1].header
    (nz,ny,nx)=SSP_data.shape
    SSP_maps={}
    SSP_key={}
    SSP_key[0]="V"
    SSP_key[1]="CS"
    SSP_key[2]="DZ"
    SSP_key[3]="med"
    SSP_key[4]="std"
    SSP_key[5]="A_L"
    SSP_key[6]="A_M"
    SSP_key[7]="e_A_M"
    SSP_key[8]="Z_L"
    SSP_key[9]="Z_M"
    SSP_key[10]="e_Z_M"
    SSP_key[11]="Av"
    SSP_key[12]="e_Av"
    SSP_key[13]="vel"
    SSP_key[14]="e_vel"
    SSP_key[15]="sig"
    SSP_key[16]="e_sig"
    SSP_key[17]="ML"
    SSP_key[18]="M"
    SSP_key[19]="Md"
    SSP_key[20]="e_M"
    for indx in arange(0,nz):
        val=SSP_key[indx]
#        print(indx,val)
        image=SSP_data[indx,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        if ((indx!=13) and (indx!=14) and (indx!=8) and (indx!=9)):
            image=np.ma.masked_array(image,\
                                 ~(image>0.0))
        else:
            image=np.ma.masked_array(image,\
                                 (image==0.0))
        SSP_maps[val]=image

        
    ######################################################
    # SFH cube                                           #
    ######################################################
    SFH_data=Pipe3D_hdu[2].data
    SFH_hdr=Pipe3D_hdu[2].header
    (nz,ny,nx)=SFH_data.shape
    SFH_maps={}
    SFH_key={}
    age_met=[]
    age=[]
    met=[]
    n_age=0
    n_met=0
    n_age_met=0
    for i in arange(0,nz):
        key='DESC_'+str(i)
        mark_labels=SFH_hdr[key]    
        mark_labels=mark_labels.replace('Luminosity Fraction for ','')
        mark_labels=mark_labels.replace(' SSP','')
        SFH_key[i]=mark_labels.replace(' ','_')
        image=SFH_data[i,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        image=np.ma.masked_array(image,\
                                 (image==0.0))               
        SFH_maps[SFH_key[i]]=image
        if (mark_labels.find('age-met')>-1):
            n_age_met=n_age_met+1
            age_met.append(mark_labels.replace('age-met ',''))
        else:
            if (mark_labels.find('age')>-1):
                n_age=n_age+1
                age.append(mark_labels.replace('age ',''))
            else:
                if (mark_labels.find('met')>-1):
                    n_met=n_met+1
                    met.append(mark_labels.replace('met ',''))
       
    age_met=np.array(age_met)
    age=np.array(age)
    met=np.array(met)    
   
    ######################################################
    # IND cube                                           #
    ######################################################
    IND_data=Pipe3D_hdu[3].data
    IND_hdr=Pipe3D_hdu[3].header
    (nz,ny,nx)=IND_data.shape
    IND_maps={}
    IND_key={}
    for i in arange(0,nz):
        key='INDEX'+str(i)
        mark_labels=IND_hdr[key]    
        IND_key[i]=mark_labels.replace(' ','_')
        image=IND_data[i,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        image=np.ma.masked_array(image,\
                                 (image==0.0))               
        IND_maps[IND_key[i]]=image

    ######################################################
    # ELINES cube                                        #
    ######################################################
    ELINES_data=Pipe3D_hdu[4].data
    ELINES_hdr=Pipe3D_hdu[4].header
    (nz,ny,nx)=ELINES_data.shape
    ELINES_maps={}
    ELINES_key={}
    for i in arange(0,nz):
        key='DESC_'+str(i)
        mark_labels=ELINES_hdr[key]    
        mark_labels=mark_labels.replace(' emission line','')
        mark_labels=mark_labels.replace('Halpha velocity','vel_Ha')    
        mark_labels=mark_labels.replace('Velocity dispersion plus instrumenta one','disp_Ha')
        mark_labels=mark_labels.replace('Halpha','Ha')
        mark_labels=mark_labels.replace('Hbeta','Hb')
        ELINES_key[i]=mark_labels.replace(' ','')
        image=ELINES_data[i,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        image=np.ma.masked_array(image,\
                                 (image==0.0))               
        ELINES_maps[ELINES_key[i]]=image

    ######################################################
    # FLUX_ELINES cube                                           #
    ######################################################
    FE_data=Pipe3D_hdu[5].data
    FE_hdr=Pipe3D_hdu[5].header
    (nz,ny,nx)=FE_data.shape
    FE_maps={}
    FE_key={}
    for i in arange(0,nz):
        key='NAME'+str(i)
        wave='WAVE'+str(i)
        mark_labels=FE_hdr[key]+'_'+FE_hdr[wave]    
        FE_key[i]=mark_labels.replace(' ','_')
        image=FE_data[i,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        image=np.ma.masked_array(image,\
                                 (image==0.0))               
        FE_maps[FE_key[i]]=image
        
    ######################################################
    # FLUX_ELINES_LONG cube                                           #
    ######################################################
    FEL_data=Pipe3D_hdu[6].data
    FEL_hdr=Pipe3D_hdu[6].header
    (nz,ny,nx)=FEL_data.shape
    FEL_maps={}
    FEL_key={}
    for i in arange(0,nz):
        key='NAME'+str(i)
        wave='WAVE'+str(i)
        mark_labels=FEL_hdr[key]+'_'+FEL_hdr[wave]    
        FEL_key[i]=mark_labels.replace(' ','_')
        image=FEL_data[i,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        image=np.ma.masked_array(image,\
                                 (image==0.0))               
        FEL_maps[FEL_key[i]]=image
        
    ######################################################
    # GAIA cube                                           #
    ######################################################
    GA_data=Pipe3D_hdu[7].data
    GA_hdr=Pipe3D_hdu[7].header        
    
    ######################################################
    # SEG map                                           #
    ######################################################
    MASK_data=Pipe3D_hdu[8].data
    MASK_hdr=Pipe3D_hdu[8].header        
        
    if (verbose==1):
        print(Pipe3D_hdu.info())
        print('SSP:',SSP_key)
        print('n_SSPs:',n_age_met,', n_ages:',n_age,n_met)
        print('SFH:',SFH_key)
        print('IND:',IND_key)
        print('ELINES:',ELINES_key)
        print('FE:',FE_key)
        print('FEL:',FEL_key)
  
    output={'hdr':Pipe3D_hdu[0].header,'SSP':SSP_maps,'SFH':SFH_maps,\
            'AGE':age,'MET':met,'AGE-MET':age_met,\
           'IND':IND_maps,'ELINES':ELINES_maps,\
           'FE':FE_maps,'FEL':FEL_maps,'GAIA_MASK':GA_data,'MASK':MASK_data}
    return output




def read_Pipe3D_CALIFA(name,DIR='',verbose=0):
#    dat=name.split("-")
#    plate=dat[1]
#    ifu=dat[2]
    DIR_plate=DIR#+"/"+plate+"/"
    Pipe3D_file=DIR_plate+name+".Pipe3D.cube.fits.gz"
    Pipe3D_hdu=fits.open(Pipe3D_file)


    ######################################################
    # SSP cube                                           #
    ######################################################    
    SELECT_REG_data=Pipe3D_hdu[8].data
    SSP_data=Pipe3D_hdu[1].data
    SSP_hdr=Pipe3D_hdu[1].header
    (nz,ny,nx)=SSP_data.shape
    SSP_maps={}
    SSP_key={}
    SSP_key[0]="V"
    SSP_key[1]="CS"
    SSP_key[2]="DZ"
    SSP_key[3]="med"
    SSP_key[4]="std"
    SSP_key[5]="A_L"
    SSP_key[6]="A_M"
    SSP_key[7]="e_A_M"
    SSP_key[8]="Z_L"
    SSP_key[9]="Z_M"
    SSP_key[10]="e_Z_M"
    SSP_key[11]="Av"
    SSP_key[12]="e_Av"
    SSP_key[13]="vel"
    SSP_key[14]="e_vel"
    SSP_key[15]="sig"
    SSP_key[16]="e_sig"
    SSP_key[17]="ML"
    SSP_key[18]="M"
    SSP_key[19]="Md"
    SSP_key[20]="e_M"
    for indx in arange(0,nz):
        val=SSP_key[indx]
#        print(indx,val)
        image=SSP_data[indx,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        if ((indx!=13) and (indx!=14) and (indx!=8) and (indx!=9)):
            image=np.ma.masked_array(image,\
                                 ~(image>0.0))
        else:
            image=np.ma.masked_array(image,\
                                 (image==0.0))
        SSP_maps[val]=image

        
    ######################################################
    # SFH cube                                           #
    ######################################################
    SFH_data=Pipe3D_hdu[2].data
    SFH_hdr=Pipe3D_hdu[2].header
    (nz,ny,nx)=SFH_data.shape
    SFH_maps={}
    SFH_key={}
    age_met=[]
    age=[]
    met=[]
    n_age=0
    n_met=0
    n_age_met=0
    for i in arange(0,nz):
        key='DESC_'+str(i)
        mark_labels=SFH_hdr[key]    
        mark_labels=mark_labels.replace('Luminosity Fraction for ','')
        mark_labels=mark_labels.replace(' SSP','')
        SFH_key[i]=mark_labels.replace(' ','_')
        image=SFH_data[i,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        image=np.ma.masked_array(image,\
                                 (image==0.0))               
        SFH_maps[SFH_key[i]]=image
        if (mark_labels.find('age-met')>-1):
            n_age_met=n_age_met+1
            age_met.append(mark_labels.replace('age-met ',''))
        else:
            if (mark_labels.find('age')>-1):
                n_age=n_age+1
                age.append(mark_labels.replace('age ',''))
            else:
                if (mark_labels.find('met')>-1):
                    n_met=n_met+1
                    met.append(mark_labels.replace('met ',''))
       
    age_met=np.array(age_met)
    age=np.array(age)
    met=np.array(met)    
   
    ######################################################
    # IND cube                                           #
    ######################################################
    IND_data=Pipe3D_hdu[3].data
    IND_hdr=Pipe3D_hdu[3].header
    (nz,ny,nx)=IND_data.shape
    IND_maps={}
    IND_key={}
    for i in arange(0,nz):
        key='INDEX'+str(i)
        mark_labels=IND_hdr[key]    
        IND_key[i]=mark_labels.replace(' ','_')
        image=IND_data[i,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        image=np.ma.masked_array(image,\
                                 (image==0.0))               
        IND_maps[IND_key[i]]=image

    ######################################################
    # ELINES cube                                        #
    ######################################################
    ELINES_data=Pipe3D_hdu[4].data
    ELINES_hdr=Pipe3D_hdu[4].header
    (nz,ny,nx)=ELINES_data.shape
    ELINES_maps={}
    ELINES_key={}
    for i in arange(0,nz):
        key='DESC_'+str(i)
        mark_labels=ELINES_hdr[key]    
        mark_labels=mark_labels.replace(' emission line','')
        mark_labels=mark_labels.replace('Halpha velocity','vel_Ha')    
        mark_labels=mark_labels.replace('Velocity dispersion plus instrumenta one','disp_Ha')
        mark_labels=mark_labels.replace('Halpha','Ha')
        mark_labels=mark_labels.replace('Hbeta','Hb')
        ELINES_key[i]=mark_labels.replace(' ','')
        image=ELINES_data[i,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        image=np.ma.masked_array(image,\
                                 (image==0.0))               
        ELINES_maps[ELINES_key[i]]=image

    ######################################################
    # FLUX_ELINES cube                                           #
    ######################################################
    FE_data=Pipe3D_hdu[5].data
    FE_hdr=Pipe3D_hdu[5].header
    (nz,ny,nx)=FE_data.shape
    FE_maps={}
    FE_key={}
    for i in arange(0,nz):
        key='NAME'+str(i)
        wave='WAVE'+str(i)
        mark_labels=FE_hdr[key]+'_'+FE_hdr[wave]    
        FE_key[i]=mark_labels.replace(' ','_')
        image=FE_data[i,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        image=np.ma.masked_array(image,\
                                 (image==0.0))               
        FE_maps[FE_key[i]]=image
        
    ######################################################
    # FLUX_ELINES_LONG cube                                           #
    ######################################################
    FEL_data=Pipe3D_hdu[6].data
    FEL_hdr=Pipe3D_hdu[6].header
    (nz,ny,nx)=FEL_data.shape
    FEL_maps={}
    FEL_key={}
    for i in arange(0,nz):
        key='NAME'+str(i)
        wave='WAVE'+str(i)
        mark_labels=FEL_hdr[key]+'_'+FEL_hdr[wave]    
        FEL_key[i]=mark_labels.replace(' ','_')
        image=FEL_data[i,:,:]
        image = np.ma.masked_invalid(image)
        image=image*SELECT_REG_data
        image=image*1.0
        image=np.ma.masked_array(image,\
                                 (image==0.0))               
        FEL_maps[FEL_key[i]]=image
        
    ######################################################
    # GAIA cube                                           #
    ######################################################
    GA_data=Pipe3D_hdu[7].data
    GA_hdr=Pipe3D_hdu[7].header        
    
    ######################################################
    # SEG map                                           #
    ######################################################
    MASK_data=Pipe3D_hdu[8].data
    MASK_hdr=Pipe3D_hdu[8].header        
        
    if (verbose==1):
        print(Pipe3D_hdu.info())
        print('SSP:',SSP_key)
        print('n_SSPs:',n_age_met,', n_ages:',n_age,n_met)
        print('SFH:',SFH_key)
        print('IND:',IND_key)
        print('ELINES:',ELINES_key)
        print('FE:',FE_key)
        print('FEL:',FEL_key)
  
    output={'hdr':Pipe3D_hdu[0].header,'SSP':SSP_maps,'SFH':SFH_maps,\
            'AGE':age,'MET':met,'AGE-MET':age_met,\
           'IND':IND_maps,'ELINES':ELINES_maps,\
           'FE':FE_maps,'FEL':FEL_maps,'GAIA_MASK':GA_data,'MASK':MASK_data}
    return output

