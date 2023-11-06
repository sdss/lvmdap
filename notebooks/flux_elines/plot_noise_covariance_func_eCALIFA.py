#!/usr/bin/python

from scipy.stats.distributions import t
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt 
from astropy.table import Table
from scipy import stats
import numpy as np
import matplotlib
import sys
import os

# ---------------------------------------------------------------------------------------

def read_table(table, dir_in=None):
    if dir_in is not None:
        table = os.path.join(dir_in, table)

    # N, s_int, n_int, s_pix, n_pix, n_teor
    if isinstance(table, str):
        if '.csv' in table:
            t = Table.read(table, format='csv')
        else:
            t = Table.read(table)
    else:
        t = table

    return t

def minmax(datas):
    if not isinstance(datas, (tuple, list)):
        datas = [datas]
    return np.min(datas), np.max(datas)

def fit_func_log(x, a):
     return 1.0 + a * np.log10(x)

def fit_func_sqrt(x, a):
     return np.sqrt(a * x / (a + x - 1.))

def get_fit(function, x, y, alpha=0.05, p=1, xlims=[1, None]):
     idn = np.isfinite(y)
     if xlims is not None:
         if xlims[0] is not None:
             idn = np.logical_and(idn, x > xlims[0])
         if xlims[1] is not None:
             idn = np.logical_and(idn, x < xlims[1])
     params, pcov = curve_fit(function, x[idn], y[idn])
     dof = max(0, x[idn].size - p)
     tval = t.ppf(1.0 - alpha/2., dof)
     sigma = np.diag(pcov)**0.5
     return params[0], sigma[0] * tval

def get_sn(table, pipe=False, nz_size=None, sn_lims=None, dir_in=None):

    # N, s_int, n_int, s_pix, n_pix, n_teor
    t = read_table(table, dir_in=dir_in)
    
    n_size = t['N']
    # sn_i: Real (integrated) empirical SN
    sn_r = t['s_int'] / t['n_int']
    # sn_p: Sum of pixels empirical SN (like in voronoi)
    sn_p = t['s_int'] / t['n_pix']
    # sn_t: Sum of pixels theoretical (pipeline errors)
    sn_t = t['s_int'] / t['n_teor']

    sn = sn_t if pipe else sn_p
    
    if nz_size is None:
        nz_size = np.arange(1.5, 110.5, 2)
    
    n_bins = len(nz_size) - 1

    SN_ratio_median = np.zeros(n_bins, dtype=np.float32)
    SN_ratio_mean   = np.zeros(n_bins, dtype=np.float32)
    SN_ratio_std    = np.zeros(n_bins, dtype=np.float32)
    bins            = np.zeros(n_bins, dtype=np.float32)
    
    at = 0.
    for i in range(n_bins):
        select = np.logical_and(n_size >= nz_size[i], n_size < nz_size[i+1])
        if sn_lims is not None:
            if sn_lims[0] is not None:
                select = np.logical_and(select, sn_r >= sn_lims[0])
            if sn_lims[1] is not None:
                select = np.logical_and(select, sn_r <= sn_lims[1])

        bins[i]            = (nz_size[i] + nz_size[i+1]) / 2.0
        sn_ratio           = sn[select] / sn_r[select]

        SN_ratio_median[i] = np.nanmedian(sn_ratio)
        SN_ratio_mean[i]   = np.nanmean(sn_ratio)
        SN_ratio_std[i]    = np.nanstd(sn_ratio)

    return bins, SN_ratio_median, SN_ratio_mean, SN_ratio_std

def plot_noise(ax, table, grating, function='log', plines=False, lw=1, median=False, **kwargs):

    bins, SN_ratio_median, SN_ratio_mean, SN_ratio_std = get_sn(table, **kwargs)
    SN_ratio = SN_ratio_median if median else SN_ratio_mean

    tp_err = ('Pipeline' if kwargs.get('pipe', False) else 'Empirical') + ' $\epsilon_{k}$'
    sn_lims = kwargs.get('sn_lims')
 
    #cs1 = '#3182bd', cs2 = '#9ecae1', cs3 = '#deebf7')
    cs1 = '#1f77b4'; cs2 = '#6aa4cd'; cs3 = '#b4d2e6'; cl = '#ffbc79'; cd = '#ff7f0e'
    cg  = cs1 # '#D62728'
 
    dpar = {'axes.linewidth': 1.5, 'xtick.major.size': 8, 'xtick.minor.size': 4, 
            'ytick.major.size': 6, 'ytick.minor.size': 3}
 
    matplotlib.rcParams.update(dpar)
    
    if plines:
        ax.plot(bins, SN_ratio + SN_ratio_std, '-k', lw=lw)
        ax.plot(bins, SN_ratio - SN_ratio_std, '-k', lw=lw)
        ax.plot(bins, SN_ratio + 2 * SN_ratio_std, '--k', lw=lw)
        ax.plot(bins, SN_ratio - 2 * SN_ratio_std, '--k', lw=lw)
        ax.plot(bins, SN_ratio + 3 * SN_ratio_std, ':k', lw=lw)
        ax.plot(bins, SN_ratio - 3 * SN_ratio_std, ':k', lw=lw)
    ax.fill_between(bins, SN_ratio + 3 * SN_ratio_std, SN_ratio + 2 * SN_ratio_std, color=cs3)
    ax.fill_between(bins, SN_ratio - 2 * SN_ratio_std, SN_ratio - 3 * SN_ratio_std, color=cs3)
    ax.fill_between(bins, SN_ratio + 2 * SN_ratio_std, SN_ratio + 1 * SN_ratio_std, color=cs2)
    ax.fill_between(bins, SN_ratio - 1 * SN_ratio_std, SN_ratio - 2 * SN_ratio_std, color=cs2)
    ax.fill_between(bins, SN_ratio + 1 * SN_ratio_std, SN_ratio,color=cs1)
    ax.fill_between(bins, SN_ratio - SN_ratio_std, SN_ratio_mean, color=cs1)
    ax.plot(bins, SN_ratio_mean, 'o', mfc=cd, lw=2, zorder=2, ms=7)
 
    #print(r'$\beta$ = 1.00 + %.2f +- %.2f log(N)' % (fit_param, sigma))
    err_sum  = r'$\epsilon_{B}^2 = \sum_{k=1}^{N} \epsilon_{k}^2$'
    err_func = r'$\epsilon^{2}_\mathrm{real,B} = \beta (N)^{2} \times \epsilon_{B}^2$'
    #err_lab  = ' | '.join([err_sum, err_func])
    #ax.text(0.95, 0.93, err_lab, fontsize=14, fontweight='medium', transform=ax.transAxes, ha='right')
    ax.text(0.95, 0.30, err_sum, fontsize=14, fontweight='medium', transform=ax.transAxes, ha='right')
    ax.text(0.95, 0.20, err_func, fontsize=14, fontweight='medium', transform=ax.transAxes, ha='right')
    
    # Function
    dfunc = {'log': fit_func_log, 'sqrt': fit_func_sqrt}
    func  = dfunc.get(function)

    fit_param, sigma = get_fit(func, bins, SN_ratio_mean)

    if function == 'log':
        #y = 1.00 + 1.11*np.log10(x)
        #sfunc = r'$\beta$ = 1.00 + 1.11 log(N)'
        sfunc  = r'$\beta$ = 1.00 + %.2f log(N)' % fit_param
    if function == 'sqrt':
        sfunc  = r'$\beta$ = $\sqrt{\frac{%.1f\, N}{%.1f + N - 1}}$' % (fit_param, fit_param)

    x = np.arange(1, 105, 1)
    y = func(x, fit_param)
    ax.plot(x, y, '-', lw=3, color=cl, zorder=1, label=sfunc)
    dleg = {'weight': 'bold', 'size': 20}
    ax.legend(frameon=False, bbox_to_anchor=[0.97, 0.1], loc='right', prop=dleg)
 
    # Limits
    #ax.set_xlim([-0.5, 50])
    #ax.set_ylim([0.7, 5.6])
    ax.set_xlim([-0.5, 110])
    ax.set_ylim([0., 9])
 
    #ax.set_ylabel(r'$\epsilon_\mathrm{real}/\epsilon_\mathrm{bin}$',fontsize=18)
    ax.set_ylabel(r'$\beta(N)$', fontsize=20)
    ax.set_xlabel('N', fontsize=20)
    #fig.text(0.02,0.45,r'$\epsilon_\mathrm{bin}/\epsilon_\mathrm{real}$',fontsize=19,rotation=90)
    ax.text(0.03, 0.92, grating, fontsize=14, fontweight='bold', transform=ax.transAxes, ha='left', color=cg)
    ax.text(0.97, 0.92, tp_err, fontsize=14, fontweight='bold', transform=ax.transAxes, ha='right', color='k')
    if sn_lims is not None:
        sn_lab = 'SN '
        if sn_lims[0] is not None:
            sn_lab += ' > %s' % sn_lims[0]
        if sn_lims[1] is not None:
            sn_lab += ' < %s' % sn_lims[1] if sn_lims[0] is None else ' & SN < %s' % sn_lims[1]
        ax.text(0.03, 0.86, sn_lab, fontsize=14, fontweight='bold', transform=ax.transAxes, ha='left', color=cg)
    
    #ax.text(5, 4.2, r'Target S/N = 20', fontsize=14)
    #ax.text(0.96,0.08,r'$\mathrm{best-fit: }\alpha=1.07$',fontsize=16,transform=ax1.transAxes,ha='right')
    #ax.set_xticklabels([])
    ax.minorticks_on()
    for line in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines() + ax.xaxis.get_minorticklines() + ax.yaxis.get_minorticklines():
        line.set_markeredgewidth(1.5)

def plot_sn(ax, table, grating, dir_in=None):

    t = read_table(table, dir_in=dir_in)

    # sn_i: Real (integrated) empirical SN
    sn_r = t['s_int'] / t['n_int']
    # sn_p: Sum of pixels empirical SN (like in voronoi)
    sn_p = t['s_int'] / t['n_pix']
    # sn_t: Sum of pixels theoretical (pipeline errors)
    sn_t = t['s_int'] / t['n_teor']

    cg  = '#1f77b4' if 'blue' in grating else '#D62728'

    xmin, xmax = minmax(sn_p)
    ymin, ymax = minmax(sn_t)
    ax.plot(sn_p, sn_t, 'o')
    ax.plot([xmin, xmax], [xmin, xmax], lw=3)
    ax.set_xlabel('SN Empirical', fontsize=18)
    ax.set_ylabel('SN Pipeline', fontsize=18)
    ax.text(0.03, 0.92, grating, fontsize=14, fontweight='bold', transform=ax.transAxes, ha='left', color=cg)
# ---------------------------------------------------------------------------------------

dir_in = 'figs'#/Users/rgb/iraf/trabajos/WEAVE/LIFU/qc/covariance/'

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
#ax1 = fig.add_axes([0.13,0.53,0.85,0.45])
#ax2 = fig.add_axes([0.13,0.07,0.85,0.45])

sn_lims  = [20, None]
function = 'log'

# Need to cut high S/N from stars (see below)
table   = 'tables/eCALIFA.beta.fits'
grating = 'V500'

plot_noise(ax, sn_lims=sn_lims, function=function, table=table, grating=grating, pipe=False, plines=True)
plt.show()
