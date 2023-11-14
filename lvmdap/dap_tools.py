import io
import sys
import warnings
import itertools
import numpy as np
from astropy.io import fits
from copy import deepcopy as copy
from os.path import basename, isfile
import os
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from astropy.io.fits.verify import VerifyWarning

import matplotlib.pyplot as plt

from pyFIT3D.common.stats import median_filter as st_median_filter
from pyFIT3D.common.stats import hyperbolic_fit_par, std_m, pdl_stats, _STATS_POS
from pyFIT3D.common.io import trim_waves, get_data_from_fits, get_wave_from_header
from pyFIT3D.common.constants import __c__, __sigma_to_FWHM__
from pyFIT3D.common.io import output_spectra, array_to_fits, write_img_header, print_verbose
from pyFIT3D.common.constants import __Hubble_constant__, __Omega_matter__, __Omega_Lambda__
from pyFIT3D.common.constants import __solar_luminosity__, __solar_metallicity__, _figsize_default

from astropy.table import Table
from astropy.table import vstack as vstack_table

warnings.simplefilter('ignore', category=VerifyWarning)

__indices__ = {
    'Hd':     [4083.500, 4122.250, 4041.600, 4079.750, 4128.500, 4161.000],
    'Hb':     [4847.875, 4876.625, 4827.875, 4847.875, 4876.625, 4891.625],
    'Mgb':    [5160.125, 5192.625, 5142.625, 5161.375, 5191.375, 5206.375],
    'Fe5270': [5245.650, 5285.650, 5233.150, 5248.150, 5285.650, 5318.150],
    'Fe5335': [5312.125, 5352.125, 5304.625, 5315.875, 5353.375, 5363.375],
    'D4000':  [4050.000, 4250.000, 3750.000, 3950.000, 0.000,    1.000],
    'Hdmod':  [4083.500, 4122.250, 4079,     4083,     4128.500, 4161.000],
    'Hg':     [4319.75,  4363.50,  4283.50,  4319.75,  4367.25,  4419.75]
}
_INDICES_POS_ = {
    'OL1': 0,
    'OL2': 1,
    'OLb1': 2,
    'OLb2': 3,
    'OLr1': 4,
    'OLr2': 5,
    'OLb': 6,
    'OLr': 7
}

__indices_sky_ = {
    'OI5577':  [5557, 5597, 5507, 5547, 5607, 5647],
    'NaID':    [5873, 5893, 5823, 5863, 5903, 5943],
    'OI6300':  [6280, 6320, 6230, 6270, 6330, 6370],
}


def dap_indices_spec(wave__w, flux_ssp__w, res__w, redshift, n_sim, plot=0, wl_half_range=200, verbose=False,\
                      __indices__=__indices__, _INDICES_POS=_INDICES_POS_):
    if plot:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        #plt.style.use('dark_background')
        f, ax = plt.subplots()
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')

    nw = wave__w.size
    cdelt = wave__w[1] - wave__w[0]
    names__i = list(__indices__.keys())
    _ind = np.array(list(__indices__.values()))
    wmin = _ind.T[_INDICES_POS['OLb1']].min() - wl_half_range
    wmax = _ind.T[_INDICES_POS['OLr2']].max() + wl_half_range
    n_ind = _ind.shape[0]

    FWHM = __sigma_to_FWHM__*3*cdelt

    stflux = pdl_stats(flux_ssp__w)
    if (stflux[_STATS_POS['mean']] == 0) and (stflux[_STATS_POS['min']] == stflux[_STATS_POS['max']]):
        flux_ssp__w = np.ones_like(flux_ssp__w)
        res__w = 100*np.ones_like(flux_ssp__w)

    med_res__w = median_filter(res__w, size=np.int(3*FWHM/cdelt), mode='reflect')

    _ind_corr = _ind*(1 + redshift)
    _ind_corr.T[_INDICES_POS['OLb']] = (_ind_corr.T[_INDICES_POS['OLb1']] + _ind_corr.T[_INDICES_POS['OLb2']])/2
    _ind_corr.T[_INDICES_POS['OLr']] = (_ind_corr.T[_INDICES_POS['OLr1']] + _ind_corr.T[_INDICES_POS['OLr2']])/2
    wmin = _ind_corr.T[_INDICES_POS['OLb1']].min() - wl_half_range
    wmax = _ind_corr.T[_INDICES_POS['OLr2']].max() + wl_half_range

    med_flux = np.median(flux_ssp__w[np.isfinite(flux_ssp__w)])
    std_flux = np.std(flux_ssp__w[np.isfinite(flux_ssp__w)])
    EW__ki = np.zeros((n_sim, n_ind))
    for k in range(n_sim):
        noise = np.random.normal(size=nw)
        noise *= res__w
        _flux__w = flux_ssp__w + noise
        _med_flux = np.median(_flux__w)
        _min_flux = -0.1*np.abs(_med_flux)
        _max_flux = 3.5*_med_flux
        for ii in range(n_ind):
            if plot:
                plt.cla()
                ax = plt.gca()
                ax.set_xlim(wmin, wmax)
                ax.set_ylim(_min_flux - 0.2*np.abs(_min_flux), _max_flux + 0.2*np.abs(_max_flux))
                ax.plot(wave__w, _flux__w, 'k-')
            name = names__i[ii]
            Lb1 = _ind_corr[ii][_INDICES_POS['OLb1']]
            Lb2 = _ind_corr[ii][_INDICES_POS['OLb2']]
            Lr1 = _ind_corr[ii][_INDICES_POS['OLr1']]
            Lr2 = _ind_corr[ii][_INDICES_POS['OLr2']]
            L1 = _ind_corr[ii][_INDICES_POS['OL1']]
            L2 = _ind_corr[ii][_INDICES_POS['OL2']]
            Lb = _ind_corr[ii][_INDICES_POS['OLb']]
            Lr = _ind_corr[ii][_INDICES_POS['OLr']]

            str_verbose = f'L1:{L1}, L2:{L2}\nblue: Lb:{Lb}, Lb1:{Lb1}, Lb2:{Lb2}\nred: Lr:{Lr}, Lr1:{Lr1}, Lr2:{Lr2}'
            print_verbose(str_verbose, verbose=verbose, level=1)

            if name != 'D4000':
                iw1_b = (Lb1 - wave__w[0] - 0.5*cdelt)/cdelt
                iw2_b = (Lb2 - wave__w[0] - 0.5*cdelt)/cdelt
                Sb = 0
                nb = 0
                for iw in range(np.int(iw1_b + 1), np.int(iw2_b)):
                    Sb += _flux__w[iw]*cdelt
                    nb += 1
                iw = np.int(iw1_b)
                ff = iw + 1 - iw1_b
                Sb += (_flux__w[iw]*ff*cdelt)
                iw = np.int(iw2_b)
                ff = iw2_b - iw
                Sb += (_flux__w[iw]*ff*cdelt)
                Sb = Sb/(Lb2 - Lb1)

                iw1_r = (Lr1 - wave__w[0] - 0.5*cdelt)/cdelt
                iw2_r = (Lr2 - wave__w[0] - 0.5*cdelt)/cdelt
                Sr = 0
                nr = 0
                for iw in range(np.int(iw1_r + 1), np.int(iw2_r)):
                    Sr += _flux__w[iw]*cdelt
                    nr += 1
                iw = np.int(iw1_r)
                ff = iw + 1 - iw1_r
                Sr += (_flux__w[iw]*ff*cdelt)
                iw = np.int(iw2_r)
                ff = iw2_r - iw
                Sr += (_flux__w[iw]*ff*cdelt)
                Sr = Sr/(Lr2 - Lr1)
                EW = 0
                CK = []
                waveK = []
                iw1 = (L1 - wave__w[0] - 0.5*cdelt)/cdelt
                iw2 = (L2 - wave__w[0] - 0.5*cdelt)/cdelt
                for iw in range(np.int(iw1 + 1), np.int(iw2)):
                    C = Sb*((Lr - wave__w[iw])/(Lr - Lb)) + Sr*((wave__w[iw] - Lb)/(Lr - Lb))
                    EW = EW + (1 - _flux__w[iw]/C)*(wave__w[iw] - wave__w[iw - 1])
                    CK.append(C)
                    waveK.append(wave__w[iw])
                iw = np.int(iw1)
                ff = iw + 1 - iw1
                C = Sb*((Lr - wave__w[iw])/(Lr - Lb)) + Sr*((wave__w[iw] - Lb)/(Lr - Lb))
                EW = EW + (1 - _flux__w[iw]/C)*(wave__w[iw] - wave__w[iw - 1])*ff
                iw = np.int(iw2)
                ff = iw2 - iw
                C = Sb*((Lr - wave__w[iw])/(Lr - Lb)) + Sr*((wave__w[iw] - Lb)/(Lr - Lb))
                EW = EW + (1 - _flux__w[iw]/C)*(wave__w[iw] - wave__w[iw - 1])*ff
                EW = EW/(1 + redshift)
                if plot:
                    _xy, dx, dy = (Lb1, _max_flux*0.5), Lb2-Lb1, _max_flux*0.3
                    rectLb1Lb2 = patches.Rectangle(_xy, dx, dy, edgecolor='b', facecolor='none')
                    ax.add_patch(rectLb1Lb2)
                    _xy, dx, dy = (Lr1, _max_flux*0.5), Lr2-Lr1, _max_flux*0.3
                    rectLr1Lr2 = patches.Rectangle(_xy, dx, dy, edgecolor='r', facecolor='none')
                    ax.add_patch(rectLr1Lr2)
                    _xy, dx, dy = (L1, _max_flux*0.5), L2-L1, _max_flux*0.3
                    rectL1L2 = patches.Rectangle(_xy, dx, dy, edgecolor='g', facecolor='none')
                    ax.add_patch(rectL1L2)
                    ax.plot(waveK, CK, ls='--', c='gray')
                    ax.scatter(Lb, Sb, marker='x', color='b', s=50)
                    ax.scatter(Lr, Sr, marker='x', color='r', s=50)
                    ax.set_xlim(Lb1 - 0.02*Lb1, Lr2 + 0.02*Lr2)
                    ax.text(0.98, 0.98, name, transform=ax.transAxes, va='top', ha='right')
                    plt.pause(0.001)
            else:
                Sb = 0
                nb = 0
                iw1_b = (Lb1 - wave__w[0] - 0.5*cdelt)/cdelt
                iw2_b = (Lb2 - wave__w[0] - 0.5*cdelt)/cdelt
                for iw in range(np.int(iw1_b + 1), np.int(iw2_b)):
                    Sb += _flux__w[iw]*cdelt
                    nb += 1
                iw = np.int(iw1_b)
                ff = iw + 1 - iw1_b
                Sb += _flux__w[iw]*ff*cdelt
                iw = np.int(iw2_b)
                ff = iw2_b - iw
                Sb += _flux__w[iw]*ff*cdelt
                Sb = Sb/(Lb2 - Lb1)

                S = 0
                K = 0
                iw1 = (L1 - wave__w[0] - 0.5*cdelt)/cdelt
                iw2 = (L2 - wave__w[0] - 0.5*cdelt)/cdelt
                for iw in range(np.int(iw1 + 1), np.int(iw2)):
                    S += _flux__w[iw]*cdelt
                    K += 1
                iw = np.int(iw1)
                ff = iw + 1 - iw1
                S += _flux__w[iw]*ff*cdelt
                iw = np.int(iw2)
                ff = iw2 - iw
                S += _flux__w[iw]*ff*cdelt
                S = S/(L2 - L1)
                if Sb != 0:
                    EW = S/Sb
                else:
                    EW = 1e16
                if plot:
                    _xy, dx, dy = (Lb1, _max_flux*0.5), Lb2-Lb1, _max_flux*0.3
                    rectLb1Lb2 = patches.Rectangle(_xy, dx, dy, edgecolor='b', facecolor='none')
                    ax.add_patch(rectLb1Lb2)
                    # _xy, dx, dy = (Lr1, _max_flux*0.5), Lr2-Lr1, _max_flux*0.3
                    # rectLr1Lr2 = patches.Rectangle(_xy, dx, dy, edgecolor='r', facecolor='none')
                    # ax.add_patch(rectLr1Lr2)
                    _xy, dx, dy = (L1, _max_flux*0.5), L2-L1, _max_flux*0.3
                    rectL1L2 = patches.Rectangle(_xy, dx, dy, edgecolor='r', facecolor='none')
                    ax.add_patch(rectL1L2)
                    # ax.plot(waveK, CK, '--k')
                    ax.scatter(Lb, Sb, marker='x', color='b', s=50)
                    ax.scatter((L1 + L2)/2, S, marker='x', color='r', s=50)
                    ax.set_xlim(Lb1 - 0.02*Lb1, L2 + 0.02*L2)
                    ax.text(0.98, 0.98, name, transform=ax.transAxes, va='top', ha='right')
                    plt.pause(0.001)
            EW__ki[k, ii] = EW
    if plot:
        plt.close(f)
    return EW__ki, med_flux, std_flux


#
# Load an standard LVM RSS file
#

def load_LVM_rss(lvm_file, m2a=10e9, flux_scale=1e16):
    """Return the RSS from the given and LVM filename in the parsed command line arguments"""
    hdu = fits.open(lvm_file, memmap=False)
    rss_f_spectra = hdu['FLUX'].data
    rss_f_hdr = hdu['FLUX'].header
    try:
        rss_e_spectra = hdu['ERROR'].data
    except:
        rss_e_spectra = np.abs(rss_f_spectra-median_filter(rss_f_spectra,size=(1,51)))        
    wl__w = np.array([rss_f_hdr["CRVAL1"] + i*rss_f_hdr["CDELT1"] for i in range(rss_f_hdr["NAXIS1"])])
    wl__w = wl__w*m2a
    rss_f_spectra=rss_f_spectra*flux_scale
    rss_e_spectra=rss_e_spectra*flux_scale
    return wl__w, rss_f_spectra, rss_e_spectra



PLATESCALE = 112.36748321030637


def rotate(xx,yy,angle):
    # rotate x and y cartesian coordinates by angle (in degrees)
    # about the point (0,0)
    theta = -1.*angle * np.pi / 180. # in radians
    xx1 = np.cos(theta) * xx - np.sin(theta) * yy
    yy1 = np.sin(theta) * xx + np.cos(theta) * yy
    return xx1, yy1

def make_radec(xx0,yy0,ra,dec,pa):
    xx, yy = rotate(xx0,yy0,pa)
    ra_fib = ra + xx*PLATESCALE/3600./np.cos(dec*np.pi/180.) 
    dec_fib = dec - yy*PLATESCALE/3600. 
    return ra_fib, dec_fib

    
def make_line(wave, r1, sci, wl_shift_vel, whichone):

    # Halpha
    wl_ha = (wave > 6560 + wl_shift_vel/3e5*6560) & (wave < 6570+ wl_shift_vel/3e5*6570)
    iis_ha = np.where(wl_ha)[0]
    ha = np.sum(r1[:,iis_ha],axis=1)[sci]

    wl_ha_cont = (wave > 6600+ wl_shift_vel/3e5*6600) & (wave < 6610+ wl_shift_vel/3e5*6610)
    iis_ha_cont = np.where(wl_ha_cont)[0]
    ha_cont = np.sum(r1[:,iis_ha_cont],axis=1)[sci]
        
        
    # SII doublet

    wl_sii1 = (wave > 6715+ wl_shift_vel/3e5*6715) & (wave < 6725+ wl_shift_vel/3e5*6725)
    iis_sii1 = np.where(wl_sii1)[0]
    sii1 = np.sum(r1[:,iis_sii1],axis=1)[sci]

    wl_sii2 = (wave > 6730+ wl_shift_vel/3e5*6730) & (wave < 6740+ wl_shift_vel/3e5*6740)
    iis_sii2 = np.where(wl_sii2)[0]
    sii2 = np.sum(r1[:,iis_sii2],axis=1)[sci]


    wl_sii_cont = (wave > 6700+ wl_shift_vel/3e5*6700) & (wave < 6710+ wl_shift_vel/3e5*6710)
    iis_sii_cont = np.where(wl_sii_cont)[0]
    sii_cont = np.sum(r1[:,iis_sii_cont],axis=1)[sci]
        
        
    # [SIII]9069

    wl_siii9068 = (wave > 9065+ wl_shift_vel/3e5*9065) & (wave < 9075+ wl_shift_vel/3e5*9075)
    iis_siii9068 = np.where(wl_siii9068)[0]
    siii9068 = np.sum(r1[:,iis_siii9068],axis=1)[sci]
    
    # [OIII]5007
    wl_oiii = (wave > 5000 + wl_shift_vel/3e5*5000) & (wave < 5015+ wl_shift_vel/3e5*5015)
    iis_oiii = np.where(wl_oiii)[0]
    oiii = np.sum(r1[:,iis_oiii],axis=1)[sci]
    
    
    # [OIII]4363    
    wl_oiii4363 = (wave > 4360 + wl_shift_vel/3e5*4360) & (wave < 4365+ wl_shift_vel/3e5*4365)
    iis_oiii4363 = np.where(wl_oiii4363)[0]
    oiii4363 = np.sum(r1[:,iis_oiii4363],axis=1)[sci]

    wl_oiii4363_cont = (wave > 4375+ wl_shift_vel/3e5*4375) & (wave < 4380+ wl_shift_vel/3e5*4380)
    iis_oiii4363_cont = np.where(wl_oiii4363_cont)[0]
    oiii4363_cont = np.sum(r1[:,iis_oiii4363_cont],axis=1)[sci]
   
    # [OII]3727
    wl_oii = (wave > 3720 + wl_shift_vel/3e5*3720) & (wave < 3732+ wl_shift_vel/3e5*3732)
    iis_oii = np.where(wl_oii)[0]
    oii = np.sum(r1[:,iis_oii],axis=1)[sci]
    
    #[NII]5755
    wl_nii5755 = (wave > 5752 + wl_shift_vel/3e5*5752) & (wave < 5758+ wl_shift_vel/3e5*5758)
    iis_nii5755 = np.where(wl_nii5755)[0]
    nii5755 = np.sum(r1[:,iis_nii5755],axis=1)[sci]

    wl_nii5755_cont = (wave > 5710+ wl_shift_vel/3e5*5710) & (wave < 5715+ wl_shift_vel/3e5*5715)
    iis_nii5755_cont = np.where(wl_nii5755_cont)[0]
    nii5755_cont = np.sum(r1[:,iis_nii5755_cont],axis=1)[sci]
    
    wl_nii5755_cont2 = (wave > 5770+ wl_shift_vel/3e5*5770) & (wave < 5775+ wl_shift_vel/3e5*5775)
    iis_nii5755_cont2 = np.where(wl_nii5755_cont2)[0]
    nii5755_cont2 = np.sum(r1[:,iis_nii5755_cont2],axis=1)[sci]
   
    #r band
    wl_r = (wave > 5500 + wl_shift_vel/3e5*5500) & (wave < 6900+ wl_shift_vel/3e5*6900)
    iis_r = np.where(wl_r)[0]
    r = np.sum(r1[:,iis_r],axis=1)[sci]


    if whichone == 'ha':
        return ha
    if whichone == 'ha_sub':
        return ha - ha_cont
    if whichone == 'sii':
        return sii1 + sii2
    if whichone == 'sii_ratio':
        ratio = sii1/sii2
        ratio[sii1+sii2 < 10] = np.nan
        return ratio
    if whichone == 'siii':
        return siii9068
    
    if whichone == 'oiii': 
        return oiii
    
    if whichone == 'oii':
        return oii
    
    if whichone == 'oiii4363':
        return oiii4363 - oiii4363_cont
    
    if whichone == 'nii5755':
        return nii5755 - (nii5755_cont + nii5755_cont2)/2.
   
    if whichone == 'r_cont':
        return r

def read_file(file_ID, mjd, whichone = 'ha', wl_shift_vel = 0., nobad=False):
    pref = DIR_redux + (mjd) + '/'
    
    # read in the rss file
    rsshdu = fits.open(pref+'lvmCFrame-'+file_ID+'.fits')

    hdr = rsshdu[0].header

    r1 = rsshdu[1].data
    r1_hdr = rsshdu[1].header
    r1_err = rsshdu[2]

    wave=rsshdu[4].data 

    tab = Table(rsshdu[6].data)
    sci = (tab['targettype']=='science')
    if nobad:
        sci = (tab['targettype']=='science') & (tab['fibstatus'] == 0)

    rsshdu.close()

    # get ra/dec measured from coadd guiders?
    agcam_coadd = DIR_agcam+mjd+'/'+'lvm.sci.coadd_s'+file_ID+'.fits'
    if os.path.isfile(agcam_coadd):
        agcam_hdu = fits.open(agcam_coadd)
        agcam_hdr = agcam_hdu[1].header
        w = WCS(agcam_hdr)
        cen = w.pixel_to_world(2500,1000)
        racen = cen.ra.deg  #agcam_hdr['RAMEAS']
        deccen = cen.dec.deg #agcam_hdr['DECMEAS']
        pa = agcam_hdr['PAMEAS'] - 180.
        agcam_hdu.close()
    else:
        racen = hdr['POSCIRA']
        deccen = hdr['POSCIDE']
        pa = hdr['POSCIPA']


#    print(hdr['OBJECT'],hdr['POSCIPA'])
    #ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci], hdr['POSCIRA'], hdr['POSCIDE'], hdr['POSCIPA'])
#    ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci], hdr['TESCIRA'], hdr['TESCIDE'], hdr['POSCIPA'])
    ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci], racen, deccen, pa)

    line_flux = make_line(wave, r1,sci, wl_shift_vel, whichone)

    return ra_fib.data, dec_fib.data, line_flux

def plotty(line_dict, vmin, vmax, title, filename, size=30):
#    size = 30

    fig = plt.figure(figsize=(8,8))
#    for dd in line_dict:
#        d = line_dict[dd]
#        print(dd,d)
    plt.scatter(line_dict['ra_fib'],line_dict['dec_fib'], c =line_dict['line'] ,s=size,vmin = vmin, vmax = vmax)

    plt.title(title)
    plt.xlabel('Ra [deg]')
    plt.ylabel('Dec [deg]')
    plt.colorbar()

    ax = plt.gca()

    xx = ax.get_xlim()
    plt.xlim(xx[1],xx[0])

    ax.set_aspect('equal', adjustable='box')
    plt.draw()
    plt.show()
    fig.savefig('figs/'+filename+'.png')

    plt.close()


def read_PT(fitsfile, agcam_coadd, nobad=False):
    rsshdu = fits.open(fitsfile)

    hdr = rsshdu[0].header
    tab = Table(rsshdu['SLITMAP'].data)
    sci = np.full(len(tab), True)
    #(tab['targettype']=='science')
    mask_bad = (tab['targettype']=='science') & (tab['fibstatus'] == 0) 
    if nobad:
        sci = (tab['targettype']=='science') & (tab['fibstatus'] == 0)
    rsshdu.close()

    # get ra/dec measured from coadd guiders?
    # agcam_coadd = DIR_agcam+mjd+'/'+'lvm.sci.coadd_s'+file_ID+'.fits'
    if os.path.isfile(agcam_coadd):
        agcam_hdu = fits.open(agcam_coadd)
        agcam_hdr = agcam_hdu[1].header
        w = WCS(agcam_hdr)
        cen = w.pixel_to_world(2500,1000)
        racen = cen.ra.deg  #agcam_hdr['RAMEAS']
        deccen = cen.dec.deg #agcam_hdr['DECMEAS']
        pa = agcam_hdr['PAMEAS'] - 180.
        agcam_hdu.close()
    else:
        racen = hdr['POSCIRA']
        deccen = hdr['POSCIDE']
        pa = hdr['POSCIPA']

    ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci], racen, deccen, pa)
    tab=Table()
    tab['ra']=ra_fib.data
    tab['dec']=dec_fib.data
    tab['mask']=mask_bad
#    print(len(sci))
#    tab['mask']=sci
#    print(sci)
    return tab

