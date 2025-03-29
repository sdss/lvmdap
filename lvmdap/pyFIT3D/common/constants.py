import numpy as np
"""This module contains definition of pyFIT3D.common.constants to be used
globally across the pyFIT3D modules
"""

__version__ = 'pyPipe3D v1.1.5'

# maximum number of free parameters in the config file
__n_models_params__ = 9

_MODELS_VOIGT_PAR = {
    'central_wavelength': 0,
    'flux': 1,
    'sigma_L': 2,
    'sigma_G': 3,
    'v0': 4
}

_MODELS_ELINE_PAR = {
    'central_wavelength': 0,
    'flux': 1,
    'sigma': 2,
    'v0': 3
}

# _MODELS_POLY1D_PAR should be dynamically created
# _MODELS_POLY1D_PAR = lambda ncoeffs: {f'coeff{i}': i for i in range(ncoeffs)}
_MODELS_POLY1D_PAR = {
    'cont': 0
}

# possible EL MODELS
_EL_MODELS = {
    # 'voigt': _MODELS_VOIGT_PAR,
    'eline': _MODELS_ELINE_PAR,
    'poly1d': _MODELS_POLY1D_PAR,
    # 'poly1d': None,
}

__shift_convolve_lnwl_frac__ = 5  # a.k.a. f_fine in shift_convolve

__mask_elines_window__ = 4  # AA
# Emission Line fine search option
__ELRND_fine_search_option__ = False

__n_Monte_Carlo__ = 20

# speed of light in km/s
__c__ = 299792.458

# convert std to FWHM
__sigma_to_FWHM__ = 2*(2*np.log(2))**0.5

# convert FWHM to sigma
__FWHM_to_sigma__ = 1/__sigma_to_FWHM__

__solar_metallicity__ = 0.02
__solar_luminosity__ = 3.826e33  # erg/s

__selected_extlaw__ = 'CCM'
__selected_R_V__ = 3.1

__selected_half_range_wl_auto_ssp__ = 50
__selected_half_range_sysvel_auto_ssp__ = 100

# Cosmological parameters
__Hubble_constant__ = 71  # km/(s Mpc)
__Omega_matter__ = 0.27
__Omega_Lambda__ = 0.73

__Ha_central_wl__ = 6562.68

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
_INDICES_POS = {
    'OL1': 0,
    'OL2': 1,
    'OLb1': 2,
    'OLb2': 3,
    'OLr1': 4,
    'OLr2': 5,
    'OLb': 6,
    'OLr': 7
}
for k, v in __indices__.items():
    OLb1 = __indices__[k][_INDICES_POS['OLb1']]
    OLb2 = __indices__[k][_INDICES_POS['OLb2']]
    OLr1 = __indices__[k][_INDICES_POS['OLr1']]
    OLr2 = __indices__[k][_INDICES_POS['OLr2']]
    OLb = (OLb1 + OLb2)/2
    OLr = (OLr1 + OLr2)/2
    __indices__[k].append(OLb)
    __indices__[k].append(OLr)

_SSP_OUTPUT_INDEX = {
    'chi_joint': 0,
    'age_min': 1, 'e_age_min': 2,
    'met_min': 3, 'e_met_min': 4,
    'AV_min': 5, 'e_AV_min': 6,
    'redshift': 7, 'e_redshift': 8,
    'sigma': 9, 'e_sigma': 10,
    'FLUX': 11, 'redshift2': 12,
    'med_flux': 13, 'rms': 14,
    'age_min_mass':15 , 'e_age_min_mass': 16,
    'met_min_mass':17 , 'e_met_min_mass': 18,
    'systemic_velocity': 19, 'lml': 20,
    'lmass': 21,
}

# plot constants
_plot_dpi = 300
_latex_ppi = 72.0
_latex_column_width_pt = 240.0
_latex_column_width = _latex_column_width_pt/_latex_ppi
# latex_column_width = latex_column_width_pt/latex_ppi/1.4
_latex_text_width_pt = 504.0
_latex_text_width = _latex_text_width_pt/_latex_ppi
_golden_mean = 0.5 * (1. + 5**0.5)
_figsize_default = (2*_latex_column_width, 2*_latex_column_width/_golden_mean)
