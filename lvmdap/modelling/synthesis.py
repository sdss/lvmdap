
import io
from astropy.table import Table
import os
import sys
import numpy as np

from pyFIT3D.common.constants import __selected_extlaw__, __selected_R_V__

from pyFIT3D.common.stats import calc_chi_sq, smooth_ratio
from pyFIT3D.common.io import print_verbose
from pyFIT3D.common.io import plot_spectra_ax
from pyFIT3D.modelling.stellar import StPopSynt
#from lvmdap.modelling.synthesis import StellarSynthesis as StPopSynt


from lvmdap.modelling.ingredients import StellarModels
from pyFIT3D.common.gas_tools import fit_elines_main
from pyFIT3D.common.io import plot_spectra_ax, array_to_fits, write_img_header
from pyFIT3D.common.io import sel_waves, trim_waves, print_verbose, get_wave_from_header
from pyFIT3D.common.stats import pdl_stats, _STATS_POS, WLS_invmat, median_box, median_filter
from pyFIT3D.common.stats import calc_chi_sq, smooth_ratio, shift_convolve, hyperbolic_fit_par
from pyFIT3D.common.constants import __c__, _MODELS_ELINE_PAR, __mask_elines_window__, __selected_R_V__
from pyFIT3D.common.constants import __selected_half_range_sysvel_auto_ssp__, _figsize_default, _plot_dpi
from pyFIT3D.common.constants import __sigma_to_FWHM__, __selected_extlaw__, __selected_half_range_wl_auto_ssp__
from copy import deepcopy

class StellarSynthesis(StPopSynt):
    def __init__(self, config, wavelength, flux, eflux, ssp_file, out_file,
                 ssp_nl_fit_file=None, sigma_inst=None, min=None, max=None,
                 w_min=None, w_max=None, nl_w_min=None, nl_w_max=None,
                 elines_mask_file=None, mask_list=None,
                 R_V=3.1, extlaw='CCM', spec_id=None, guided_errors=None,
                 plot=None, verbose=False, ratio_master=None, fit_gas=True):
        """
        Instantiates :class: `StPopSynt`.
        Reads the config file and SSP models. Creates all wavelength masks and
        a spectra dictionary used through the fit.

        Parameters
        ----------
        config : ConfigAutoSSP class
            The class which configures the whole SSP fit process.

        wavelength : array like
            Observed wavelengths.

        flux : array like
            Observed flux.

        eflux : array like
            Error in observed flux.

        ssp_file : str
            Path to the SSP models fits file.

        out_file : str
            File to outputs the result.

        ssp_nl_fit_file : str, optional
            Path to the SSP models fits file used in the non-linear round of fit.
            The non-linear procedure search for the kinematics parameters (redshift
            and sigma) and the dust extinction (AV).
            Defaults to None, i.e., `self.ssp_nl_fit` is equal to `self.ssp`.

        sigma_inst : float, optional
            Instrumental dispersion. Defaults to None.

        w_min : int, optional

        w_max : int, optional

        nl_w_min : int, optional

        nl_w_max : int, optional

        elines_mask_file : str, optional

        mask_list : str, optional

        R_V : float, optional
            Selective extinction parameter (roughly "slope"). Default value 3.1.

        extlaw : str {'CCM', 'CAL'}, optional
            Which extinction function to use.
            CCM will call `Cardelli_extlaw`.
            CAL will call `Calzetti_extlaw`.
            Default value is CCM.

        spec_id : int or tuple, optional
            Used only for cube or rss fit. Defaults to None.

            ..see also::

                :func:`pyFIT3D.common.auto_ssp_tools.auto_ssp_elines_rnd_rss_main`.

        guided_errors : array like, optional
            Input the errors in non linear fit. Defaults to None.
            TODO: EL: This option should be moved to `StPopSynt.non_linear_fit`.

        plot : int, optional
            Plots the fit. Defaults to 0.

        verbose : bools, optional
            If True produces a nice text output.

        ratio_master : bool, optional

        fit_gas : bool, optional

        """
        self.spec_id = spec_id
        self.fit_gas = fit_gas
        self.config = config

        self.verbose = verbose
        self.R_V = __selected_R_V__ if R_V is None else R_V
        self.extlaw =  __selected_extlaw__ if extlaw is None else extlaw
        self.n_loops_nl_fit = 0

#        self.wavelength = self.get_wavelength()
#        self.normalization_wavelength = self.get_normalization_wavelength()
        
        self.sigma_inst = sigma_inst
        self.sigma_mean = None
        self.filename = ssp_file
        self.filename_nl_fit = ssp_file if ssp_nl_fit_file is None else ssp_nl_fit_file
        self.out_file = out_file
        self.plot = 0 if plot is None else plot
        self.guided_errors = guided_errors
        self.spectra = None
        self._load_masks(w_min, w_max, nl_w_min, nl_w_max, mask_list, elines_mask_file)

        if self.verbose:
            self._greet()

        self._create_spectra_dict(wavelength, flux, eflux, min, max, ratio_master)

        # load SSP FITS File
        # Everytime you do a fit, you load the SSP fits files (!)
        # This is very slow!
        self._load_ssp_fits()

        # Not working right now!
        # Multi AVs paliative solution:
        # all the process should assume a different AV for
        # each SSP models.
        self._multi_AV()
        self._fitting_init()
        self.ssp_init()

    def _load_ssp_fits(self):
        self.models = StellarModels(self.filename)

        # deprecated the use of self.ssp.
        # in order to keep working the code at first instance:
        self.ssp = self.models

        if self.filename_nl_fit:
            self.models_nl_fit = StellarModels(self.filename_nl_fit)
            # deprecated the use of self.ssp_nl_fit
            # in order to keep working the code at first instance:
            self.ssp_nl_fit = self.models_nl_fit

    def rsp_single_fit(self):
        cf = self.config
        models = self.models

        print_verbose('', verbose=self.verbose)
        print_verbose('-----------------------------------------------------------------------', verbose=self.verbose)
        print_verbose('--[ BEGIN non-linear fit ]---------------------------------------------', verbose=self.verbose)

        sigma_inst = self.sigma_inst
        sigma = self.best_sigma
        redshift = self.best_redshift
        AV = self.best_AV
        R_V = self.R_V
        extlaw = self.extlaw
        wavelength = self.spectra['raw_wave']

        models.to_observed(wavelength, sigma_inst=sigma_inst, sigma=sigma, redshift=redshift)
        models.apply_dust_to_flux_models_obsframe(wavelength/(1 + redshift), AV, R_V=R_V, extlaw=extlaw)
        flux_models_obsframe_dust = models.flux_models_obsframe_dust

        half_norm_range = 45  # Angstrom
        l_wave = models.wavenorm - half_norm_range
        r_wave = models.wavenorm + half_norm_range
        self.spectra['sel_norm_window'] = (self.spectra['raw_wave'] > l_wave) & (self.spectra['raw_wave'] < r_wave)

        sel_norm = self.spectra['sel_norm_window'] & self.spectra['sel_wl']
        norm_mean_flux = self.spectra['raw_flux_no_gas'][sel_norm].mean()
        norm_median_flux = np.median(self.spectra['raw_flux_no_gas'][sel_norm])
        self.spectra['raw_flux_no_gas_norm_mean'] = np.divide(self.spectra['raw_flux_no_gas'], norm_mean_flux, where=norm_mean_flux!=0)
        self.spectra['raw_eflux_norm_mean'] = np.divide(self.spectra['raw_eflux'], norm_mean_flux, where=norm_mean_flux!=0)
        self.spectra['raw_flux_no_gas_norm_median'] = np.divide(self.spectra['raw_flux_no_gas'], norm_median_flux, where=norm_median_flux!=0)
        self.spectra['raw_eflux_norm_median'] = np.divide(self.spectra['raw_eflux'], norm_median_flux, where=norm_median_flux!=0)

        chi_sq_mean = []
        chi_sq_median = []
        coeffs_single=[]
        for M__w in flux_models_obsframe_dust:
            _chi_sq, _ = calc_chi_sq(f_obs=self.spectra['raw_flux_no_gas_norm_mean'], f_mod=M__w, ef_obs=self.spectra['raw_eflux_norm_mean'])
            chi_sq_mean.append(_chi_sq)
            _chi_sq, _ = calc_chi_sq(f_obs=self.spectra['raw_flux_no_gas_norm_median'], f_mod=M__w, ef_obs=self.spectra['raw_eflux_norm_median'])
            chi_sq_median.append(_chi_sq)
            coeffs_single.append(0.0)
        coeffs_single=np.array(coeffs_single)
        chi_sq_mean = np.array(chi_sq_mean)
        chi_sq_median = np.array(chi_sq_median)
        print(f'len_chi_sq_mean={len(chi_sq_mean)}')
        chi_sq_mean_norm = chi_sq_mean / chi_sq_mean.sum()
        chi_sq_median_norm = chi_sq_median / chi_sq_median.sum()

        # f_out_coeffs = open(filename, 'a')
        cols = 'ID,TEFF,LOGG,MET,ALPHAM,MEAN(CHISQ),MEDIAN(CHISQ)'
        # cols_out_coeffs = cols.replace(',', '\t')
        # print(f'#{cols_out_coeffs}', file=f_out_coeffs)
        fmt_cols = '| {0:^4} | {1:^7} | {2:^7} | {3:^6} | {3:^6} | {4:^11} | {5:^13} |'
        fmt_numbers = '| {:=04d} | {:=7.4f} | {:=7.4f} | {:=6.4f} | {:=6.4f} | {:=11.4f} | {:=13.4f} |'
        # fmt_numbers_out_coeffs = '{:=04d},{:=7.4f},{:=6.4f},{:=6.4f},{:=9.4f},{:=8.4f},{:=4.2f},{:=7.4f},{:=9.4f},{:=6.4f},{:=6.4f}'
        fmt_numbers_out_coeffs = '{:=04d}\t{:=7.4f}\t{:=7.4f}\t{:=6.4f}\t{:=6.4f}\t{:=6.4f}\t{:=6.4f}'

        cols_split = cols.split(',')
        tbl_title = fmt_cols.format(*cols_split)
        ntbl = len(tbl_title)
        tbl_border = ntbl*'-'
        # output coeffs table
        print(tbl_border)
        print(tbl_title)
        print(tbl_border)

        i_C_mean_min = chi_sq_mean_norm.argmin()
        i_C_median_min = chi_sq_mean_norm.argmin()

        model_min=flux_models_obsframe_dust[i_C_median_min]*norm_median_flux
        coeffs_single[i_C_median_min]=1.0
        self.coeffs_ssp_MC=coeffs_single
        
        self.spectra['model_ssp_min']=model_min
        self.spectra['model_min']=model_min


        
        #res_ssp = s['raw_flux_no_gas'] - model_ssp_min
        # smooth model_ssp_min (not tested)
        #ratio = np.divide(s['raw_flux_no_gas'], model_ssp_min, where=model_ssp_min!=0)
        #ratio = np.where(np.isfinite(ratio), ratio, 0)
        #ratio = np.where(model_ssp_min == 0, 0, ratio)
        #sm_rat = smooth_ratio(ratio, int(self.sigma_mean))
       # model_ssp_min *= sm_rat
       # model_joint = model_ssp_min + s['raw_model_elines']
        # _rat = (s['orig_flux_ratio']/self.ratio_master)
        # _rat = np.where(self.ratio_master > 0, _rat, s['orig_flux_ratio'])
        #res_joint = (res_ssp - s['raw_model_elines'])

        #s['model_joint'] = model_joint
        #s['res_joint'] = res_joint
        #s['res_ssp'] = res_ssp
        #s['res_ssp_no_corr'] = s['orig_flux'] - s['model_ssp_min_uncorr']
     
        tbl_row = []
        tbl_row.append(i_C_mean_min)
        tbl_row.append(self.ssp.teff_models[i_C_mean_min])
        tbl_row.append(self.ssp.logg_models[i_C_mean_min])
        tbl_row.append(self.ssp.meta_models[i_C_mean_min])
        tbl_row.append(self.ssp.alph_models[i_C_mean_min])
        tbl_row.append(chi_sq_mean_norm[i_C_mean_min])  # a_coeffs_N
        tbl_row.append(chi_sq_median_norm[i_C_median_min])  # a_min_coeffs
        print(fmt_numbers.format(*tbl_row))
        if (i_C_median_min != i_C_mean_min):
            tbl_row = []
            tbl_row.append(i_C_median_min)
            tbl_row.append(self.ssp.teff_models[i_C_median_min])
            tbl_row.append(self.ssp.logg_models[i_C_median_min])
            tbl_row.append(self.ssp.meta_models[i_C_median_min])
            tbl_row.append(self.ssp.alph_models[i_C_median_min])
            tbl_row.append(chi_sq_mean_norm[i_C_mean_min])  # a_coeffs_N
            tbl_row.append(chi_sq_median_norm[i_C_median_min])  # a_min_coeffs
            print(fmt_numbers.format(*tbl_row))

        self.output_table = []
        for i in range(models.n_models):
            C_mean = chi_sq_mean_norm[i]
            C_median = chi_sq_median_norm[i]
            tbl_row = []
            tbl_row.append(i)
            tbl_row.append(self.ssp.teff_models[i])
            tbl_row.append(self.ssp.logg_models[i])
            tbl_row.append(self.ssp.meta_models[i])
            tbl_row.append(self.ssp.alph_models[i])
            tbl_row.append(C_mean)  # a_coeffs_N
            tbl_row.append(C_median)  # a_min_coeffs
            self.output_table.append(tbl_row)
        print(tbl_border)

        return chi_sq_median_norm[i_C_median_min]

    def rsp_fit(self, n_MC=5):
        """
        Generates minimal ssp model through a Monte-Carlo search of the coefficients.
        I.e. go through fit_WLS_invmat() `n_MC` times (using fit.WLS_invmat_MC).

        Parameters
        ----------
        n_MC : int
            Number of Monte-Carlos loops. Default value is 20.

        See also
        --------
        fit_WLS_invmat() and fit_WLS_invmat_MC()

        """
        ssp = self.models
        s = self.spectra
        # XXX: Should be fit_WLS_invmat with Monte-Carlo
        #   2020-01-16: EADL@laptop-XPS13-9333
        #       DONE
                    # coeffs_now, chi_sq, msk_model_min = self.fit_WLS_invmat(ssp=ssp)
        coeffs_MC, chi_sq_MC, models_MC = self.fit_WLS_invmat_MC(ssp=ssp, n_MC=n_MC)
        #   2020-01-16: EADL@laptop-XPS13-9333
        #       21:46 - PASSED WITHOUT ERRORS FOR THE FIRST TIME

        for i in range(n_MC):
            self.update_ssp_parameters(coeffs=coeffs_MC[i], chi_sq=chi_sq_MC[i])

        # first_model = copy(models_MC[-1])

        chi_sq = self._calc_coeffs_MC(ssp, coeffs_MC, chi_sq_MC, models_MC)

        coeffs_now = self.coeffs_ssp_MC
        print_verbose(f'coeffs_now ------------------ {coeffs_now}', verbose=self.verbose)

        model_ssp_min = s['model_ssp_min']
        res_ssp = s['raw_flux_no_gas'] - model_ssp_min

        # smooth model_ssp_min (not tested)
        ratio = np.divide(s['raw_flux_no_gas'], model_ssp_min, where=model_ssp_min!=0)
        ratio = np.where(np.isfinite(ratio), ratio, 0)
        ratio = np.where(model_ssp_min == 0, 0, ratio)
        sm_rat = smooth_ratio(ratio, int(self.sigma_mean))
        model_ssp_min *= sm_rat
        model_joint = model_ssp_min + s['raw_model_elines']
        # _rat = (s['orig_flux_ratio']/self.ratio_master)
        # _rat = np.where(self.ratio_master > 0, _rat, s['orig_flux_ratio'])
        res_joint = (res_ssp - s['raw_model_elines'])

        s['model_joint'] = model_joint
        s['res_joint'] = res_joint
        s['res_ssp'] = res_ssp
        s['res_ssp_no_corr'] = s['orig_flux'] - s['model_ssp_min_uncorr']

        self._MC_averages()
        return chi_sq

    def _MC_averages(self):
        """
        Calc. of the mean age, metallicity and AV weighted by light and mass.
        """
        ssp = self.models

        coeffs_input_zero = self.coeffs_input_MC
        _coeffs = self.coeffs_ssp_MC
        # print(_coeffs)
        norm = _coeffs.sum()
        _coeffs_norm = _coeffs/norm
        _sigma = self.coeffs_ssp_MC_rms
        _sigma_norm = np.divide(_sigma*_coeffs_norm, _coeffs, where=_coeffs > 0.0, out=np.zeros_like(_coeffs))
        _min_coeffs = self.orig_best_coeffs
        _min_coeffs_norm = _min_coeffs/norm
        # _sigma_norm = np.where(_coeffs > 0, _sigma*(_coeffs_norm/_coeffs), 0)

        self.final_AV = np.array([self.best_AV]*len(_coeffs), dtype='float')
        l_teff_min = np.dot(_coeffs, ssp.teff_models)
        l_logg_min = np.dot(_coeffs, ssp.logg_models)
        l_meta_min = np.dot(_coeffs, ssp.meta_models)
        l_alph_min = np.dot(_coeffs, ssp.alph_models)
        l_AV_min = np.dot(_coeffs, self.final_AV)
        l_teff_min_mass = np.dot(_coeffs_norm*ssp.mass_to_light, ssp.teff_models)
        l_logg_min_mass = np.dot(_coeffs_norm*ssp.mass_to_light, ssp.logg_models)
        l_meta_min_mass = np.dot(_coeffs_norm*ssp.mass_to_light, ssp.meta_models)
        l_alph_min_mass = np.dot(_coeffs_norm*ssp.mass_to_light, ssp.alph_models)
        l_AV_min_mass = np.dot(_coeffs_norm*ssp.mass_to_light, self.final_AV)
        e_l_teff_min = np.dot(_sigma, ssp.teff_models)
        e_l_logg_min = np.dot(_sigma, ssp.logg_models)
        e_l_meta_min = np.dot(_sigma, ssp.meta_models)
        e_l_alph_min = np.dot(_sigma, ssp.alph_models)
        e_l_AV_min = np.dot(_sigma, self.final_AV)
        e_l_teff_min_mass = np.dot(_sigma_norm*ssp.mass_to_light, ssp.teff_models)
        e_l_logg_min_mass = np.dot(_sigma_norm*ssp.mass_to_light, ssp.logg_models)
        e_l_meta_min_mass = np.dot(_sigma_norm*ssp.mass_to_light, ssp.meta_models)
        e_l_alph_min_mass = np.dot(_sigma_norm*ssp.mass_to_light, ssp.alph_models)
        e_l_AV_min_mass = np.dot(_sigma_norm*ssp.mass_to_light, self.final_AV)




        
        self.mass_to_light = np.dot(ssp.mass_to_light, _coeffs_norm)
        self.teff_min = 10**l_teff_min
        self.logg_min = l_logg_min
        self.meta_min = l_meta_min
        self.alph_min = l_alph_min
        self.AV_min = l_AV_min
        if self.mass_to_light == 0:
            self.mass_to_light = 1
        self.teff_min_mass = 10**(l_teff_min_mass/self.mass_to_light)
        self.logg_min_mass = (l_logg_min_mass/self.mass_to_light)
        self.meta_min_mass = (l_meta_min_mass/self.mass_to_light)
        self.alph_min_mass = (l_alph_min_mass/self.mass_to_light)
        self.AV_min_mass = l_AV_min_mass/self.mass_to_light
        self.e_teff_min = np.abs(0.43*e_l_teff_min*self.teff_min)
        self.e_logg_min = np.abs(0.43*e_l_teff_min*self.logg_min)
        self.e_meta_min = np.abs(0.43*e_l_teff_min*self.meta_min)
        self.e_alph_min = np.abs(0.43*e_l_teff_min*self.alph_min)
        self.e_AV_min = np.abs(0.43*e_l_AV_min*self.AV_min)
        self.e_teff_min_mass = np.abs(0.43*e_l_teff_min*self.teff_min_mass)
        self.e_logg_min_mass = np.abs(0.43*e_l_logg_min*self.logg_min_mass)
        self.e_meta_min_mass = np.abs(0.43*e_l_meta_min*self.meta_min_mass)
        self.e_alph_min_mass = np.abs(0.43*e_l_alph_min*self.alph_min_mass)
        self.e_AV_min_mass = np.abs(0.43*e_l_AV_min*self.AV_min_mass)

    def output_coeffs_MC_to_screen(self):
        cols = 'ID,TEFF,LOGG,META,ALPHAM,COEFF,Min.Coeff,log(M/L),AV,N.Coeff,Err.Coeff'
        fmt_cols = '| {0:^4} | {1:^7} | {2:^7} | {3:^7} | {4:^7} | {5:^6} | {6:^9} | {7:^8} | {8:^4} | {9:^7} | {10:^9} |'
        fmt_numbers = '| {:=04d} | {:=7.4f} | {:=7.4f} | {:=7.4f} | {:=7.4f} | {:=6.4f} | {:=9.4f} | {:=8.4f} | {:=4.2f} | {:=7.4f} | {:=9.4f} | {:=6.4f} | {:=6.4f}'
        cols_split = cols.split(',')
        tbl_title = fmt_cols.format(*cols_split)
        ntbl = len(tbl_title)
        tbl_border = ntbl*'-'
        print(tbl_border)
        print(tbl_title)
        print(tbl_border)
        table = Table(names=cols_split)
        for i in range(self.ssp.n_models):
            try:
                C = self.coeffs_ssp_MC[i]
            except (IndexError,TypeError):
                C = 0
            if np.isnan(C):
                C = 0
            if C < 1e-5:
                continue
        # for i, C in enumerate(_coeffs):
            tbl_row = []
            tbl_row.append(i)
            tbl_row.append(self.ssp.teff_models[i])
            tbl_row.append(self.ssp.logg_models[i])
            tbl_row.append(self.ssp.meta_models[i])
            tbl_row.append(self.ssp.alph_models[i])
            tbl_row.append(self.coeffs_norm[i])  # a_coeffs_N
            tbl_row.append(self.min_coeffs_norm[i])  # a_min_coeffs
            tbl_row.append(np.log10(self.ssp.mass_to_light[i]))
            tbl_row.append(self.best_AV)
            tbl_row.append(C)  # a_coeffs
            tbl_row.append(self.coeffs_ssp_MC_rms[i])  # a_e_coeffs
            table.add_row(tbl_row)
            tbl_row.append(self.coeffs_input_MC[i])  # ???
            tbl_row.append(self.coeffs_input_MC_rms[i])  # ???
            print(fmt_numbers.format(*tbl_row))

        print(tbl_border)

        return table

    def output_coeffs_MC(self, filename, write_header=True):
        """
        Outputs the SSP coefficients table to the screen and to the output file `filename`.

        Parameters
        ----------
        filename : str
            The output filename to the coefficients table.
        """

        if isinstance(filename, io.TextIOWrapper):
            f_out_coeffs = filename
        else:
            f_out_coeffs = open(filename, 'a')

        cols = 'ID,TEFF,LOGG,META,ALPHAM,COEFF,Min.Coeff,log(M/L),AV,N.Coeff,Err.Coeff'
        fmt_cols = '| {0:^4} | {1:^7} | {2:^7} | {3:^7} | {4:^7} | {5:^6} | {6:^9} | {7:^8} | {8:^4} | {9:^7} | {10:^9} |'
        fmt_numbers = '| {:=04d} | {:=7.4f} | {:=7.4f} | {:=7.4f} | {:=7.4f} | {:=6.4f} | {:=9.4f} | {:=8.4f} | {:=4.2f} | {:=7.4f} | {:=9.4f} | {:=6.4f} | {:=6.4f}'
        fmt_numbers_out_coeffs = ' {:=04d}  {:=7.4f}  {:=7.4f}  {:=7.4f}  {:=7.4f}  {:=6.4f}  {:=9.4f}  {:=8.4f}  {:=4.2f}  {:=7.4f}  {:=9.4f}'

        cols_split = cols.split(',')
        tbl_title = fmt_cols.format(*cols_split)
        ntbl = len(tbl_title)
        tbl_border = ntbl*'-'
        if write_header:
            cols_out_coeffs = tbl_title.replace("|", "")
            print(f'#{cols_out_coeffs}', file=f_out_coeffs)

        # output coeffs table
        coeffs_input_zero = self.coeffs_input_MC
        coeffs_rms = self.coeffs_input_MC_rms
        _coeffs = self.coeffs_ssp_MC
        _sigma = self.coeffs_ssp_MC_rms
        _min_coeffs = self.orig_best_coeffs
        norm = _coeffs.sum()
        if norm == 0:
            norm = 1
        _coeffs_norm = _coeffs/norm
        _min_coeffs_norm = _min_coeffs/norm
        # _sigma_norm = np.divide(_sigma*_coeffs_norm, _coeffs, where=_coeffs > 0.0, out=np.zeros_like(_coeffs))

        # if not ((_coeffs == 0).all() or ((np.isfinite(_coeffs)).sum() == 0)):
        print(tbl_border)
        print(tbl_title)
        print(tbl_border)
        for i in range(self.ssp.n_models):
            try:
                C = _coeffs[i]
            except (IndexError,TypeError):
                C = 0
            if np.isnan(C):
                C = 0
        # for i, C in enumerate(_coeffs):
            tbl_row = []
            tbl_row.append(i)
            tbl_row.append(self.ssp.teff_models[i])
            tbl_row.append(self.ssp.logg_models[i])
            tbl_row.append(self.ssp.meta_models[i])
            tbl_row.append(self.ssp.alph_models[i])
            tbl_row.append(_coeffs_norm[i])  # a_coeffs_N
            tbl_row.append(_min_coeffs_norm[i])  # a_min_coeffs
            tbl_row.append(np.log10(self.ssp.mass_to_light[i]))
            tbl_row.append(self.best_AV)
            tbl_row.append(C)  # a_coeffs
            tbl_row.append(_sigma[i])  # a_e_coeffs
            print(fmt_numbers_out_coeffs.format(*tbl_row), file=f_out_coeffs)
            if C > 1e-5:
                tbl_row.append(coeffs_input_zero[i])  # ???
                tbl_row.append(coeffs_rms[i])  # ???
                print(fmt_numbers.format(*tbl_row))
        print(tbl_border)

        if not isinstance(filename, io.TextIOWrapper):
            f_out_coeffs.close()

    def _print_header(self, filename):
        """
        Writes the main output file header.

        Parameters
        ----------
        filename : str
            Output filename.
        """
        cf = self.config
        wavenorm = self.ssp.wavenorm

        if isinstance(filename, io.TextIOWrapper):
            f_outfile = filename
        else:
            f_outfile = open(filename, 'a')
        #  outbuf = f'{chi_joint},'
        #outbuf = f'{outbuf}{self.teff_min},{self.e_teff_min},{self.logg_min},{self.e_logg_min},{self.meta_min},'
        #outbuf = f'{outbuf}{self.e_meta_min},{self.alph_min},{self.e_alph_min},'
        #outbuf = f'{outbuf}{self.AV_min},{self.e_AV_min},{self.best_redshift},{self.e_redshift},'
        #outbuf = f'{outbuf}{self.best_sigma},{self.e_sigma},{FLUX},{self.best_redshift},'
        #outbuf = f'{outbuf}{med_flux},{rms},{self.teff_min_mass},{self.e_teff_min_mass},{self.logg_min_mass},{self.e_logg_min_mass},'
        #outbuf = f'{outbuf}{self.meta_min_mass},{self.e_meta_min_mass},{self.alph_min_mass},{self.e_alph_min_mass},{self.systemic_velocity},'
        #outbuf = f'{outbuf}{lml},{lmass}'
        print(f'# (1) MIN_CHISQ', file=f_outfile)
        print(f'# (2) Teff', file=f_outfile)
        print(f'# (3) e_Teff', file=f_outfile)
        print(f'# (4) Log_g', file=f_outfile)
        print(f'# (5) e_Log_g', file=f_outfile)
        print(f'# (6) Fe', file=f_outfile)
        print(f'# (7) e_Fe', file=f_outfile)
        print(f'# (8) alpha', file=f_outfile)
        print(f'# (9) e_alpha', file=f_outfile)
        print(f'# (10) Av', file=f_outfile)
        print(f'# (11) e_Av', file=f_outfile)
        print(f'# (12) z', file=f_outfile)
        print(f'# (13) e_z', file=f_outfile)
        print(f'# (14) disp', file=f_outfile)
        print(f'# (15) e_disp', file=f_outfile)
        print(f'# (16) flux', file=f_outfile)
        print(f'# (17) redshift', file=f_outfile)
        print(f'# (18) med_flux', file=f_outfile)
        print(f'# (19) e_med_flux', file=f_outfile)
        print(f'# (20) Teff_MW', file=f_outfile)
        print(f'# (21) e_Teff_MW', file=f_outfile)
        print(f'# (22) Log_g_MW', file=f_outfile)
        print(f'# (23) e_Log_g_MW', file=f_outfile)
        print(f'# (24) Fe_MW', file=f_outfile)
        print(f'# (25) e_Fe_MW', file=f_outfile)
        print(f'# (26) alpha_MW', file=f_outfile)
        print(f'# (27) e_alpha_MW', file=f_outfile)
        print(f'# (28) sys_vel', file=f_outfile)
        print(f'# (29) log_ML', file=f_outfile)
        print(f'# (30) log_Mass', file=f_outfile)
        if not isinstance(filename, io.TextIOWrapper):
            f_outfile.close()


            
    def output(self, filename, write_header=True, block_plot=True):
        """
        Summaries the run in a csv file.

        Parameters
        ----------
        filename : str
            Output filename.
        """
        cf = self.config
        ssp = self.ssp
        s = self.spectra
        model_joint = s['model_joint']
        res_joint = s['res_joint']
        # if chi_sq_msk > 0:
        #     delta_chi = np.abs()
        spectra_list = [
            s['orig_flux_ratio'],
            s['model_ssp_min'],
            s['model_joint'],
            s['orig_flux_ratio'] - s['model_ssp_min_uncorr'],
            s['res_joint'],
            s['orig_flux_ratio'] - (s['res_ssp'] - s['res_joint']),
        ]

        # chi joint
        _chi = np.nansum((model_joint[s['sel_wl']] - s['msk_flux'])**2/s['msk_eflux']**2)
        n_wave_orig = len(s['sel_wl'])
        # print(f'n_wave_orig:{n_wave_orig}')
        # print(f'ssp.flux_models_obsframe_dust.shape[0]:{ssp.flux_models_obsframe_dust.shape[0]}')
        # print(f'self.n_models_elines:{self.n_models_elines}')
        chi_joint = _chi/(n_wave_orig-ssp.n_models-self.n_models_elines-1)

        if self.plot:
            if 'matplotlib.pyplot' not in sys.modules:
                from matplotlib import pyplot as plt
            else:
                plt = sys.modules['matplotlib.pyplot']

            labels = [
                'orig_flux_ratio',
                'model_min',
                'model_joint',
                'orig_flux_ratio - model_min_uncorr',
                'res_joint',
                'orig_flux_ratio - (res_min - res_joint)'
            ]
            plt.cla()
            title = f'X={chi_joint:.4f} T={self.teff_min:.4f} ({self.teff_min_mass:.4f})'
            title = f'{title} G={self.logg_min:.4f} ({self.logg_min_mass:.4f})'
            title = f'{title} Z={self.meta_min:.4f} ({self.meta_min_mass:.4f})'
            title = f'{title} A={self.alph_min:.4f} ({self.alph_min_mass:.4f})'
            wave_list = [s['raw_wave']]*len(spectra_list)
            plot_spectra_ax(plt.gca(), wave_list, spectra_list, title=title, labels_list=labels)
            plt.pause(0.001)
            plt.show(block=block_plot)

        rms = res_joint[s['sel_norm_window']].std()
        med_flux = np.median(s['raw_flux_no_gas'][s['sel_norm_window']])
        FLUX = s['orig_flux'][s['sel_wl']].sum()

        # XXX:
        #   Why a 3500 factor?
        mass = self.mass_to_light*med_flux*3500
        if mass > 0:
            lmass = np.log10(mass)
            lml = np.log10(self.mass_to_light/3500)
        else:
            lmass = 0
            lml = 0

        report_vals = f'MSP CHISQ={chi_joint} TEFF={self.teff_min}+-{self.e_teff_min}'
        report_vals = f'{report_vals} LOGG={self.logg_min}+-{self.e_logg_min}'
        report_vals = f'{report_vals} META={self.meta_min}+-{self.e_meta_min}'
        report_vals = f'{report_vals} ALPHAM={self.alph_min}+-{self.e_alph_min}'
        report_vals = f'{report_vals} AV={self.AV_min}+-{self.e_AV_min}'
        report_vals = f'{report_vals} REDSHIFT={self.best_redshift}+-{self.e_redshift}'
        report_vals = f'{report_vals} SIGMA_DISP_km_s={self.best_sigma}+-{self.e_sigma}'
        report_vals = f'{report_vals} RMS={rms} MED_FLUX={med_flux}'
        report_vals = f'{report_vals} TEFF_mass={self.teff_min_mass}+-{self.e_teff_min_mass}'
        report_vals = f'{report_vals} LOGG_mass={self.logg_min_mass}+-{self.e_logg_min_mass}'
        report_vals = f'{report_vals} META_mass={self.meta_min_mass}+-{self.e_meta_min_mass}'
        report_vals = f'{report_vals} ALPHAM_mass={self.alph_min_mass}+-{self.e_alph_min_mass}'
        report_vals = f'{report_vals} MASS={mass} log_M/L={lml} log_Mass={lmass}'
        print(report_vals)

        print('--------------------------------------------------------------');

        if isinstance(filename, io.TextIOWrapper):
            if write_header:
                self._print_header(filename)
            f_outfile = filename
        else:
            if not os.path.exists(filename):
                self._print_header(filename)
            f_outfile = open(filename, 'a')


        outbuf = f'{chi_joint},'
        outbuf = f'{outbuf}{self.teff_min},{self.e_teff_min},{self.logg_min},{self.e_logg_min},{self.meta_min},'
        outbuf = f'{outbuf}{self.e_meta_min},{self.alph_min},{self.e_alph_min},'
        outbuf = f'{outbuf}{self.AV_min},{self.e_AV_min},{self.best_redshift},{self.e_redshift},'
        outbuf = f'{outbuf}{self.best_sigma},{self.e_sigma},{FLUX},{self.best_redshift},'
        outbuf = f'{outbuf}{med_flux},{rms},{self.teff_min_mass},{self.e_teff_min_mass},{self.logg_min_mass},{self.e_logg_min_mass},'
        outbuf = f'{outbuf}{self.meta_min_mass},{self.e_meta_min_mass},{self.alph_min_mass},{self.e_alph_min_mass},{self.systemic_velocity},'
        outbuf = f'{outbuf}{lml},{lmass}'
        print(f'{outbuf}', file=f_outfile)

        if not isinstance(filename, io.TextIOWrapper):
            f_outfile.close()

        
    def non_linear_fit_rsp(self, guide_sigma=False, fit_sigma_rnd=True,
                       sigma_rnd_medres_merit=False, correct_wl_ranges=False):
    # def non_linear_fit(self, guided_sigma=None):
        """
        Do the non linear fit in order to find the kinematics parameters and the dust
        extinction. This procedure uses the set of SSP models, `self.models_nl_fit`.
        At the end will set the first entry to the ssp fit chain with the coefficients
        for the best model after the non-linear fit.
        """
        cf = self.config
        ssp = self.models_nl_fit
        s = self.spectra
           # if guided_sigma is not None: self.best_sigma = guided_sigma

        print_verbose('', verbose=self.verbose)
        print_verbose('-----------------------------------------------------------------------', verbose=self.verbose)
        print_verbose('--[ BEGIN non-linear fit ]---------------------------------------------', verbose=self.verbose)
        coeffs_now, chi_sq, msk_model_min = self.fit_WLS_invmat(ssp=ssp, sel_wavelengths=s['sel_AV'])
        # TODO: check mask from med_flux in perl version
        msk_flux = s['raw_flux'][s['sel_AV']]
        med_flux = np.median(msk_flux)
        print_verbose('-[ non-linear fit report ]-----------------------------', verbose=self.verbose)
        if self.guided_errors is not None:
            self.best_redshift = cf.redshift
            self.e_redshift = self.guided_errors[0]
            print_verbose(f'- Redshift: {self.best_redshift:.8f} +- {self.e_redshift:.8f}', verbose=True)
            self.best_sigma = cf.sigma
            self.e_sigma = self.guided_errors[1]
            print_verbose(f'- Sigma:    {self.best_sigma:.8f} +- {self.e_sigma:.8f}', verbose=True)
            self.best_AV = cf.AV
            self.e_AV = self.guided_errors[2]
            print_verbose(f'- AV:       {self.best_AV:.8f} +- {self.e_AV:.8f}', verbose=True)
            # self.correct_elines_mask(redshift=self.best_redshift)
            self.redshift_correct_masks(redshift=self.best_redshift, correct_wl_ranges=correct_wl_ranges)
            if self.sigma_inst is None:
                self.sigma_mean = self.best_sigma
            else:
                self.sigma_mean = np.sqrt(self.sigma_inst**2 + (5000*self.best_sigma/__c__)**2)
        else:
#            print(f'Deriving redshift, sigma, AV...')
            # self.update_redshift_params(coeffs=coeffs_now, chi_sq=chi_sq, redshift=self.best_redshift)
            # self.best_chi_sq_redshift = self.get_last_chi_sq_redshift()
            # self.best_coeffs_redshift = self.get_last_coeffs_redshift()
            msg_cut = f' - cut value: {cf.CUT_MEDIAN_FLUX}'
            if cf.delta_redshift > 0:
                self._fit_redshift(correct_wl_ranges=correct_wl_ranges,ssp=ssp)
                self.best_chi_sq_nl_fit = self.get_last_chi_sq_redshift()
                self.best_coeffs_nl_fit = self.get_last_coeffs_redshift()
            print_verbose(f'- Redshift: {self.best_redshift:.8f} +- {self.e_redshift:.8f}', verbose=True)
            # self.correct_elines_mask(redshift=self.best_redshift)
            self.redshift_correct_masks(redshift=self.best_redshift, correct_wl_ranges=correct_wl_ranges)
            coeffs_now, chi_sq, _ = self.fit_WLS_invmat(ssp=ssp, smooth_cont=True, sel_wavelengths=s['sel_nl_wl'])
            self.best_coeffs_sigma = coeffs_now
            if cf.delta_sigma > 0:
                if not fit_sigma_rnd:
                    self._fit_sigma(guided=guide_sigma)
                else:
                    print_verbose(f'- fit_sigma_rnd', verbose=True)
                    self._fit_sigma_rnd(guided=guide_sigma,
                                        medres_merit=sigma_rnd_medres_merit)
                # self._fit_sigma(guided_sigma)
                self.best_chi_sq_nl_fit = self.get_last_chi_sq_sigma()
                self.best_coeffs_nl_fit = self.get_last_coeffs_sigma()
            print_verbose(f'- Sigma:    {self.best_sigma:.8f} +- {self.e_sigma:.8f}', verbose=True)
            if self.sigma_inst is None:
                self.sigma_mean = self.best_sigma
            else:
                self.sigma_mean = np.sqrt(self.sigma_inst**2 + (5000*self.best_sigma/__c__)**2)
            if cf.delta_AV > 0:
                self._fit_AV()
                self.best_chi_sq_nl_fit = self.get_last_chi_sq_AV()
                self.best_coeffs_nl_fit = self.get_last_coeffs_AV()
            print_verbose(f'- AV:       {self.best_AV:.8f} +- {self.e_AV:.8f}', verbose=True)
#            print(f'Deriving redshift, sigma, AV... DONE!')
        self.best_chi_sq_nl_fit = chi_sq
        self.best_coeffs_nl_fit = coeffs_now
        print_verbose('------------------------[ END non-linear fit report]--', verbose=self.verbose)

        print_verbose('', verbose=self.verbose)
        print_verbose('-----------------------------------------------[ END non-linear fit ]--', verbose=self.verbose)
        print_verbose('-----------------------------------------------------------------------', verbose=self.verbose)


    def non_linear_fit_kin(self, guide_sigma=False, fit_sigma_rnd=True,
                       sigma_rnd_medres_merit=False, correct_wl_ranges=False):
    # def non_linear_fit(self, guided_sigma=None):
        """
        Do the non linear fit in order to find the kinematics parameters and the dust
        extinction. This procedure uses the set of SSP models, `self.models_nl_fit`.
        At the end will set the first entry to the ssp fit chain with the coefficients
        for the best model after the non-linear fit.
        """
        cf = self.config
        nl_w_min=self.nl_w_min
        nl_w_max=self.nl_w_max
        guess_redshift = cf.redshift_set[0]

        SPS_kin=deepcopy(self)
        mask_w = (self.wavelength>nl_w_min*(1+guess_redshift)) & (self.wavelength<nl_w_max*(1+guess_redshift))
        SPS_kin.wavelength=self.wavelength[mask_w]
        SPS_kin.flux=self.flux[mask_w]
        SPS_kin.eflux=self.eflux[mask_w]
        SPS_kin.ratio_master=SPS_kin.ratio_master[mask_w]
        SPS_kin._create_spectra_dict(SPS_kin.wavelength, SPS_kin.flux, SPS_kin.eflux, SPS_kin.min, SPS_kin.max, SPS_kin.ratio_master)

        #SPS_kin._multi_AV()
        #SPS_kin._fitting_init()
        #SPS_kin.ssp_init()
  
        mask_w_nl = (self.models_nl_fit.wavelength>nl_w_min*0.9) & (self.models_nl_fit.wavelength<nl_w_max*1.1)
        ssp = deepcopy(self.models_nl_fit)
        ssp.wavelength=self.models_nl_fit.wavelength[mask_w_nl]
        ssp.flux_models=np.transpose(np.transpose(self.models_nl_fit.flux_models)[mask_w_nl])
        ssp.n_wave=len(ssp.wavelength)
        s = deepcopy(self.spectra)

                  
        s['sel_AV']=s['sel_AV'][mask_w]
        s['raw_flux']=s['raw_flux'][mask_w]
        s['sel_nl_wl']=s['sel_nl_wl'][mask_w]
#        print(len(s['sel_AV']),len(s['raw_flux']))
        
           # if guided_sigma is not None: self.best_sigma = guided_sigma

        print_verbose('', verbose=self.verbose)
        print_verbose('-----------------------------------------------------------------------', verbose=self.verbose)
        print_verbose('--[ BEGIN non-linear fit ]---------------------------------------------', verbose=self.verbose)
        coeffs_now, chi_sq, msk_model_min = SPS_kin.fit_WLS_invmat(ssp=ssp, sel_wavelengths=s['sel_AV'])
 #       print(f'Paso {coeffs_now}')
        # TODO: check mask from med_flux in perl version
        msk_flux = s['raw_flux'][s['sel_AV']]
        med_flux = np.median(msk_flux)
        print_verbose('-[ non-linear fit report ]-----------------------------', verbose=self.verbose)
        if self.guided_errors is not None:
            self.best_redshift = cf.redshift
            self.e_redshift = self.guided_errors[0]
            print_verbose(f'- Redshift: {self.best_redshift:.8f} +- {self.e_redshift:.8f}', verbose=True)
            self.best_sigma = cf.sigma
            self.e_sigma = self.guided_errors[1]
            print_verbose(f'- Sigma:    {self.best_sigma:.8f} +- {self.e_sigma:.8f}', verbose=True)
            self.best_AV = cf.AV
            self.e_AV = self.guided_errors[2]
            print_verbose(f'- AV:       {self.best_AV:.8f} +- {self.e_AV:.8f}', verbose=True)
            # self.correct_elines_mask(redshift=self.best_redshift)
            self.redshift_correct_masks(redshift=self.best_redshift, correct_wl_ranges=correct_wl_ranges)
            if self.sigma_inst is None:
                self.sigma_mean = self.best_sigma
            else:
                self.sigma_mean = np.sqrt(self.sigma_inst**2 + (5000*self.best_sigma/__c__)**2)
        else:
#            print(f'Deriving redshift sigma...')
            # self.update_redshift_params(coeffs=coeffs_now, chi_sq=chi_sq, redshift=self.best_redshift)
            # self.best_chi_sq_redshift = self.get_last_chi_sq_redshift()
            # self.best_coeffs_redshift = self.get_last_coeffs_redshift()
            msg_cut = f' - cut value: {cf.CUT_MEDIAN_FLUX}'
            if cf.delta_redshift > 0:
                SPS_kin._fit_redshift(correct_wl_ranges=correct_wl_ranges,ssp=ssp)
                self.best_redshift=SPS_kin.best_redshift
                self.e_redshift=SPS_kin.e_redshift
                self.best_chi_sq_nl_fit = self.get_last_chi_sq_redshift()
                self.best_coeffs_nl_fit = self.get_last_coeffs_redshift()
            print_verbose(f'- Redshift: {self.best_redshift:.8f} +- {self.e_redshift:.8f}', verbose=True)
            # self.correct_elines_mask(redshift=self.best_redshift)
            SPS_kin.redshift_correct_masks(redshift=SPS_kin.best_redshift, correct_wl_ranges=correct_wl_ranges)
            coeffs_now, chi_sq, _ = SPS_kin.fit_WLS_invmat(ssp=ssp, smooth_cont=True, sel_wavelengths=s['sel_nl_wl'])
            self.best_coeffs_sigma = coeffs_now
            if cf.delta_sigma > 0:
                if not fit_sigma_rnd:
                    SPS_kin._fit_sigma(guided=guide_sigma)
                else:
                    print_verbose(f'- fit_sigma_rnd', verbose=True)
                    SPS_kin._fit_sigma_rnd(guided=guide_sigma,
                                        medres_merit=sigma_rnd_medres_merit)
                # self._fit_sigma(guided_sigma)
                self.best_sigma=SPS_kin.best_sigma
                self.e_sigma=SPS_kin.e_sigma
                self.best_chi_sq_nl_fit = self.get_last_chi_sq_sigma()
                self.best_coeffs_nl_fit = self.get_last_coeffs_sigma()
            print_verbose(f'- Sigma:    {self.best_sigma:.8f} +- {self.e_sigma:.8f}', verbose=True)
            if self.sigma_inst is None:
                self.sigma_mean = self.best_sigma
            else:
                self.sigma_mean = np.sqrt(self.sigma_inst**2 + (5000*self.best_sigma/__c__)**2)
#            print(f'Deriving redshift, sigma... DONE!')
        self.best_chi_sq_nl_fit = chi_sq
        self.best_coeffs_nl_fit = coeffs_now
        print_verbose('------------------------[ END non-linear fit report]--', verbose=self.verbose)

        print_verbose('', verbose=self.verbose)
        print_verbose('-----------------------------------------------[ END non-linear fit ]--', verbose=self.verbose)
        print_verbose('-----------------------------------------------------------------------', verbose=self.verbose)
        

    
    def gas_fit_no_rsp(self, ratio=True):
        """
        Prepares the observed spectra in order to fit systems of emission lines
        to the residual spectra.

        Attributes
        ----------
        spectra['raw_flux_no_gas'] : array like
            The raw observed spectrum without the model of the emission lines.
        """
        #print('*** PASO****')
        s = self.spectra
        ssp = self.models
        sigma_mean = self.sigma_mean


        print_verbose('', verbose=self.verbose)
        #print('*** P here ***')

        print_verbose('-----------------------------------------------------------------------', verbose=self.verbose)
        print_verbose('--[ BEGIN EL fit ]-----------------------------------------------------', verbose=self.verbose)

#        coeffs=self.get_last_coeffs_ssp()
#        print('coeffs=',coeffs)
        model_min = s['model_min'] / ratio
        #self.get_best_model_from_coeffs(ssp=ssp, coeffs=self.get_last_coeffs_ssp())
#        print('model_min=',model_min)
        res_min = s['raw_flux']- model_min
        #if self.SN_norm_window > 10:
        #    self._subtract_continuum(model_min)

        # fit Emission Lines
        self._EL_fit(model_min=model_min)

#        print('EL done')
        #   s[raw_flux] is not used anymore. Idk why this is made.
        #   Inside perl code SFS rewrites raw_flux when creates the
        #   spectra[raw_flux_no_gas], but here I create a spectra[raw_flux_no_gas]
        # TODO: THIS SHOULD BE FIXED!
        #   2019.12.19: EADL@laptop-XPS13-9333
        #       The need to save the raw_flux in the spectra dictionary is because
        #       self.fit_WLS_invmat uses the spectra['raw_flux']. We have to create
        #       a _fit_WLS_invmat() that receives the SSP base and the spectrum to
        #       fit.
        #   2020.01.14: EADL@laptop-XPS13-9333
        #       The need to create _fit_WLS_invmat() proposed before helps also to
        #       create the Monte-Carlo loop!
        # FIXED: now raw_flux back to be raw_flux and raw_model_elines should be used.
        #   2020.01.15: EADL@laptop-XPS13-9333
        #       _fit_WLS_invmat() Done!
        # s['orig_raw_flux'] = s['raw_flux'].copy()

        # Removes the gas from the raw_flux.
        s['raw_flux_no_gas'] = s['orig_flux'] - s['raw_model_elines']
        #s['raw_flux'] -= s['raw_model_elines']

        # plt.cla()
        # wave_list = [s['raw_wave']]*5
        # res = s['orig_raw_flux'] - ssp_model_joint
        # spectra_list = [s['orig_raw_flux'], model_min, s['orig_raw_flux'],
        #                 s['raw_model_elines'], ssp_model_joint, res]
        # plot_spectra_ax(plt.gca(), wave_list, spectra_list)
        # # plt.ylim(-0.2, 1.5)
        
        # plt.pause(0.001)
        # plt.show(block=True)

        # re-scale
#        print('Ratio master')
#        try:
 #           y_ratio = self.ratio_master if not ratio else self._rescale(model_min)
 #       except:
  
 #       print('Ratio master')
#        ratio = 
        #ratio = np.divide(s['orig_flux'], y_ratio, where=y_ratio!=0)
        s['orig_flux_ratio'] = s['orig_flux'] / ratio
#        ratio = np.divide(s['raw_flux_no_gas'], y_ratio)#, where=y_ratio!=0)
        #s['raw_flux_no_gas'] = self._rescale(model_min)
       #np.where(y_ratio > 0, ratio, s['raw_flux_no_gas'])
        # s['msk_flux_no_gas'] = s['raw_flux_no_gas'][s['sel_wl']]
        #return model_min,s['raw_flux_no_gas'],s['raw_model_elines'],s['orig_flux_ratio'] 


        # plt.show(block=True)

        
