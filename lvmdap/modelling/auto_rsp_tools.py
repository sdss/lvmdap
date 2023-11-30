import sys
import numpy as np
from astropy.io import fits
from copy import deepcopy as copy
from os.path import basename, isfile

from pyFIT3D.modelling.stellar import StPopSynt
#from lvmdap.modelling.synthesis import StellarSynthesis as StPopSynt
from lvmdap.dap_tools import binArray,bin1D

from  pyFIT3D.common.io import read_spectra, print_time, clean_preview_results_files, ReadArguments
from  pyFIT3D.common.io import array_to_fits, trim_waves, sel_waves, print_verbose, write_img_header
from  pyFIT3D.common.gas_tools import append_emission_lines_parameters
from  pyFIT3D.common.gas_tools import ConfigEmissionModel, create_emission_lines_parameters

from  pyFIT3D.common.constants import __selected_extlaw__, __selected_R_V__, __n_Monte_Carlo__
from pyFIT3D.common.constants import __c__, _MODELS_ELINE_PAR, __mask_elines_window__
from pyFIT3D.common.constants import __selected_half_range_sysvel_auto_ssp__, _figsize_default, _plot_dpi
from pyFIT3D.common.constants import __sigma_to_FWHM__, __selected_half_range_wl_auto_ssp__

from pyFIT3D.common.stats import pdl_stats, _STATS_POS, WLS_invmat, median_box, median_filter

from copy import deepcopy

__verbose__ = True
__verbose__ = False

class ConfigAutoSSP(object):
    """Reads, stores and process the configuration of AutoSSP script. Also, load masks and
    SSP models.
    This class

    Attributes
    ----------
    args : ReadArgumentsAutoSSP class
    filename :

    Methods
    -------
    _load :
    _load_ssp_fits :
    _multi_AV :
    _load_masks :

    """

    def __init__(self, config_file,
                 redshift_set=None, sigma_set=None, AV_set=None,
                 w_min=None, w_max=None, nl_w_min=None, nl_w_max=None,
                 mask_list=None, elines_mask_file=None,
                 sigma_inst=None, verbose=False):
        self.filename = config_file
        self.redshift_set = redshift_set
        self.sigma_set = sigma_set
        self.AV_set = AV_set
        self._verbose = verbose
        self._load()

    def _load(self):
        """
        Loads the configuration file. Also, reads the configuration file of
        each to-be-fitted system.
        """
        config_keys = [
            'redshift', 'delta_redshift','min_redshift','max_redshift',
            'DV','RV','DS','RS','MIN_W','MAX_W',
            'sigma','delta_sigma','min_sigma','max_sigma',
            'AV','delta_AV','min_AV','max_AV',
        ]
        self.systems = []
        self.n_systems = 0
        # Array of `ConfigEmissionModel`
        self.systems_config = []
        self.start_w = 1e12
        self.end_w = -1e12

        with open(self.filename, "r") as f:
            # get file size
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)

            # reads each line till systems config
            for k, v in zip(config_keys[0:10], f.readline().split()):
                setattr(self, k, eval(v))
            for k, v in zip(config_keys[10:14], f.readline().split()):
                setattr(self, k, eval(v))
            for k, v in zip(config_keys[14:18], f.readline().split()):
                setattr(self, k, eval(v))
            self.n_systems = eval(f.readline())

            # redefine parameters setted by user
            if self.redshift_set is not None:
                self.redshift = self.redshift if self.redshift_set[0] is None else self.redshift_set[0]
                self.delta_redshift = self.delta_redshift if self.redshift_set[1] is None else self.redshift_set[1]
                self.min_redshift = self.min_redshift if self.redshift_set[2] is None else self.redshift_set[2]
                self.max_redshift = self.max_redshift if self.redshift_set[3] is None else self.redshift_set[3]
            if self.sigma_set is not None:
                self.sigma = self.sigma if self.sigma_set[0] is None else self.sigma_set[0]
                self.delta_sigma = self.delta_sigma if self.sigma_set[1] is None else self.sigma_set[1]
                self.min_sigma = self.min_sigma if self.sigma_set[2] is None else self.sigma_set[2]
                self.max_sigma = self.max_sigma if self.sigma_set[3] is None else self.sigma_set[3]
            if self.AV_set is not None:
                self.AV = self.AV if self.AV_set[0] is None else self.AV_set[0]
                self.delta_AV = self.delta_AV if self.AV_set[1] is None else self.AV_set[1]
                self.min_AV = self.min_AV if self.AV_set[2] is None else self.AV_set[2]
                self.max_AV = self.max_AV if self.AV_set[3] is None else self.AV_set[3]

            # reads each system config
            for i in range(self.n_systems):
                l = f.readline().split()
                tmp = {
                    'start_w': eval(l[0]), 'end_w': eval(l[1]), 'mask_file': l[2],
                    'config_file': l[3], 'npoly': eval(l[4]),
                    'mask_poly': l[5], 'nmin': eval(l[6]), 'nmax': eval(l[7])
                }
                if tmp['start_w'] < self.start_w:
                    self.start_w = tmp['start_w']
                if tmp['end_w'] > self.end_w:
                    self.end_w = tmp['end_w']
                if not isfile(tmp['mask_file']):
                    tmp['mask_file'] = None
                self.systems.append(tmp)
                self.systems_config.append(ConfigEmissionModel(tmp['config_file'], verbose=self._verbose))
            l = f.readline().split()
            self.MIN_DELTA_CHI_SQ = eval(l[0])
            self.MAX_N_ITER = eval(l[1])
            self.CUT_MEDIAN_FLUX = eval(l[2])
            self.ABS_MIN = 0.5*self.CUT_MEDIAN_FLUX

            l = f.readline().split()
            self.start_w_peak = eval(l[0])
            self.end_w_peak = eval(l[1])

            # Some configuration files could have this line
            # Not tested yet
            if (f.tell() != file_size):
                l = f.readline().split()
                if len(l) > 0:
                    self.wave_norm = eval(l[0])
                    self.w_wave_norm = eval(l[1])
                    self.new_ssp_file = l[2]

        print_verbose(f'{self.n_systems} Number of systems', verbose=self._verbose)

        return None

def auto_rsp_elines_single_main(
        wavelength, flux, eflux, ssp_file, config_file, out_file,
        ssp_nl_fit_file=None, sigma_inst=None, mask_list=None, elines_mask_file=None,
        min=None, max=None, w_min=None, w_max=None, nl_w_min=None, nl_w_max=None,
        input_redshift=None, delta_redshift=None, min_redshift=None, max_redshift=None,
    input_sigma=None, delta_sigma=None, min_sigma=None, max_sigma=None,
        input_AV=None, delta_AV=None, min_AV=None, max_AV=None,
        R_V=None, extlaw=None, plot=None, single_ssp=False,
        is_guided_sigma=False,
        # guided_sigma=None,
        ratio=True, spec_id=None, y_ratio=None, guided_errors=None, fit_sigma_rnd=True,
        fit_gas=True, losvd_rnd_medres_merit=False, verbose=None, sps_class=StPopSynt,
        SPS_master=None,bin_AV=51,SN_CUT=  2 ):

    verbose = __verbose__ if verbose is None else verbose
    redshift_set = [input_redshift, delta_redshift, min_redshift, max_redshift]
    sigma_set = [input_sigma, delta_sigma, min_sigma, max_sigma]
    AV_set = [input_AV, delta_AV, min_AV, max_AV]
    if plot:
        import matplotlib.pyplot
        import seaborn as sns
        sns.set(context="paper", style="ticks", palette="colorblind", color_codes=True)
    
    

    if (SPS_master==None):
        cf = ConfigAutoSSP(config_file, redshift_set=redshift_set, sigma_set=sigma_set, AV_set=AV_set)
        if guided_errors is not None:
            fit_gas = False
        SPS = sps_class(config=cf,
                        wavelength=wavelength, flux=flux, eflux=eflux,
                        mask_list=mask_list, elines_mask_file=elines_mask_file,
                        sigma_inst=sigma_inst, ssp_file=ssp_file,
                        ssp_nl_fit_file=ssp_nl_fit_file, out_file=out_file,
                        w_min=w_min, w_max=w_max, nl_w_min=nl_w_min, nl_w_max=nl_w_max,
                        R_V=R_V, extlaw=extlaw, spec_id=spec_id, min=min, max=max,
                        guided_errors=guided_errors, ratio_master=y_ratio,
                        fit_gas=fit_gas, plot=plot, verbose=verbose)
        SPS.wavelength=wavelength
        SPS.flux=flux
        SPS.eflux=eflux
        SPS.mask_list=mask_list
        SPS.elines_mask_file=elines_mask_file
        SPS.sigma_inst=sigma_inst
        SPS.out_file=out_file
        SPS.w_min=w_min
        SPS.w_max=w_max
        SPS.nl_w_min=nl_w_min
        SPS.nl_w_max=nl_w_max
        SPS.R_V=R_V
        SPS.extlaw=extlaw
        SPS.min=min
        SPS.max=max
        SPS.fit_gas=fit_gas
        SPS.spec_id=spec_id
        SPS.guided_errors=guided_errors
        SPS.ratio_master=y_ratio
        SPS.plot=plot
        SPS.verbose=verbose
        SPS._create_spectra_dict(wavelength, flux, eflux, min, max, y_ratio)
        SPS._multi_AV()
        SPS._fitting_init()
        SPS.ssp_init()

    else:
        cf = SPS_master.config
        cf.redshift_set=redshift_set
        cf.sigma_set=sigma_set
        cf.AV_set=AV_set
        if guided_errors is not None:
            fit_gas = False
        SPS = SPS_master
        SPS.wavelength=wavelength
        SPS.flux=flux
        SPS.eflux=eflux
        SPS.mask_list=mask_list
        SPS.elines_mask_file=elines_mask_file
        SPS.sigma_inst=sigma_inst
        SPS.out_file=out_file
        SPS.w_min=w_min
        SPS.w_max=w_max
        SPS.nl_w_min=nl_w_min
        SPS.nl_w_max=nl_w_max
        SPS.R_V=R_V
        SPS.extlaw=extlaw
        SPS.min=min
        SPS.max=max
        SPS.fit_gas=fit_gas
        SPS.spec_id=spec_id
        SPS.guided_errors=guided_errors
        SPS.ratio_master=y_ratio
        SPS.plot=plot
        SPS.verbose=verbose
        SPS._create_spectra_dict(wavelength, flux, eflux, min, max, y_ratio)
        SPS._multi_AV()
        SPS._fitting_init()
        SPS.ssp_init()
        
        
    msg_cut = f' - cut value: {cf.CUT_MEDIAN_FLUX:6.4f}'
    if cf.CUT_MEDIAN_FLUX == 0:
        msg_cut = ' - Warning: no cut (CUT_MEDIAN_FLUX = 0)'
    print(f'-> median raw flux = {SPS.median_flux:6.4f}{msg_cut}')
    SPS.cut = False
    # valid_flux = 2 - everything OK
    # valid_flux = 1 - missing valid flux (flux > 0) inside nl-fit wavelength window.
    # valid_flux = 0 - missing valid flux (flux > 0) inside linear fit wavelength window

    mask_w = (SPS.wavelength>SPS.nl_w_min) & (SPS.wavelength<SPS.nl_w_max)
    nl_wavelength=SPS.wavelength[mask_w]
    nl_flux=SPS.flux[mask_w]
    SPS.med_flux = np.median(nl_flux)
    SPS.rms = np.std(nl_flux)
    SPS.calc_SN_norm_window()
    print(f'-> MED_FLUX: {SPS.med_flux} +- {SPS.rms} SN:{SPS.SN_norm_window}')
 #   SN_CUT=3
 #   print(f'# SN_CUT = {SN_CUT}')
   


    if ((SPS.med_flux<=0) or (SPS.med_flux<0.5*SPS.rms) or (SPS.SN_norm_window<SN_CUT)):
        SPS.valid_flux=0
#        print('-> **** not fitting the continuum ****')
    
    if (SPS.valid_flux > 0) and (SPS.median_flux > cf.CUT_MEDIAN_FLUX):  # and (median_flux > cf.ABS_MIN):

        SPS_AV = deepcopy(SPS)
        try:
            print('-> NL kin fitting')
            SPS.non_linear_fit_kin(is_guided_sigma, fit_sigma_rnd=fit_sigma_rnd,
                                   sigma_rnd_medres_merit=losvd_rnd_medres_merit)
        except:
            print('-> NL kin not fitted')
            SPS.best_redshift = cf.redshift
            SPS.e_redshift = 0.0
            print_verbose(f'- Redshift: {SPS.best_redshift:.8f} +- {SPS.e_redshift:.8f}', verbose=True)
            SPS.best_sigma = cf.sigma
            SPS.e_sigma = 0.0
            print_verbose(f'- Sigma:    {SPS.best_sigma:.8f} +- {SPS.e_sigma:.8f}', verbose=True)
            
        # SPS.non_linear_fit(guided_sigma)


        
        if cf.delta_AV > 0:
            print('# Fitting Av')
            SPS_AV.ssp_nl_fit.flux_models=binArray(SPS.ssp_nl_fit.flux_models, 1, bin_AV, bin_AV, func=np.nanmean)
            nx_new = SPS_AV.ssp_nl_fit.flux_models.shape[1]
            crval=SPS.ssp_nl_fit.wavelength[0]
            cdelt=SPS.ssp_nl_fit.wavelength[1]-crval
            crpix=1
            SPS_AV.ssp_nl_fit.wavelength=crval+cdelt*bin_AV*0.5+cdelt*bin_AV*(np.arange(0,nx_new)-(crpix-1))
            SPS_AV.ssp_nl_fit.n_wave=nx_new

            SPS_AV.wavelength=bin1D(SPS.wavelength,bin_AV)
            SPS_AV.flux=bin1D(SPS.flux,bin_AV)
            SPS_AV.eflux=bin1D(SPS.eflux,bin_AV)
            SPS_AV.ratio_master=bin1D(SPS.ratio_master,bin_AV)

            #SPS_kin.wavelength=SPS.wavelength[mask_w]
            #SPS_kin.flux=SPS.flux[mask_w]
            #SPS_kin.eflux=SPS.eflux[mask_w]
            #SPS_kin.ratio_master=SPS_kin.ratio_master[mask_w]


            SPS_AV._multi_AV()
            SPS_AV._fitting_init()
            SPS_AV.ssp_init()
            SPS_AV._fit_AV()
            SPS.best_chi_sq_nl_fit = SPS_AV.get_last_chi_sq_AV()
            SPS.best_coeffs_nl_fit = SPS_AV.get_last_coeffs_AV()
            SPS.best_AV=SPS_AV.best_AV
            SPS.e_AV=SPS_AV.e_AV
        print_verbose(f'- AV:       {SPS.best_AV:.8f} +- {SPS.e_AV:.8f}', verbose=True)



        SPS.calc_SN_norm_window()
        min_chi_sq = 1e12
        n_iter = 0
        while ((min_chi_sq > cf.MIN_DELTA_CHI_SQ) & (n_iter < cf.MAX_N_ITER)):
            print(f'# Deriving SFH... attempt {n_iter + 1} of {cf.MAX_N_ITER}')
            # This part is repeated in auto_ssp perl script
            # if cf.CUT_MEDIAN_FLUX == 0:
            #     msg_cut = ' - Warning: no cut (CUT_MEDIAN_FLUX = 0)'
            # print(f'-> median masked flux = {med_flux:6.4f}{msg_cut}')
            # if SPS.med_flux >= cf.CUT_MEDIAN_FLUX:

            # Emission lines fit
            SPS.gas_fit(ratio=ratio)

            # stellar population synthesis based solely on chi_sq determination
            # if median_flux > cf.CUT_MEDIAN_FLUX:
            if single_ssp:
                min_chi_sq_now = SPS.rsp_single_fit()
            else:
                min_chi_sq_now = SPS.rsp_fit(n_MC=__n_Monte_Carlo__)
            SPS.resume_results()
            print(f'# Deriving SFH... attempt {n_iter + 1} DONE!')
            if not single_ssp:
                SPS.output_coeffs_MC_to_screen()
                #SPS.output_to_screen(block_plot=False)
            if min_chi_sq_now < min_chi_sq:
                min_chi_sq = min_chi_sq_now
            n_iter += 1
    else:
#        SPS.cut = True
        print(f'-> Single SSP fit ')
        SPS.best_redshift = cf.redshift
        SPS.e_redshift = 0.0
        print_verbose(f'- Redshift: {SPS.best_redshift:.8f} +- {SPS.e_redshift:.8f}', verbose=True)
        SPS.best_sigma = cf.sigma
        SPS.e_sigma = 0.0
        print_verbose(f'- Sigma:    {SPS.best_sigma:.8f} +- {SPS.e_sigma:.8f}', verbose=True)
        SPS.best_Av = cf.AV
        SPS.e_Av = 0.0
        print_verbose(f'- Av:    {SPS.best_Av:.8f} +- {SPS.e_Av:.8f}', verbose=True)
        SPS.spectra['raw_flux_no_gas']=SPS.spectra['orig_flux']
        min_chi_sq_now = SPS.rsp_single_fit()
        model_ssp_min = SPS.spectra['model_ssp_min']
        #print(f'model_ssp_min: {model_ssp_min}')

        ratio_now = np.divide(SPS.spectra['orig_flux'] - model_ssp_min,  model_ssp_min, where= model_ssp_min !=0) + 1
        median_ratio = median_filter(15, ratio_now)


        #SPS.spectra['model_ssp_min'],SPS.spectra['raw_flux_no_gas'],SPS.spectra['raw_model_elines'],SPS.spectra['orig_flux_ratio']=
        SPS.gas_fit_no_rsp(ratio=median_ratio)
        res_ssp = SPS.spectra['raw_flux_no_gas'] - model_ssp_min     
        model_joint = model_ssp_min + SPS.spectra['raw_model_elines']
        res_joint = (res_ssp - SPS.spectra['raw_model_elines'])     
        SPS.spectra['model_joint'] = model_joint
        SPS.spectra['res_joint'] = res_joint
        SPS.spectra['res_ssp'] = res_ssp
        SPS.spectra['res_ssp_no_corr'] = SPS.spectra['orig_flux'] - SPS.spectra['model_ssp_min_uncorr']

        SPS._MC_averages()
        SPS.resume_results()
        SPS.output_coeffs_MC_to_screen()

        
        #print('PASO')
        #SPS.output_to_screen(block_plot=False)
        # TODO: What to do when cut ?
        # SPS.output_gas_emission will fail because
        # SPS.config.systems[:]['EL'] it will not be defined.
        print('-> median flux below cut: unable to perform analysis.')
    return cf, SPS

def test_gas_fit_no_rsp(self, ratio=True):
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

        coeffs=self.get_last_coeffs_ssp()
        print('coeffs=',coeffs)
        model_min = self.get_best_model_from_coeffs(ssp=ssp, coeffs=self.get_last_coeffs_ssp())
        print('model_min=',model_min)
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
        #y_ratio = self.ratio_master if not ratio else self._rescale(model_min)

#        ratio = 
        #ratio = np.divide(s['orig_flux'], y_ratio, where=y_ratio!=0)
        s['orig_flux_ratio'] = s['orig_flux']
        #ratio = np.divide(s['raw_flux_no_gas'], y_ratio, where=y_ratio!=0)
        #s['raw_flux_no_gas'] = np.where(y_ratio > 0, ratio, s['raw_flux_no_gas'])
        # s['msk_flux_no_gas'] = s['raw_flux_no_gas'][s['sel_wl']]
        return model_min,s['raw_flux_no_gas'],s['raw_model_elines'],s['orig_flux_ratio'] 


# TODO: ADD SINGLE SSP FIT TO AUTO_SSP_SPEC
def auto_ssp_spec(
    spec_file, ssp_models_file, out_file, config_file,
    error_file=None, variance_error_column=False,
    nl_ssp_models_file=None, instrumental_dispersion=None,
    wl_range=None, nl_wl_range=None, mask_list=None, elines_mask_file=None,
    redshift_set=None, losvd_set=None, losvd_in_AA=False, AV_set=None,
    fit_gas=True, fit_sigma_rnd=True, single_ssp=False,
    plot=None, min=None, max=None, seed=None, ratio=None,
    losvd_rnd_medres_merit=False, verbose=0,
    R_V=None, extlaw=None,
    ):
    time_ini_run = print_time(print_seed=False, get_time_only=True)
    # initial time in seconds since Epoch.
    seed = print_time() if seed is None else print_time(time_ini=seed)
    # initial time used as the seed of the random number generator.
    np.random.seed(seed)
    ratio = True if ratio is None else ratio
    nl_ssp_models_file = ssp_models_file if nl_ssp_models_file is None else nl_ssp_models_file
    out_file_elines = 'elines_' + out_file
    out_file_single = 'single_' + out_file
    out_file_coeffs = 'coeffs_' + out_file
    out_file_fit = 'output.' + out_file + '.fits'
    out_file_ps = out_file
    w_min = None if wl_range is None else wl_range[0]
    w_max = None if wl_range is None else wl_range[1]
    nl_w_min = None if nl_wl_range is None else nl_wl_range[0]
    nl_w_max = None if nl_wl_range is None else nl_wl_range[1]
    if redshift_set is not None:
        input_redshift, delta_redshift, min_redshift, max_redshift = redshift_set
    else:
        input_redshift, delta_redshift, min_redshift, max_redshift = None, None, None, None
    if losvd_set is not None:
        input_sigma, delta_sigma, min_sigma, max_sigma = losvd_set
    else:
        input_sigma, delta_sigma, min_sigma, max_sigma = None, None, None, None
    if AV_set is not None:
        input_AV, delta_AV, min_AV, max_AV = AV_set
    else:
        input_AV, delta_AV, min_AV, max_AV = None, None, None, None

    # remove old files
    clean_preview_results_files(out_file, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)

    # read spectrum
    wl__w, f__w, ef__w = read_spectra(spec_file, f_error=lambda x: 0.1*np.sqrt(np.abs(x)), variance_column=variance_error_column)

    input_SN = np.divide(f__w, ef__w, where=ef__w!=0)
    print_verbose(f'-> mean input S/N: {input_SN[np.isfinite(input_SN)].mean()}', verbose=verbose)

    cf, SPS = auto_ssp_elines_single_main(
        wl__w, f__w, ef__w, ssp_models_file,
        config_file=config_file,
        ssp_nl_fit_file=nl_ssp_models_file, sigma_inst=instrumental_dispersion, out_file=out_file,
        mask_list=mask_list, elines_mask_file=elines_mask_file,
        min=min, max=max, w_min=w_min, w_max=w_max, nl_w_min=nl_w_min, nl_w_max=nl_w_max,
        input_redshift=input_redshift, delta_redshift=delta_redshift,
        min_redshift=min_redshift, max_redshift=max_redshift,
        input_sigma=input_sigma, delta_sigma=delta_sigma, min_sigma=min_sigma, max_sigma=max_sigma,
        input_AV=input_AV, delta_AV=delta_AV, min_AV=min_AV, max_AV=max_AV,
        plot=plot, single_ssp=single_ssp, ratio=ratio, fit_sigma_rnd=fit_sigma_rnd,
        fit_gas=fit_gas, losvd_rnd_medres_merit=losvd_rnd_medres_merit,
        verbose=verbose, R_V=R_V, extlaw=extlaw,

        # TODO: need to be implemented in StPopSynth
        # sigma_in_AA=losvd_in_AA,

    )

    # write outputs
    if fit_gas:
        SPS.output_gas_emission(filename=out_file_elines)
    if not single_ssp:
        SPS.output_fits(filename=out_file_fit)
        SPS.output_coeffs_MC(filename=out_file_coeffs)
        SPS.output(filename=out_file, block_plot=False)
    else:
        SPS.output_single_ssp(filename=out_file_coeffs.replace('coeffs', 'chi_sq'))

    time_end = print_time(print_seed=False)
    time_total = time_end - time_ini_run
    print(f'# SECONDS = {time_total}')

def auto_ssp_elines_single(
    spec_file, ssp_file, out_file, config_file, plot=None,
    error_file=None, variance_error_column=False,
    ssp_nl_fit_file=None, sigma_inst=None, mask_list=None,
    min=None, max=None, w_min=None, w_max=None,
    nl_w_min=None, nl_w_max=None, elines_mask_file=None,
    input_redshift=None, delta_redshift=None, min_redshift=None, max_redshift=None,
    input_sigma=None, delta_sigma=None, min_sigma=None, max_sigma=None,
    input_AV=None, delta_AV=None, min_AV=None, max_AV=None, seed=None, ratio=None,
    fit_sigma_rnd=True, fit_gas=True):
    auto_ssp_spec(
        spec_file, ssp_file, out_file, config_file,
        error_file=error_file, nl_ssp_models_file=ssp_nl_fit_file,
        instrumental_dispersion=sigma_inst, mask_list=mask_list,
        elines_mask_file=elines_mask_file, plot=plot, min=min, max=max,
        wl_range=[w_min, w_max], nl_wl_range=[nl_w_min, nl_w_max],
        redshift_set=[input_redshift, delta_redshift, min_redshift, max_redshift],
        losvd_set=[input_sigma, delta_sigma, min_sigma, max_sigma],
        AV_set=[input_AV, delta_AV, min_AV, max_AV],
        seed=seed, ratio=ratio, fit_sigma_rnd=fit_sigma_rnd,
        variance_error_column=variance_error_column, fit_gas=fit_gas,

        # single_ssp
        single_ssp=True,

        # TODO: Needs to be implemented in StPopSynt
        losvd_in_AA=False,
    )
    #
    # ratio = True if ratio is None else ratio
    # ssp_nl_fit_file = ssp_file if ssp_nl_fit_file is None else ssp_nl_fit_file
    # out_file_elines = 'elines_' + out_file
    # out_file_single = 'single_' + out_file
    # out_file_coeffs = 'coeffs_' + out_file
    # out_file_fit = 'output.' + out_file + '.fits'
    # out_file_ps = out_file
    # time_ini_run = print_time(print_seed=False, get_time_only=True)
    # # initial time in seconds since Epoch.
    # seed = print_time() if seed is None else print_time(time_ini=seed)
    # # initial time used as the seed of the random number generator.
    # # np.random.seed(1573492732)
    # np.random.seed(seed)
    # # read arguments
    # clean_preview_results_files(out_file, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
    # wl__w, f__w, ef__w = read_spectra(spec_file, f_error=lambda x: 0.1*np.sqrt(np.abs(x)), variance_column=variance_error_column)
    # cf, SPS = auto_ssp_elines_single_main(
    #     wl__w, f__w, ef__w, ssp_file,
    #     config_file=config_file,
    #     ssp_nl_fit_file=ssp_nl_fit_file, sigma_inst=sigma_inst, out_file=out_file,
    #     mask_list=mask_list, elines_mask_file=elines_mask_file,
    #     min=min, max=max, w_min=w_min, w_max=w_max, nl_w_min=nl_w_min, nl_w_max=nl_w_max,
    #     input_redshift=input_redshift, delta_redshift=delta_redshift,
    #     min_redshift=min_redshift, max_redshift=max_redshift,
    #     input_sigma=input_sigma, delta_sigma=delta_sigma, min_sigma=min_sigma, max_sigma=max_sigma,
    #     input_AV=input_AV, delta_AV=delta_AV, min_AV=min_AV, max_AV=max_AV,
    #     plot=plot, single_ssp=True, ratio=ratio
    # )
    # # write outputs
    # SPS.output_gas_emission(filename=out_file_elines)
    # SPS.output_single_ssp(filename=out_file_coeffs.replace('coeffs', 'chi_sq'))
    # time_end = print_time(print_seed=False)
    # time_total = time_end - time_ini_run
    # print(f'# SECONDS = {time_total}')

# deprecated
def auto_ssp_elines_rnd(
    spec_file, ssp_file, out_file, config_file, plot=None,
    error_file=None, ssp_nl_fit_file=None, sigma_inst=None, mask_list=None,
    min=None, max=None, w_min=None, w_max=None,
    nl_w_min=None, nl_w_max=None, elines_mask_file=None,
    input_redshift=None, delta_redshift=None, min_redshift=None, max_redshift=None,
    input_sigma=None, delta_sigma=None, min_sigma=None, max_sigma=None,
    input_AV=None, delta_AV=None, min_AV=None, max_AV=None, seed=None, ratio=None,
    fit_sigma_rnd=True, variance_error_column=False, fit_gas=True):
    auto_ssp_spec(
        spec_file, ssp_file, out_file, config_file,
        error_file=error_file, nl_ssp_models_file=ssp_nl_fit_file,
        instrumental_dispersion=sigma_inst, mask_list=mask_list,
        elines_mask_file=elines_mask_file, plot=plot, min=min, max=max,
        wl_range=[w_min, w_max], nl_wl_range=[nl_w_min, nl_w_max],
        redshift_set=[input_redshift, delta_redshift, min_redshift, max_redshift],
        losvd_set=[input_sigma, delta_sigma, min_sigma, max_sigma],
        AV_set=[input_AV, delta_AV, min_AV, max_AV],
        seed=seed, ratio=ratio, fit_sigma_rnd=fit_sigma_rnd,
        variance_error_column=variance_error_column, fit_gas=fit_gas,

        # TODO: Needs to be implemented in StPopSynt
        losvd_in_AA=False,
    )

def load_rss(spec_file, error_file=None, output_seds=False):
    """Return the RSS from the given filename in the parsed command line arguments"""
    rss_f = fits.open(spec_file, memmap=False)
    if output_seds:
        rss_f_spectra = rss_f[0].data[0, :] - (rss_f[0].data[2, :] - rss_f[0].data[1, :])
    else:
        rss_f_spectra = rss_f[0].data
    if error_file is not None:
        rss_e = fits.open(error_file, memmap=False)
        rss_e_spectra = rss_e[0].data
        rss_e_spectra[~np.isfinite(rss_e_spectra)] = 1
    else:
        rss_e_spectra = 0.1*np.sqrt(np.abs(rss_f_spectra))
    wl__w = np.array([rss_f[0].header["CRVAL1"] + i*rss_f[0].header["CDELT1"] for i in range(rss_f[0].header["NAXIS1"])])
    return wl__w, rss_f_spectra, rss_e_spectra

def dump_rss_output(out_file_fit, wavelength, model_spectra):
    """Dump the RSS models into a FITS"""
    fits_name = out_file_fit if out_file_fit.endswith(".gz") else out_file_fit+".gz"
    # hdr = fits.Header()
    h = {}
    h['CRPIX1'] = 1
    h['CRVAL1'] = wavelength[0]
    h['CDELT1'] = wavelength[1] - wavelength[0]
    h['NAME0'] = 'org_spec'
    h['NAME1'] = 'model_spec'
    h['NAME2'] = 'mod_joint_spec'
    h['NAME3'] = 'gas_spec'
    h['NAME4'] = 'res_joint_spec'
    h['NAME5'] = 'no_gas_spec'
    h["FILENAME"] = fits_name
    h['COMMENT'] = f'OUTPUT {basename(sys.argv[0])} FITS'
    array_to_fits(fits_name, model_spectra, overwrite=True)
    write_img_header(fits_name, list(h.keys()), list(h.values()))
    # fits.PrimaryHDU(data=np.array(model_spectra).transpose(1,0,2), header=hdr).writeto(fits_name, overwrite=True)
    return model_spectra

def auto_ssp_elines_rnd_rss_main(
    wavelength, rss_flux, rss_eflux, output_files,
    ssp_file, config_file, out_file,
    ssp_nl_fit_file=None, sigma_inst=None, mask_list=None, elines_mask_file=None,
    min=None, max=None, w_min=None, w_max=None, nl_w_min=None, nl_w_max=None,
    input_redshift=None, delta_redshift=None, min_redshift=None, max_redshift=None,
    input_sigma=None, delta_sigma=None, min_sigma=None, max_sigma=None,
    input_AV=None, delta_AV=None, min_AV=None, max_AV=None,
    R_V=None, extlaw=None, plot=None, is_guided_sigma=False, ratio=True, spec_id=None,
    input_guided=None, input_guided_errors=None, fit_sigma_rnd=True, sps_class=StPopSynt):
    """Returns the model spectra, results and coefficient the analysis of a RSS.
    Also, returns the maps of the emission lines analysis."""
    elines_out, coeffs_out, summary_out = output_files
    guided_nl = False
    guided_errors = None
    if input_guided is not None:
        guided_nl = True
    # guided_sigma, sigma_seq = None, []
    sigma_seq = []
    input_delta_sigma = delta_sigma
    input_max_sigma = max_sigma
    input_min_sigma = min_sigma
    model_spectra = []
    results = []
    results_coeffs = []
    y_ratio = None
    ns = rss_flux.shape[0]
    output_el_models = {}
    if not guided_nl:
        _tmpcf = ConfigAutoSSP(config_file)
        for i_s in range(_tmpcf.n_systems):
            system = _tmpcf.systems[i_s]
            elcf = _tmpcf.systems_config[i_s]
            k = f'{system["start_w"]}_{system["end_w"]}'
            output_el_models[k] = create_emission_lines_parameters(elcf, ns)
        del _tmpcf
    for i, (f__w, ef__w) in enumerate(zip(rss_flux, rss_eflux)):
        print(f"\n# ID {i}/{ns - 1} ===============================================\n")
        if guided_nl:
            input_redshift = input_guided[0][i]
            delta_redshift = 0
            input_sigma = input_guided[1][i]
            delta_sigma = 0
            input_AV = input_guided[2][i]
            delta_AV = 0
            # (e_AV, e_sigma, e_redshift)
            guided_errors = (input_guided_errors[0][i], input_guided_errors[1][i], input_guided_errors[2][i])
            print(f'-> Forcing non-linear fit parameters (input guided):')
            print(f'-> input_guided_redshift:{input_redshift} e:{guided_errors[0]}')
            print(f'-> input_guided_sigma:{input_sigma} e:{guided_errors[1]}')
            print(f'-> input_guided_AV:{input_AV} e:{guided_errors[2]}')
        if i > 0 and (not guided_nl and is_guided_sigma):
            if SPS.best_sigma > 0:
                sigma_seq.append(SPS.best_sigma)
            guided_sigma = SPS.best_sigma
            k_seq = len(sigma_seq)
            n_seq_last = int(0.2*i)
            if n_seq_last < 10:
                n_seq_last = 10
            if k_seq > n_seq_last:
                guided_sigma = np.median(np.asarray(sigma_seq)[-n_seq_last:])
            input_sigma = guided_sigma
            min_sigma = guided_sigma - input_delta_sigma
            max_sigma = guided_sigma + input_delta_sigma
            delta_sigma = 0.25*input_delta_sigma
            if min_sigma < input_min_sigma:
                min_sigma = input_min_sigma
            if max_sigma > input_max_sigma:
                max_sigma = input_max_sigma
        cf, SPS = auto_ssp_elines_single_main(
            wavelength=wavelength, flux=f__w, eflux=ef__w, ssp_file=ssp_file, config_file=config_file,
            ssp_nl_fit_file=ssp_nl_fit_file, sigma_inst=sigma_inst, out_file=out_file,
            mask_list=mask_list, elines_mask_file=elines_mask_file,
            min=min, max=max, w_min=w_min, w_max=w_max, nl_w_min=nl_w_min, nl_w_max=nl_w_max,
            input_redshift=input_redshift, delta_redshift=delta_redshift,
            min_redshift=min_redshift, max_redshift=max_redshift,
            input_sigma=input_sigma, delta_sigma=delta_sigma, min_sigma=min_sigma, max_sigma=max_sigma,
            input_AV=input_AV, delta_AV=delta_AV, min_AV=min_AV, max_AV=max_AV,
            plot=plot, single_ssp=False,
            is_guided_sigma=is_guided_sigma,
            # guided_sigma=guided_sigma,
            spec_id=spec_id,
            guided_errors=guided_errors, ratio=ratio, y_ratio=y_ratio,
            fit_sigma_rnd=fit_sigma_rnd, sps_class=sps_class
        )
        y_ratio = SPS.ratio_master
        if not guided_nl:
            SPS.output_gas_emission(filename=elines_out, spec_id=i)
        SPS.output_coeffs_MC(filename=coeffs_out, write_header=i==0)
        SPS.output(filename=summary_out, write_header=i==0, block_plot=False)
        if not guided_nl:
            for system in SPS.config.systems:
                if system['EL'] is not None:
                    k = f'{system["start_w"]}_{system["end_w"]}'
                    append_emission_lines_parameters(system['EL'], output_el_models[k], i)
        model_spectra.append(SPS.output_spectra_list)
        results.append(SPS.output_results)
        results_coeffs.append(SPS.output_coeffs)

    # Transpose output
    # output model_spectra has dimensions (n_output_spectra_list, n_s, n_wavelength)
    model_spectra = np.array(model_spectra).transpose(1, 0, 2)
    # output results has dimensions (n_results, n_s)
    results = np.array(results).T
    # output results_coeffs has dimensions (n_results_coeffs, n_models, n_s)
    results_coeffs = np.array(results_coeffs).transpose(1, 2, 0)
    return model_spectra, results, results_coeffs, output_el_models

# TODO:
# def auto_ssp_rss: -> replaces auto_ssp_elines_rnd_rss
# def auto_ssp_cube

def auto_ssp_elines_rnd_rss(
    spec_file, ssp_file, out_file, config_file, plot=None,
    error_file=None, ssp_nl_fit_file=None, sigma_inst=None, mask_list=None,
    min=None, max=None, w_min=None, w_max=None,
    nl_w_min=None, nl_w_max=None, elines_mask_file=None,
    input_redshift=None, delta_redshift=None, min_redshift=None, max_redshift=None,
    input_sigma=None, delta_sigma=None, min_sigma=None, max_sigma=None,
    input_AV=None, delta_AV=None, min_AV=None, max_AV=None, seed=None, is_guided_sigma=False,
    ratio=None, guided_nl_file=None, sps_class=StPopSynt):
    ratio = True if ratio is None else ratio
    ssp_nl_fit_file = ssp_file if ssp_nl_fit_file is None else ssp_nl_fit_file
    out_file_elines = 'elines_' + out_file
    out_file_single = 'single_' + out_file
    out_file_coeffs = 'coeffs_' + out_file
    out_file_fit = 'output.' + out_file + '.fits.gz'
    out_file_ps = out_file
    time_ini_run = print_time(print_seed=False, get_time_only=True)
    # initial time in seconds since Epoch.
    seed = print_time() if seed is None else print_time(time_ini=seed)
    np.random.seed(seed)
    guided_nl = False
    input_guided = None
    input_guided_errors = None
    if guided_nl_file is not None:
        guided_nl = True
        AV, e_AV, redshift, e_redshift, sigma, e_sigma = np.genfromtxt(
            fname=guided_nl_file,
            comments="#", delimiter=",", usecols=(5,6,7,8,9,10), unpack=True
        )
        input_guided = (redshift, sigma, AV)
        input_guided_errors = (e_redshift, e_sigma, e_AV)
    # read arguments
    clean_preview_results_files(out_file, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
    # read RSS spectra
    wl__w, rss_f_spectra, rss_e_spectra = load_rss(spec_file, error_file, output_seds=guided_nl)
    # perform the RSS modelling
    with open(out_file_elines,"w") as elines_out, open(out_file_coeffs,"w") as coeffs_out, open(out_file,"w") as summary_out:
        model_spectra, _, _, _ = auto_ssp_elines_rnd_rss_main(
            wavelength=wl__w, rss_flux=rss_f_spectra, rss_eflux=rss_e_spectra,
            output_files=(elines_out,coeffs_out,summary_out),
            ssp_file=ssp_file, config_file=config_file,
            ssp_nl_fit_file=ssp_nl_fit_file, sigma_inst=sigma_inst, out_file=out_file,
            mask_list=mask_list, elines_mask_file=elines_mask_file,
            min=min, max=max, w_min=w_min, w_max=w_max, nl_w_min=nl_w_min, nl_w_max=nl_w_max,
            input_redshift=input_redshift, delta_redshift=delta_redshift,
            min_redshift=min_redshift, max_redshift=max_redshift,
            input_sigma=input_sigma, delta_sigma=delta_sigma, min_sigma=min_sigma, max_sigma=max_sigma,
            input_AV=input_AV, delta_AV=delta_AV, min_AV=min_AV, max_AV=max_AV,
            plot=plot, is_guided_sigma=is_guided_sigma, ratio=ratio,
            input_guided=input_guided, input_guided_errors=input_guided_errors, sps_class=sps_class
        )
    # write FITS RSS output
    dump_rss_output(out_file_fit=out_file_fit, wavelength=wl__w, model_spectra=model_spectra)
    time_end = print_time(print_seed=False)
    time_total = time_end - time_ini_run
    print(f'# SECONDS = {time_total}')
