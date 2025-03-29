import sys
import numpy as np
from astropy.io import fits
from copy import deepcopy as copy
from os.path import basename, isfile

from pyFIT3D.modelling.stellar import StPopSynt

from .io import read_spectra, print_time, clean_preview_results_files, ReadArguments
from .io import array_to_fits, trim_waves, sel_waves, print_verbose, write_img_header
from .gas_tools import append_emission_lines_parameters
from .gas_tools import ConfigEmissionModel, create_emission_lines_parameters

from .constants import __selected_extlaw__, __selected_R_V__, __n_Monte_Carlo__

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

def auto_ssp_elines_single_main(
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
    fit_gas=True, losvd_rnd_medres_merit=False, verbose=None, sps_class=StPopSynt):

    verbose = __verbose__ if verbose is None else verbose
    redshift_set = [input_redshift, delta_redshift, min_redshift, max_redshift]
    sigma_set = [input_sigma, delta_sigma, min_sigma, max_sigma]
    AV_set = [input_AV, delta_AV, min_AV, max_AV]

    cf = ConfigAutoSSP(config_file, redshift_set=redshift_set, sigma_set=sigma_set, AV_set=AV_set)
    if plot:
        import matplotlib.pyplot
        import seaborn as sns
        sns.set(context="paper", style="ticks", palette="colorblind", color_codes=True)
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
    msg_cut = f' - cut value: {cf.CUT_MEDIAN_FLUX:6.4f}'
    if cf.CUT_MEDIAN_FLUX == 0:
        msg_cut = ' - Warning: no cut (CUT_MEDIAN_FLUX = 0)'
    print(f'-> median raw flux = {SPS.median_flux:6.4f}{msg_cut}')
    SPS.cut = False
    # valid_flux = 2 - everything OK
    # valid_flux = 1 - missing valid flux (flux > 0) inside nl-fit wavelength window.
    # valid_flux = 0 - missing valid flux (flux > 0) inside linear fit wavelength window
    if (SPS.valid_flux > 0) and (SPS.median_flux > cf.CUT_MEDIAN_FLUX):  # and (median_flux > cf.ABS_MIN):
        # redshift, sigma and AV fit
        SPS.non_linear_fit(is_guided_sigma, fit_sigma_rnd=fit_sigma_rnd,
                           sigma_rnd_medres_merit=losvd_rnd_medres_merit)
        # SPS.non_linear_fit(guided_sigma)
        SPS.calc_SN_norm_window()
        min_chi_sq = 1e12
        n_iter = 0
        while ((min_chi_sq > cf.MIN_DELTA_CHI_SQ) & (n_iter < cf.MAX_N_ITER)):
            print(f'Deriving SFH... attempt {n_iter + 1} of {cf.MAX_N_ITER}')

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
                min_chi_sq_now = SPS.ssp_single_fit()
            else:
                min_chi_sq_now = SPS.ssp_fit(n_MC=__n_Monte_Carlo__)
            SPS.resume_results()
            print(f'Deriving SFH... attempt {n_iter + 1} DONE!')
            if not single_ssp:
                SPS.output_coeffs_MC_to_screen()
                SPS.output_to_screen(block_plot=False)
            if min_chi_sq_now < min_chi_sq:
                min_chi_sq = min_chi_sq_now
            n_iter += 1
    else:
        SPS.cut = True
        # TODO: What to do when cut ?
        # SPS.output_gas_emission will fail because
        # SPS.config.systems[:]['EL'] it will not be defined.
        print('-> median flux below cut: unable to perform analysis.')
    return cf, SPS

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
