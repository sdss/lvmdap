import io
import sys
import numpy as np
from os import remove
from astropy.io import fits
from os.path import basename, isfile, exists
from scipy.interpolate import interp1d
import scipy.optimize as opt
from scipy.stats import norm
from copy import deepcopy as copy

# local imports
from .dust import spec_apply_dust
from .gas import output_config_final_fit
from pyFIT3D.common.gas_tools import fit_elines_main
from pyFIT3D.common.io import plot_spectra_ax, array_to_fits, write_img_header
from pyFIT3D.common.io import sel_waves, trim_waves, print_verbose, get_wave_from_header
from pyFIT3D.common.stats import pdl_stats, _STATS_POS, WLS_invmat, median_box, median_filter
from pyFIT3D.common.stats import calc_chi_sq, smooth_ratio, shift_convolve, hyperbolic_fit_par
from pyFIT3D.common.constants import __c__, _MODELS_ELINE_PAR, __mask_elines_window__, __selected_R_V__
from pyFIT3D.common.constants import __selected_half_range_sysvel_auto_ssp__, _figsize_default, _plot_dpi
from pyFIT3D.common.constants import __sigma_to_FWHM__, __selected_extlaw__, __selected_half_range_wl_auto_ssp__

def convolve(wl, sed, kernel, nx=100):
    from scipy import signal
    if kernel is None:
        return sed

    nwl = wl.size + 2*nx
    sed_ = np.zeros(nwl)
#     extent the wavelength range for interpolations and redshifting
#     wl_ = np.zeros(nwl)
#     wl_[:nx] = np.arange(wl[0] - nx*(wl[1]-wl[0]), wl[0], wl[1]-wl[0])
#     wl_[nwl-nx:] = np.arange(wl[-1]+(wl[-1]-wl[-2]), wl[-1] + (nx+1)*(wl[-1]-wl[-2]), wl[-1]-wl[-2])
#     wl_[nx:nwl-nx] = wl
    sed_[:nx] = sed[0]
    sed_[nx:nwl-nx] = sed
    sed_[nwl-nx:] = sed[-1]

    sed_ = signal.convolve(sed_, kernel / kernel.sum(), mode="same")
    return sed_[nx:nwl-nx]

def _fit_WLS_invmat(flux, eflux, flux_models, verbose=False):
    '''
    A wrapper to choose the fit_WLS function. It's build for future implementations
    of fit process. This is the funcion called inside the :class:`StPopSynt`.

    Parameters
    ----------
    flux : array
        Observed spectrum.

    eflux : array
        Observed error in spectrum.

    flux_models : array like
        Models spectra at the observed frame.

    verbose : bool, optional
        If True produces a nice text output. Default value is False.

    See also
    --------
    `_fit_WLS_invmat_default`, `_fit_WLS_invmat_alternative`
    '''
    return _fit_WLS_invmat_default(flux, eflux, flux_models, verbose)

def _fit_WLS_invmat_default(flux, eflux, flux_models, verbose=False):
    """
    Fits an observed spectrum a with a linear combination of SSP models using
    Weighted Least Squares (WLS) through matrix inversion. This process consider
    measured errors of the observed spectrum.

    Parameters
    ----------
    flux : array
        Observed spectrum.

    eflux : array
        Observed error in spectrum.

    flux_models : array like
        Models spectra at the observed frame.

    verbose : bool, optional
        If True produces a nice text output. Default value is False.

    Returns
    -------
    array like
        Coefficients of the WLS fit.

    float
        Chi square of the fit.

    array like
        The `flux` modelled.
    """
    print_verbose('------------------------------------------------------------------', verbose=verbose)
    print_verbose('--[ BEGIN fit SSP with WLS_invmat ]-------------------------------', verbose=verbose)

    n_models = flux_models.shape[0]

    n_search_loops = 0
    search_coeffs = True
    select_positive_coeffs = np.ones(n_models, dtype='bool')
    weights = 1/eflux**2
    while search_coeffs:
        coeffs_now = np.zeros(n_models, dtype='float')
        print_verbose(f'n_models -------------------- {flux_models.shape[0]}', verbose=verbose)

        # invert matrix
        r = WLS_invmat(flux, flux_models[select_positive_coeffs, :], dy=weights)
        model_now, coeffs_now[select_positive_coeffs] = r
        print_verbose(f'coeffs_now ------------------ {coeffs_now}', verbose=verbose)

        # check negative coeffs
        n_neg_coeffs = (coeffs_now < 0).sum()
        search_coeffs = n_neg_coeffs > 0
        select_positive_coeffs = coeffs_now > 0
        n_positive_coeffs = select_positive_coeffs.sum()
        print_verbose(f'n_positive_coeffs ----------- {n_positive_coeffs}', verbose=verbose)
        print_verbose(f'n_neg_coeffs ---------------- {n_neg_coeffs}', verbose=verbose)

        n_search_loops += 1

    # calc chi square
    n_real_models = (coeffs_now > 0).sum()
    chi_sq, n_free_param = calc_chi_sq(flux, model_now, eflux, n_models + 1)

    # final report
    print_verbose(f'n_free_param ---------------- {n_free_param}', verbose=verbose)
    print_verbose(f'chi_sq ---------------------- {chi_sq}', verbose=verbose)
    print_verbose(f'n_search_loops --------------------- {n_search_loops}', verbose=verbose)
    print_verbose('---------------------------------[ END fit SSP with WLS_invmat ]--', verbose=verbose)
    print_verbose('------------------------------------------------------------------', verbose=verbose)

    return coeffs_now, chi_sq, model_now

def _fit_WLS_invmat_alternative(flux, eflux, flux_models, verbose=False):
    """
    A copy of perl fit process.

    Fits an observed spectrum a with a linear combination of SSP models using
    Weighted Least Squares (WLS) through matrix inversion. This process consider
    measured errors of the observed spectrum.

    Parameters
    ----------
    flux : array
        Observed spectrum.

    eflux : array
        Observed error in spectrum.

    flux_models : array like
        Models spectra at the observed frame.

    verbose : bool, optional
        If True produces a nice text output. Default value is False.

    Returns
    -------
    array like
        Coefficients of the WLS fit.

    float
        Chi square of the fit.
    """
    print_verbose('------------------------------------------------------------------', verbose=verbose)
    print_verbose('--[ BEGIN fit SSP with WLS_invmat ]-------------------------------', verbose=verbose)

    n_models = flux_models.shape[0]
    n_search_loops = 0
    search_coeffs = True
    select_positive_coeffs = np.ones(n_models, dtype='bool')
    weights = 1/eflux**2

    r = WLS_invmat(flux, flux_models, dy=weights)
    model_now, coeffs = r
    n_new = 0
    n_neg = 0
    for i, c in enumerate(coeffs):
        if c > 0:
            n_new += 1
        else:
            n_neg += 1
    if n_new > 0:
        while n_neg > 0:
            _flux_models = np.zeros((n_new, len(flux)), dtype='float')
            ic = 0
            for i, c in enumerate(coeffs):
                if c > 0:
                    _flux_models[ic] = flux_models[i]
                    ic += 1
                else:
                    coeffs[i] = 0
            print_verbose(f'n_models -------------------- {_flux_models.shape[0]}', verbose=verbose)
            # invert matrix
            r = WLS_invmat(flux, _flux_models, dy=weights)
            model_now, coeffs_now = r
            print_verbose(f'coeffs_now ------------------ {coeffs_now}', verbose=verbose)
            ic = 0
            n_neg = 0
            n_new = 0
            for i, c in enumerate(coeffs):
                if c > 0:
                    v = coeffs_now[ic]
                    if v > 0:
                        coeffs[i] = v
                        n_new += 1
                    else:
                        coeffs[i] = 0
                        n_neg += 1
            if n_new == 0:
                n_neg = 0
            n_search_loops += 1

    # calc chi square
    n_real_models = (coeffs > 0).sum()
    chi_sq, n_free_param = calc_chi_sq(flux, model_now, eflux, n_models + 1)

    # final report
    print_verbose(f'n_free_param ---------------- {n_free_param}', verbose=verbose)
    print_verbose(f'chi_sq ---------------------- {chi_sq}', verbose=verbose)
    print_verbose(f'n_search_loops --------------------- {n_search_loops}', verbose=verbose)
    print_verbose('---------------------------------[ END fit SSP with WLS_invmat ]--', verbose=verbose)
    print_verbose('------------------------------------------------------------------', verbose=verbose)

    return coeffs, chi_sq, model_now

class SSPModels(object):
    """
    A helper class to deal with SSP models. It reads the models directly
    from FITS file.

    Attributes
    ----------
    _header : FITS header
        Header of the SSP Models FITS file.

    wavelength : array like
        Wavelenghts at rest-frame for SSP models.

    n_wave : int
        Number of wavelengths.

    n_models : int
        Number of SSP models.

    flux_models : array like
        Spectra of SSP models.

    age_models : array like
        Ages of SSP models.

    metallicity_models : array like
        Metallicity of SSP models.

    mass_to_light : array like
        Mass-to-light ratio of SSP models

    normalization_wavelength : int
        The normalization wavelength.

    flux_models_obsframe : array like
        The spectra of SSP models shifted and convolved to the observed
        frame with observed and instrumental dispersion of velocities (sigma).
        Is produced when `SSPModels.to_observed` is called.

    flux_models_dust : array like
        The spectra of SSP models extincted by dust.
        Is produced when `SSPModels.apply_dust_to_flux_models` is called.

    flux_models_obsframe_dust : array like
        The spectra of SSP models shifted and convolved to the observed
        frame with observed and instrumental dispersion of velocities (sigma)
        and extincted by dust.
        Is produced when `SSPModels.apply_dust_to_flux_models_obsframe`
        is called.

    Methods
    -------
    get_normalization_wavelength :
        Returns the normalization wavelength which is present in the header
        of the FITS file. If no WAVENORM key is find in the header, sweeps
        all the models looking for the wavelengths where the flux is closer
        to 1, calculates the median of those wavelengths and return it.

    get_wavelength :
        Creates wavelength array from FITS header. Also shifts the
        wavelengths to `redshift`. Applies a mask to wavelengths range
        `mask` is set.

    get_tZ_models :
        Reads the values of age, metallicity and mass-to-light at the
        normalization flux from the ssp base FITS file.

    get_tZ_from_coeffs :
        Return the value from the age and metallicity weighted by light
        and mass from the age and metallicity of each model weighted by
        the `coeffs`.

    write_tZ_header :
        Write in the stored FITS header the `age` and `metallicity` for
        `i_tZ` spectrum.

    to_observed :
        Shift and convolves `self.flux_models` to wavelengths `wave` using
        `sigma` and `sigma_inst`.

    get_model_from_coeffs :
        Shift and convolves SSP model fluxes (i.e. `self.flux_models`) to
        wavelengths `wave_obs` using `sigma` and `sigma_inst`. After this,
        corrects the SSPs for dust extinction following the extinction law
        `extlaw` with `AV` attenuance. At the end, returns the SSP model
        spectra using `coeffs`. If `fit` is True gets the coeffs fitting
        the observed data provided by `spectra`.

    apply_dust_to_flux_models_obsframe :
        Applies dust extinction to the observed-frame SSPs models (i.e. with
        spectra shifted and convolved) following the ratio of the total
        selective extinction (`R_V`, roughly "slope"), an extinction law
        `extlaw` with `AV` attenuance.

    apply_dust_to_flux_models :
        Applies dust extinction to the rest-frame SSPs models following the
        ratio of the total selective extinction (`R_V`, roughly "slope"),
        an extinction law `extlaw` with `AV` attenuance.

    """
    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str
            FITS filename of the SSP models spectral library.
        """
        # XXX: To the future (2019-10-18 EADL) selects initial models instead
        # load another FITS file.
        self._t = fits.open(filename)[0]
        self._header = self._t.header
        self.n_wave = self._header['NAXIS1']
        self.n_models = self._header['NAXIS2']
        self.flux_models = self._t.data

        self.wavelength = self.get_wavelength()
        self.normalization_wavelength = self.get_normalization_wavelength()
        self.age_models, self.metallicity_models, self.mass_to_light = self.get_tZ_models()

        # deprecated the use of self.wavenorm
        # in order to keep working the code at first instance:
        self.wavenorm = self.normalization_wavelength

    def get_normalization_wavelength(self):
        """ Search for the normalization wavelength at the FITS header.
        If the key WAVENORM does not exists in the header, sweeps all the
        models looking for the wavelengths where the flux is closer to 1,
        calculates the median of those wavelengths and return it.

        TODO: defines a better normalization wavelength if it's not present
        in the header.

        Returns
        -------
        float
            The normalization wavelength.
        """
        try:
            wave_norm = self._header['WAVENORM']
        except Exception as ex:
            _closer = 1e-6
            probable_wavenorms = np.hstack([self.wavelength[(np.abs(self.flux_models[i] - 1) < _closer)]
                                            for i in range(self.n_models)])
            wave_norm = np.median(probable_wavenorms)
            # wave_norm = self.get_wavelength()[int(self.n_wave/2)]
            print(f'[SSPModels] {ex}')
            print(f'[SSPModels] setting normalization wavelength to {wave_norm} A')
        return wave_norm

    def get_wavelength(self, mask=None):
        """ Creates wavelength array from FITS header. Applies a mask to
        wavelengths if `mask` is set.

        Parameters
        ----------
        mask : array like, optional
            Masked wavelengths.

        Returns
        -------
        array like
            Wavelenght array from FITS header.

        See also
        --------
        :func:`pyFIT3D.common.io.get_wave_from_header`
        """
        # crval = self._header['CRVAL1']
        # cdelt = self._header['CDELT1']
        # crpix = self._header['CRPIX1']
        # pixels = np.arange(self.n_wave) + 1 - crpix
        # w = crval + cdelt*pixels
        w = get_wave_from_header(self._header, wave_axis=1)
        if mask is None:
            mask = np.ones_like(w).astype('bool')
        return w[mask]

    def get_tZ_models(self):
        """ Reads the values of age, metallicity and mass-to-light at the
        normalization flux from the SSP models FITS file.

        Returns
        -------
        array like
            Ages, in Gyr, in the sequence as they appear in FITS data.

        array like
            Metallicities in the sequence as they appear in FITS data.

        array like
            Mass-to-light value at the normalization wavelength.
        """
        ages = np.zeros(self.n_models, dtype='float')
        Zs = ages.copy()
        mtol = ages.copy()
        for i in range(self.n_models):
            mult = {'Gyr': 1, 'Myr': 1/1000}
            name_read_split = self._header[f'NAME{i}'].split('_')
            # removes 'spec_ssp_' from the name
            name_read_split = name_read_split[2:]
            _age = name_read_split[0]
            if 'yr' in _age:
                mult = mult[_age[-3:]]  # Gyr or Myr
                _age = _age[:-3]
            else:
                mult = 1  # Gyr
            age = mult*np.float(_age)
            _Z = name_read_split[1].split('.')[0]
            Z = np.float(_Z.replace('z', '0.'))
            ages[i] = age
            Zs[i] = Z
            mtol[i] = 1/np.float(self._header[f'NORM{i}'])
            self.write_tZ_header(i, age, Z)
        return ages, Zs, mtol

    def get_tZ_from_coeffs(self, coeffs, mean_log=False):
        """
        Return the value from the age and metallicity weighted by light and mass
        from the age and metallicity of each model weighted by the `coeffs`.

        Parameters
        ----------
        coeffs : array like
            Coefficients of each SSP model.

        mean_log : bool, optional
            If True computes the parameters in logscale. Defaults to False.
            E.g.::

                age = np.dot(coeffs, np.log10(age_models))

        Returns
        -------
        float
            Mean age weighted by light.

        float
            Mean met weighted by light.

        float
            Mean age weighted by mass.

        float
            Mean met weighted by mass.
        """
        coeffs = np.array(coeffs)
        coeffs[~np.isfinite(coeffs)] = 0
        if (coeffs == 0).all():
            age, met, age_mass, met_mass = 0, 0, 0, 0
        else:
            norm_C = coeffs.sum()
            coeffs_normed = coeffs/norm_C
            if mean_log:
                age = np.dot(coeffs, np.log10(self.age_models))
                met = np.dot(coeffs, np.log10(self.metallicity_models))
                age_mass = np.dot(self.mass_to_light*coeffs_normed, np.log10(self.age_models))
                met_mass = np.dot(self.mass_to_light*coeffs_normed, np.log10(self.metallicity_models))
            else:
                age = np.dot(coeffs, self.age_models)
                met = np.dot(coeffs, self.metallicity_models)
                age_mass = np.dot(coeffs_normed, self.mass_to_light*self.age_models)
                met_mass = np.dot(coeffs_normed, self.mass_to_light*self.metallicity_models)
            # # age =  np.dot(coeffs, np.log10(self.age_models)/norm_C)
            # age =  np.dot(coeffs_normed, np.log10(self.age_models))
            # met =  np.dot(coeffs_normed, self.metallicity_models)
            # norm_C_mass = np.dot(coeffs, self.mass_to_light)
            # coeffs_normed_mass = coeffs/norm_C_mass
            # age_mass =  np.dot(coeffs_normed_mass, np.log10(self.age_models*self.mass_to_light))
            # met_mass =  np.dot(coeffs_normed_mass, self.metallicity_models*self.mass_to_light)
        return age, met, age_mass, met_mass

    def write_tZ_header(self, i_tZ, age, metallicity):
        """
        Write in the stored FITS header the `age` and `metallicity`
        for `i_tZ` spectrum.

        TODO: in future write also to the FITS file.

        header['AGE`i_tZ`'] = `age`
        header['MET`i_tZ`'] = `metallicity`

        Parameters
        ----------
        i_tZ : int
            Index of the spectrum of a SSP model.

        age : float
            Age of the SSP model in Gyr.

        metallity : float
            Metallicity of the SSP model.
        """
        self._header[f'AGE{i_tZ}'] = age
        self._header[f'MET{i_tZ}'] = metallicity

    def to_observed(self, wavelength, sigma_inst, sigma, redshift, coeffs=None):
        """
        Shift and convolves `self.flux_models` to wavelengths `wavelength` using `sigma`
        and `sigma_inst`. It will fill the `self.flux_models_obsframe` attribute.
        If `coeffs` is passed, only the models with coeff > 0 will be shift and convolved,
        i.e., `self.flux_models_obsframe` will have dimension
        ``(n_positive_coeffs, n_wavelength)`` instead of ``(n_models, n_wavelength)``.

        Parameters
        ----------
        wavelength : array like
            Wavelenghts at observed frame.

        sigma_inst : float
            Sigma instrumental defined by user in the program call.

        sigma : float
            Sigma of the Observed frame.

        redshift : float
            Redshift of the Observed frame.

        coeffs : array like, optional
            A `self.n_models` dimension array with the coefficients of each model.
            Only models with coeff > 0 will be shift convolved, i.e., the
            created `self.flux_models_obsframe` will be a 2D array
            ``(n_positive_coeffs, n_wavelength)``.
        """
        if coeffs is None:
            flux_models_obsframe = np.asarray([
                shift_convolve(wavelength, self.wavelength, self.flux_models[i], redshift,
                               sigma, sigma_inst=sigma_inst)
                for i in range(self.n_models)
            ])
        else:
            # generates a dictionary with i_tZ as key and the coeff as the dict[i_tZ] = coeff
            tZcoeff_dict = {i: c for i, c in enumerate(coeffs) if c > 0}
            flux_models_obsframe = np.asarray([
                shift_convolve(wavelength, self.wavelength, self.flux_models[i], redshift,
                               sigma, sigma_inst=sigma_inst)
                for i in tZcoeff_dict.keys()
            ])
        self.flux_models_obsframe = flux_models_obsframe
        m = flux_models_obsframe[0] > 0
        self._msk_wavelength_obsframe = wavelength[m]

    def get_model_from_coeffs(self, coeffs, wavelength, sigma, redshift, AV, sigma_inst,
                              R_V=3.1, extlaw='CCM', return_tZ=False):
        """
        Shift and convolves SSP model fluxes (i.e. `self.flux_models`) to
        wavelengths `wave_obs` using `sigma` and `sigma_inst`. After this,
        applies dust extinction to the SSPs following the extinction law
        `extlaw` with `AV` attenuance. At the end, returns the SSP model
        spectra using `coeffs`.

        Parameters
        ----------
        coeffs : array like
            Coefficients of each SSP model.

        wavelength : array like
            Wavelenghts at observed frame.

        sigma : float
            Velocity dispersion (i.e. sigma) at observed frame.

        redshift : float
            Redshift of the Observed frame.

        AV : float or array like
            Dust extinction in mag.
            TODO: If AV is an array, will create an (n_AV, n_wave) array of dust spectra.

        sigma_inst : float or None
            Sigma instrumental.

        R_V : float, optional
            Selective extinction parameter (roughly "slope"). Default value 3.1.

        extlaw : str {'CCM', 'CAL'}, optional
            Which extinction function to use.
            CCM will call `Cardelli_extlaw`.
            CAL will call `Calzetti_extlaw`.
            Default value is CCM.

        return_tZ : bool, optional
            Also returns the age and metallicity for the model.

        Returns
        -------
        array like
            SSP model spectrum created by coeffs.

        list of floats
            Only returned if `return_tZ` is True.

            The list carries:
            [t_LW, Z_LW, t_MW, Z_MW]
            Age, metallicity light- and mass-weighted.

        See also
        --------
        :func:`apply_dust_to_flux_models`, :func:`to_observed`,
        :func:`get_tZ_from_coeffs`
        """
        # if sigma_inst is None:
        #     sigma_inst = config.args.sigma_inst
        # if sigma is None:
        #     sigma = config.sigma
        # if redshift is None:
        #     redshift = config.redshift
        # if AV is None:
        #     AV = config.AV
        coeffs = np.array(coeffs)
        coeffs[~np.isfinite(coeffs)] = 0
        if (coeffs == 0).all():
            model = np.zeros(wavelength.size)
        else:
            self.to_observed(wavelength, sigma_inst=sigma_inst, sigma=sigma, redshift=redshift, coeffs=coeffs)
            self.apply_dust_to_flux_models_obsframe(wavelength/(1 + redshift), AV, R_V=R_V, extlaw=extlaw)
            model = np.dot(coeffs[coeffs > 0], self.flux_models_obsframe_dust)
            if return_tZ:
                age_LW, met_LW, age_MW, met_MW = self.get_tZ_from_coeffs(coeffs)
                return model, [age_LW, met_LW, age_MW, met_MW]
        return model

    def apply_dust_to_flux_models_obsframe(self, wavelength_rest_frame, AV, R_V=3.1, extlaw='CCM'):
        self.flux_models_obsframe_dust = self._apply_dust_to_models(
            wavelength_rest_frame, self.flux_models_obsframe, AV, R_V=R_V, extlaw=extlaw
        )

    def apply_dust_to_flux_models(self, wavelength_rest_frame, AV, R_V=3.1, extlaw='CCM'):
        self.flux_models_dust = self._apply_dust_to_models(
            wavelength_rest_frame, self.flux_models, AV, R_V=R_V, extlaw=extlaw
        )

    def _apply_dust_to_models(self, wavelength_rest_frame, flux_models, AV, R_V=3.1, extlaw='CCM'):
        """
        Applies dust extinction to the SSPs following the ratio of the total selective
        extinction (`R_V`, roughly "slope"), an extinction law `extlaw` with `AV`
        attenuance.

        Parameters
        ----------
        wavelength_rest_frame : array like
            Wavelenghts at observed frame. if `fit` is True uses the one set by
            `fit_kwargs`.

        flux_models : array like
            The SSP models spectra.

        AV : float or array like
            Dust extinction in mag.

            TODO: If AV is an array, will create an (n_AV, n_wave) array of dust spectra.

        R_V : float, optional
            Selective extinction parameter (roughly "slope"). Default value 3.1.

        extlaw : str {'CCM', 'CAL'}, optional
            Which extinction function to use.
            CCM will call `Cardelli_extlaw`.
            CAL will call `Calzetti_extlaw`.
            Default value is CCM.

        See also
        --------
        :func:`pyFIT3D.modelling.dust.spec_apply_dust`
        """
        return spec_apply_dust(wavelength_rest_frame, flux_models, AV, R_V=R_V, extlaw=extlaw)

class StPopSynt(object):
    """
    This class is created to fit an observed spectrum using a decomposition of the
    underlying stellar population through a set of simple stellar population
    (SSP) models.

    The process starts with the search for the redshift, sigma and dust extinction
    (AV) of the observed spectrum. After this kinemactic evaluation, the gas spectrum
    is derived using a set of emission line models preseted in `config` and fitted by
    `pyFIT3D.common.gas_tools.fit_elines_main`.

    Then, the gas spectrum is subtracted from the original spectrum and the stellar
    population synthesis is performed, with the calculation of the coefficients of
    a set of SSP models. The search for the best coefficients is made through a
    Monte-Carlo perturbation loop of the observed spectrum.

    This class uses :class:`SSPModels` to manage the simple stellar population models
    and `pyFIT3D.common.gas_tools.fit_elines_main` for the emission line fitting.

    An example of utilization of :class:`StPopSynt` can be found in:
    `pyFIT3D.common.auto_ssp_tools.auto_ssp_elines_single_main`

    Attributes
    ----------
    config : ConfigAutoSSP class
        The class which configures the whole SSP fit process.

    spectra : dict
        A dictionary with the observed spectral information (wavelengths,
        fluxes and the standard deviation of the fluxes, with the masks).

    R_V : float, optional
        Selective extinction parameter (roughly "slope"). Default value 3.1.

    extlaw : str {'CCM', 'CAL'}, optional
        Which extinction function to use.
        CCM will call `Cardelli_extlaw`.
        CAL will call `Calzetti_extlaw`.
        Default value is CCM.

    verbose : bools, optional
        If True produces a nice text output.

    filename : str
        Path to the SSP models fits file.

    filename_nl_fit : str or None
        Path to the SSP models fits file used in the non-linear round of fit.
        The non-linear procedure search for the kinematics parameters (redshift
        and sigma) and the dust extinction (AV).
        If None, `self.ssp_nl_fit` is equal to `self.ssp`. Default value is None.

    n_loops : int
        Counts the number of loops in the whole fit process.

    Attributes
    ----------
    n_loops_nl_fit : int
        Counts the number of loops in the non-linear fit process.

    n_loops_redshift : int
        Counts the number of loops in the redshift search process.

    n_loops_sigma : int
        Counts the number of loops in the sigma search process.

    n_loops_AV : int
        Counts the number of loops in the A_V search process.

    best_coeffs_redshift :
        Best SSP Models coefficients of redshift fit.

    best_chi_sq_redshift :
        Best Chi squared of redshift fit.

    best_redshift :
        Best redshift.

    e_redshift :
        Error in best redshift.

    redshift_chain :
        Chain of inspected redshifts.

    best_coeffs_sigma :
        Best SSP Models coefficients of sigma fit.

    best_chi_sq_sigma :
        Best Chi squared of sigma fit.

    best_sigma :
        Best sigma.

    e_sigma :
        Error in best sigma.

    sigma_chain :
        Chain of inspected sigmas.

    best_coeffs_AV :
        Best SSP Models coefficients of AV fit.

    best_chi_sq_AV :
        Best Chi squared of AV fit.

    best_AV :
        Best AV.

    e_AV :
        Error in best AV.

    AV_chain :
        Chain of inspected AVs.

    coeffs_ssp_chain :
        All coefficients of the ssp search process.

    chi_sq_ssp_chain :
        All Chi squared of the ssp search process.

    Methods
    -------
    non_linear_fit :

    gas_fit :

    ssp_fit :

    fit_WLS_invmat :

    fit_WLS_invmat_MC :

    output_coeffs_MC :

    output_fits :

    output :

    fit_WLS_invmat :
        Fits an observed spectrum a with a linear combination of SSP models
        using Weighted Least Squares (WLS) through matrix inversion. This
        process consider measured errors of the observed spectrum.

    get_last_redshift :

    get_last_chi_sq_redshift :

    get_last_coeffs_redshift :

    update_redshift_params :

    plot_last :

    get_last_sigma :

    get_last_chi_sq_sigma :

    get_last_coeffs_sigma :

    update_sigma_params :

    get_last_AV :

    get_last_chi_sq_AV :

    get_last_coeffs_AV :

    update_AV_params :

    get_last_chi_sq_ssp :

    get_last_coeffs_ssp :

    update_ssp_params :

    """
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
        self.config = config

        self.verbose = verbose
        self.R_V = __selected_R_V__ if R_V is None else R_V
        self.extlaw =  __selected_extlaw__ if extlaw is None else extlaw
        self.n_loops_nl_fit = 0

        self.sigma_inst = sigma_inst
        self.sigma_mean = None
        self.filename = ssp_file
        self.filename_nl_fit = ssp_file if ssp_nl_fit_file is None else ssp_nl_fit_file
        self.out_file = out_file
        self.plot = 0 if plot is None else plot
        self.guided_errors = guided_errors
        self.fit_gas = fit_gas
        self.spectra = None
        self._load_masks(w_min, w_max, nl_w_min, nl_w_max, mask_list, elines_mask_file)

        self._greet()

        self._create_spectra_dict(wavelength, flux, eflux, min, max, ratio_master)

        # load SSP FITS File
        self._load_ssp_fits()

        # Not working right now!
        # Multi AVs paliative solution:
        # all the process should assume a different AV for
        # each SSP models.
        self._multi_AV()
        self._fitting_init()
        self.ssp_init()

    def _greet(self):
        cf = self.config
        sys_str = 'system'
        if cf.n_systems > 1:
            sys_str += 's'
        msg = '---[ StPopSynt ]'
        msg_config = '| Input config: \n'
        msg_config += f'| - MIN_DELTA_CHI_SQ = {cf.MIN_DELTA_CHI_SQ}\n'
        msg_config += f'| - MAX_N_ITER = {cf.MAX_N_ITER}\n'
        msg_config += f'| - CUT_MEDIAN_FLUX = {cf.CUT_MEDIAN_FLUX}\n'
        msg_config += f'| - Wavelength range - [{self.w_min:.2f}, {self.w_max:.2f}] - Non-linear analysis: [{self.nl_w_min:.2f}, {self.nl_w_max:.2f}]\n'
        msg_config += f'| - Instrumental dispersion: {self.sigma_inst}'
        msg_config_red = f'| - Redshift - guess:{cf.redshift:.6f} delta:{cf.delta_redshift:.6f} min:{cf.min_redshift:.6f} max:{cf.max_redshift:.6f}'
        msg_config_sig = f'| - Sigma    - guess:{cf.sigma:.6f} delta:{cf.delta_sigma:.6f} min:{cf.min_sigma:.6f} max:{cf.max_sigma:.6f}'
        msg_config_av  = f'| - AV       - guess:{cf.AV:.6f} delta:{cf.delta_AV:.6f} min:{cf.min_AV:.6f} max:{cf.max_AV:.6f}'
        max_msg_size = np.max([len(msg_config_red), len(msg_config_sig), len(msg_config_av)])
        if self.fit_gas:
            msg_el_config = '| Emission lines analysis:\n'
            msg_el_config += f'| - Number of {sys_str} = {cf.n_systems}\n'
            for i in range(cf.n_systems):
                sw = cf.systems[i]['start_w']
                ew = cf.systems[i]['end_w']
                elcf_file = cf.systems[i]['config_file']
                _msg = f'| - wavelength range: {sw}-{ew} - config file: {elcf_file}\n'
                _msg_size = len(_msg)
                if _msg_size > max_msg_size:
                    max_msg_size = _msg_size
                msg_el_config += _msg
            msg_el_config = msg_el_config[:-1]
        bar = '-'*max_msg_size
        print(msg + '-'*(max_msg_size-len(msg)))
        print(msg_config)
        print(msg_config_red)
        print(msg_config_sig)
        print(msg_config_av)
        if self.fit_gas:
            print(bar)
            print(msg_el_config)
        print(bar)

    # deprecated, use redshift_correct_masks instead
    def correct_elines_mask(self, redshift, window_size=None, update_nl_range=False):
        if self.elines_mask_file is not None:
            print_verbose(f'[correct_elines_mask]: input redshift: {redshift:.8f}', verbose=self.verbose)
            e_masks = np.empty((self.e_w.size, 2), dtype=np.float)
            z_fact = 1 + redshift
            window_size = __mask_elines_window__ if window_size is None else window_size
            if self.sigma_inst is not None:
                window_size *= self.sigma_inst
            e_masks[:,0] = self.e_w*z_fact - window_size
            e_masks[:,1] = self.e_w*z_fact + window_size
            self.e_masks = e_masks
            # if needed a complete mask:
            if self.masks is not None:
                self.comp_masks = np.sort(np.append(self.masks, self.e_masks, axis=0), axis=0)
            if self.spectra is not None:
                s = self.spectra
                sel_nl_wl_range = s['sel_nl_wl_range']
                if update_nl_range:
                    self.nl_wl_range = [self.nl_wl_range[0]*z_fact, self.nl_wl_range[1]*z_fact]
                    sel_nl_wl_range = trim_waves(s['raw_wave'], self.nl_wl_range)
                s['sel_wl_emasks'] = sel_waves(self.e_masks, s['raw_wave'])
                s['sel_nl_wl'] = (s['raw_flux'] > 0) & s['sel_wl_masks'] & sel_nl_wl_range & s['sel_wl_emasks']
                s['sel_AV'] = (s['raw_flux'] > 0) & s['sel_wl_range'] & s['sel_wl_masks'] & s['sel_wl_emasks']
        else:
            print_verbose('[correct_elines_mask]: no mask elines file', verbose=self.verbose)

    def redshift_correct_masks(self, redshift, eline_half_range=None, correct_wl_ranges=False):
        print_verbose(f'[redshift_correct_masks]: input redshift: {redshift:.8f}', verbose=self.verbose)
        # correct_wl_ranges = True
        z_fact = 1 + redshift
        eline_half_range = __mask_elines_window__ if eline_half_range is None else eline_half_range
        # Correct ranges by redshift
        if correct_wl_ranges:
            w_min = self.w_min*z_fact
            w_max = self.w_max*z_fact
            nl_w_min = self.nl_w_min*z_fact
            nl_w_max = self.nl_w_max*z_fact
            self.wl_range = [w_min, w_max]
            self.nl_wl_range = [nl_w_min, nl_w_max]
            print_verbose(f'- New wavelength range: [{w_min:.2f}, {w_max:.2f}] - Non-linear analysis: [{nl_w_min:.2f}, {nl_w_max:.2f}]', verbose=self.verbose)
        if self.e_w is not None:
            e_masks = np.empty((self.e_w.size, 2), dtype=np.float)
            eline_half_range *= 1 if self.sigma_inst is None else (self.sigma_inst if self.sigma_mean is None else self.sigma_mean)
            if eline_half_range < 4:
                eline_half_range = 4
            e_masks[:,0] = self.e_w*z_fact - eline_half_range
            e_masks[:,1] = self.e_w*z_fact + eline_half_range
            self.e_masks = e_masks
            print_verbose(f'- Update emission line masks: ', verbose=self.verbose)
            print_verbose(self.e_masks, verbose=self.verbose)
            # if needed a complete mask:
            if self.masks is not None:
                self.comp_masks = np.sort(np.append(self.masks, self.e_masks, axis=0), axis=0)
        if self.spectra is not None:
            s = self.spectra
            sel_valid_flux = (s['raw_flux'] > 0)
            if self.e_masks is not None:
                s['sel_wl_emasks'] = sel_waves(self.e_masks, s['raw_wave'])
            s['sel_wl_range'] = trim_waves(s['raw_wave'], self.wl_range)
            s['sel_nl_wl_range'] = trim_waves(s['raw_wave'], self.nl_wl_range)
            s['sel_wl'] = sel_valid_flux & s['sel_wl_range'] & s['sel_wl_masks']
            s['sel_nl_wl'] = sel_valid_flux & s['sel_wl_masks'] & s['sel_nl_wl_range'] & s['sel_wl_emasks']
            s['sel_AV'] = (s['raw_flux'] > 0) & s['sel_wl_range'] & s['sel_wl_masks'] & s['sel_wl_emasks']
            # s['sel_AV'] &= s['sel_AV_range']
            # print(s['raw_wave'][s['']])

    def _load_masks(self, w_min=None, w_max=None, nl_w_min=None, nl_w_max=None,
                    mask_list=None, elines_mask_file=None):
        """ Loads and creates the masks from the arguments and mask files
        passed by the user.
        """
        self.w_min = -np.inf if w_min is None else w_min
        self.w_max = np.inf if w_max is None else w_max
        self.nl_w_min = w_min if nl_w_min is None else nl_w_min
        self.nl_w_max = w_max if nl_w_max is None else nl_w_max
        self.elines_mask_file = elines_mask_file
        self.wl_range = [self.w_min, self.w_max]
        self.nl_wl_range = [self.nl_w_min, self.nl_w_max]
        self.e_masks = None
        self.e_w = None
        self.mask_list = mask_list
        self.masks = None
        if self.mask_list is not None:
            if isfile(self.mask_list):
                self.masks = np.loadtxt(self.mask_list)
            else:
                print(f'{basename(sys.argv[0])}: {self.mask_list}: mask list file not found')
                self.mask_list = None
        else:
            print(f'{basename(sys.argv[0])}: no mask list file')
            # read elines mask
        self.comp_masks = self.masks
        if self.elines_mask_file is not None:
            if isfile(self.elines_mask_file):
                self.e_w = np.loadtxt(self.elines_mask_file, usecols=(0))
            else:
                print(f'{basename(sys.argv[0])}: {self.elines_mask_file}: emission lines mask file not found')
                self.elines_mask_file = None
        else:
            print(f'{basename(sys.argv[0])}: no elines mask file')
        self.redshift_correct_masks(redshift=self.config.redshift)

    def _create_spectra_dict(self, wavelength, flux, eflux, min=None, max=None, ratio_master=None):
        """ Creates a dictionary with the spectral info (wavelengths, fluxes, and error in fluxes).

        XXX: EL: SFS fill the NaNs in input spectrum with the last value, e.g.:
            if f[i] == NaN:
                f[i] = f[i-1]
        """

        raw_wave = copy(wavelength)
        raw_flux = copy(flux)
        raw_eflux = copy(eflux)
        if not raw_eflux.any():
            # print_verbose('all zeros in error spectrum. Setting error_flux = 5% flux', verbose=self.verbose)
            print_verbose(f'{basename(sys.argv[0])}: eflux: all zeros in error spectrum. Setting error_flux = 5% flux', verbose=1)
            raw_eflux = 0.05*flux

        redef_max = 0
        redef_min = 0
        if (min is not None):
            redef_min = 1
        if (max is not None):
            redef_max = 1
        # if (w_min is not None) and (w_max is not None):
        #     redef = 2

        sigma_e = np.median(raw_eflux)
        print(f'-> median error in flux = {sigma_e:6.4f}')

        # correct errors, i.e. defines the maximum error
        raw_eflux = np.where(raw_eflux > sigma_e*1.5, sigma_e*1.5, raw_eflux)

        # create selection of to-fit wavelenghts
        # (i.e., it will define the number of free parameters).
        sel_valid_flux = raw_flux > 0
        sel_wl_range = trim_waves(raw_wave, self.wl_range)
        sel_nl_wl_range = trim_waves(raw_wave, self.nl_wl_range)
        sel_wl_masks = sel_waves(self.masks, raw_wave)
        sel_wl_emasks = sel_waves(self.e_masks, raw_wave)

        # create mask arrays
        sel_wl = sel_valid_flux & sel_wl_range & sel_wl_masks
        sel_nl_wl = sel_valid_flux & sel_wl_masks & sel_nl_wl_range & sel_wl_emasks
        sel_AV = sel_valid_flux & sel_wl_range & sel_wl_masks & sel_wl_emasks

        # sel_AV_range = trim_waves(raw_wave, [self.wl_range[0], 6000])
        # sel_AV &= sel_AV_range

        self.valid_flux = 2
        if not (sel_nl_wl).any():
            print('-> no valid flux inside non-linear fit wavelength range.')
            self.valid_flux = 1
            if not (sel_wl & sel_wl_emasks).any():
                print('-> no valid flux inside wavelength range.')
                self.valid_flux = 0
            else:
                print('-> force use linear fit wavelength range during non-linear fit.')
                sel_nl_wl = sel_wl & sel_wl_emasks
                self.nl_wl_range = self.wl_range

        self.ratio_master = np.ones_like(raw_wave) if ratio_master is None else ratio_master
        raw_flux /= self.ratio_master
        raw_flux[~np.isfinite(raw_flux)] = 0
        flux = copy(raw_flux)

        self.min_flux = np.nanmin(raw_flux)#.min()
        self.max_flux = np.nanmax(raw_flux)#.max()
        self.median_flux = np.median(raw_flux)
        if redef_min > 0:
            self.min_flux = min
        if redef_max > 0:
            _tmp = 3*np.median(raw_flux[sel_wl])
            self.max_flux = _tmp
            if max < _tmp:
                self.max_flux = max

        # deprecated, masked elements should be created on-the-fly
        # because all masks (sel_*) change with redshift.
        msk_wave = raw_wave[sel_wl]
        msk_flux = raw_flux[sel_wl]
        msk_eflux = raw_eflux[sel_wl]

        # change max passed by user by 3 times
        # the median of the non-masked flux
        self.spectra = {
            'orig_wave': wavelength, 'orig_flux': flux, 'orig_sigma_flux': eflux,
            'sel_wl': sel_wl, 'sel_nl_wl': sel_nl_wl, 'sel_AV': sel_AV,
            'sel_wl_range': sel_wl_range, 'sel_nl_wl_range': sel_nl_wl_range,
            'sel_wl_masks': sel_wl_masks, 'sel_wl_emasks': sel_wl_emasks,
            'raw_wave': raw_wave, 'raw_flux': raw_flux, 'raw_sigma_flux': raw_eflux,

            # deprecated, masked elements should be created on-the-fly
            'msk_wave': msk_wave, 'msk_flux': msk_flux, 'msk_sigma_flux': msk_eflux,

            # deprecated, please use 'orig_sigma_flux'
            'orig_eflux': eflux,
            # deprecated, please use 'raw_sigma_flux'
            'raw_eflux': raw_eflux,
            # deprecated, please use 'msk_sigma_flux'
            'msk_eflux': msk_eflux,

            # 'sel_AV_range': sel_AV_range,
        }

    def get_last_redshift(self):
        r = self.config.redshift
        if len(self.redshift_chain) > 0:
            r = self.redshift_chain[-1]
        return r

    def get_last_chi_sq_redshift(self):
        r = 1e12
        if len(self.chi_sq_redshift_chain) > 0:
            r = self.chi_sq_redshift_chain[-1]
        return r

    def get_last_coeffs_redshift(self):
        r = np.zeros(self.ssp_nl_fit.n_models, dtype='float')
        if len(self.coeffs_redshift_chain) > 0:
            r = self.coeffs_redshift_chain[-1]
        return r

    def get_last_sigma(self):
        r = self.config.sigma
        if len(self.sigma_chain) > 0:
            r = self.sigma_chain[-1]
        return r

    def get_last_chi_sq_sigma(self):
        r = 1e12
        if len(self.chi_sq_sigma_chain) > 0:
            r = self.chi_sq_sigma_chain[-1]
        return r

    def get_last_coeffs_sigma(self):
        r = np.zeros(self.ssp_nl_fit.n_models, dtype='float')
        if len(self.coeffs_sigma_chain) > 0:
            r = self.coeffs_sigma_chain[-1]
        return r

    def get_last_AV(self):
        r = self.config.AV
        if len(self.AV_chain) > 0:
            r = self.AV_chain[-1]
        return r

    def get_last_chi_sq_AV(self):
        r = 1e12
        if len(self.chi_sq_AV_chain) > 0:
            r = self.chi_sq_AV_chain[-1]
        return r

    def get_last_coeffs_AV(self):
        r = np.zeros(self.ssp_nl_fit.n_models, dtype='float')
        if len(self.coeffs_AV_chain) > 0:
            r = self.coeffs_AV_chain[-1]
        return r

    def _init_redshift_fit(self):
        self.redshift_chain = []
        self.coeffs_redshift_chain = []
        self.chi_sq_redshift_chain = []
        self.best_coeffs_redshift = self.get_last_coeffs_redshift()
        self.best_chi_sq_redshift = self.get_last_chi_sq_redshift()
        self.best_redshift = self.get_last_redshift()
        self.n_loops_redshift = 0
        self.e_redshift = 0

    def _init_sigma_fit(self):
        self.sigma_chain = []
        self.coeffs_sigma_chain = []
        self.chi_sq_sigma_chain = []
        self.best_coeffs_sigma = self.get_last_coeffs_sigma()
        self.best_chi_sq_sigma = self.get_last_chi_sq_sigma()
        self.best_sigma = self.get_last_sigma()
        self.n_loops_sigma = 0
        self.e_sigma = 0

    def _init_AV_fit(self):
        self.AV_chain = []
        self.coeffs_AV_chain = []
        self.chi_sq_AV_chain = []
        self.best_coeffs_AV = self.get_last_coeffs_AV()
        self.best_chi_sq_AV = self.get_last_chi_sq_AV()
        self.best_AV = self.get_last_AV()
        self.n_loops_AV = 0
        self.e_AV = 0

    def _fitting_init(self):
        self.best_chi_sq_nl_fit = 1e12
        self.best_coeffs_nl_fit = np.zeros(self.ssp_nl_fit.n_models, dtype='float')
        self._init_ssp_fit()

        # non-lin pars
        self._init_redshift_fit()
        self._init_sigma_fit()
        self._init_AV_fit()

    def update_redshift_params(self, coeffs, chi_sq, redshift):
        """
        Save the redshift fit parameters chain of tries.

        Parameters
        ----------
        coeffs : array like
            The last calculed coefficients for the SSP models.
        chi_sq : float
            The chi squared of the fit.
        redshift : float
            The last redshift.
        """
        self.coeffs_redshift_chain.append(coeffs)
        self.chi_sq_redshift_chain.append(chi_sq)
        self.redshift_chain.append(redshift)

    def update_sigma_params(self, coeffs, chi_sq, sigma):
        """
        Save the sigma fit parameters chain of tries.

        Parameters
        ----------
        coeffs : array like
            The last calculed coefficients for the SSP models.
        chi_sq : float
            The chi squared of the fit.
        sigma : float
            The last sigma.
        """
        self.coeffs_sigma_chain.append(coeffs)
        self.chi_sq_sigma_chain.append(chi_sq)
        self.sigma_chain.append(sigma)

    def update_AV_params(self, coeffs, chi_sq, AV):
        """
        Save the AV fit parameters chain of tries.

        Parameters
        ----------
        coeffs : array like
            The last calculed coefficients for the SSP models.
        chi_sq : float
            The chi squared of the fit.
        AV : float
            The last AV.
        """
        self.coeffs_AV_chain.append(coeffs)
        self.chi_sq_AV_chain.append(chi_sq)
        self.AV_chain.append(AV)

    def _multi_AV(self):
        """
        Creates the array of AVs using the AV passed by user.

        TODO: The program should be ready to work with a different
              extinction for each SSP model.
        """
        # ready to work with a different AV for each SSP.
        self.AV_arr = np.array([self.config.AV]*self.ssp.n_models, dtype=np.float)
        self.ini_AV_arr = np.array([self.config.AV]*self.ssp_nl_fit.n_models, dtype=np.float)
        self.delta_AV_arr = np.array([self.config.AV]*self.ssp.n_models, dtype=np.float)
        self.ini_delta_AV_arr = np.array([self.config.AV]*self.ssp_nl_fit.n_models, dtype=np.float)

    def _load_ssp_fits(self):
        """
        Loads the SSP models to the ConfigAutoSSP.

        The class SSPModels is used to deal the SSP models.
        """
        # if self.filename.endswith('fits'):
        self.models = SSPModels(self.filename)

        # deprecated the use of self.ssp.
        # in order to keep working the code at first instance:
        self.ssp = self.models

        if self.filename_nl_fit:
            self.models_nl_fit = SSPModels(self.filename_nl_fit)

            # deprecated the use of self.ssp_nl_fit
            # in order to keep working the code at first instance:
            self.ssp_nl_fit = self.models_nl_fit

    def fit_WLS_invmat(self, ssp, sigma_inst=None, sigma=None, redshift=None,
                       AV=None, smooth_cont=False, sel_wavelengths=None):
        """
        A wrapper to _fit_WLS_invmat setting up the spectra, masks, and extinction law.
        If smooth_cont is True smooths the continuum modeled by sigma_mean, where:

            sigma_mean^2 = sigma_inst^2 + (5000*(sigma/c))^2

        Obs.: This method calls `ssp.apply_dust_to_flux_models_obsframe()` which will set
        the models spectra to the input `AV` `ssp.flux_models_obsframe_dust`.

        Parameters
        ----------
        ssp : SSPModels class
            The class with ssp models used during the fit process.
        sigma_inst : float or None
            Sigma instrumental. If the `sigma_inst` is None, is used `self.sigma_inst`.
        redshift : float or None
            Redshift of the Observed frame. If the `redshift` is None, is used
            `self.best_redshift`.
        sigma : float or None
            Velocity dispersion (i.e. sigma) at observed frame.
            If the sigma is not set None, is used `self.best_sigma`.
        AV : float, array like or None
            Dust extinction in mag. If AV is an array, will create
            an (n_AV, n_wave) array of dust spectra. If None, is used `self.best_AV`
        smooth_cont : bool, optional
            If True smooths the continuum modeled by sigma_mean, where
            sigma_mean^2 = sigma_inst^2 + (5000*(sigma/c))^2
        sel_wavelengths : array like or None
            The selection of valid wavelengths. If None uses `self.spectra['sel_wl']` as
            the wavelength selection.

        Returns
        -------
        array like
            Coefficients of the WLS fit.
        float
            Chi square of the fit.
        """
        if sigma_inst is None:
            sigma_inst = self.sigma_inst
        if redshift is None:
            redshift = self.best_redshift
        if sigma is None:
            sigma = self.best_sigma
        if AV is None:
            AV = self.best_AV
        if sel_wavelengths is None:
            sel_wavelengths = self.spectra['sel_wl']
        msk_sigma_flux = self.spectra['raw_sigma_flux'][sel_wavelengths]
        msk_flux = self.spectra['raw_flux'][sel_wavelengths]
        # SSP models at observed frame
        ssp.to_observed(self.spectra['raw_wave'], sigma_inst=sigma_inst, sigma=sigma, redshift=redshift)

        # SSP models at observed frame with considered dust
        ssp.apply_dust_to_flux_models_obsframe(self.spectra['raw_wave']/(1 + redshift), AV, R_V=self.R_V, extlaw=self.extlaw)
        # SSP models at observed frame with considered dust and masked
        ssp_flux_models_obsframe_dust = ssp.flux_models_obsframe_dust[:, sel_wavelengths]
        coeffs_now, chi_sq, msk_model_now = _fit_WLS_invmat(flux=msk_flux,
                                                            eflux=msk_sigma_flux,
                                                            flux_models=ssp_flux_models_obsframe_dust,
                                                            verbose=self.verbose)
        if smooth_cont:
            # _w = self.spectra['raw_wave'][sel_wavelengths]
            # wl_cent = _w[_w.size//2]
            wl_cent = 5000
            sigma_mean = sigma if sigma_inst is None else np.sqrt(sigma_inst**2 + (wl_cent*sigma/__c__)**2)
            ratio = np.divide(msk_flux, msk_model_now, where=msk_model_now!=0)
            msk_model_now *= smooth_ratio(ratio, int(sigma_mean), kernel_size_factor=7*__sigma_to_FWHM__)
            chi_sq, n_free_param = calc_chi_sq(msk_flux, msk_model_now, msk_sigma_flux,
                                               ssp.n_models + 1)
        return coeffs_now, chi_sq, msk_model_now

    def fit_WLS_invmat_MC(self, ssp,
                          n_MC=20, sigma_inst=None, sigma=None, redshift=None,
                          AV=None, sel_wavelengths=None):
        """
        This method wraps a  Monte-Carlo realisation of _fit_WLS_invmat() `n_MC` times used
        during the SSP fit of an observed spectra. This process consider measured errors of
        the observed spectrum. This method models the observed spectrum without emission
        lines, i.e. it uses `self.spectra['raw_flux_no_gas']` as the input spectra).

        Obs.: This method calls `ssp.apply_dust_to_flux_models_obsframe()` which will set
        the models spectra to the input `AV` `ssp.flux_models_obsframe_dust`.

        Parameters
        ----------
        ssp : SSPModels class
            The class with ssp models used during the fit process.
        n_MC : int
            Number of Monte-Carlo loops
        sigma_inst : float or None
            Sigma instrumental. If the `sigma_inst` is None, is used `self.sigma_inst`.
        redshift : float or None
            Redshift of the Observed frame. If the `redshift` is None, is used
            `self.best_redshift`.
        sigma : float or None
            Velocity dispersion (i.e. sigma) at observed frame.
            If the sigma is not set None, is used `self.best_sigma`.
        AV : float, array like or None
            Dust extinction in mag. If AV is an array, will create
            an (n_AV, n_wave) array of dust spectra. If None, is used `self.best_AV`
        sel_wavelengths : array like or None
            The selection of valid wavelengths. If None uses `self.spectra['sel_wl']` as
            the wavelength selection.

        Returns
        -------
        array like
            Coefficients from the MC realisation.
        array like
            Chi squareds from the MC realisation.
        array like
            Models from the MC realisation.
        """

        if sigma_inst is None:
            sigma_inst = self.sigma_inst
        if redshift is None:
            redshift = self.best_redshift
        if sigma is None:
            sigma = self.best_sigma
        if AV is None:
            AV = self.best_AV
        if sel_wavelengths is None:
            sel_wavelengths = self.spectra['sel_wl']

        s = self.spectra

        # SSP models at observed frame
        ssp.to_observed(s['raw_wave'], sigma_inst=sigma_inst, sigma=sigma, redshift=redshift)
        # SSP models at observed frame with considered dust
        ssp.apply_dust_to_flux_models_obsframe(s['raw_wave']/(1 + redshift), AV, R_V=self.R_V, extlaw=self.extlaw)
        # SSP models at observed frame with considered dust and masked
        ssp_flux_models_obsframe_dust = ssp.flux_models_obsframe_dust[:, sel_wavelengths]

        msk_sigma_flux = s['raw_sigma_flux'][sel_wavelengths]  #1/(np.abs(s['msk_eflux'])**2)
        msk_flux = s['raw_flux_no_gas'][sel_wavelengths]

        coeffs = np.zeros((n_MC, ssp.n_models), dtype='float')
        chi_sqs = np.zeros(n_MC, dtype='float')
        models = np.zeros((n_MC, msk_flux.size), dtype='float')

        for i_MC in range(n_MC):
            msk_wave = s['raw_wave'][sel_wavelengths]
            rnd = np.clip(np.random.randn(msk_wave.size), -1, 1)
            noise = rnd*msk_sigma_flux
            # noise = rnd*msk_sigma_flux**2
            perturbed_flux = msk_flux + noise
            print_verbose(f'i_MC: {i_MC} - perturbed flux SN: {(perturbed_flux/msk_sigma_flux).mean()}', verbose=self.verbose)
            coeffs[i_MC], chi_sqs[i_MC], models[i_MC] = _fit_WLS_invmat(flux=perturbed_flux,
                                                                        eflux=msk_sigma_flux,
                                                                        flux_models=ssp_flux_models_obsframe_dust,
                                                                        verbose=self.verbose)
        return coeffs, chi_sqs, models

    def plot_last(self, model, sel, chi_sq, redshift, sigma, AV, filename=None):
        wvm = self.spectra['raw_wave'][sel]
        xlim = (np.nanmin(wvm)*0.95, np.nanmax(wvm)*1.05)
        if self.plot:
            if 'matplotlib.pyplot' not in sys.modules:
                from matplotlib import pyplot as plt
            else:
                plt = sys.modules['matplotlib.pyplot']
            title = r'$\chi^2={:f}$'.format(chi_sq)
            title += r' z={:f}'.format(redshift)
            title += r' $\sigma=${:f}'.format(sigma)
            title += r' A$_V$={:f}'.format(AV)
            wv = self.spectra['raw_wave']
            msk_wave = np.ma.masked_array(wv, mask=~sel)
            wave_list = [wv, wv, msk_wave, wv]
            msk_flux = np.ma.masked_array(self.spectra['raw_flux'], mask=~sel)
            msk_model = np.ma.masked_array(model, mask=~sel)
            spectra_list = [self.spectra['raw_flux'], model, msk_flux - msk_model, self.spectra['raw_flux'] - model]
            alpha_list = [0.3, 0.7, 1, 0.3]
            lw_list = [1, 2, 2, 1]
            labels = ['obs', 'model', 'msk res', 'res']
            #
            # 30.07.24
            # PATCH
            
            if (~(np.isfinite(self.max_flux))):
                self.max_flux=1000.0
            if (~(np.isfinite(self.min_flux))):
                self.min_flux=-10.0
            ymax = 1.15*self.max_flux
            ymin = 0.75*self.min_flux
            
            if self.plot == 1:
                plt.cla()
#                print(f'tests: {xlim},{ymin},{ymax}')
                plot_spectra_ax(plt.gca(), wave_list, spectra_list, title=title, xlim=xlim, ylim=(ymin, ymax), lw=lw_list, alpha=alpha_list, labels_list=labels)
                # plot_spectra_ax(plt.gca(), wave_list, spectra_list, title=title, labels_list=labels, ylim=(-1, ymax), lw=lw_list, alpha=alpha_list)
                plt.pause(0.001)
            elif (self.plot == 2) and (filename is not None):
                f, ax = plt.subplots(figsize=(_figsize_default))
                plot_spectra_ax(ax, wave_list, spectra_list, title=title, xlim=xlim, ylim=(ymin, ymax), lw=lw_list, alpha=alpha_list, labels_list=labels)
                f.savefig(filename, dpi=_plot_dpi)
                plt.close(f)

    def plot_best(self, ssp, coeffs, sel, filename=None):
        """
        Plots spectra with recent best redshift, sigma and AV.

        Parameters
        ----------
        ssp : SSPModels class
            The class with ssp models used during the fit process.
        coeffs : array like
            The last calculed coefficients for the SSP models.
        """
        if self.plot:
            if 'matplotlib.pyplot' not in sys.modules:
                from matplotlib import pyplot as plt
            else:
                plt = sys.modules['matplotlib.pyplot']
            model = self.get_best_model_from_coeffs(ssp, coeffs)
            chi_sq, _ = calc_chi_sq(self.spectra['raw_flux'][sel], model[sel], self.spectra['raw_sigma_flux'][sel],
                                    ssp.n_models + 1)
            title = r'$BEST \chi^2={:f}$'.format(chi_sq)
            title += r' A$_V$={:f}'.format(self.best_AV)
            title += r' z={:f}'.format(self.best_redshift)
            title += r' $\sigma=${:f}'.format(self.best_sigma)
            wv = self.spectra['raw_wave']
            xlim = [np.nanmin(wv[sel])*0.95, np.nanmax(wv[sel])*1.05]
            msk_wave = np.ma.masked_array(wv, mask=~sel)
            wave_list = [wv, wv, msk_wave, wv]
            msk_flux = np.ma.masked_array(self.spectra['raw_flux'], mask=~sel)
            msk_model = np.ma.masked_array(model, mask=~sel)
            spectra_list = [self.spectra['raw_flux'], model, msk_flux - msk_model, self.spectra['raw_flux'] - model]
            alpha_list = [0.3, 0.7, 1, 0.3]
            lw_list = [1, 1, 1, 1]
            labels = ['obs', 'model', 'msk res', 'res']
            ymax = 1.15*self.max_flux
            ymin = 0.75*self.min_flux
            if self.plot == 1:
                plt.cla()
                plot_spectra_ax(plt.gca(), wave_list, spectra_list, title=title, xlim=xlim, ylim=(ymin, ymax), lw=lw_list, alpha=alpha_list, labels_list=labels)
                plt.pause(0.001)
            elif (self.plot == 2) and (filename is not None):
                f, ax = plt.subplots(figsize=(_figsize_default))
                plot_spectra_ax(ax, wave_list, spectra_list, title=title, xlim=xlim, ylim=(ymin, ymax), lw=lw_list, alpha=alpha_list, labels_list=labels)
                f.savefig(filename, dpi=_plot_dpi)
                plt.close(f)


    def get_best_model_from_coeffs(self, ssp, coeffs):
        return ssp.get_model_from_coeffs(coeffs=coeffs, wavelength=self.spectra['raw_wave'],
                                         redshift=self.best_redshift, sigma=self.best_sigma, AV=self.best_AV,
                                         sigma_inst=self.sigma_inst, R_V=self.R_V, extlaw=self.extlaw)

    def loop_par_func(self, min_par, max_par, delta_par,
                      update_parameter_function=None, update_delta_function=None):
        """
        Creates the array of tries of a parameter following an `update_parameter_function` and a
        `update_delta_function`.

        Parameters
        ----------
        min_par : float
        max_par : float
        delta_par : float
        update_parameter_function : function or None
        update_delta_function : function or None

        Returns
        -------
        iterator
            Returns an iterator that loops a parameter from `min_par` to `max_par` values
            following an `update_parameter_function`.
        """
        if update_parameter_function is None:
            update_parameter_function = lambda p, d: p + d

        if update_delta_function is None:
            update_delta_function = lambda d: d

        parameter = min_par
        while (parameter < max_par):
            yield parameter, delta_par
            parameter = update_parameter_function(parameter, delta_par)
            delta_par = update_delta_function(delta_par)

    def _loop_redshift_broad(self):
        cf = self.config
        for red, delta in self.loop_par_func(cf.min_redshift, cf.max_redshift, cf.delta_redshift):
            self.n_loops_nl_fit += 1
            self.n_loops_redshift += 1
            yield red, delta

    def _loop_redshift_narrow(self):
        cf = self.config
        for red, delta in self.loop_par_func(min_par=self.best_redshift - 1.5*cf.delta_redshift,
                                             max_par=self.best_redshift + 1.5*cf.delta_redshift,
                                             delta_par=0.1*cf.delta_redshift,
                                             update_parameter_function=lambda p, d: p + d*np.random.rand()):
            self.n_loops_nl_fit += 1
            self.n_loops_redshift += 1
            yield red, delta

    def _fit_redshift(self, ssp=None, correct_wl_ranges=False):
        """
        Fit the redshift of an observed spectra. This method updates the redshift chain.

        Parameters
        ..........
        ssp : SSPModels class or None
            The class with ssp models used during the fit process. Defaults to `self.models_nl_fit`.
        """
        if ssp is None:
            ssp = self.models_nl_fit
        cf = self.config
        s = self.spectra
        print_verbose('', verbose=self.verbose)
        print_verbose('-=[ BEGIN fit redshift ]=---------------------------------------------', verbose=self.verbose)
        # print(f'D_REDSHIFT = {cf.delta_redshift} MIN = {cf.min_redshift} MAX = {cf.max_redshift}')
        # if cf.MIN_W == 0:
        #     cf.MIN_W = cf.wl_range[0]
        # if cf.MAX_W == 0:
        #     cf.MAX_W = cf.wl_range[1]
        n_loops = 0
        min_chi_sq = self.best_chi_sq_redshift
        print_verbose('', verbose=self.verbose)
        for red, delta_red in self._loop_redshift_broad():
            self.redshift_correct_masks(redshift=red, correct_wl_ranges=correct_wl_ranges)
            # self.correct_elines_mask(redshift=red)
            n_loops += 1
            print_verbose(f'redshift ---------------- {red}', verbose=self.verbose)
            coeffs_now, chi_sq, msk_model_now = self.fit_WLS_invmat(ssp=ssp, redshift=red,
                                                                    smooth_cont=True,
                                                                    sel_wavelengths=s['sel_nl_wl'])
            self.update_redshift_params(coeffs=coeffs_now, chi_sq=chi_sq, redshift=red)
            model_now = np.dot(coeffs_now, ssp.flux_models_obsframe_dust)
            # model_now = ssp.get_model_from_coeffs(
            #     coeffs=coeffs_now, wavelength=s['raw_wave'],
            #     redshift=red, sigma=self.best_sigma, AV=self.best_AV, sigma_inst=self.sigma_inst,
            #     R_V=self.R_V, extlaw=self.extlaw,
            # )
            self.plot_last(model=model_now, sel=s['sel_nl_wl'], redshift=red, chi_sq=chi_sq, AV=self.best_AV, sigma=self.best_sigma)
            print_verbose(f'last_chi_sq ------------- {self.get_last_chi_sq_redshift()}', verbose=self.verbose)
            print_verbose(f'last_coeffs ------------- {self.get_last_coeffs_redshift()}', verbose=self.verbose)
            print_verbose(f'best_chi_sq_redshift ---- {self.best_chi_sq_redshift}', verbose=self.verbose)
            print_verbose(f'best_coeffs_redshift ---- {self.best_coeffs_redshift}', verbose=self.verbose)
            # if found a lower chi squared: update_bests
            if (n_loops > 1) and chi_sq < self.best_chi_sq_redshift:
                print_verbose(f'found min chi_sq ]=-------- {chi_sq} ', verbose=self.verbose)
                self.best_chi_sq_redshift = chi_sq
                print_verbose(f'updating best coeffs ]=---- {coeffs_now} ')
                self.best_coeffs_redshift = coeffs_now
                print_verbose(f'updating best redshift ]=-- {red} ')
                self.best_redshift = red
            print_verbose('', verbose=self.verbose)
        print_verbose(f'n_loops ------------------ {n_loops}', verbose=self.verbose)
        print_verbose(f'n_loops_redshift --------- {self.n_loops_redshift}', verbose=self.verbose)
        print_verbose(f'n_loops_nl_fit ----------- {self.n_loops_nl_fit}', verbose=self.verbose)
        print_verbose(f'best_chi_sq_redshift ----- {self.best_chi_sq_redshift}', verbose=self.verbose)
        print_verbose(f'best_redshift ------------ {self.best_redshift}', verbose=self.verbose)
        print_verbose(f'best_coeffs_redshift ----- {self.best_coeffs_redshift}', verbose=self.verbose)
        print_verbose('-------------------------------------------=[ END 1st fit redshift ]=-', verbose=self.verbose)
        self.e_redshift = delta_red
        # self.plot_best(ssp, coeffs=self.best_coeffs_redshift, sel=s['sel_nl_wl'])
        print_verbose('', verbose=self.verbose)
        print_verbose('-=[ BEGIN redshift fine tune ]=---------------------------------------', verbose=self.verbose)
        print_verbose('-------------------------------------------=[ END 1st fit redshift ]=-', verbose=self.verbose)
        n_first_loop = n_loops
        n_loops = 0
        if self.plot > 0:
            if 'matplotlib.pyplot' not in sys.modules:
                from matplotlib import pyplot as plt
            else:
                plt = sys.modules['matplotlib.pyplot']
            if self.plot == 1:
                plt.cla()
                ax = plt.gca()
            elif self.plot == 2:
                f, ax = plt.subplots(figsize=(_figsize_default))
            ax.scatter(self.redshift_chain, self.chi_sq_redshift_chain)
            ax.plot(self.best_redshift, self.best_chi_sq_redshift, 'kx')
            ax.set_xlabel('z')
            ax.set_ylabel(r'$\chi^2$')
            if self.plot == 1:
                plt.pause(0.001)
            if self.plot == 2:
                f.savefig('redshift_fit_broad.png', dpi=_plot_dpi)
                plt.close(f)

        # search for the redshift around best value previously found.
        for red, delta_red in self._loop_redshift_narrow():
            self.redshift_correct_masks(redshift=red, correct_wl_ranges=correct_wl_ranges)
            # self.correct_elines_mask(redshift=red)
            n_loops += 1
            print_verbose(f'redshift ------------------ {red}', verbose=self.verbose)
            coeffs_now, chi_sq, msk_model_now = self.fit_WLS_invmat(ssp=ssp, redshift=red,
                                                                    smooth_cont=True,
                                                                    sel_wavelengths=s['sel_nl_wl'])
            self.update_redshift_params(coeffs=coeffs_now, redshift=red, chi_sq=chi_sq)
            # model_now = ssp.get_model_from_coeffs(
            #     coeffs=coeffs_now, wavelength=s['raw_wave'],
            #     redshift=red, sigma=self.best_sigma, AV=self.best_AV, sigma_inst=self.sigma_inst,
            #     R_V=self.R_V, extlaw=self.extlaw,
            # )
            model_now = np.dot(coeffs_now, ssp.flux_models_obsframe_dust)
            self.plot_last(model=model_now, redshift=red, chi_sq=chi_sq, AV=self.best_AV, sigma=self.best_sigma, sel=s['sel_nl_wl'])
            print_verbose(f'last_chi_sq --------------- {self.get_last_chi_sq_redshift()}', verbose=self.verbose)
            print_verbose(f'last_coeffs --------------- {self.get_last_coeffs_redshift()}', verbose=self.verbose)
            print_verbose(f'best_chi_sq_redshift ------ {self.best_chi_sq_redshift}', verbose=self.verbose)
            print_verbose(f'best_coeffs_redshift ------ {self.best_coeffs_redshift}', verbose=self.verbose)
            redshift_chain = self.redshift_chain[n_first_loop + 1:]
            chi_sq_chain = self.chi_sq_redshift_chain[n_first_loop + 1:]
            if len(chi_sq_chain) > 2:
                fa, fb, fc = chi_sq_chain[-3:]
                if (fb < fa) & (fb < fc) & (fb <= self.best_chi_sq_redshift):
                    _red, _e_red = hyperbolic_fit_par(redshift_chain, chi_sq_chain, verbose=self.verbose)
                    # coeffs, chi_sq, msk_model_now = self.fit_WLS_invmat(ssp=ssp,
                    #                                                     redshift=_red,
                    #                                                     smooth_cont=True,
                    #                                                     sel_wavelengths=s['sel_nl_wl'])
                    print_verbose(f'found min chi_sq ]=-------- ' , verbose=self.verbose)
                    print_verbose(f'update best chi_sq ]=------ {chi_sq}' , verbose=self.verbose)
                    self.best_chi_sq_redshift = chi_sq
                    print_verbose(f'updating best coeffs ]=---- {coeffs_now}', verbose=self.verbose)
                    self.best_coeffs_redshift = coeffs_now
                    print_verbose(f'updating best redshift ]=-- {_red}', verbose=self.verbose)
                    self.best_redshift = _red
                    print_verbose(f'sigma best redshift ]=----- {_e_red}', verbose=self.verbose)
                    self.e_redshift = _e_red
                    # self.update_nl_fit_params(coeffs=coeffs_now, redshift=red, chi_sq=chi_sq)
            print_verbose('', verbose=self.verbose)
        print_verbose('---------------------------------------=[ END redshift fine tune ]=-', verbose=self.verbose)
        # print(f'REDSHIFT = {self.best_redshift} +- {self.e_redshift}')
        print_verbose(f'n_loops ------------------- {n_loops}', verbose=self.verbose)
        print_verbose(f'n_loops_redshift ---------- {self.n_loops_redshift}', verbose=self.verbose)
        print_verbose(f'n_loops_nl_fit ------------ {self.n_loops_nl_fit}', verbose=self.verbose)
        print_verbose(f'best_chi_sq_redshift ------ {self.best_chi_sq_redshift}', verbose=self.verbose)
        print_verbose(f'best_coeffs_redshift ------ {self.best_coeffs_redshift}', verbose=self.verbose)
        print_verbose('-----------------------------------------------=[ END fit redshift ]=-', verbose=self.verbose)
        self.plot_best(ssp, coeffs=self.best_coeffs_redshift, sel=s['sel_nl_wl'], filename='best_redshift.png')
        if self.plot > 0:
            if 'matplotlib.pyplot' not in sys.modules:
                from matplotlib import pyplot as plt
            else:
                plt = sys.modules['matplotlib.pyplot']
            if self.plot == 1:
                plt.cla()
                ax = plt.gca()
            elif self.plot == 2:
                f, ax = plt.subplots(figsize=(_figsize_default))
            ax.scatter(self.redshift_chain[n_first_loop + 1:], self.chi_sq_redshift_chain[n_first_loop + 1:])
            ax.plot(self.best_redshift, self.best_chi_sq_redshift, 'kx')
            ax.set_xlabel('z')
            ax.set_ylabel(r'$\chi^2$')
            if self.plot == 1:
                plt.pause(0.001)
            elif self.plot == 2:
                f.savefig('redshift_fit_narrow.png', dpi=_plot_dpi)
                plt.close(f)

    def _loop_sigma_broad(self, guided=False):
        cf = self.config

        update_parameter_function = lambda p, d: p + d*(np.random.rand() - 0.25*np.random.rand())
        # update_parameter_function = lambda p, d: p + d),

        # use a user-defined middle value for the sigma (useful for guiding)
        if guided:
            # update_parameter_function = lambda p, d: p + d*np.random.rand()
            update_delta_function = lambda d: d*(1 + 0.2*np.random.rand())
        else:
            # update_delta_function = lambda d: d*(1+0.05*np.random.normal()):
            update_delta_function = lambda d: d*(1 + 0.15*np.random.rand())

        min_sigma = cf.min_sigma
        max_sigma = cf.max_sigma
        delta_sigma = cf.delta_sigma

        for sigma, delta in self.loop_par_func(min_sigma, max_sigma, delta_sigma,
                                               update_parameter_function=update_parameter_function,
                                               update_delta_function=update_delta_function):
            self.n_loops_sigma += 1
            self.n_loops_nl_fit += 1
            yield sigma, delta

    def _loop_sigma_narrow(self, guided=False):
        cf = self.config
        half_range = 1.5
        delta_frac = 0.33
        min_sigma = self.best_sigma - half_range*cf.delta_sigma
        max_sigma = self.best_sigma + half_range*cf.delta_sigma
        if min_sigma <= 0.0: min_sigma = 10.0
        if max_sigma > cf.max_sigma: max_sigma = cf.max_sigma
        delta_sigma = delta_frac*cf.delta_sigma

        if guided:
            update_delta_function = lambda d: d + 0.05*d*np.random.rand()
        else:
            update_delta_function = lambda d: d*(1 + 0.05*np.random.rand())
        for sigma, delta in self.loop_par_func(min_par=min_sigma,
                                               max_par=max_sigma,
                                               delta_par=delta_sigma,
                                               update_delta_function=update_delta_function):
            self.n_loops_sigma += 1
            self.n_loops_nl_fit += 1
            yield sigma, delta

    def _fit_sigma(self, guided=False, ssp=None):
        # observations
        wl = self.spectra["raw_wave"]
        fl = self.spectra["raw_flux"]
        sg = self.spectra["raw_eflux"]
        valid_mask = self.spectra["sel_nl_wl"] & (sg != 0)
        # model templates
        if ssp is None:
            ssp = self.models_nl_fit

        # dusty models
        models_obs = spec_apply_dust(
            ssp.wavelength,
            ssp.flux_models,
            self.best_AV, R_V=self.R_V, extlaw=self.extlaw
            )
        wl_cent = ssp.wavelength[ssp.wavelength.size//2]
        # instrumental models
        if self.sigma_inst is not None and self.sigma_inst > 0.01:
            kernel = norm.pdf(ssp.wavelength, loc=wl_cent, scale=self.sigma_inst)
            for j in range(ssp.flux_models.shape[0]):
                models_obs[j] = convolve(ssp.wavelength, models_obs[j], kernel)

        for losvd, _ in self._loop_sigma_broad(guided):
        # for losvd in np.arange(self.config.min_sigma, self.config.max_sigma+self.config.delta_sigma, self.config.delta_sigma):

            # EADL: If no sigma_inst the input sigma should be in AA
            # In this case here we convert from Angstroms -> km/s
            if self.sigma_inst is None:
                losvd = losvd*__c__/wl_cent

            if losvd > 0:
                kernel = norm.pdf(
                    ssp.wavelength,
                    loc=wl_cent,
                    scale=losvd/__c__*wl_cent
                    )
            else:
                kernel = None

            models_conv = np.zeros((models_obs.shape[0], wl.size))
            for j in range(models_obs.shape[0]):
                models_conv[j] = np.interp(
                    wl,
                    ssp.wavelength * (1+self.best_redshift),
                    convolve(ssp.wavelength, models_obs[j], kernel)
                    )

            A = models_conv[:, valid_mask] / sg[valid_mask]
            b = fl[valid_mask] / sg[valid_mask]
            w, _ = opt.nnls(A.T, b)

            model = (models_conv * w[:, None]).sum(axis=0)
            x = ((model - fl)[valid_mask]**2 / sg[valid_mask]**2).sum()
            x /= (valid_mask.sum() - 1)

            self.coeffs_sigma_chain.append(w)
            self.chi_sq_sigma_chain.append(x)
            self.sigma_chain.append(losvd)

            # self.plot_last(model)
            self.plot_last(model=model, sel=self.spectra["sel_nl_wl"], redshift=self.best_redshift, chi_sq=x, AV=self.best_AV, sigma=losvd)

        idx_best = np.argmin(self.chi_sq_sigma_chain)
        self.best_coeffs_sigma = self.coeffs_sigma_chain[idx_best]
        self.best_chi_sq_sigma = self.chi_sq_sigma_chain[idx_best]
        self.best_sigma = self.sigma_chain[idx_best]

        # here we convert from km/s -> Angstroms
        if self.sigma_inst is None:
            self.best_sigma = self.best_sigma*wl_cent/__c__

        self.e_sigma = self.config.delta_sigma
        self.plot_best(ssp, coeffs=self.best_coeffs_sigma, sel=self.spectra['sel_nl_wl'], filename='best_sigma.png')

        # write fitting chain for debugging --------------------------------------------------------
        # summary_sigma = np.column_stack((
        #     self.sigma_chain,
        #     self.chi_sq_sigma_chain,
        #     self.chi_sq_sigma_chain
        #     ))
        # np.savetxt(self.out_file+".txt", summary_sigma)
        # ------------------------------------------------------------------------------------------

    def _fit_sigma_rnd(self, guided=False, ssp=None, calc_coeffs=True, medres_merit=False):
        """
        Fit the velocity dispersion (sigma) of an observed spectra. This method updates
        the sigma chain.

        Parameters
        ..........
        ssp : SSPModels class or None
            The class with ssp models used during the fit process. Defaults to `self.models_nl_fit`.
        """
        if ssp is None:
            ssp = self.models_nl_fit
        cf = self.config
        s = self.spectra
        print_verbose('', verbose=self.verbose)
        print_verbose('-=[ BEGIN fit sigma ]=------------------------------------------------', verbose=self.verbose)
        if medres_merit:
            print_verbose('- using median(abs(res/(np.abs(res) + obs))) as merit function', verbose=self.verbose, level=0)
        ### Here is used flux_masked2 in the perl version.
        # masked2 is a mask that includes other wavelengths limits passed by user.
        # TODO: min_wave2 and max_wave2 it is now implemented
        n_loops = 0
        med_chain = []
        _best_med = 1e12
        msk_flux = s['raw_flux'][s['sel_nl_wl']]
        msk_sigma_flux = s['raw_sigma_flux'][s['sel_nl_wl']]
        delta_sigma = cf.delta_sigma
        delta_chain = []
        for sigma, delta_sigma in self._loop_sigma_broad(guided=guided):
            delta_chain.append(delta_sigma)
            n_loops += 1
            print_verbose(f'sigma -------------------- {sigma}', verbose=self.verbose)
            ##############################################################################
            # EL: The calculation of coefficients in the first loop of the sigma
            # search is a bad idea. does not work because creates the 'box effect'
            # see: http://132.248.1.15:8001/~lacerda/LOSVD_input_x_sim.png
            ##############################################################################
            # if self.sigma_inst is None:
            #     coeffs_now, chi_sq, msk_model_now = self.fit_WLS_invmat(ssp=ssp,
            #                                                             sigma=sigma,
            #                                                             smooth_cont=True,
            #                                                             sel_wavelengths=s['sel_nl_wl'])
            #     model_now = np.dot(coeffs_now[coeffs_now > 0], ssp.flux_models_obsframe_dust)
            # else:
            coeffs_now = self.best_coeffs_sigma
            model_now = ssp.get_model_from_coeffs(coeffs=coeffs_now,
                                                  wavelength=s['raw_wave'],
                                                  sigma=sigma,
                                                  redshift=self.best_redshift,
                                                  sigma_inst=self.sigma_inst,
                                                  AV=self.best_AV,
                                                  R_V=self.R_V,
                                                  extlaw=self.extlaw)
            msk_model_now = model_now[s['sel_nl_wl']]
            chi_sq, _ = calc_chi_sq(msk_flux, msk_model_now, msk_sigma_flux, ssp.n_models + 1)
            ##############################################################################
            # minimize the median residual instead the chisq
            _res = s['raw_flux'] - model_now
            _res = _res[s['sel_nl_wl']]
            # _res = msk_flux - msk_model_now
            _res = _res/(np.abs(_res) + msk_flux)
            _res[~np.isfinite(_res)] = 0
            _med = pdl_stats(np.abs(_res))[_STATS_POS['median']]

            med_chain.append(_med)
            if (len(med_chain) > 1) & (_med <_best_med):
                _best_med = _med

            if medres_merit:
                chi_sq = _med
            ##############################################################################
            self.update_sigma_params(coeffs=coeffs_now, sigma=sigma, chi_sq=chi_sq)
            # self.plot_last_sigma(model_now)
            self.plot_last(model=model_now, sel=s['sel_nl_wl'], redshift=self.best_redshift, chi_sq=chi_sq, AV=self.best_AV, sigma=sigma)
            print_verbose(f'last_chi_sq -------------- {self.get_last_chi_sq_sigma()}', verbose=self.verbose)
            print_verbose(f'best_chi_sq_sigma -------- {self.best_chi_sq_sigma}', verbose=self.verbose)
            chi_chain = self.chi_sq_sigma_chain
            sigma_chain = self.sigma_chain
            n_chi = len(chi_chain)
            # if n_chi == 1:
            #     print_verbose(f'first min chi_sq ]=-------- {chi_sq}', verbose=self.verbose)
            #     self.best_chi_sq_sigma = chi_sq
            # elif n_chi > 1:
            if (len(sigma_chain) > 1) & (chi_sq < self.best_chi_sq_sigma):
            # if (len(med_chain) > 1) & (_med <_best_med):
                # _best_med = _med
                print_verbose(f'found min chi_sq ]=-------- {chi_sq}', verbose=self.verbose)
                self.best_chi_sq_sigma = chi_sq
                print_verbose(f'updating best coeffs ]=---- {coeffs_now} ')
                self.best_coeffs_sigma = coeffs_now
                print_verbose(f'updating best sigma ]=----- {sigma}', verbose=self.verbose)
                self.best_sigma = sigma
            print_verbose('', verbose=self.verbose)
        print_verbose(f'n_loops ------------------- {n_loops}', verbose=self.verbose)
        print_verbose(f'n_loops_sigma ------------- {self.n_loops_sigma}', verbose=self.verbose)
        print_verbose(f'n_loops_nl_fit ------------ {self.n_loops_nl_fit}', verbose=self.verbose)
        print_verbose(f'best_chi_sq_sigma --------- {self.best_chi_sq_sigma}', verbose=self.verbose)
        print_verbose(f'best_sigma ---------------- {self.best_sigma}', verbose=self.verbose)
        print_verbose('----------------------------------------------=[ END 1st fit sigma ]=-', verbose=self.verbose)
        self.e_sigma = delta_sigma
        # self.plot_best(ssp, coeffs=self.best_coeffs_sigma, sel=s['sel_nl_wl'])
        if self.plot > 0:
            if 'matplotlib.pyplot' not in sys.modules:
                from matplotlib import pyplot as plt
            else:
                plt = sys.modules['matplotlib.pyplot']
            if self.plot == 1:
                plt.cla()
                ax = plt.gca()
            elif self.plot == 2:
                f, ax = plt.subplots(figsize=(_figsize_default))
            ax.plot(self.sigma_chain, self.chi_sq_sigma_chain, 'o-')
            ax.axvline(self.best_sigma, c='k', ls='--', label='best sigma')
            ax.set_xlabel(r'$\sigma^*$')
            ax.set_ylabel(r'$\chi^2$')
            if self.plot == 1:
                plt.pause(0.001)
            elif self.plot == 2:
                f.savefig('sigma_fit_broad.png', dpi=_plot_dpi)
                plt.close(f)

            # TO COMPARE med_res AND chisq MERIT FUNCTION UNCOMMENT HERE
            # f, ax = plt.subplots(figsize=(_figsize_default))
            # ax.plot(self.sigma_chain, med_chain/_best_med, 'go-', label='SSMF')
            # ax.plot(self.sigma_chain, self.chi_sq_sigma_chain/self.best_chi_sq_sigma, 'bo-', label='chisq')
            # ax.axvline(self.sigma_chain[np.argmin(med_chain)], c='g', ls='--', lw=3, label='best SSMF sigma')
            # ax.axvline(self.best_sigma, c='k', ls='--', label='best sigma')
            # ax.set_xlabel(r'$\sigma^*$')
            # ax.set_ylabel(r'merit/best_merit')
            # ax.legend()
            # f.savefig('sigma_fit_broad_compare.png', dpi=_plot_dpi)
            # plt.close(f)

        n_first_loop = n_loops
        n_loops = 0
        print_verbose('-=[ sigma fine tune ]=-', verbose=self.verbose)
        smooth_cont = False
        if self.sigma_inst is None:
            smooth_cont = True
            calc_coeffs = True
        delta_chain = []
        for sigma, delta_sigma in self._loop_sigma_narrow(guided=guided):
            delta_chain.append(delta_sigma)
            print_verbose(f'sigma -------------------- {sigma}', verbose=self.verbose)
            n_loops += 1
            # BUFGIX EL: (2020-06-08)
            #   added smooth_cont=True to fit_WLS_invmat (which will do the equivalent
            #   job of fit_ssp_lin_no_zero_no_cont() in perl)
            if calc_coeffs:
                coeffs_now, chi_sq, msk_model_now = self.fit_WLS_invmat(ssp=ssp,
                                                                        sigma=sigma,
                                                                        smooth_cont=smooth_cont,
                                                                        sel_wavelengths=s['sel_nl_wl'])
                model_now = np.dot(coeffs_now, ssp.flux_models_obsframe_dust)
            else:
                coeffs_now = self.best_coeffs_sigma
                model_now = ssp.get_model_from_coeffs(coeffs=coeffs_now,
                                                      wavelength=s['raw_wave'],
                                                      sigma=sigma,
                                                      redshift=self.best_redshift,
                                                      sigma_inst=self.sigma_inst,
                                                      AV=self.best_AV,
                                                      R_V=self.R_V,
                                                      extlaw=self.extlaw)
                msk_model_now = model_now[s['sel_nl_wl']]
                chi_sq, _ = calc_chi_sq(msk_flux, msk_model_now, msk_sigma_flux, ssp.n_models + 1)
            ##############################################################################
            # minimize the median residual instead the chisq
            _res = s['raw_flux'] - model_now
            _res = _res[s['sel_nl_wl']]
            # _res = msk_flux - msk_model_now
            _res = _res/(np.abs(_res) + msk_flux)
            _res[~np.isfinite(_res)] = 0
            _med = pdl_stats(np.abs(_res))[_STATS_POS['median']]
            med_chain.append(_med)
            if guided and n_loops == 1:
                _best_med = _med
            if (n_loops > 0) and (_med < _best_med):
                _best_med = _med
            if medres_merit:
                chi_sq = _med
            ##############################################################################
            self.update_sigma_params(coeffs=coeffs_now, sigma=sigma, chi_sq=chi_sq)
            self.plot_last(model=model_now, sel=s['sel_nl_wl'], redshift=self.best_redshift, chi_sq=chi_sq, AV=self.best_AV, sigma=sigma)
            # self.plot_last_sigma(model_now)
            print_verbose(f'last_chi_sq ------------- {self.get_last_chi_sq_sigma()}', verbose=self.verbose)
            print_verbose(f'last_coeffs ------------- {self.get_last_coeffs_sigma()}', verbose=self.verbose)
            print_verbose(f'best_chi_sq_sigma ------- {self.best_chi_sq_sigma}', verbose=self.verbose)
            print_verbose(f'best_coeffs_sigma ------- {self.best_coeffs_sigma}', verbose=self.verbose)
            # Perl version of disp_min forces the best sigma to be, at least,
            # the first sigma of this narrow sigma search loop.
            if guided and n_loops == 1:
                self.best_sigma = sigma
                # _best_med = _med
                self.best_chi_sq_sigma = chi_sq
                self.best_coeffs_sigma = coeffs_now
            # if (n_loops > 0) and (_med < _best_med):
            if (n_loops > 0) and (chi_sq < self.best_chi_sq_sigma):
                self.best_sigma = sigma
                # _best_med = _med
                self.best_chi_sq_sigma = chi_sq
                self.best_coeffs_sigma = coeffs_now
                print_verbose(f'found min chi_sq ]=-------- {chi_sq}', verbose=self.verbose)
                print_verbose(f'updating best coeffs ]=---- {coeffs_now}', verbose=self.verbose)
                print_verbose(f'updating best sigma ]=----- {sigma}', verbose=self.verbose)
        _sigma = self.get_last_sigma()
        if _sigma != self.best_sigma:
            # delta_y = _med - _best_med
            delta_y = chi_sq - self.best_chi_sq_sigma
            # print(delta_y)
            # delta_y /= (msk_flux/msk_sigma_flux).mean() if not medres_merit else 1
            delta_x = _sigma - self.best_sigma
            slope = delta_y/delta_x
            if slope != 0:
                e_sigma = self.best_sigma/slope
                if self.sigma_inst is not None:
                    e_sigma /= __c__
                if guided or (self.sigma_inst is None):
                    e_sigma /= 10
            else:
                e_sigma = np.median(delta_chain)
                # e_sigma = cf.delta_sigma
        else:
            e_sigma = np.median(delta_chain)
            # e_sigma = cf.delta_sigma
        e_sigma = np.abs(e_sigma)
        self.best_sigma = np.abs(self.best_sigma)
        # The error in the sigma guided version
        if guided:
            sigma_chain = self.sigma_chain[n_first_loop + 1:]
            chi_chain = self.chi_sq_sigma_chain[n_first_loop + 1:]
            sigma_s = np.asarray(sigma_chain)
            chi_s = np.asarray(chi_chain)
            # med_s = np.asarray(med_chain[n_first_loop + 1:])
            # _s = sigma_s/med_s**4
            _s = sigma_s/chi_s**4
            # _m = 1/med_s**4
            _c = 1/chi_s**4
            sigma_sum = _s.sum()
            chi_sum = _c.sum()
            # med_sum = _m.sum()
            if chi_sum != 0:
                sigma_sum /= chi_sum
            # if med_sum != 0:
            #     sigma_sum /= med_sum
            st_sigma = pdl_stats(_s)
            st_chi = pdl_stats(_c)
            # st_med = pdl_stats(_m)
            # e_sigma = 0.33*(st_sigma[_STATS_POS['pRMS']]/st_med[_STATS_POS['mean']])
            e_sigma = 0.33*(st_sigma[_STATS_POS['pRMS']]/st_chi[_STATS_POS['mean']])
            if np.isnan(e_sigma):
                e_sigma = 0.5*sigma
        self.e_sigma = e_sigma
        print_verbose(f'n_loops ------------------- {n_loops}', verbose=self.verbose)
        print_verbose(f'n_loops_sigma ------------- {self.n_loops_sigma}', verbose=self.verbose)
        print_verbose(f'n_loops_nl_fit ------------ {self.n_loops_nl_fit}', verbose=self.verbose)
        print_verbose(f'best_chi_sq_sigma --------- {self.best_chi_sq_sigma}', verbose=self.verbose)
        print_verbose(f'best_sigma ---------------- {self.best_sigma}', verbose=self.verbose)
        print_verbose(f'best_coeffs_sigma --------- {self.best_coeffs_sigma}', verbose=self.verbose)
        print_verbose('--------------------------------------------------=[ END fit sigma ]=-', verbose=self.verbose)
        self.plot_best(ssp, coeffs=self.best_coeffs_sigma, sel=s['sel_nl_wl'], filename='best_sigma.png')
        if self.plot > 0:
            if 'matplotlib.pyplot' not in sys.modules:
                from matplotlib import pyplot as plt
            else:
                plt = sys.modules['matplotlib.pyplot']

            sigma_chain_narrow = self.sigma_chain[n_first_loop + 1:]
            med_chain_narrow = med_chain[n_first_loop + 1:]
            chisq_chain_narrow = self.chi_sq_sigma_chain[n_first_loop + 1:]

            if self.plot == 1:
                plt.cla()
                ax = plt.gca()
            elif self.plot == 2:
                f, ax = plt.subplots(figsize=(_figsize_default))
            ax.plot(sigma_chain_narrow, chisq_chain_narrow, 'o-')
            ax.axvline(self.best_sigma, c='k', ls='--', label='best sigma')
            ax.set_xlabel(r'$\sigma^*$')
            ax.set_ylabel(r'$\chi^2$')
            if self.plot == 1:
                plt.pause(0.001)
            elif self.plot == 2:
                f.savefig('sigma_fit_narrow.png', dpi=_plot_dpi)
                plt.close(f)

            # TO COMPARE med_res AND chisq MERIT FUNCTION UNCOMMENT HERE
            # f, ax = plt.subplots(figsize=(_figsize_default))
            # ax.plot(sigma_chain_narrow, med_chain_narrow/_best_med, 'go-', label='SSMF')
            # ax.plot(sigma_chain_narrow, chisq_chain_narrow/self.best_chi_sq_sigma, 'bo-', label='chisq')
            # ax.axvline(sigma_chain_narrow[np.argmin(med_chain_narrow)], c='g', ls='--', label='best SSMF sigma')
            # ax.axvline(self.best_sigma, c='k', ls='--', label='best sigma')
            # ax.set_xlabel(r'$\sigma^*$')
            # ax.set_ylabel(r'merit/best_merit')
            # ax.legend()
            # f.savefig('sigma_fit_narrow_compare.png', dpi=_plot_dpi)
            # plt.close(f)

    def _fit_AV(self, ssp=None):
        """
        Fits the dust extinction parameter (AV) of an observed spectra. This method updates
        the AV chain.

        Parameters
        ----------
        ssp : SSPModels class or None
            The class with ssp models used during the fit process. Defaults to `self.models_nl_fit`.
        """
        if ssp is None:
            ssp = self.models_nl_fit
        cf = self.config
        s = self.spectra
        print_verbose('', verbose=self.verbose)
        print_verbose('-=[ BEGIN fit AV ]=---------------------------------------------------', verbose=self.verbose)
        # print(f'D_AV = {cf.delta_AV} MIN = {cf.min_AV} MAX = {cf.max_AV}')
        self.best_chi_sq_AV = self.get_last_chi_sq_AV()
        force_update = True
        n_loops = 0
        update_parameter_function = lambda x, d: x + d*np.random.random_sample(2)[0]
        for AV, delta_AV in self.loop_par_func(cf.min_AV, cf.max_AV, cf.delta_AV,
                                               update_parameter_function=update_parameter_function):
            print_verbose(f'AV ----------------------- {AV}', verbose=self.verbose)
            # print(f'AV ----------------------- {AV}')
            n_loops += 1
            self.n_loops_AV += 1
            self.n_loops_nl_fit += 1
            coeffs_now, chi_sq, msk_model_now = self.fit_WLS_invmat(ssp=ssp, AV=AV,
                                                                    sel_wavelengths=s['sel_AV'])
            self.update_AV_params(coeffs=coeffs_now, AV=AV, chi_sq=chi_sq)
            # model_now = ssp.get_model_from_coeffs(cf, coeffs_now, s['raw_wave'],
            #                                       redshift=self.best_redshift, sigma=self.best_sigma,
            #                                       AV=AV, R_V=self.R_V, extlaw=self.extlaw)
            model_now = np.dot(coeffs_now, ssp.flux_models_obsframe_dust)
            # model_now = ssp.get_model_from_coeffs(coeffs=coeffs_now, wavelength=s['raw_wave'], AV=AV,
            #                                       redshift=self.best_redshift, sigma=self.best_sigma,
            #                                       sigma_inst=self.sigma_inst, R_V=self.R_V, extlaw=self.extlaw)
            self.plot_last(model=model_now, sel=s['sel_AV'], redshift=self.best_redshift, chi_sq=chi_sq, AV=AV, sigma=self.best_sigma)
            # self.plot_last_AV(model_now)
            print_verbose(f'last_chi_sq -------------- {self.get_last_chi_sq_AV()}', verbose=self.verbose)
            print_verbose(f'last_coeffs -------------- {self.get_last_coeffs_AV()}', verbose=self.verbose)
            print_verbose(f'best_chi_sq_AV ----------- {self.best_chi_sq_AV}', verbose=self.verbose)
            print_verbose(f'best_coeffs_AV ----------- {self.best_coeffs_AV}', verbose=self.verbose)
            if force_update and (chi_sq < self.best_chi_sq_AV):
                # print(f'force update AV: {AV} chi_sq: {chi_sq}')
                self.best_chi_sq_AV = chi_sq
                self.best_coeffs_AV = coeffs_now
                self.best_AV = AV
            if len(self.chi_sq_AV_chain) > 2:
                fa, fb, fc = self.chi_sq_AV_chain[-3:]
                if (fb < fa) & (fb < fa) & (fb <= self.best_chi_sq_AV):
                    _AV, _e_AV = hyperbolic_fit_par(self.AV_chain, self.chi_sq_AV_chain, verbose=self.verbose)
                    # print(f'found best: {_AV} chi_sq: {chi_sq}', end=' ')
                    # coeffs_now, chi_sq, msk_model_now = self.fit_WLS_invmat(ssp=ssp, AV=_AV, sel_wavelengths=s['sel_AV'])
                    print_verbose(f'found min chi_sq ]=-------- ' , verbose=self.verbose)
                    print_verbose(f'update best chi_sq ]=------ {chi_sq}' , verbose=self.verbose)
                    # print(f' new chi_sq: {chi_sq}')
                    self.best_chi_sq_AV = chi_sq
                    print_verbose(f'updating best coeffs ]=---- {coeffs_now}', verbose=self.verbose)
                    self.best_coeffs_AV = coeffs_now
                    print_verbose(f'updating best AV ]=-------- {_AV}', verbose=self.verbose)
                    self.best_AV = _AV
                    force_update = False
            print_verbose('', verbose=self.verbose)
        _AV = self.get_last_AV()
        _chi = self.get_last_chi_sq_AV()
        if _AV != self.best_AV:
            slope = (_chi**0.5 - self.best_chi_sq_AV**0.5)/(_AV - self.best_AV)
            if slope > 0:
                self.e_AV = np.abs(self.best_AV/slope/3)
            else:
                self.e_AV = cf.delta_AV
        else:
            self.e_AV = cf.delta_AV
        if self.e_AV < (0.25*cf.delta_AV):
            self.e_AV = cf.delta_AV*0.5
        if self.e_AV < 0.0001 or np.isnan(self.e_AV):
            self.e_AV = cf.delta_AV*0.5

        # print(f'AV = {self.best_AV} +- {self.e_AV}')
        print_verbose(f'n_loops ------------------- {n_loops}', verbose=self.verbose)
        print_verbose(f'n_loops_AV ---------------- {self.n_loops_AV}', verbose=self.verbose)
        print_verbose(f'n_loops_nl_fit ------------ {self.n_loops_nl_fit}', verbose=self.verbose)
        print_verbose(f'best_chi_sq_AV ------------ {self.best_chi_sq_AV}', verbose=self.verbose)
        print_verbose(f'best_AV ------------------- {self.best_AV}', verbose=self.verbose)
        print_verbose(f'best_coeffs_AV ------------ {self.best_coeffs_AV}', verbose=self.verbose)
        print_verbose('-----------------------------------------------------=[ END fit AV ]=-', verbose=self.verbose)
        self.plot_best(ssp, coeffs=self.best_coeffs_AV, sel=s['sel_AV'], filename='best_AV.png')
        if self.plot > 0:
            if 'matplotlib.pyplot' not in sys.modules:
                from matplotlib import pyplot as plt
            else:
                plt = sys.modules['matplotlib.pyplot']
            if self.plot == 1:
                plt.cla()
                ax = plt.gca()
            elif self.plot == 2:
                f, ax = plt.subplots(figsize=(_figsize_default))
            ax.scatter(self.AV_chain, self.chi_sq_AV_chain)
            ax.plot(self.best_AV, self.best_chi_sq_AV, 'kx')
            ax.set_xlabel(r'A$_V^*$')
            ax.set_ylabel(r'$\chi^2$')
            if self.plot == 1:
                plt.pause(0.001)
            if self.plot == 2:
                f.savefig('AV_fit_broad.pdf', dpi=_plot_dpi)
                plt.close(f)

    def get_last_chi_sq_ssp(self):
        r = 1e12
        if len(self.chi_sq_ssp_chain) > 0:
            r = self.chi_sq_ssp_chain[-1]
        return r

    def get_last_coeffs_ssp(self):
        r = np.zeros(self.models_nl_fit.n_models, dtype='float')
        if len(self.coeffs_ssp_chain) > 0:
            r = self.coeffs_ssp_chain[-1]
        return r

    def _init_ssp_fit(self):
        self.ssp_chain = []
        self.coeffs_ssp_chain = []
        self.chi_sq_ssp_chain = []
        self.best_coeffs_ssp = self.get_last_coeffs_ssp()
        self.best_chi_sq_ssp = self.get_last_chi_sq_ssp()
        self.n_loops_ssp = 0

    def update_ssp_parameters(self, coeffs, chi_sq):
        """
        Save the SSP fit parameters chain of tries.

        Parameters
        ----------
        coeffs : array like
            The last calculed coefficients for the SSP models.
        chi_sq : float
            The chi squared of the fit.
        """
        self.coeffs_ssp_chain.append(coeffs)
        self.chi_sq_ssp_chain.append(chi_sq)

    def calc_SN_norm_window(self):
        """
        Calculates the signal-to-noise.

        Attributes
        ----------
        sel_norm_window : array like
            An array of booleans selecting the wavelength range of the
            normalization window.

        SN_norm_window : float
            The signal-to-noise at the normalization window.
        """
        s = self.spectra
        ssp = self.models

        # invert matrix in order to obtain the coefficients (weights)
        coeffs_now, chi_sq, msk_model_min = self.fit_WLS_invmat(ssp=ssp,
                                                                sel_wavelengths=s['sel_AV'])
        # update SSP parameters in self.
        self.update_ssp_parameters(coeffs=coeffs_now, chi_sq=chi_sq)
        # create raw model with the coefficients
        model_min = self.get_best_model_from_coeffs(ssp=self.models, coeffs=coeffs_now)

        # calculates S/N in the range defined by SFS in perl version
        # TODO: the normalization window should be defined inside ssp FITS header
        half_norm_range = 45  # Angstrom
        # l_wave = self.models.normalization_wavelength - half_norm_range
        # r_wave = self.models.normalization_wavelength + half_norm_range
        # l_wave = ssp.wavenorm - half_norm_range
        # r_wave = ssp.wavenorm + half_norm_range
        wavenorm_corr = ssp.wavenorm*(1 + self.best_redshift)
        l_wave = wavenorm_corr - half_norm_range
        r_wave = wavenorm_corr + half_norm_range
        s['sel_norm_window'] = (s['raw_wave'] >= l_wave) & (s['raw_wave'] <= r_wave)
        # n_w = len(s['raw_wave'])
        # i0 = int(0.4*n_w)
        # i1 = int(0.6*n_w)
        # if i1 > ssp.wavelength[-1]:
        #     i1 = ssp.n_wave - 1
        res_min = s['raw_flux'] - model_min
        res_norm_window = res_min[s['sel_norm_window'] & (model_min > 0)]
        model_norm_window = model_min[s['sel_norm_window'] & (model_min > 0)]
        # stats_res_min = pdl_stats(res_min[i0:i1+1])
        # stats_mod = pdl_stats(model[i0:i1+1])
        stats_res_norm_window = pdl_stats(res_norm_window)
        stats_model_norm_window = pdl_stats(model_norm_window)
        # print(f'stats_res_norm_window ---- {stats_res_norm_window}')
        # print(f'stats_model_norm_window ---- {stats_model_norm_window}')
        SN = 0
        if stats_res_norm_window[_STATS_POS['pRMS']] > 0:
            SN = stats_model_norm_window[_STATS_POS['mean']]/stats_res_norm_window[_STATS_POS['pRMS']]
        self.SN_norm_window = SN
        print(f'Normalization window = [{l_wave}, {r_wave}]')
        print(f'Signal-to-Noise inside normalization window = {SN}')

    def non_linear_fit(self, guide_sigma=False, fit_sigma_rnd=True,
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
            print(f'Deriving redshift, sigma, AV...')
            # self.update_redshift_params(coeffs=coeffs_now, chi_sq=chi_sq, redshift=self.best_redshift)
            # self.best_chi_sq_redshift = self.get_last_chi_sq_redshift()
            # self.best_coeffs_redshift = self.get_last_coeffs_redshift()
            msg_cut = f' - cut value: {cf.CUT_MEDIAN_FLUX}'
            if cf.delta_redshift > 0:
                self._fit_redshift(correct_wl_ranges=correct_wl_ranges)
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
            print(f'Deriving redshift, sigma, AV... DONE!')
        self.best_chi_sq_nl_fit = chi_sq
        self.best_coeffs_nl_fit = coeffs_now
        print_verbose('------------------------[ END non-linear fit report]--', verbose=self.verbose)

        print_verbose('', verbose=self.verbose)
        print_verbose('-----------------------------------------------[ END non-linear fit ]--', verbose=self.verbose)
        print_verbose('-----------------------------------------------------------------------', verbose=self.verbose)
        # plt.show(block=True)

    def _subtract_continuum(self, model, ratio_range=None, ratio_std=None):
        """
        Subtract the continuum of `self.spectra['raw_flux']` in order to perform the
        emission-lines fit.

        Parameters
        ----------
        model : array like
            Fit of the observed spectrum.
        """
        ratio_range = 0.2 if ratio_range is None else ratio_range
        ratio_std = 0.02 if ratio_std is None else ratio_std
        s = self.spectra
        sigma_mean = self.sigma_mean

        res_min = s['raw_flux'] - model
        # smooth process (not tested)
        # TODO: test smooth process (NaNs, zeros, inf)

        ratio = np.divide(res_min, model, where=model>0) + 1
        # from pyFIT3D.common.stats import median_filter
        # print(sigma_mean)
        # print(int(7*sigma_mean*__sigma_to_FWHM__))
        # median_ratio = median_filter(7*sigma_mean*__sigma_to_FWHM__, copy(ratio))
        # np.savetxt('array_python.txt', [ratio, median_ratio])
        # sys.exit(1)
        median_ratio = median_filter(int(7*__sigma_to_FWHM__*sigma_mean), ratio)
        # median_ratio = smooth_ratio(ratio, sigma_mean, kernel_size_factor=8*sigma_mean)
        median_sigma = int(1.5*sigma_mean)
        if median_sigma < 3:
            median_sigma = 3
        median_ratio_box = median_box(median_sigma, median_ratio)
        median_wave_box = median_box(median_sigma, s['raw_wave'])
        med_wave_box_size = median_wave_box.size
        med_ratio_box_size = median_ratio_box.size
        if med_wave_box_size > med_ratio_box_size:
            median_wave_box = median_wave_box[0:med_ratio_box_size]
        elif med_wave_box_size < med_ratio_box_size:
            median_ratio_box = median_ratio_box[0:med_wave_box_size]
        f = interp1d(median_wave_box, median_ratio_box, bounds_error=False, fill_value='extrapolate')
        y_ratio = f(s['raw_wave'])
        # plot smooth process
        if self.plot:
            if 'matplotlib.pyplot' not in sys.modules:
                from matplotlib import pyplot as plt
            else:
                plt = sys.modules['matplotlib.pyplot']

            # spectra_list = [s['raw_flux'], model, res_min, ratio, y_ratio]
            # spectra_list = [res_min, ratio, y_ratio]
            spectra_list = [res_min, ratio, y_ratio]
            wave_list = [s['raw_wave']]*len(spectra_list)
            colors = ["0.4", "r", "b"]
            ylim = (-0.2, 1.5)
            if self.plot == 1:
                plt.cla()
                plot_spectra_ax(plt.gca(), wave_list, spectra_list, ylim=ylim, color=colors)
                plt.pause(0.001)
            elif self.plot == 2:
                f, ax = plt.subplots(figsize=(_figsize_default))
                plot_spectra_ax(ax, wave_list, spectra_list, ylim=ylim, color=colors)
                f.savefig('yratio.png', dpi=_plot_dpi)
                plt.close(f)
            # plt.show(block=False)
        # XXX: EADL:
        st_y_ratio_nw = pdl_stats(y_ratio[s['sel_norm_window'] & (model > 0)])

        if (((st_y_ratio_nw[_STATS_POS['mean']] > (1 - ratio_range))
            & (st_y_ratio_nw[_STATS_POS['mean']] < (1 + ratio_range)))
            & (st_y_ratio_nw[_STATS_POS['pRMS']] > ratio_std)):
            _where = y_ratio > (st_y_ratio_nw[_STATS_POS['min']] - st_y_ratio_nw[_STATS_POS['pRMS']])
            _where &= y_ratio < (st_y_ratio_nw[_STATS_POS['min']] + st_y_ratio_nw[_STATS_POS['pRMS']])
            _where &= y_ratio != 0
            # s['raw_flux'] = np.where(y_ratio > 0, s['raw_flux']/y_ratio, s['raw_flux'])
            s['raw_flux'] = np.divide(s['raw_flux'], y_ratio, s['raw_flux'], where=_where)
            s['orig_flux'] = np.divide(s['orig_flux'], y_ratio, s['orig_flux'], where=_where)

    def _EL_fit(self, model_min, half_range_sysvel=None, correct_wl_range=0, guide_vel=True):
        """
        Performs the EL fit to the residual spectrum.

        Parameters
        ----------
        model_min : array
            The best SSP model to the observed spectrum.

        half_range_sysvel : float, optional
            Defines the range of the velocity of the emission lines.
            Defaults to __selected_half_range_sysvel_auto_ssp__.
            .. code-block:: python

                systemic_velocity = light_velocity*redshift
                vel_range = [
                    systemic_velocity - half_range_sysvel,
                    systemic_velocity + half_range_sysvel,
                ]

        correct_wl_range : int {0, 1, 2}, optional
            Determine the wavelength range redshift correction of the
            emission line system search. The accepted values are::

                0 : Do not correct wavelength range.
                1 : correct wavelength range using `self.best_redshift`
                2 : correct the range based on the system of emission lines.
                    wl_range = [
                        (eml_central_wave_min - (half_range_sysvel/10)),
                        (eml_central_wave_max + (half_range_sysvel/10))
                    ]

            Defaults to 0.

        guide_vel : bool, optional
            Guide velocity guess of the fit around self.best_redshift. Defaults to True.

        Attributes
        ----------
        spectra['raw_model_elines'] : array like
            The raw model of the emission lines fit.

        n_models_elines : int
            The number of emission lines models.

        See also
        --------
        pyFIT3D.common.constants.__selected_half_range_sysvel_auto_ssp__` and
        :func:`pyFIT3D.common.gas_tools.fit_elines_main`
        """
        # This could be an argument. Need it?
        if half_range_sysvel is None:
            half_range_sysvel = __selected_half_range_sysvel_auto_ssp__

        if correct_wl_range > 2:
            print(f'{basename(sys.argv[0])}: [StPopSynth._EL_fit()]: {correct_wl_range}: correct mode not recognized. Setting up to zero.')
            correct_wl_range = 0

        self.systemic_velocity = __c__*self.best_redshift
        self.spectra['raw_model_elines'] = np.zeros_like(self.spectra["raw_wave"])
        self.n_models_elines = 0
        if not self.fit_gas:
            return
        print('Gas fit...')
        model_elines = np.zeros(self.spectra['raw_flux'].shape)
        n_models = 0
        for i_s, (els, els_config) in enumerate(zip(self.config.systems, self.config.systems_config)):
            start_w = els['start_w']
            end_w = els['end_w']
            mask_file = els['mask_file']
            _m_str = 'model'
            if els_config.n_models > 1:
                _m_str += 's'
            # do not correct wl range
            start_w_corr = start_w
            end_w_corr = end_w
            # correct wl range
            if correct_wl_range > 0:
                z_fact = 1 + self.best_redshift
                _start_w = start_w
                _end_w = end_w
                # correct wl range based on emission lines in `els_config` system.
                if correct_wl_range == 2:
                    # 0 is _EL_MODELS['eline']['central_wavelength']
                    # for this to work, all models (eline, voigt, ...)
                    # should have _MODELS_model_PAR['central_wavelength'] = 0
                    # configured at pyFIT3D/common/constants.py
                    _tmp = np.asarray(els_config.guess).T[0]
                    _el_models = np.asarray(els_config.model_types)
                    _s_wl = np.nanmin(_tmp[_el_models != 'poly1d'])
                    _e_wl = np.nanmax(_tmp[_el_models != 'poly1d'])
                    _start_w = _s_wl - __selected_half_range_wl_auto_ssp__
                    _end_w = _e_wl + __selected_half_range_wl_auto_ssp__
                    # too tight range
                    # _z = half_range_sysvel/__c__
                    # _start_w = _s_wl*(1 - _z)
                    # _end_w = _e_wl*(1 + _z)
                start_w_corr = _start_w*z_fact
                end_w_corr = _end_w*z_fact
            print(f'-> analyzing {els_config.n_models} {_m_str} in {start_w_corr}-{end_w_corr} wavelength range')

            # number of models different from 'poly1d'
            n_models += np.sum(list(map(lambda x: x != 'poly1d', els_config.model_types)))

            flux_elines = self.spectra['orig_flux'] - model_min
            eflux_elines = self.spectra['raw_sigma_flux']
            wave_elines = self.spectra['raw_wave']

            sel_wl_range = trim_waves(wave_elines, [start_w_corr, end_w_corr])
            sel_wl_masks = np.ones_like(flux_elines, dtype='bool')
            if mask_file is not None:
                if isfile(mask_file):
                    _m = np.loadtxt(mask_file)
                    sel_wl_masks = sel_waves(_m, wave_elines)
                else:
                    print(f'{basename(sys.argv[0])}: [StPopSynth._EL_fit()]: {mask_file}: mask list file not found')
            else:
                print(f'{basename(sys.argv[0])}: [StPopSynth._EL_fit()]: no mask list file')

            flux_elines = flux_elines[sel_wl_range & sel_wl_masks]
            eflux_elines = eflux_elines[sel_wl_range & sel_wl_masks]
            wave_elines = wave_elines[sel_wl_range & sel_wl_masks]

            # guide_vel = False

            EL = fit_elines_main(
                wavelength=wave_elines,
                flux=flux_elines,
                sigma_flux=eflux_elines,
                config=els_config,
                plot=self.plot,

                # EmissionLinesRND class fit parameters.
                # `n_MC` and `n_loops` will also be used by the EmissionLinesRM class
                n_MC=30, n_loops=5, fine_search=False,

                # guide the velocity of the emission lines using the
                # derived stellar redshift
                vel_mask=1, vel_fixed=1,
                vel_guide=None if not guide_vel else self.systemic_velocity,
                vel_guide_half_range=half_range_sysvel,

                # redefine the range of the emission lines integrated flux
                redefine_max=True, max_factor=2,
                redefine_min=True, min_factor=0.01*1.2,

                # Run RND and uses the RND best fit params as LM input guesses.
                # The fit_elines_main() uses EmissionLines.update_config() method
                # to generate the LM input config. The range of each free parameter
                # is defined by the input variable frac_range.
                #
                # par_range = par * [1 - frac_range, 1 + frac_range]
                #
                run_mode='BOTH',
                update_config_frac_range=0.25,
            )

            # saving EmissionLinesLM system to self.config.systems[i]['EL']
            self.config.systems[i_s]['EL'] = EL

            # Build up the emission spectrum model using the entire wavelength range.
            EL.wavelength = self.spectra['raw_wave']
            model_elines += EL.create_system_model(parameters=EL.final_fit_params_mean,
                                                   ignore_poly1d=True)

        # create joint spectrum
        self.spectra['raw_model_elines'] = model_elines
        print('Gas fit... DONE!');
        self.n_models_elines = n_models

    def _rescale(self, model_min, ratio_range=None, ratio_std=None):
        ratio_range = 0.2 if ratio_range is None else ratio_range
        ratio_std = 0.02 if ratio_std is None else ratio_std
        s = self.spectra
        sigma_mean = self.sigma_mean
        res_min = s['raw_flux'] - model_min
        ssp_model_joint = model_min + s['raw_model_elines']
        ssp_res_joint = res_min - s['raw_model_elines']
        y_ratio = copy(self.ratio_master)
        if self.SN_norm_window > 10:
            ratio = np.divide(ssp_res_joint, ssp_model_joint, where=ssp_model_joint!=0) + 1
            median_ratio = median_filter(int(5*__sigma_to_FWHM__*sigma_mean), ratio)
            # median_ratio = smooth_ratio(ratio, sigma_mean, kernel_size_factor=8*sigma_mean)
            median_sigma = int(1.5*sigma_mean)
            if median_sigma < 3:
                median_sigma = 3
            median_ratio_box = median_box(median_sigma, median_ratio)
            median_wave_box = median_box(median_sigma, s['raw_wave'])
            med_wave_box_size = median_wave_box.size
            med_ratio_box_size = median_ratio_box.size
            if med_wave_box_size > med_ratio_box_size:
                median_wave_box = median_wave_box[0:med_ratio_box_size]
            elif med_wave_box_size < med_ratio_box_size:
                median_ratio_box = median_ratio_box[0:med_wave_box_size]
            # median_ratio = smooth_ratio(ratio, sigma_mean, kernel_size_factor=5*__sigma_to_FWHM__)
            # median_sigma = int(1.5*sigma_mean)
            # if median_sigma < 3:
            #     median_sigma = 3
            # median_ratio_box = median_box(median_sigma, median_ratio)
            # median_wave_box = median_box(median_sigma, s['raw_wave'])
            f = interp1d(median_wave_box, median_ratio_box, bounds_error=False,fill_value=0.)
            y_ratio = f(s['raw_wave'])
            # stats_y_ratio_norm_window = pdl_stats(y_ratio[s['sel_norm_window'] & (model_min > 0)])
            st_y_ratio_nw = pdl_stats(y_ratio[s['sel_norm_window'] & (model_min > 0)])
            if self.plot:
                if 'matplotlib.pyplot' not in sys.modules:
                    from matplotlib import pyplot as plt
                else:
                    plt = sys.modules['matplotlib.pyplot']

                plt.cla()
                title = f"ratio = {st_y_ratio_nw[_STATS_POS['mean']]}, rms = {st_y_ratio_nw[_STATS_POS['pRMS']]}"
                spectra_list = [res_min, ratio, y_ratio]
                wave_list = [s['raw_wave']]*len(spectra_list)
                colors = ["0.4", "r", "b"]
                ylim = (-0.2, 1.5)
                if self.plot == 1:
                    plt.cla()
                    plot_spectra_ax(plt.gca(), wave_list, spectra_list, title=title, ylim=ylim, color=colors)
                    plt.pause(0.001)
                elif self.plot == 2:
                    f, ax = plt.subplots(figsize=(_figsize_default))
                    plot_spectra_ax(ax, wave_list, spectra_list, ylim=ylim, title=title, color=colors)
                    f.savefig('yratio.png', dpi=_plot_dpi)
                    plt.close(f)

            st_y_ratio_nw = pdl_stats(y_ratio[s['sel_norm_window'] & (model_min > 0)])
            if (((st_y_ratio_nw[_STATS_POS['mean']] > (1 - ratio_range))
                & (st_y_ratio_nw[_STATS_POS['mean']] < (1 + ratio_range)))
                & (st_y_ratio_nw[_STATS_POS['pRMS']] > ratio_std)):
                if self.spec_id == 0:
                    self.ratio_master = copy(y_ratio)
                else:
                    y_ratio = self.ratio_master
            else:
                y_ratio = self.ratio_master
        return y_ratio

    def gas_fit(self, ratio=True):
        """
        Prepares the observed spectra in order to fit systems of emission lines
        to the residual spectra.

        Attributes
        ----------
        spectra['raw_flux_no_gas'] : array like
            The raw observed spectrum without the model of the emission lines.
        """
        s = self.spectra
        ssp = self.models
        sigma_mean = self.sigma_mean

        print_verbose('', verbose=self.verbose)
        print_verbose('-----------------------------------------------------------------------', verbose=self.verbose)
        print_verbose('--[ BEGIN EL fit ]-----------------------------------------------------', verbose=self.verbose)

        model_min = self.get_best_model_from_coeffs(ssp=ssp, coeffs=self.get_last_coeffs_ssp())

        res_min = s['raw_flux'] - model_min
        if self.SN_norm_window > 10:
            self._subtract_continuum(model_min)

        # fit Emission Lines
        self._EL_fit(model_min=model_min)

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
        y_ratio = self.ratio_master if not ratio else self._rescale(model_min)

        ratio = np.divide(s['orig_flux'], y_ratio, where=y_ratio!=0)
        s['orig_flux_ratio'] = np.where(y_ratio > 0, ratio, s['orig_flux'])
        ratio = np.divide(s['raw_flux_no_gas'], y_ratio, where=y_ratio!=0)
        s['raw_flux_no_gas'] = np.where(y_ratio > 0, ratio, s['raw_flux_no_gas'])
        # s['msk_flux_no_gas'] = s['raw_flux_no_gas'][s['sel_wl']]

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
        l_AV_min = np.dot(_coeffs, self.final_AV)
        l_AV_min_mass = np.dot(_coeffs_norm, ssp.mass_to_light*self.final_AV)
        e_l_AV_min = np.dot(_sigma, self.final_AV)
        e_l_AV_min_mass = np.dot(_sigma_norm, ssp.mass_to_light*self.final_AV)

        l_age_min, l_met_min, l_age_min_mass, l_met_min_mass = ssp.get_tZ_from_coeffs(_coeffs, mean_log=True)
        e_l_age_min, e_l_met_min, e_l_age_min_mass, e_l_met_min_mass = ssp.get_tZ_from_coeffs(_sigma, mean_log=True)

        # l_age_min = np.dot(_coeffs, np.log10(ssp.age_models))
        # l_met_min = np.dot(_coeffs, np.log10(ssp.metallicity_models))
        # l_age_min_mass = np.dot(_coeffs_norm, ssp.mass_to_light*np.log10(ssp.age_models))
        # l_met_min_mass = np.dot(_coeffs_norm, ssp.mass_to_light*np.log10(ssp.metallicity_models))
        # e_l_age_min = np.dot(_sigma, np.log10(ssp.age_models))
        # e_l_met_min = np.dot(_sigma, np.log10(ssp.metallicity_models))
        # e_l_age_min_mass = np.dot(_sigma_norm, ssp.mass_to_light*np.log10(ssp.age_models))
        # e_l_met_min_mass = np.dot(_sigma_norm, ssp.mass_to_light*np.log10(ssp.metallicity_models))

        self.mass_to_light = np.dot(ssp.mass_to_light, _coeffs_norm)

        self.age_min = 10**l_age_min
        self.met_min = 10**l_met_min
        self.AV_min = l_AV_min
        if self.mass_to_light == 0:
            self.mass_to_light = 1
        self.age_min_mass = 10**(l_age_min_mass/self.mass_to_light)
        self.met_min_mass = 10**(l_met_min_mass/self.mass_to_light)
        self.AV_min_mass = l_AV_min_mass/self.mass_to_light
        # XXX:
        #   SFS, why 0.43 ?
        self.e_age_min = np.abs(0.43*e_l_age_min*self.age_min)
        self.e_met_min = np.abs(0.43*e_l_met_min*self.met_min)
        self.e_AV_min = np.abs(0.43*e_l_AV_min*self.AV_min)
        self.e_age_min_mass = np.abs(0.43*e_l_age_min*self.age_min_mass)
        self.e_met_min_mass = np.abs(0.43*e_l_met_min*self.met_min_mass)
        self.e_AV_min_mass = np.abs(0.43*e_l_AV_min*self.AV_min_mass)

    def _calc_coeffs_MC(self, ssp, coeffs_MC, chi_sq_MC, models_MC):
        """
        Calculates the mean coefficients and the standard deviation from the Monte-Carlo (MC)
        realisation of `self.fit_WLS_invmat_MC()`. Also calculates the final SSP coefficients
        performing a MC perturbing the mean coefficients of the later MC realisation. This
        method defines the ssp coefficients resulting attributes and also the best ssp model.

        Parameters
        ----------
        ssp : SSPModels class
            The class with ssp models used during the fit process.
        coeffs_MC : array like
            The coefficients from the MC realisation.
        chi_sq_MC : array like
            Chi squareds from the MC realisation.
        models_MC : array like
            Models from the MC realisation.

        Attributes
        ----------
        coeffs_input_MC : array like
            The mean coefficients of the MC realisation of `self.fit_WLS_invmat_MC()`.
        coeffs_input_MC_rms : array like
            The stddev of `coeffs_input_MC`.
        coeffs_ssp_MC : array like
            The mean coefficients of the MC realisation of `self.coeffs_input_MC`.
        coeffs_ssp_rms : array like
            The stddev of `coeffs_ssp_MC`
        spectra['model_ssp_min'] : array like
            The best ssp model smoothed by `self.sigma_mean`.
        spectra['model_ssp_min_uncorr'] : array like
            The best ssp model without the smooth process.
        """
        s = self.spectra
        # normalization window of ssp.models
        half_norm_range = 45  # Angstrom
        wavenorm_corr = ssp.wavenorm*(1 + self.best_redshift)
        l_wave = wavenorm_corr - half_norm_range
        r_wave = wavenorm_corr + half_norm_range
        s['sel_norm_window'] = (s['raw_wave'] > l_wave) & (s['raw_wave'] < r_wave)

        ssp.to_observed(self.spectra['raw_wave'], sigma_inst=self.sigma_inst, sigma=self.best_sigma, redshift=self.best_redshift)
        # SSP models at observed frame with considered dust
        ssp.apply_dust_to_flux_models_obsframe(self.spectra['raw_wave']/(1 + self.best_redshift), self.best_AV, R_V=self.R_V, extlaw=self.extlaw)

        coeffs_MC = np.asarray(coeffs_MC)
        coeffs_input_zero = np.zeros(ssp.n_models, dtype='float')
        coeffs_rms = np.zeros(ssp.n_models, dtype='float')
        best_coeffs = np.zeros(ssp.n_models, dtype='float')
        best_coeffs_rms = np.zeros(ssp.n_models, dtype='float')
        model_ssp_min = np.zeros((ssp.flux_models_obsframe_dust.shape[-1]), dtype='float')
        model_ssp_min_uncorr = np.zeros((ssp.flux_models_obsframe_dust.shape[-1]), dtype='float')
        chi_sq_min_now = 1e12

        if (coeffs_MC != 0).astype('int').sum():
            n_MC = coeffs_MC.shape[0]
            coeffs_input = coeffs_MC.sum(axis=0)
            norm = coeffs_input.sum()
            coeffs_input /= norm
            coeffs_input_zero = copy(coeffs_input)
            coeffs_MC /= norm

            # coeffs_rms = coeffs_MC.std(axis=0)
            # pRMS calc
            coeffs_rms = np.asarray([np.sqrt(np.nansum((_c - np.nanmean(_c))**2)/(_c.size - 1)) for _c in coeffs_MC.T])

            i_MC = 0
            ini_cat = 0
            chi_sq_cat = np.zeros(n_MC, dtype='float')
            coeffs_cat = np.zeros((n_MC, ssp.n_models), dtype='float')
            model_cat = np.zeros((n_MC, ssp.flux_models_obsframe_dust.shape[-1]), dtype='float')

            for j_MC in range(n_MC):
                if i_MC == 0:
                    fact_q = 0
                elif i_MC == 1:
                    fact_q = 1
                i_MC += 1

                if i_MC > 1:
                    coeffs_now = coeffs_input_zero + 2*fact_q*coeffs_rms
                    coeffs_now = np.clip(coeffs_now, 0, 1)
                else:
                    coeffs_now = copy(coeffs_input)

                coeffs_now /= coeffs_now.sum()
                # model_now without mask
                model_now = np.dot(coeffs_now, ssp.flux_models_obsframe_dust)

                # XXX: EADL: Why re-scale the model?
                msk_flux_no_gas = copy(s['raw_flux_no_gas'][s['sel_norm_window']])
                msk_model_now = copy(model_now[s['sel_norm_window']])
                m = (msk_flux_no_gas > 0) & (msk_model_now > 0)
                msk_flux_no_gas = msk_flux_no_gas[m]
                msk_model_now = msk_model_now[m]
                med_msk_model_now = np.median(msk_model_now)
                if med_msk_model_now != 0:
                    med_norm = np.median(msk_flux_no_gas)/med_msk_model_now
                else:
                    med_norm = 1
                model_now *= med_norm

                # Calculate chi-squared
                m = (s['raw_flux_no_gas'] != 0)
                m &= (model_now != 0)
                m &= (s['raw_sigma_flux'] != 0)
                m &= s['sel_wl']

                msk_flux = s['raw_flux_no_gas'][m]
                sigma_flux = s['raw_sigma_flux'][m]

                chi_sq, _ = calc_chi_sq(msk_flux, model_now[m], sigma_flux)  #,
                                        # ssp.n_models + 1)

                # if coeffs_now.sum() != 0:
                #     chi_sq, _ = calc_chi_sq(msk_flux, model_now[m], msk_eflux,
                #                             len(coeffs_now[coeffs_now > 0]) - 1)
                # else:
                #     chi_sq = 2e12

                if chi_sq < (1.1*chi_sq_min_now):
                    coeffs_input = copy(coeffs_now)
                    fact_q *= 0.95
                    j_MC = 0
                    if (fact_q < 0.05) & (i_MC > 1):
                        j_MC = n_MC  # a.k.a stops
                    if chi_sq < chi_sq_min_now:
                        chi_sq_min_now = chi_sq
                    best_coeffs = copy(coeffs_now)
                    model_end = copy(model_now)

                    # XXX: EADL: SFS, wtf is cat?
                    chi_sq_cat[ini_cat] = chi_sq
                    coeffs_cat[ini_cat] = copy(coeffs_now)
                    model_cat[ini_cat] = copy(model_end)
                    ini_cat += 1

                    if ini_cat > (n_MC - 2):
                        j_MC = n_MC  # a.k.a stops

            # define output coefficients
            sum_W = 0
            n_cases = 0
            out_coeffs = []
            out_model_ssp_now = np.zeros((ssp.flux_models_obsframe_dust.shape[-1]), dtype='float')
            for i in range(ini_cat):
                _chi_sq = chi_sq_cat[i]
                if _chi_sq < (1.1*chi_sq_min_now):
                    out_model_ssp_now += model_cat[i]/_chi_sq
                    out_coeffs.append(coeffs_cat[i])
                    sum_W += 1/_chi_sq
            out_coeffs = np.array(out_coeffs)
            if out_coeffs.size == 0 or (~np.isnan(out_coeffs)).sum() == 0:
                out_coeffs = coeffs_input_zero

            if sum_W == 0:
                # No better solution found than the best model for the MC realisation!
                # EADL:
                #   2020-01-16: this model is masked, out_model_ssp_now NO!
                sum_W = 1
                _coeffs = np.asarray(coeffs_MC)[np.argmin(chi_sq_MC)]
                model_ssp_min = np.dot(_coeffs, ssp.flux_models_obsframe_dust)
                # model_ssp_min = np.asarray(models_MC)[np.argmin(chi_sq_MC)]
                # ratio = np.divide(s['raw_flux_no_gas'][s['sel_wl']], model_ssp_min, where=model_ssp_min!=0)
            else:
                model_ssp_min = out_model_ssp_now/sum_W

        # final coefficients
        # if (best_coeffs.sum() == 0):
        if (best_coeffs == 0).all():
            _c = np.zeros_like(coeffs_input_zero)
        else:
            _c = copy(best_coeffs)
            best_coeffs = out_coeffs.mean(axis=0)
            best_coeffs_rms = np.sqrt((out_coeffs.std(axis=0)**2+coeffs_rms**2))

        self.coeffs_input_MC = coeffs_input_zero
        self.coeffs_input_MC_rms = coeffs_rms
        self.coeffs_ssp_MC = best_coeffs
        self.coeffs_ssp_MC_rms = best_coeffs_rms
        self.orig_best_coeffs = _c

        norm = self.coeffs_ssp_MC.sum()
        if norm == 0:
            norm = 1
        self.coeffs_norm = self.coeffs_ssp_MC/norm
        self.min_coeffs_norm = self.orig_best_coeffs/norm

        s['model_ssp_min'] = model_ssp_min
        s['model_ssp_min_uncorr'] = copy(model_ssp_min)

        return chi_sq_min_now

    def ssp_fit(self, n_MC=20):
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
        coeffs_MC, chi_sq_MC, models_MC = self.fit_WLS_invmat_MC(ssp=ssp, n_MC=20)
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

    def ssp_init(self):
        ssp = self.models
        self.n_models_elines = 0
        s = self.spectra
        half_norm_range = 45  # Angstrom
        l_wave = ssp.wavenorm - half_norm_range
        r_wave = ssp.wavenorm + half_norm_range
        s['orig_flux_ratio'] = copy(s['raw_flux'])
        s['sel_norm_window'] = (s['raw_wave'] > l_wave) & (s['raw_wave'] < r_wave)
        s['model_ssp_min'] = np.zeros((s['raw_wave'].shape), dtype='float')
        s['model_ssp_min_uncorr'] = np.zeros((s['raw_wave'].shape), dtype='float')
        s['res_ssp_no_corr'] = np.zeros((s['raw_wave'].shape), dtype='float')
        s['model_joint'] = np.zeros((s['raw_wave'].shape), dtype='float')
        s['res_joint'] = np.zeros((s['raw_wave'].shape), dtype='float')
        s['res_ssp'] = np.zeros((s['raw_wave'].shape), dtype='float')
        s['raw_flux_no_gas'] = np.zeros_like(s['raw_wave'])
        self.output_spectra_list = [
            s['orig_flux_ratio'],
            s['model_ssp_min'],
            s['model_joint'],
            s['res_ssp_no_corr'],
            s['orig_flux_ratio'] - s['model_joint'],
            s['orig_flux_ratio'] - (s['model_joint'] - s['model_ssp_min']),
        ]
        self.coeffs_input_MC = np.zeros(ssp.n_models, dtype='float')
        self.coeffs_input_MC_rms = np.zeros(ssp.n_models, dtype='float')
        self.coeffs_ssp_MC = np.zeros(ssp.n_models, dtype='float')
        self.coeffs_ssp_MC_rms = np.zeros(ssp.n_models, dtype='float')
        self.orig_best_coeffs = np.zeros(ssp.n_models, dtype='float')
        self.coeffs_norm = np.zeros(ssp.n_models, dtype='float')
        self.min_coeffs_norm = np.zeros(ssp.n_models, dtype='float')
        self.output_coeffs = [
            self.coeffs_norm, self.min_coeffs_norm,
            self.coeffs_ssp_MC, self.coeffs_ssp_MC_rms,
            self.coeffs_input_MC, self.coeffs_input_MC_rms,
            self.orig_best_coeffs,
        ]
        self.final_AV = np.zeros(ssp.n_models, dtype='float')
        self.mass_to_light = 0
        self.age_min = 0
        self.met_min = 0
        self.AV_min = 0
        self.age_min_mass = 0
        self.met_min_mass = 0
        self.AV_min_mass = 0
        self.e_age_min = 0
        self.e_met_min = 0
        self.e_AV_min = 0
        self.e_age_min_mass = 0
        self.e_met_min_mass = 0
        self.e_AV_min_mass = 0
        self.systemic_velocity = __c__*self.best_redshift
        self.rms = 0
        self.med_flux = 0
        self.output_results = [
            0,
            self.age_min, self.e_age_min,
            self.met_min, self.e_met_min,
            self.AV_min, self.e_AV_min,
            self.best_redshift, self.e_redshift,
            self.best_sigma, self.e_sigma,
            0,
            self.best_redshift,
            self.med_flux, self.rms,
            self.age_min_mass, self.e_age_min_mass,
            self.met_min_mass, self.e_met_min_mass,
            self.systemic_velocity,
            0, 0,
        ]
        self.sigma_mean = None
        for system in self.config.systems:
            system['EL'] = None

    def output_gas_emission(self, filename, spec_id=None, append=True):
        if self.fit_gas:
            for system, elcf in zip(self.config.systems, self.config.systems_config):
                EL = system['EL']
                if EL is not None:
                    EL.output(filename_output=filename, spec_id=spec_id, append_output=append)
                else:
                    output_config_final_fit(elcf, filename, chi_sq=np.nan,
                                            parameters=[np.zeros_like(x) for x in elcf.guess],
                                            e_parameters=[np.zeros_like(x) for x in elcf.guess],
                                            spec_id=spec_id, append_output=append)

    def output_coeffs_MC_to_screen(self):
        cols = 'ID,AGE,MET,COEFF,Min.Coeff,log(M/L),AV,N.Coeff,Err.Coeff'
        fmt_cols = '| {0:^4} | {1:^7} | {2:^6} | {3:^6} | {4:^9} | {5:^8} | {6:^4} | {7:^7} | {8:^9} |'
        fmt_numbers = '| {:=04d} | {:=7.4f} | {:=6.4f} | {:=6.4f} | {:=9.4f} | {:=8.4f} | {:=4.2f} | {:=7.4f} | {:=9.4f} | {:=6.4f} | {:=6.4f}'
        # fmt_numbers_out_coeffs = '{:=04d},{:=7.4f},{:=6.4f},{:=6.4f},{:=9.4f},{:=8.4f},{:=4.2f},{:=7.4f},{:=9.4f},{:=6.4f},{:=6.4f}'
        cols_split = cols.split(',')
        tbl_title = fmt_cols.format(*cols_split)
        ntbl = len(tbl_title)
        tbl_border = ntbl*'-'
        print(tbl_border)
        print(tbl_title)
        print(tbl_border)
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
            tbl_row.append(self.ssp.age_models[i])
            tbl_row.append(self.ssp.metallicity_models[i])
            tbl_row.append(self.coeffs_norm[i])  # a_coeffs_N
            tbl_row.append(self.min_coeffs_norm[i])  # a_min_coeffs
            tbl_row.append(np.log10(self.ssp.mass_to_light[i]))
            tbl_row.append(self.best_AV)
            tbl_row.append(C)  # a_coeffs
            tbl_row.append(self.coeffs_ssp_MC_rms[i])  # a_e_coeffs
            tbl_row.append(self.coeffs_input_MC[i])  # ???
            tbl_row.append(self.coeffs_input_MC_rms[i])  # ???
            print(fmt_numbers.format(*tbl_row))
        print(tbl_border)

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

        cols = 'ID,AGE,MET,COEFF,Min.Coeff,log(M/L),AV,N.Coeff,Err.Coeff'
        cols_out_coeffs = cols.replace(',', '\t')
        fmt_numbers_out_coeffs = '{:=04d}\t{:=7.4f}\t{:=6.4f}\t{:=6.4f}\t{:=9.4f}\t{:=8.4f}\t{:=4.2f}\t{:=7.4f}\t{:=9.4f}'

        if write_header:
            print(f'#{cols_out_coeffs}', file=f_out_coeffs)

        for i in range(self.ssp.n_models):
            try:
                C = self.coeffs_ssp_MC[i]
            except (IndexError,TypeError):
                C = 0
            if np.isnan(C):
                C = 0
        # for i, C in enumerate(_coeffs):
            tbl_row = []
            tbl_row.append(i)
            tbl_row.append(self.ssp.age_models[i])
            tbl_row.append(self.ssp.metallicity_models[i])
            tbl_row.append(self.coeffs_norm[i])  # a_coeffs_N
            tbl_row.append(self.min_coeffs_norm[i])  # a_min_coeffs
            tbl_row.append(np.log10(self.ssp.mass_to_light[i]))
            tbl_row.append(self.best_AV)
            tbl_row.append(C)  # a_coeffs
            tbl_row.append(self.coeffs_ssp_MC_rms[i])  # a_e_coeffs
            print(fmt_numbers_out_coeffs.format(*tbl_row), file=f_out_coeffs)

        if not isinstance(filename, io.TextIOWrapper):
            f_out_coeffs.close()

    def output_fits(self, filename):
        """
        Writes the FITS file with the output spectra (original, model, residual and joint).

        Parameters
        ----------
        filename : str
            Output FITS filename.
        """
        s = self.spectra
        table = np.array(self.output_spectra_list)
        array_to_fits(filename, table, overwrite=True)
        h = {}
        h['CRPIX1'] = 1
        h['CRVAL1'] = s['raw_wave'][0]
        h['CDELT1'] = s['raw_wave'][1] - s['raw_wave'][0]
        h['NAME0'] = 'org_spec'
        h['NAME1'] = 'model_spec'
        h['NAME2'] = 'mod_joint_spec'
        h['NAME3'] = 'gas_spec'
        h['NAME4'] = 'res_joint_spec'
        h['NAME5'] = 'no_gas_spec'
        h['COMMENT'] = f'OUTPUT {basename(sys.argv[0])} FITS'
        write_img_header(filename, list(h.keys()), list(h.values()))

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

        print(f'# (1) MIN_CHISQ', file=f_outfile)
        print(f'# (2) LW Age (Gyr)', file=f_outfile)
        print(f'# (3) LW Age error', file=f_outfile)
        print(f'# (4) LW metallicity', file=f_outfile)
        print(f'# (5) LW metallicity error', file=f_outfile)
        print(f'# (6) Av', file=f_outfile)
        print(f'# (7) AV error', file=f_outfile)
        print(f'# (8) redshift', file=f_outfile)
        print(f'# (9) redshift error', file=f_outfile)
        print(f'# (10) velocity dispersion sigma, in AA', file=f_outfile)
        print(f'# (11) velocity dispersion error', file=f_outfile)
        print(f'# (12) median_FLUX', file=f_outfile)
        print(f'# (13) redshift_ssp', file=f_outfile)
        print(f'# (14) med_flux', file=f_outfile)
        print(f'# (15) StdDev_residual', file=f_outfile)
        print(f'# (16) MW Age (Gyr)', file=f_outfile)
        print(f'# (17) MW Age error', file=f_outfile)
        print(f'# (18) MW metallicity', file=f_outfile)
        print(f'# (19) MW metallicity error', file=f_outfile)
        print(f'# (20) Systemic Velocity km/s ', file=f_outfile)
        print(f'# (21) Log10 Average Mass-to-Light Ratio', file=f_outfile)
        print(f'# (22) Log10 Mass', file=f_outfile)
        # print(f'# SSP_SFH {cf.args.ssp_file} ', file=f_outfile)
        # print(f'# SSP_KIN {cf.args.ssp_nl_fit_file} ', file=f_outfile)
        print(f'# SSP_SFH {self.filename} ', file=f_outfile)
        print(f'# SSP_KIN {self.filename_nl_fit} ', file=f_outfile)
        print(f'# WAVE_NORM {wavenorm} AA', file=f_outfile)

        if not isinstance(filename, io.TextIOWrapper):
            f_outfile.close()

    def resume_results(self):
        ssp = self.ssp
        s = self.spectra
        sel_wl = s['sel_wl']
        model_joint = s['model_joint']
        res_joint = s['orig_flux_ratio'] - s['model_joint']
        # if chi_sq_msk > 0:
        #     delta_chi = np.abs()
        spectra_list = self.output_spectra_list
        # chi joint
        msk_sigma_flux = s['raw_sigma_flux'][sel_wl]
        _chi = np.nansum((res_joint[sel_wl])**2/msk_sigma_flux**2)
        n_obs = (msk_sigma_flux != 0).sum()
        # print(f'n_obs:{n_obs}')
        # print(f'ssp.flux_models_obsframe_dust.shape[0]:{ssp.flux_models_obsframe_dust.shape[0]}')
        # print(f'self.n_models_elines:{self.n_models_elines}')
        chi_joint = _chi/(n_obs - ssp.n_models - self.n_models_elines - 1)
        self.rms = res_joint[s['sel_norm_window']].std()
        self.med_flux = np.median(s['raw_flux_no_gas'][s['sel_norm_window']])
        FLUX = s['orig_flux'][sel_wl].sum()
        # XXX:
        #   Why a 3500 factor?
        mass = self.mass_to_light*self.med_flux*3500
        if mass > 0:
            lmass = np.log10(mass)
            lml = np.log10(self.mass_to_light/3500)
        else:
            lmass = 0
            lml = 0
        self.output_spectra_list = [
            s['orig_flux_ratio'],
            s['model_ssp_min'],
            s['model_joint'],
            s['res_ssp_no_corr'],
            s['orig_flux_ratio'] - s['model_joint'],
            s['orig_flux_ratio'] - (s['model_joint'] - s['model_ssp_min']),
        ]
        self.output_results = [
            chi_joint,
            self.age_min, self.e_age_min,
            self.met_min, self.e_met_min,
            self.AV_min, self.e_AV_min,
            self.best_redshift, self.e_redshift,
            self.best_sigma, self.e_sigma,
            FLUX,
            self.best_redshift,
            self.med_flux, self.rms,
            self.age_min_mass, self.e_age_min_mass,
            self.met_min_mass, self.e_met_min_mass,
            self.systemic_velocity,
            lml, lmass,
        ]
        self.output_coeffs = [
            self.coeffs_norm, self.min_coeffs_norm,
            self.coeffs_ssp_MC, self.coeffs_ssp_MC_rms,
            self.coeffs_input_MC, self.coeffs_input_MC_rms,
            self.orig_best_coeffs,
        ]

    def plot_final_resume(self, cmap='Blues', vmin=0.1, vmax=50, sigma_unit='km/s',
                          percent=True, Zsun=0.019, block_plot=False):
        import pandas as pd
        if 'matplotlib' not in sys.modules:
            import matplotlib as mpl
        else:
            mpl = sys.modules['matplotlib']
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = 'Times New Roman'
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['legend.numpoints'] = 1
        if 'matplotlib.pyplot' not in sys.modules:
            from matplotlib import pyplot as plt
        else:
            plt = sys.modules['matplotlib.pyplot']
        if 'seaborn' not in sys.modules:
            import seaborn as sns
        else:
            sns = sys.modules['seaborn']
        sns.set(context='paper', style='ticks', palette='colorblind',
                color_codes=True, font_scale=1.5)

        latex_ppi = 72.0
        latex_column_width_pt = 240.0
        latex_column_width = latex_column_width_pt/latex_ppi
        # latex_column_width = latex_column_width_pt/latex_ppi/1.4
        latex_text_width_pt = 504.0
        latex_text_width = latex_text_width_pt/latex_ppi
        golden_mean = 0.5 * (1. + 5**0.5)
        aspect = 1/golden_mean

        s = self.spectra
        wl = s['raw_wave']
        spec__tw = self.output_spectra_list
        output_results = self.output_results

        d_to_df = dict(ID=np.arange(self.models.n_models), AGE=self.models.age_models,
                       MET=self.models.metallicity_models, COEFF=self.coeffs_norm)
        df = pd.DataFrame(data=d_to_df)
        idx = np.lexsort((df.MET, df.AGE))
        df = df.iloc[idx]
        df.MET = df.MET.apply(lambda x: np.log10(x/Zsun))
        coeffs = df.reset_index(drop=True).pivot_table(index="AGE", columns="MET", values="COEFF")
        met = coeffs.columns.values
        met_weights = coeffs / coeffs.sum(axis="columns", skipna=True).values[:,None]
        met_weights = met_weights.values[:,:,None].transpose(0,2,1)
        age_weights = coeffs.sum(axis="columns", skipna=True)
        age_weights /= np.nansum(age_weights)
        age_weights = age_weights.values[None,:]
        pwei = coeffs
        masks = self.masks
        e_masks = self.e_masks
        el_wl_fit = []
        for elcf in self.config.systems_config:
            n_wl_fit = np.sum([
                elcf.check_par_fit('eline', a) for a in range(len(_MODELS_ELINE_PAR))
            ])
            for i, mt in enumerate(elcf.model_types):
                if ((mt == 'eline') and n_wl_fit):
                    el_wl_fit.append(elcf.guess[i][_MODELS_ELINE_PAR['central_wavelength']]*(1+self.best_redshift))

        wl_range = [3800, 6850]
        fmt = '.1f' if percent else '.3f'
        wlmsk = np.ones_like(wl, dtype='bool')
        if wl_range is not None:
            wlmsk = trim_waves(wl, wl_range)
        else:
            wl_range = [np.nanmin(w), np.nanmax(wl)]

        if self.plot == 1:
            plt.cla()
            f = plt.gcf()
            ax = plt.gca()
        elif self.plot == 2:
            f = plt.figure()
            width = 2*latex_text_width
            f.set_size_inches(width, 0.7*width*aspect)

        gs = f.add_gridspec(3,4)
        ax_sp = f.add_subplot(gs[:2, :3])
        ax = f.add_subplot(gs[2, :])

        ax_lw = f.add_subplot(gs[:2, 3])
        ax_lw.set_axis_off()

        msg = r'$\chi^2$ = %6.2f' % output_results[0]
        msg += '\n'
        msg += r'RMS Flux = %6.2f' % output_results[13]
        msg += '\n'
        msg += '\n'
        msg += r'sys vel = %6.2f +/- %.2f km/s' % (output_results[7]*__c__, output_results[8]*__c__)
        msg += '\n'
        msg += r'$\sigma^\star$ = %6.2f +/- %0.2f' % (output_results[9], output_results[10])
        msg += sigma_unit
        msg += '\n'
        msg += r'${\rm A}_{\rm V}^\star$ = %6.2f +/- %0.2f' % (output_results[5], output_results[6])
        msg += '\n'
        msg += '\n'
        msg += r'$\left< \log {\rm t^\star} \right>_{\rm L}$ = %6.2f +/- %0.2f Gyr' % (output_results[1], output_results[2])
        msg += '\n'
        msg += r'$\left< \log {\rm Z}^\star \right>_{\rm L}$ = %6.2f +/- %0.2f [Z/H]' % (output_results[3], output_results[4])
        msg += '\n'
        msg += r'$\left< \log {\rm t^\star} \right>_{\rm M}$ = %6.2f +/- %0.2f Gyr' % (output_results[15], output_results[16])
        msg += '\n'
        msg += r'$\left< \log {\rm Z}^\star \right>_{\rm M}$ = %6.2f +/- %0.2f [Z/H]' % (output_results[17], output_results[18])
        # msg += '\n'
        # msg += r'$\log {\rm M}/{\rm M}$ = %6.2f' % output_results[-2]

        ax_lw.text(0.02, 0.9, msg, transform=ax_lw.transAxes, ha='left', va='top', fontsize=13)

        in_spec = spec__tw[0][wlmsk]
        out_spec = spec__tw[1][wlmsk]
        out_spec_joint = spec__tw[2][wlmsk]

        res = in_spec - out_spec
        wl = wl[wlmsk]
        ax_sp.set_ylabel('Flux [$10^{-16}$ erg/s/cm$^2$]')
        ax_sp.set_xlabel('wavelength [\AA]')
        ax_sp.plot(wl, in_spec, '-k', lw=3)
        ax_sp.plot(wl, out_spec, '-y', lw=0.7)
        ax_sp.plot(wl, (in_spec - out_spec), '-r', lw=1)

        if masks is not None:
            _ = [ax_sp.axvspan(msk[0], msk[1], alpha=0.3, color='gray') for msk in masks if msk[0] < wl_range[1]]
        if e_masks is not None:
            for msk in e_masks:
                l, r = msk
                if r > wl_range[1]:
                    r = wl_range[1]
                if l > wl_range[0]:
                    ax_sp.axvspan(l, r, alpha=0.2, color='blue')

        # Plot EL
        _ = [ax_sp.axvline(x=wlc, color='k', ls='--') for wlc in el_wl_fit]

        # COEFFS HEATMAP
        pwei = pwei.T
        pwei.sort_index(axis='index', ascending=False, inplace=True)
        sns.heatmap((pwei / pwei.values.sum() * 100 if percent else pwei), cmap=cmap,
                    center=None, vmin=vmin, vmax=vmax, square=False, linewidths=1.0,
                    linecolor='w', annot=True, fmt=fmt, annot_kws={'fontsize':12},
                    cbar=False, ax=ax,
                    xticklabels=list(map(lambda v: '$%5.2f$'%v, np.log10(pwei.columns*1e9))),
                    yticklabels=list(map(lambda v: '$%5.2f$'%v, pwei.index)))

        ax.set_xlabel(r'$\log {\rm t}^\star$ [yr]')
        ax.set_ylabel(r'$\log {\rm Z}^\star$ [${\rm Z}/{\rm Z}_\odot$]')
        ax_sp.set_xlim(wl_range)
        f.tight_layout()

        if self.plot == 1:
            plt.pause(0.001)
            plt.show(block=block_plot)
        elif self.plot == 2:
            f.savefig('final_resume.png', dpi=_plot_dpi)
            plt.close(f)

    def output_to_screen(self, block_plot=True):
        s = self.spectra
        spectra_list = self.output_spectra_list
        chi_joint = self.output_results[0]
        FLUX = self.output_results[11]
        mass = self.mass_to_light*self.med_flux*3500
        lmass = self.output_results[-1]
        lml = self.output_results[-2]
        if self.plot:
            self.plot_final_resume(cmap='Blues', vmin=0.1, vmax=50,
                                   sigma_unit='km/s' if self.sigma_inst is not None else r'$\AA$',
                                   percent=True, Zsun=0.019, block_plot=block_plot)
            # labels = [
            #     'orig_flux_ratio',
            #     'model_min',
            #     'model_joint',
            #     'orig_flux - model_min_uncorr',
            #     'orig_flux_ratio - model_joint',
            #     'orig_flux_ratio - (model_joint - model_min)'
            # ]
            # plt.cla()
            # title = f'X={chi_joint:.4f} T={self.age_min:.4f} ({self.age_min_mass:.4f})'
            # title = f'{title} Z={self.met_min:.4f} ({self.met_min_mass:.4f})'
            # wave_list = [s['raw_wave']]*len(spectra_list)
            # if self.plot == 1:
            #     plot_spectra_ax(plt.gca(), wave_list, spectra_list, title=title, labels_list=labels)
            #     plt.pause(0.001)
            #     plt.show(block=block_plot)
            # elif self.plot == 2:
            #     f, ax = plt.subplots(figsize=(_figsize_default))
            #     plot_spectra_ax(ax, wave_list, spectra_list, title=title, labels_list=labels)
            #     f.savefig('final_fit.png', dpi=_plot_dpi)
            #     plt.close(f)

        msg_ini = f'| Chi^2 = {chi_joint:6.4f} RMS = {self.rms:6.4f} median(no gas flux) = {self.med_flux:6.4f}'
        msg_nl = '|---[ Non-linear analysis properties ]'
        msg_nl += '-'*(len(msg_ini)-len(msg_nl)) + '\n'
        msg_nl += f'|     REDSHIFT = {self.best_redshift:6.4f} +/- {self.e_redshift:0.4f}\n'
        msg_nl += f'|     SIGMA = {self.best_sigma:8.4f} +/- {self.e_sigma:0.4f} '
        if self.sigma_inst is not None:
            msg_nl += 'km/s\n'
        else:
            msg_nl += 'A\n'
        msg_nl += f'|     AV = {self.AV_min:6.4} +/- {self.e_AV_min:0.4f}'
        msg_ssp = '|---[ Resolved properties ]'
        msg_ssp += '-'*(len(msg_ini)-len(msg_ssp)) + '\n'
        msg_ssp += f'|     <AGE>_L = {self.age_min:6.4f} +/- {self.e_age_min:0.4f} Gyr\n'
        msg_ssp += f'|     <MET>_L = {self.met_min:6.4f} +/- {self.e_met_min:0.4f} Z/H\n'
        msg_ssp += f'|     <AGE>_M = {self.age_min_mass:6.4f} +/- {self.e_age_min_mass:0.4f} Gyr\n'
        msg_ssp += f'|     <MET>_M = {self.met_min_mass:6.4f} +/- {self.e_met_min_mass:0.4f} Z/H\n'
        msg_ssp += f'|     log(M/L) = {lml:6.4f}'

        # max_msg = np.max([len(x.strip()) for x in (msg_ini + msg_nl + msg_ssp).split('\n')])
        bar = '='*len(msg_ini)
        print('')
        print(bar)
        print(msg_ini)
        print(bar)
        print(msg_nl)
        print(msg_ssp)
        print(bar)
        print('')

    def output(self, filename, write_header=True, block_plot=True):
        """
        Summaries the run in a csv file.

        Parameters
        ----------
        filename : str
            Output filename.
        """
        s = self.spectra
        spectra_list = self.output_spectra_list
        chi_joint = self.output_results[0]
        FLUX = self.output_results[11]
        mass = self.mass_to_light*self.med_flux*3500
        lmass = self.output_results[-1]
        lml = self.output_results[-2]
        if isinstance(filename, io.TextIOWrapper):
            if write_header:
                self._print_header(filename)
            f_outfile = filename
        else:
            if not exists(filename):
                self._print_header(filename)
            f_outfile = open(filename, 'a')
        outbuf = f'{chi_joint},'
        outbuf = f'{outbuf}{self.age_min},{self.e_age_min},{self.met_min},{self.e_met_min},'
        outbuf = f'{outbuf}{self.AV_min},{self.e_AV_min},{self.best_redshift},{self.e_redshift},'
        outbuf = f'{outbuf}{self.best_sigma},{self.e_sigma},{FLUX},{self.best_redshift},'
        outbuf = f'{outbuf}{self.med_flux},{self.rms},{self.age_min_mass},{self.e_age_min_mass},'
        outbuf = f'{outbuf}{self.met_min_mass},{self.e_met_min_mass},{self.systemic_velocity},'
        outbuf = f'{outbuf}{lml},{lmass}'
        print(outbuf, file=f_outfile)
        if not isinstance(filename, io.TextIOWrapper):
            f_outfile.close()

    def ssp_single_fit(self):
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
        self.spectra['raw_eflux_norm_mean'] = np.divide(self.spectra['raw_sigma_flux'], norm_mean_flux, where=norm_mean_flux!=0)
        self.spectra['raw_flux_no_gas_norm_median'] = np.divide(self.spectra['raw_flux_no_gas'], norm_median_flux, where=norm_median_flux!=0)
        self.spectra['raw_eflux_norm_median'] = np.divide(self.spectra['raw_sigma_flux'], norm_median_flux, where=norm_median_flux!=0)

        chi_sq_mean = []
        chi_sq_median = []
        for M__w in flux_models_obsframe_dust:
            _chi_sq, _ = calc_chi_sq(f_obs=self.spectra['raw_flux_no_gas_norm_mean'], f_mod=M__w, ef_obs=self.spectra['raw_eflux_norm_mean'])
            chi_sq_mean.append(_chi_sq)
            _chi_sq, _ = calc_chi_sq(f_obs=self.spectra['raw_flux_no_gas_norm_median'], f_mod=M__w, ef_obs=self.spectra['raw_eflux_norm_median'])
            chi_sq_median.append(_chi_sq)

        chi_sq_mean = np.array(chi_sq_mean)
        chi_sq_median = np.array(chi_sq_median)
        print(f'len_chi_sq_mean={len(chi_sq_mean)}')
        chi_sq_mean_norm = chi_sq_mean / chi_sq_mean.sum()
        chi_sq_median_norm = chi_sq_median / chi_sq_median.sum()

        # f_out_coeffs = open(filename, 'a')
        cols = 'ID,AGE,MET,MEAN(CHISQ),MEDIAN(CHISQ)'
        # cols_out_coeffs = cols.replace(',', '\t')
        # print(f'#{cols_out_coeffs}', file=f_out_coeffs)

        fmt_cols = '| {0:^4} | {1:^7} | {2:^6} | {3:^11} | {4:^13} |'
        fmt_numbers = '| {:=04d} | {:=7.4f} | {:=6.4f} | {:=11.4f} | {:=13.4f} |'
        # fmt_numbers_out_coeffs = '{:=04d},{:=7.4f},{:=6.4f},{:=6.4f},{:=9.4f},{:=8.4f},{:=4.2f},{:=7.4f},{:=9.4f},{:=6.4f},{:=6.4f}'
        fmt_numbers_out_coeffs = '{:=04d}\t{:=7.4f}\t{:=6.4f}\t{:=6.4f}\t{:=6.4f}'

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
        tbl_row = []
        tbl_row.append(i_C_mean_min)
        tbl_row.append(self.ssp.age_models[i_C_mean_min])
        tbl_row.append(self.ssp.metallicity_models[i_C_mean_min])
        tbl_row.append(chi_sq_mean_norm[i_C_mean_min])  # a_coeffs_N
        tbl_row.append(chi_sq_median_norm[i_C_median_min])  # a_min_coeffs
        print(fmt_numbers.format(*tbl_row))
        if (i_C_median_min != i_C_mean_min):
            tbl_row = []
            tbl_row.append(i_C_median_min)
            tbl_row.append(self.ssp.age_models[i_C_median_min])
            tbl_row.append(self.ssp.metallicity_models[i_C_median_min])
            tbl_row.append(chi_sq_mean_norm[i_C_mean_min])  # a_coeffs_N
            tbl_row.append(chi_sq_median_norm[i_C_median_min])  # a_min_coeffs
            print(fmt_numbers.format(*tbl_row))

        self.output_table = []
        for i in range(models.n_models):
            C_mean = chi_sq_mean_norm[i]
            C_median = chi_sq_median_norm[i]
            tbl_row = []
            tbl_row.append(i)
            tbl_row.append(self.ssp.age_models[i])
            tbl_row.append(self.ssp.metallicity_models[i])
            tbl_row.append(C_mean)  # a_coeffs_N
            tbl_row.append(C_median)  # a_min_coeffs
            self.output_table.append(tbl_row)
        print(tbl_border)

        return chi_sq_median_norm[i_C_median_min]

    def output_single_ssp(self, filename):
        # filename = cf.args.out_file_coeffs.replace('coeffs', 'chi_sq')
        # remove old output_file
        if isfile(filename):
            remove(filename)

        f_out_coeffs = open(filename, 'a')
        cols = 'ID,AGE,MET,MEAN(CHISQ),MEDIAN(CHISQ)'
        cols_out_coeffs = cols.replace(',', '\t')
        print(f'#{cols_out_coeffs}', file=f_out_coeffs)
        fmt_numbers_out_coeffs = '{:=04d}\t{:=7.4f}\t{:=6.4f}\t{:=6.4f}\t{:=6.4f}'

        # output coeffs table
        for tbl_row in self.output_table:
            print(fmt_numbers_out_coeffs.format(*tbl_row), file=f_out_coeffs)

        f_out_coeffs.close()

        return None
