import io
import sys
# import warnings
import itertools
import numpy as np
from copy import deepcopy as copy
from scipy.optimize import least_squares
from astropy.modeling.functional_models import Voigt1D

np.set_printoptions(precision=4, suppress=True, linewidth=200)
# warnings.filterwarnings('ignore', category=UserWarning, append=True)

# local imports
from lvmdap.pyFIT3D.common.io import output_spectra, plot_spectra_ax
from lvmdap.pyFIT3D.common.stats import _STATS_POS, pdl_stats, WLS_invmat, calc_chi_sq
from lvmdap.pyFIT3D.common.constants import _MODELS_ELINE_PAR, _EL_MODELS
from lvmdap.pyFIT3D.common.constants import __c__, __sigma_to_FWHM__, __ELRND_fine_search_option__

latex_ppi = 72.0
latex_column_width_pt = 240.0
latex_column_width = latex_column_width_pt/latex_ppi
# latex_column_width = latex_column_width_pt/latex_ppi/1.4
latex_text_width_pt = 504.0
latex_text_width = latex_text_width_pt/latex_ppi
golden_mean = 0.5 * (1. + 5**0.5)

def output_config_final_fit(config, filename_output, chi_sq,
                            parameters=None, e_parameters=None,
                            append_output=False, spec_id=None):
    """ Outputs the parameters and e_parameters formatted.

    Parameters
    ----------
    config : :class:`ConfigEmissionModel` class
        The :class:`ConfigEmissionModel` instance.

    filename_output : str
        The output filename for the final result.

    parameters : array like, optional
        Parameters to be outputted. Defaults to ``config.guess``.

    e_parameters : array like, optional
        Sigma of the parameters to be outputted. Defaults to ``zeros_like(config.guess)``.

    append_output : bool
        When True, open `filename_output` for append. Otherwise, rewrite the file.

    spec_id : int or tuple
        The spec coordinate id. For cubes is a tuple with (ix, iy).
    """
    parameters = config.guess if parameters is None else parameters
    e_parameters = [np.zeros_like(par) for par in config.guess] if e_parameters is None else e_parameters
    _fname = filename_output.name if isinstance(filename_output, io.TextIOWrapper) else filename_output
    i_w = _MODELS_ELINE_PAR['central_wavelength']
    par_to_get_wave = parameters
    if par_to_get_wave[0][i_w] == 0:
        par_to_get_wave = config.guess
    waves = [par_to_get_wave[i_m][i_w] for i_m in range(config.n_models) if config.model_types[i_m] == 'eline']
    sys_str = 'system'
    if config.n_models > 1:
        sys_str += 's'
    print(f'{int(np.min(waves))}-{int(np.max(waves))} wavelength range: Saving {config.n_models} EML {sys_str} to file {_fname}')
    n_pars = len(parameters[0])
    end = (n_pars - 1)*[' '] + ['\n']
    if isinstance(filename_output, io.TextIOWrapper):
        f = filename_output
    else:
        open_mode = 'a' if append_output else 'w'
        f = open(filename_output, open_mode)
    if spec_id is not None:
        if isinstance(spec_id, tuple):
            ix, iy = spec_id
            print(f'#ID {ix},{iy}', file=f)
        else:
            ix = spec_id
            print(f'#ID {ix}', file=f)
    print(f'{config.n_models} {chi_sq}', file=f)
    for i_m in range(config.n_models):
        print(f'{config.model_types[i_m]}', end=' ', file=f)
        n_pars = len(parameters[i_m])
        for i_p in range(n_pars):
            print(f'{parameters[i_m][i_p]:.4f} {e_parameters[i_m][i_p]:.4f}',
                  end=end[i_p], file=f)
    if not isinstance(filename_output, io.TextIOWrapper):
        f.close()

def _voigt_model(parameters, wavelength):
    """Return a Voigt profile evaluated at the given wavelength and parameters

    This model assumes a Voigt profile defined by the parameters array, where::

        mu            -> parameters[0]
        peak          -> parameters[1]
        sigma_L       -> parameters[2]
        sigma_G       -> parameters[3]
        systemic vel  -> parameters[4]

    Parameters
    ----------
    parameters : array like
        The list of parameters to define the current model.

    wavelength : array like
        The wavelengths of the flux to be modeled.

    normed : bool, optional
        If True assumes that the peak is one. Default value is False.

    Returns
    -------
    array like
        The model evaluated at the given wavelengths using the
        given parameters.
    """
    mu = parameters[0]
    peak = parameters[1]
    sigma_L = parameters[2]
    sigma_G = parameters[3]
    f = 1 + parameters[4] / __c__

    model = np.zeros(wavelength.size)
    if sigma_L != 0 or sigma_G != 0:
        model = Voigt1D(mu*f, peak, 2.355*sigma_L, 2.355*sigma_G)(wavelength)
    else:
        # if the velocity dispersion is zero, then
        # the Voigt turns into a flat distribution
        # QUESTION: this should be a delta function, but it
        #           implies interpolation to evaluate at the
        #           position of the peak if not present in the
        #           given wavelength array
        model = np.ones((wavelength.size))
    return model

def _eline_model(parameters, wavelength, normed=False):
    """Return a Gaussian profile evaluated at the given wavelength and
    parameters

    This models assumes a Gaussian profile defined by the parameters
    array, where::

        mu           -> parameters[0]
        peak         -> parameters[1]
        sigma        -> parameters[2]
        systemic vel -> parameters[3]

    Parameters
    ----------
    parameters : array like
        The list of parameters to define the current model.

    wavelength : array like
        The wavelengths of the flux to be modeled.

    normed : bool, optional
        If True assumes that the peak is one. Default value is False.

    Returns
    -------
    array like
        The model evaluated at the given wavelengths using the
        given parameters.
    """
    f = 1 + parameters[3] / __c__  # + parameters[5]
    # sys.exit()
    mu = parameters[0]
    sigma = parameters[2]
    peak = parameters[1]
    if normed:
        peak = 1

    model = np.zeros(wavelength.size)
    if sigma != 0:
        # evaluate the Gaussian function into
        gaussian = np.exp( -0.5 * ( ( wavelength - mu * f ) / sigma )**2 )
        # QUESTION: shouldn't be this normalized to one and then multiplied by the
        #           peak parameter?
        model = peak * gaussian / ( sigma * ( ( 2 * np.pi )**0.5 ) )
    else:
        # if the velocity dispersion is zero, then
        # the Gaussian turns into a flat distribution
        # QUESTION: this should be a delta function, but it
        #           implies interpolation to evaluate at the
        #           position of the peak if not present in the
        #           given wavelength array
        model = np.ones((wavelength.size))
    return model

def _poly1d_model(parameters, wavelength):
    """Return a Gaussian profile evaluated at the given wavelength and
    parameters

    This model assumes a polynomial function where each given parameter
    is a coefficient and the number of parameters is the degree of the
    polynomial function

    Parameters
    ----------
    parameters : array like
        The list of parameters to define the current model
        
    wavelength : array like
        The wavelengths of the flux to be modeled.

    Returns
    -------
    array like
        The model evaluated at the given wavelengths using the
        given parameters
    """
    # assume a 9-degree polynomial model
    # TODO: make this more general, using the size of the parameter
    #       space to infere the degree of the polynomial model taking
    #       into account the fixed parameters
    model = np.zeros(wavelength.size)
    for j in range(len(parameters)):
        model += parameters[j] * wavelength**j
    return model

class EmissionLines(object):
    """
    Class created to fit combined models for a system of emission lines in a spectrum.

    If the user wants to add other models should create a method like `self._elines_model()`
    or `self._poly1d_model()`.

    Attributes
    ----------
    config
    best_config
    _config_list
    wavelength
    flux
    sigma_flux

    Methods
    -------
    update_config
    create_system_model
    _create_single_model
    """
    def __init__(self, wavelength, flux, sigma_flux, config):
        self.config = copy(config)
        # print(self.config.guess)
        # sys.exit()
        self.flux = flux
        # self.flux[~np.isfinite(self.flux)] = 0
        self.sigma_flux = sigma_flux
        # self.sigma_flux[~np.isfinite(self.sigma_flux)] = 0
        self.wavelength = wavelength

        # make a copy of the guess parameters for the fitting
        # set linking in fitting parameters and update boundaries accordingly
        # print(self.config.guess)
        self.latest_fit = self.config._set_linkings()
        # print(self.latest_fit)
        # sys.exit()
        self.latest_chi_sq = 1e12
        # set the initial fitting chain which will be updated on every call to
        # update fitting
        self.fitting_chain = []
        self.chi_sq_chain = []
        self.final_fit_params_mean = [np.zeros_like(par) for par in self.config.guess]
        self.final_fit_params_std = [np.zeros_like(par) for par in self.config.guess]

    def redefine_max_flux(self, flux_max=None, redefine_min=False, max_factor=1.2, min_factor=0.012):
        """
        Redefines the max value of the flux in eline models.

        Parameters
        ----------
        flux_max : double, optional
            Max flux value. Defaults to `self.flux.max()`.

        redefine_min : bool, optional
            If True redefines the minimum treshold of the parameters range.
            Defaults to False.

        max_factor : double, optional
            Defines the factor to multiply the maximum value of the integrated
            flux. It defaults to 1.2.

        min_factor : double, optional
            Defines the factor to multiply the minimum value of the integrated
            flux. It defaults to 0.012.
        """
        cf = self.config
        # print(self.flux)
        # print(self.flux.shape)
        flux_max = self.flux.max() if flux_max is None else flux_max
        flux_min = self.flux.min()
        pars_0 = cf.pars_0
        pars_1 = cf.pars_1
        new_pars_0 = []
        new_pars_1 = []
        i_p_flux = _MODELS_ELINE_PAR['flux']
        i_p_sigma = _MODELS_ELINE_PAR['sigma']
        #i_p_cont = _MODELS_POLY1D_PAR['cont']
        for pars, fit_mask, model, links, p0, p1 in zip(cf.guess, cf.to_fit, cf.model_types, cf.links, copy(cf.pars_0), copy(cf.pars_1)):
            m = (links[i_p_flux] == -1)
            if (model == 'eline') & m:
                factor_1 = flux_max*pars[i_p_sigma]*((2*np.pi)**0.5)
                factor_0 = min_factor*factor_1
                factor_1 *= max_factor
                if p1[i_p_flux] > factor_1:
                    p1[i_p_flux] = factor_1
                    if redefine_min:
                        p0[i_p_flux] = factor_0
            elif (model == 'poly1d') & m:
                n_pars = fit_mask.astype(int).sum()
                for i_p_cont in range(n_pars):
                    if p0[i_p_cont] < flux_min:
                        p0[i_p_cont] = flux_min
                    if p1[i_p_cont] > flux_max:
                        p1[i_p_cont] = flux_max
            new_pars_0.append(p0)
            new_pars_1.append(p1)
        cf.update_ranges(max_values=new_pars_1, min_values=new_pars_0)
        cf._correct_guess()
        self.latest_fit = copy(cf.guess)

    def _voigt_model(self, parameters):
        return _voigt_model(parameters, wavelength=self.wavelength)

    def _eline_model(self, parameters, normed=False):
        """Return a Gaussian profile evaluated at the given wavelength and
        parameters

        This models assumes a Gaussian profile defined by the parameters
        array, where::

            mu           -> parameters[0]
            peak         -> parameters[1]
            sigma        -> parameters[2]
            systemic vel -> parameters[3]

        Parameters
        ----------
        parameters : array like
            The list of parameters to define the current model.

        normed : bool, optional
            If True assumes that the peak is one. Default value is False.

        Returns
        -------
        array like
            The model evaluated at the given wavelengths using the
            given parameters.
        """
        wavelength = self.wavelength
        return _eline_model(parameters, wavelength=wavelength, normed=normed)

    def _poly1d_model(self, parameters):
        """Return a Gaussian profile evaluated at the given wavelength and
        parameters

        This model assumes a polynomial function where each given parameter
        is a coefficient and the number of parameters is the degree of the
        polynomial function

        Parameters
        ----------
        parameters : array like
            The list of parameters to define the current model

        Returns
        -------
        array like
            The model evaluated at the given wavelengths using the
            given parameters
        """
        wavelength = self.wavelength
        return _poly1d_model(parameters, wavelength)

    def _create_single_model(self, i_model, normed=False, ignore_poly1d=False, parameters=None):
        """Return the emission line model evaluated at the given wavelengths at the
        given config parameters `self.config.fitting_pars`.

        This function evaluate the i-model parametrized by the type and the
        parameters.

        Parameters
        ----------
        i_model : int
            The location of the model to evaluate in the config file.

        normed : bool
            Sets the peak of eline models to 1.

        ignore_poly1d : bool
            Ignore any poly1d model.

        parameters : array like
            The list of parameters to define the current model.

        Returns
        -------
        array like
            The model evaluated at the given wavelengths using the
            given parameters.
        """
        parameters = copy(self.latest_fit) if parameters is None else parameters
        model = np.zeros(self.wavelength.size)
        if self.config.model_types[i_model] == "voigt":
            model = self._voigt_model(parameters[i_model])
        elif self.config.model_types[i_model] == "eline":
            model = self._eline_model(parameters[i_model], normed=normed)
        elif self.config.model_types[i_model] == "poly1d":
            if not ignore_poly1d:
                model = self._poly1d_model(parameters[i_model])
        else:
            script_name = 'fit_elines'
            raise ValueError(f"[{script_name}] '{self.config.model_types[i_model]}' model not implemented")
        return model

    def create_system_model(self, normed=False, ignore_poly1d=False, parameters=None):
        """Return the emission system model evaluated at the given wavelengths at the
        given parameters.

        This function evaluate the all the models found in the config file, i.e., the
        models defined for the emission line system, taking into account the links and
        fixed parameters.

        Parameters
        ----------
        normed : bool
            Sets the peak of eline models to 1.

        ignore_poly1d : bool
            Ignore any poly1d model.

        parameters : array like
            The list of parameters to define the current model.

        Returns
        -------
        array like
            The model evaluated at the given wavelengths using the given parameters.
        """
        # TODO: deal with fixed parameters:
        #       split all parameters into fixed and to be fitted
        #       replace fixed parameters by the initial guess
        #       join back in all parameters
        # TODO: deal with links:
        #       check if linked models are the same type
        #       split parameters into linked and not linked
        #       replace linked parameters according to link type
        wavelength = self.wavelength
        config = self.config
        system_model = np.zeros(wavelength.size)
        for i_model in range(config.n_models):
            system_model += self._create_single_model(i_model, normed=normed,
                                                      ignore_poly1d=ignore_poly1d,
                                                      parameters=parameters)

        return system_model

    def update_fitting(self, parameters, placeholder=None, inplace=False):
        """Return a copy of the fitting parameters in each model updated

        This method takes into account fixed parameters and links, so that
        only parameters meant to be fitted are updated and linked accordingly.
        If inplace is True `self.fitting_chain` will be updated with the given
        parameters inplace.

        Parameters
        ----------
        parameters: array like
            A list of the parameter arrays to update the current values in
            the parameters to be fitted.

        placeholder: array like or None
            A list of parameters like `parameters` to be used as the initial
            values to the updated parameters instead `self.latest_fit`.
            Default value is None.

        inplace: boolean
            Whether to update `self.latest_fit` and `self.fitting_chain`
            inplace or not. Defaults to False.

        Returns
        -------
        array like
            A list of the full parameter space (including fixed parameters)
            updated taking into account links.
        """
        updated_parameters = self.config.get_updated_guess(parameters, placeholder=placeholder)

        if inplace:
            self.latest_fit = updated_parameters
            self.fitting_chain.append(self.latest_fit)

        return updated_parameters

    def update_chi_square(self, residuals, inplace=False):
        """Return the reduced chi square computed from the given residuals

        The residuals are assumed to be weighted by the standard deviation
        of the observations, so that::

            chi**2 = sum( residuals**2 ) / (n_residuals - 1 - n_free_param)

        Parameters
        ----------
        residuals : array like
            A vector of residuals between an assumed model and a set of observations,
            weighted by the standard deviation on those observations.

        inplace : boolean
            Whether to update `self.latest_chi_sq` and `self.chi_sq_chain` inplace
            or not. Defaults to False.

        Returns
        -------
        float
            The reduced chi square score
        """
        n_obs = np.count_nonzero(residuals != 0.0)
        reduced_chi_sq = ( residuals**2 ).sum() / (n_obs - 1 - self.config.n_free_param)

        if inplace:
            self.latest_chi_sq = reduced_chi_sq
            self.chi_sq_chain.append(self.latest_chi_sq)

        return reduced_chi_sq

    def update_config(self, new_guess=None, update_ranges=False, frac_range=0.2):
        """Update the original config parameters guess to the
        given fitted parameters

        This method will update the guess parameters so that the config
        object can be used as a guess of a new fitting procedure.

        Parameters
        ----------
        new_guess : array like
            A list of the fitted parameters to be used as new guess. If
            None (default), the latest fit will be used as guess.

        update_ranges : array like, optional
            Updates the range of the generated config. It will use the
            `frac_range` option to set the parameter range.

        frac_range : float, optional
            Set the half range of the parameter interval in the config. I.e.::

                min = (1 - frac_range)*guess
                max = (1 + frac_range)*guess

        Returns
        -------
        :class:`ConfigEmissionModel` class
            An updated version of the initial configuration object
            using the latest fit as the guess.
        """
        if new_guess is None:
            new_guess = self.latest_fit

        cf = self.config

        if update_ranges:
            pars = np.asarray(new_guess)
            pars_0 = copy(pars)
            pars_1 = copy(pars)
            sel_model = (np.asarray(cf.model_types) == 'eline')
            delta = pars*frac_range
            delta[:, _MODELS_ELINE_PAR['v0']] *= 0.5
            pars_0[sel_model] = pars[sel_model] - delta[sel_model]
            pars_1[sel_model] = pars[sel_model] + delta[sel_model]
            cf.update_ranges(min_values=pars_0, max_values=pars_1)

        self.config.guess = self.config._set_linkings(new_guess)
        self.config._correct_guess()

        return self.config

class EmissionLinesRND(EmissionLines):
    """
    Fits a system of emission lines (EL) with a pseudo-random search of the input parameters.

    Attributes
    ----------
    wavelength : array like
        The observed wavelengths array.

    flux : array like
        The observed spectrum array.

    sigma_flux : array like
        The array with the standard deviation of the observed spectrum.

    config : :class:`ConfigEmissionModel` class
        The class which configures the emission lines system.

    n_MC : int
        The input number of Monte-Carlo realisations.

    n_loops : int
        The maximum number of loops of the main search of the best EL parameters.

    scale_ini : float
        Controls the size of the random search step.

    plot : bool
        Nice plot of the entire process.

    fine_search : bool
        Adds a final Monte-Carlo loop to `self._MC_search` with a fine search for
        the fitted parameters. The default value is False.

    spec_id : int, tuple or None
        The index coordinate(s) of the spectrum if looping from fits data.
        Default is None.

    Methods
    -------
    fit :
        The main function of the RND method. It makes a loop searching the fitting
        parameters narrowing the parameters range at each model that have a lower chi
        square.

    output :
        Prints the results to output files and to /dev/stdout.

    See Also
    --------
    :class:`EmissionLines`
    """
    def __init__(self, wavelength, flux, sigma_flux, config,
                 n_MC, n_loops, scale_ini, plot, fine_search=None):
        EmissionLines.__init__(self, wavelength, flux, sigma_flux, config)
        self.n_loops = n_loops
        self.n_MC = n_MC
        self.scale_ini = scale_ini
        self.plot = plot
        self.n_loops_real = 0
        # self.model = None
        self.model = np.zeros(wavelength.size)
        self.model_cont = np.zeros(wavelength.size)
        self.fine_search = __ELRND_fine_search_option__ if fine_search is None else fine_search

    def _update_parameters(self, i_param, i_MC, n_MC):
        """Updates a parameter through a randomly perturbed procedure, taking
        in account the parameter max and min values.

        * For models of type `eline`:

            * It assumes that the new guess of `i_param` parameter is the min value
              plus (max - min)/`ratio`. This additive factor is then multiplied by
              a random value within [0, 1).
              ::

                new_guess = min + random*(max - min)/ratio

        * For models of type `poly1d`:
            * It assumes the new guess as the last best guess perturbed by a random
              number from a univariate "normal" (Gaussian) distribution of mean 0
              and variance 1.
              ::

                new_guess = old_guess*(1 + scale*normal_random)

        Parameters
        ----------
        i_param : int
            The index of the guess parameter to be perturbed.

        i_MC : int
            Index of MC realisation.

        n_MC : int
            Total number of MC realisations.
        """
        cf = self.config
        old_pars = copy(self.latest_fit)
        # print(f'self.latest_fit ---- {cf.get_tofit(self.latest_fit)}')
        new_pars = []
        update_iter = zip(cf.model_types, old_pars, cf.to_fit, cf.links, cf.pars_0, cf.pars_1)
        # print(f'delta_factor={delta_factor}')
        i_m = 0
        for model, pars, fit_mask, links, pars_0, pars_1 in update_iter:
            # print(f'{model} {pars} {fit_mask} {pars_0} {pars_1} {links}')
            m = fit_mask & (links == -1)
            if model != 'poly1d':
                delta_factor = i_MC/n_MC
                d_pars = delta_factor*(pars_1 - pars_0)
                rnd_lin = self.rnd_lin[(i_param + (len(pars)*i_m))*i_MC]
                pars[i_param] = pars_0[i_param] + rnd_lin*d_pars[i_param]
            else:
                n_pars = len(pars)
                rnd = self.rnd[(np.arange(n_pars) + (len(pars)*i_m))*i_MC]
                pars *= (1 + self.scale*rnd)
            m_pars_0 = m & (pars < pars_0)
            pars[m_pars_0] = pars_0[m_pars_0]
            m_pars_1 = m & (pars > pars_1)
            pars[m_pars_1] = pars_1[m_pars_1]
            new_pars.append(pars[m])
            i_m += 1
        self.update_fitting(parameters=new_pars, inplace=True)

    def _update_parameters_final(self, i_MC, n_MC):
        """Return a randomly perturbed realisation of the given parameter set within
        a range defined by the max and min values of the given parameter.

        * For models of type `eline`:

            * For the initial velocity (v0):
              It assumes that the new guess of `i_param` parameter is the min value
              plus (max - min)/(5*n_MC), wher n_MC is the number of Monte-Carlo loops.
              This additive factor is then multiplied by a random value within [0, 1).
              ::

                new_guess = min + random*(max - min)/(5*n_MC)

            * For other parameters (central wavelength, flux and sigma):
              It assumes the new guess as the last best guess perturbed by a random
              number from a univariate "normal" (Gaussian) distribution of mean 0
              and variance 1.
              ::

                new_guess = old_guess*(1 + scale*normal_random)

        * For models of type `poly1d`:

            * Do not change the guess value.
        """
        cf = self.config
        old_pars = copy(self.latest_fit)
        # print(f'cf.latest_fit ---- {cf.get_tofit(cf.latest_fit)}')
        new_pars = []
        update_iter = zip(cf.model_types, old_pars, cf.to_fit, cf.links, cf.pars_0, cf.pars_1)
        i_m = 0
        for model, pars, fit_mask, links, pars_0, pars_1 in update_iter:
            m = fit_mask & (links == -1)
            a0 = np.where(pars_0 < 0.7*pars, 0.7*pars, pars_0)
            a1 = np.where(pars_1 > 1.3*pars, 1.3*pars, pars_1)
            delta = self.scale*(a1 - a0)/(5*n_MC)
            if model == 'eline':
                n_pars = len(pars)
                for i, p in enumerate(pars):
                    rnd = self.rnd[(i + (n_pars*i_m))*i_MC]
                    if (i == _MODELS_ELINE_PAR['v0']) | (i == _MODELS_ELINE_PAR['sigma']):
                        pars[i] += rnd*delta[i]
                    else:
                        # rnd_lin = self.rnd_lin[(i_param + (__n_models_params__*i_m))*i_MC]
                        pars[i] *= (1 + self.scale*rnd)
                pars[pars < a0] = a0[pars < a0]
                pars[pars > a1] = a1[pars > a1]
                # get flux inside
                i_p = _MODELS_ELINE_PAR['flux']
                if pars[i_p] < pars_0[i_p]:
                    pars[i_p] = pars_0[i_p]
                if pars[i_p] > pars_1[i_p]:
                    pars[i_p] = pars_1[i_p]
            new_pars.append(pars[m])
            i_m += 1
        # print(f'new_pars --------- {new_pars}')
        self.update_fitting(parameters=new_pars, inplace=True)
        # print(f'cf.latest_fit ---- {cf.get_tofit(cf.latest_fit)}')

    def _models_flux_stacked(self, normed=False):
        """Creates a list with the models that will be used to fit the spectra by
        `WLS_invmat`.

        Parameter
        ---------
        normed : bool
            Sets the peak of eline models to 1.
        """
        models_stacked = []
        cf = self.config
        # n_models_free = 0
        for i_m in range(cf.n_models):
            i_p = _MODELS_ELINE_PAR['flux']
            if cf.model_types[i_m] == 'eline':
                if cf.to_fit[i_m][i_p] == 1:
                    tmp = self._eline_model(parameters=self.latest_fit[i_m], normed=normed)
                    if cf.links[i_m][i_p] == -1:
                        models_stacked.append(tmp)
                        # n_models_free += 1
                    else:
                        i_m_linked = int(cf.links[i_m][i_p] - 1)
                        model_flux_linked = models_stacked[i_m_linked]
                        tmp = tmp*cf.pars_0[i_m][i_p]
                        models_stacked[i_m_linked] = tmp + model_flux_linked
            elif cf.model_types[i_m] == 'poly1d':
                # n_models_free += cf.to_fit[i_m].astype('int').sum()
                n_pars = len(self.latest_fit[i_m])
                pars = self.latest_fit[i_m]
                # XXX:
                # WHY NOT MULTIPLY BY THE PARAMETER
                tmp = [self.wavelength**i_p for i_p in range(n_pars)]
                # XXX:
                # n_pars is the n_coeffs of poly1d
                #tmp = [pars[i_p]*self.wavelength**i_p for i_p in range(n_pars)]
                models_stacked += [tmp[i] for i in range(n_pars) if cf.to_fit[i_m][i]]
        return models_stacked

    def _WLS_models(self, models_stacked):
        """ Applies the weighted least-squares inverse matrix method in the list
        of models in order to fit the new parameters.

        Parameters
        ----------
        models_stacked : list
            A list of the models of the emission lines system.

        See Also
        --------
        :func:`pyFIT3D.common.stats.WLS_invmat`, :func:`pyFIT3D.common.stats.pdl_stats`
        """
        n_w = len(self.wavelength)
        observed = self.flux_fit + np.random.randn(n_w)*self.sigma_flux  # - cont__w
        if len(models_stacked) > 1:
            model, coeffs = WLS_invmat(observed, models_stacked)
        else:
            # print('lonely place here...')
            # print(f'len(models_stacked) <= 1:{len(models_stacked)}')
            model = models_stacked[0]
            flux_rat__w = np.divide(observed, model, where=model!=0, out=np.zeros_like(observed))
            # flux_rat__w = observed / model
            flux_rat__w[~np.isfinite(flux_rat__w)] = 0
            tmp = pdl_stats(flux_rat__w)
            model *= tmp[_STATS_POS['median']]
            coeffs = np.array([tmp[_STATS_POS['median']] for _ in range(2)])
        return model, coeffs

    def _coeffs_to_models(self, coeffs, threshold=True):
        """ Rewrites the parameters array with the integrated flux (`coeffs`)
        derived by the WLS minimization.

        Parameters
        ----------
        coeffs : array like
            A list of coefficients of the models that fit the emission lines
            system.
        threshold : bool
            Limits the integrated flux value with a threshold defined by
            `self.config.pars_0` and `self.config.pars_1`. It is defined as
            True by default.
        """
        cf = self.config
        old_pars = copy(self.latest_fit)
        new_pars = []
        update_iter = zip(cf.model_types, old_pars, cf.to_fit, cf.links, cf.pars_0, cf.pars_1)
        n_flux_free = 0
        for model, pars, fit_mask, links, pars_0, pars_1 in update_iter:
            i_p = _MODELS_ELINE_PAR['flux']
            if model == 'eline':
                if fit_mask[i_p]:
                    if links[i_p] == -1:
                        pars[i_p] = coeffs[n_flux_free]
                        if threshold:
                            if pars[i_p] < pars_0[i_p]:
                                pars[i_p] = pars_0[i_p]
                            if pars[i_p] > pars_1[i_p]:
                                pars[i_p] = pars_1[i_p]
                        n_flux_free += 1
            elif model == 'poly1d':
                # 2021-06-25: Bug corrected missing coefficients
                pars[fit_mask] = coeffs[n_flux_free:]
                n_flux_free += fit_mask.astype('int').sum()
            new_pars.append(pars[fit_mask & (links == -1)])
        self.update_fitting(parameters=new_pars, inplace=True)

    def _invert_matrix(self, normed=False, threshold=True):
        """ Do all the fitting process: creates and fit the models, fill the fitted
        coeffs to the parameters array and calculates the new chi square.

        Parameter
        ---------
        normed : bool
            Sets the peak of eline models to 1.

        threshold : bool
            Limits the integrated flux from coefficients to a threshold defined
            by `self.config.pars_0` and `self.config.pars_1`. It is defined as
            True by default.

        Returns
        -------
        array like
            The fitted model.
        """
        models_stacked = self._models_flux_stacked(normed=normed)
        _, coeffs = self._WLS_models(models_stacked)
        self._coeffs_to_models(coeffs, threshold=threshold)
        model = self.create_system_model()
        return model

    def _narrow_range(self):
        """ Decreases the range of the fitted parameters."""
        cf = self.config
        pars = np.asarray(self.latest_fit)
        pars_0 = np.asarray(copy(cf.pars_0))
        pars_1 = np.asarray(copy(cf.pars_1))
        sel_model = (np.asarray(cf.model_types) == 'eline')
        delta = 0.5*np.abs(pars_1 - pars_0)
        delta[:, _MODELS_ELINE_PAR['v0']] *= 0.5
        pars_0[sel_model] = pars[sel_model] - delta[sel_model]
        pars_1[sel_model] = pars[sel_model] + delta[sel_model]
        pars_0 = np.where(pars_0 < cf.pars_0, cf.pars_0, pars_0)
        pars_1 = np.where(pars_1 > cf.pars_1, cf.pars_1, pars_1)
        cf.update_ranges(min_values=pars_0, max_values=pars_1)
        cf._correct_guess()

    def _stats_results(self):
        """ Generates the mean and the standard deviation for the best system
        configurations (parameters) of each global Monte-Carlo loop.
        """
        cf = self.config
        best_fits = np.asarray(self.fitting_chain)[self.best_fits_loc]
        final_fit_params_mean = best_fits[0]
        n = best_fits.shape[0]
        # if n > 1:
        #     best_fits = best_fits[1:]
        #     n = best_fits.shape[0]
        if n > 1:
            self.final_fit_params_std = best_fits.std(axis=0, ddof=1)
            if n > 3:
                final_fit_params_mean = np.median(best_fits, axis=0)
            else:
                final_fit_params_mean = best_fits.mean(axis=0)
        else:
            final_fit_params_mean = best_fits.mean(axis=0)
            self.final_fit_params_std = best_fits.std(axis=0, ddof=0)
        # update the self.latest_fit with the final_fit_params
        self.final_fit_params_mean = self.update_fitting(parameters=cf.get_tofit(final_fit_params_mean), inplace=True)
        self.model = self.create_system_model(parameters=self.final_fit_params_mean)
        self.final_chi_sq, _ = calc_chi_sq(self.flux, self.model, self.sigma_flux,
                                           self.config.n_free_param + 1)

    def _add_back_noise(self, noise):
        """Adds background noise to the stddev of the final flux fitted.

        Parameters
        ----------
        noise : float
            The noise factor to be multiplied by the final (mean) flux parameter.
        """
        i_flux = _MODELS_ELINE_PAR['flux']
        i_sigma = _MODELS_ELINE_PAR['sigma']
        for i_m in range(self.config.n_models):
            if self.config.model_types[i_m] == 'eline':
                e_F = noise*__sigma_to_FWHM__*self.final_fit_params_mean[i_m][i_sigma]
                a = self.final_fit_params_std[i_m][i_flux]
                b = e_F
                self.final_fit_params_std[i_m][i_flux] = np.sqrt(a**2 + b**2)

    def _MC_search(self,
                   oversize_chi=True,
                   vel_fixed=False, sigma_fixed=False,
                   redshift_flux_threshold=True,
                   sigma_flux_threshold=False):
        """ Search the best fit parameters of the RND method. Perform all the fitting
        process and the Monte-Carlo exploration of each parameter space.

        Parameters
        ----------
        oversize_chi : bool, optional

        vel_fixed : bool, optional

        sigma_fixes : bool, optional

        redshift_flux_threhsold : bool, optional

        sigma_flux_threhsold : bool, optional

        Returns
        -------
        array like
            The best fitted model for the emission lines system.
        """

        debug = False

        self.scale = self.scale_ini

        # 1st guess #
        model_flux = self.create_system_model()
        chi_sq_ini, _ = calc_chi_sq(self.flux_fit, model_flux, self.sigma_flux, self.config.n_free_param + 1)
        if oversize_chi:
            chi_sq_ini *= 3

        latest_fit = copy(self.latest_fit)

        n_rand = self.config.n_models*len(self.config.guess[0])*self.n_MC
        self.rnd = np.random.randn(n_rand)
        self.rnd_lin = 0.8+0.4*np.random.rand(n_rand)

        if debug:
            print(f'begin MC search - chi_sq_ini={chi_sq_ini}')
            for i_m in range(self.config.n_models):
                print(f'{np.asarray(self.latest_fit)[i_m, 0:4]}')

        # Fit redshift #
        n_MC_redshift = int(self.n_MC/2)
        if vel_fixed == 0:
            n_MC_redshift = 3
        for i_MC in range(n_MC_redshift):
            self.n_loops_real += 1
            self._update_parameters(i_param=_MODELS_ELINE_PAR['v0'],
                                    i_MC=i_MC, n_MC=n_MC_redshift)
            # self.update_parameters(_MODELS_ELINE_PAR['v0'], i_MC/n_MC_redshift)
            model = self._invert_matrix(normed=True, threshold=redshift_flux_threshold)
            chi_sq, _ = calc_chi_sq(self.flux_fit, model, self.sigma_flux, self.config.n_free_param + 1)
            if debug:
                print(f'{i_MC} {chi_sq}')
            asterx = ''
            if chi_sq <= chi_sq_ini:
                latest_fit = copy(self.latest_fit)
                chi_sq_ini = chi_sq
                model_flux = model
                asterx = '*'
            if debug:
                print(f'{asterx}fit_redshift: {len(self.fitting_chain) - 1}, chi_sq={chi_sq}')
                for i_m in range(self.config.n_models):
                    print(f'{np.asarray(self.latest_fit)[i_m, 0:4]}')
        self.latest_fit = latest_fit

        if debug:
            print(f'>>{chi_sq_ini}')
            for i_m in range(self.config.n_models):
                print(f'{np.asarray(self.latest_fit)[i_m, 0:4]}')

        if self.config.check_par_fit('eline', _MODELS_ELINE_PAR['sigma']):
            # Fit sigma #
            n_MC_sigma = int(self.n_MC/3)
            if sigma_fixed == 0:
                n_MC_sigma = 3
            for i_MC in range(n_MC_sigma):
                self.n_loops_real += 1
                self._update_parameters(i_param=_MODELS_ELINE_PAR['sigma'],
                                        i_MC=i_MC, n_MC=n_MC_sigma)
                model = self._invert_matrix(normed=True, threshold=sigma_flux_threshold)
                chi_sq, _ = calc_chi_sq(self.flux_fit, model, self.sigma_flux, self.config.n_free_param + 1)
                if debug:
                    print(f'{i_MC} {chi_sq}')
                asterx = ''
                if chi_sq <= chi_sq_ini:
                    latest_fit = copy(self.latest_fit)
                    chi_sq_ini = chi_sq
                    model_flux = model
                    asterx = '*'
                if debug:
                    print(f'{asterx}fit_sigma: {len(self.fitting_chain) - 1}, chi_sq={chi_sq}')
                    for i_m in range(self.config.n_models):
                        print(f'{np.asarray(self.latest_fit)[i_m, 0:4]}')
            self.latest_fit = latest_fit

        if debug:
            print(f'>>{chi_sq_ini}')
            for i_m in range(self.config.n_models):
                print(f'{np.asarray(self.latest_fit)[i_m, 0:4]}')

        fine_search = self.fine_search
        i_MC = 0
        n_MC = int(self.n_MC/3)
        while(fine_search and (i_MC < n_MC)):
            self.n_loops_real += 1
            self._update_parameters_final(i_MC, n_MC)
            model = self.create_system_model()
            chi_sq, _ = calc_chi_sq(self.flux_fit, model, self.sigma_flux, self.config.n_free_param + 1)
            if oversize_chi:
                chi_sq *= 3
            asterx = ''
            if chi_sq <= chi_sq_ini:
                asterx = '*'
                chi_sq_ini = chi_sq
                model_flux = model
                latest_fit = copy(self.latest_fit)
                self.scale *= 0.99
                if self.scale < (0.1*self.scale_ini):
                    self.scale = 0.1*self.scale_ini
            else:
                self.scale = self.scale_ini
                if ((np.abs(chi_sq - chi_sq_ini) < self.config.chi_step) | (chi_sq_ini < self.config.chi_goal)):
                    i_MC = self.n_MC
            i_MC += 1
            if debug:
                print(f'{asterx}fine_search: {len(self.fitting_chain) - 1}, chi_sq={chi_sq}')
                for i_m in range(self.config.n_models):
                    print(f'{np.asarray(self.latest_fit)[i_m, 0:4]}')
        self.latest_fit = latest_fit

        return model_flux

    def fit(self, vel_fixed=False, sigma_fixed=False, randomize_flux=False, check_stats=True,
            redshift_flux_threshold=True, sigma_flux_threshold=False, oversize_chi=True):
        """ The main function of the RND method. It makes a loop searching the fitting
        parameters narrowing the parameters range at each model that have a lower chi
        square.
        """
        if self.plot:
            if 'matplotlib.pyplot' not in sys.modules:
                from matplotlib import pyplot as plt
            else:
                plt = sys.modules['matplotlib.pyplot']
            if 'seaborn' not in sys.modules:
                import seaborn as sns
            else:
                sns = sys.modules['seaborn']
            sns.set(context="paper",
                    style="ticks",
                    palette="colorblind",
                    color_codes=True,
                    font_scale=1.5,
                    rc={'figure.figsize':(latex_text_width, latex_text_width/golden_mean)})
            sns.despine()
            # fig = plt.gcf()
            # ax = plt.gca()
            # sns.despine()
            # ax.set_xlabel("Wavelength")
            # ax.set_ylabel("Flux")

        stop = 0
        i_iter = 0
        i_loops = 0
        chi_sq = 1e12
        self.fitting_chain = []
        self.best_fits_loc = []

        guess_model = self.create_system_model()
        model_flux = guess_model

        # if np.isnan(self.flux).all():
        #     stop = 2
        # else:
        if self.plot:
            # display original data
            title = r'$\chi^2=$-- [{:.1f} $\AA$ - {:.1f} $\AA$]'.format(self.wavelength[0], self.wavelength[-1])
            wave = [self.wavelength]
            spectra = [self.flux, self.sigma_flux]
            labels = ['observed', r'$\sigma$(flux)']
            colors = ["0.4", "0.8"]
            if self.plot == 1:
                ax = plt.gca()
                plt.cla()
                plot_spectra_ax(ax, wave_list=wave, spectra_list=spectra, labels_list=labels, color=colors, title=title)
                plt.pause(0.001)
                ylim = ax.get_ylim()
                ylim = list(ylim)
                ax.set_xlabel("Wavelength")
                ax.set_ylabel("Flux")
                #plt.show()
            elif self.plot == 2:
                f, ax = plt.subplots()
                plot_spectra_ax(ax, wave_list=wave, spectra_list=spectra, labels_list=labels, color=colors, title=title)
                ylim = ax.get_ylim()
                ylim = list(ylim)
                ax.set_xlabel("Wavelength")
                ax.set_ylabel("Flux")
                config = self.config
                par_to_get_wave = config.guess
                i_w = _MODELS_ELINE_PAR['central_wavelength']
                if par_to_get_wave[0][i_w] == 0:
                    par_to_get_wave = config.guess
                waves = [par_to_get_wave[i_m][i_w] for i_m in range(config.n_models) if config.model_types[i_m] == 'eline']
                wlrange_str = ''
                if len(waves):
                    wlrange_str = f'{int(np.min(waves))}_{int(np.max(waves))}_'
                f.savefig(f'{wlrange_str}fit_elines_rnd_input.png')
                plt.close(f)

        if check_stats:
            st_flux = pdl_stats(self.flux)
            mean_st_flux = st_flux[_STATS_POS['mean']]
            pRMS_st_flux = st_flux[_STATS_POS['pRMS']]
            if not ((mean_st_flux != pRMS_st_flux) | (mean_st_flux != 0)):
                stop = 2

        self.flux_fit = self.flux
        # for i_m in range(self.config.n_models):
        #     print(f'>>>{np.asarray(self.config.pars_0)[i_m, 0:4]}')
        #     print(f'>>>{np.asarray(self.config.pars_1)[i_m, 0:4]}')

        while ((i_iter < self.n_loops) & (not stop)):
            chi_sq_iter = self.latest_chi_sq
            # if (len(self.fitting_chain) > 0):
            #     self.latest_fit = copy(self.fitting_chain[-1])

            # print(f'chi_sq_iter={chi_sq_iter}')
            # for i_m in range(self.config.n_models):
            #     print(f'{np.asarray(self.latest_fit)[i_m, 0:4]}')

            if randomize_flux:
                self.flux_fit = self.flux + np.random.randn(self.flux.size)*self.sigma_flux*0.5

            if self.config.n_free_param > 0:
                model = self._MC_search(vel_fixed=vel_fixed, sigma_fixed=sigma_fixed,
                                        redshift_flux_threshold=redshift_flux_threshold,
                                        sigma_flux_threshold=sigma_flux_threshold,
                                        oversize_chi=oversize_chi)
            else:
                # Uses the first guess as the final model and sets the final chi
                # square, i.e. calculates the first guess and do not proceed with the search
                # of best parameters.
                # model = self._MC_search(fix=True)
                model = self.create_system_model()

            # chi_sq, _ = calc_chi_sq(self.flux, model, self.sigma_flux, self.config.n_models + 1)
            # weighted_residuals = np.divide(model - self.flux, self.sigma_flux,
            #                                where=self.sigma_flux != 0.0,
            #                                out=np.zeros_like(self.sigma_flux))
            # # update the current chi square chain with the latest model
            # self.update_chi_square(residuals=weighted_residuals, inplace=True)
            self.fitting_chain.append(self.latest_fit)
            self.latest_chi_sq, _ = calc_chi_sq(self.flux, model, self.sigma_flux, self.config.n_free_param + 1)
            self.chi_sq_chain.append(self.latest_chi_sq)

            # print(f'End MC search: chi_sq={self.latest_chi_sq}')
            # for i_m in range(self.config.n_models):
            #     print(f'{np.asarray(self.latest_fit)[i_m, 0:4]}')

            model_flux = model
            if ((self.latest_chi_sq < chi_sq_iter) | (self.config.n_free_param == 0)):
                if self.plot == 1:
                    plt.cla()
                    title = r'$\chi^2$={:f} [{:.1f} $\AA$ - {:.1f} $\AA$]'.format(self.latest_chi_sq,
                                                                                  self.wavelength[0],
                                                                                  self.wavelength[-1])
                    wave = [self.wavelength]
                    spectra = [self.flux, model, self.flux-model]
                    labels = ['observed', 'latest best fit', 'residual']
                    colors = ["0.4", "b", "r"]
                    ax = plt.gca()
                    plt.cla()
                    plot_spectra_ax(ax, wave_list=wave, spectra_list=spectra, color=colors, labels_list=labels, title=title)
                    ylimtmp = ax.get_ylim()
                    ylim = ax.get_ylim()
                
                    if ylimtmp[0] < ylim[0]:
                        ylim[0] = ylimtmp[0]
                    if ylimtmp[1] > ylim[1]:
                        ylim[1] = ylimtmp[1]
                    ax.set_xlabel("Wavelength")
                    ax.set_ylabel("Flux")
                    ax.set_ylim(ylim)
                    plt.pause(0.001)
                    # plt.show()
                model_flux = model
                self._narrow_range()
                # self.fitting_chain.append(copy(self.latest_fit))
                self.best_fits_loc.append(len(self.fitting_chain) - 1)
                # for i_m in range(self.config.n_models):
                #     print(f'*{np.asarray(self.latest_fit)[i_m, 0:4]}')
                # for i_m in range(self.config.n_models):
                #     print(f'>>>{np.asarray(self.config.pars_0)[i_m, 0:4]}')
                #     print(f'>>>{np.asarray(self.config.pars_1)[i_m, 0:4]}')
                i_iter += 1
            else:
                i_loops += 1
                if i_loops > (5*self.n_loops):
                    stop = 1

        if (stop > 1) or (not (np.isfinite(model_flux).any())) or (len(self.best_fits_loc) == 0):
            print('RND unable to fit...')
            self.final_chi_sq = np.nan
            self.model = guess_model
        else:
            print(f'-> real number of loops = {self.n_loops_real}')
            self.model = model_flux
            self.residuals = self.flux - self.model
            # weighted_residuals = np.divide(self.residuals, self.sigma_flux,
            #                                where=self.sigma_flux != 0.0,
            #                                out=np.zeros_like(self.sigma_flux))
            # # update the current chi square chain with the latest model
            # self.update_chi_square(residuals=weighted_residuals, inplace=True)
            # # self.chi_sq, _ = calc_chi_sq(self.flux, self.model, self.sigma_flux, self.config.n_models + 1)
            self._stats_results()
            stback = pdl_stats(self.residuals)
            self._add_back_noise(noise=stback[_STATS_POS['pRMS']])

            if self.plot:
                plt.cla()
                title = r'$\chi^2$={:f} [{:.1f} $\AA$ - {:.1f} $\AA$]'.format(self.latest_chi_sq, self.wavelength[0], self.wavelength[-1])
                wave = [self.wavelength]
                spectra = [self.flux, self.sigma_flux, self.model, self.flux - self.model]
                labels = ['observed', r'$\sigma$(flux)', 'fitted model', 'residual']
                colors = ["0.4", "0.8", "b", "r"]
                if self.plot == 1:
                    ax = plt.gca()
                    plt.cla()
                    plot_spectra_ax(ax, wave_list=wave, spectra_list=spectra, labels_list=labels, color=colors, title=title)
                    ylimtmp = ax.get_ylim()
                    ylim = ax.get_ylim()
                    if ylimtmp[0] < ylim[0]:
                        ylim[0] = ylimtmp[0]
                    if ylimtmp[1] > ylim[1]:
                        ylim[1] = ylimtmp[1]
                    ax.set_ylim(ylim)
                    ax.set_xlabel("Wavelength")
                    ax.set_ylabel("Flux")
                    plt.pause(0.001)
                elif self.plot == 2:
                    f, ax = plt.subplots()
                    plot_spectra_ax(ax, wave_list=wave, spectra_list=spectra, labels_list=labels, color=colors, title=title)
                    ylimtmp = ax.get_ylim()
                    ylim = ax.get_ylim()
                    if ylimtmp[0] < ylim[0]:
                        ylim[0] = ylimtmp[0]
                    if ylimtmp[1] > ylim[1]:
                        ylim[1] = ylimtmp[1]
                    ax.set_ylim(ylim)
                    ax.set_xlabel("Wavelength")
                    ax.set_ylabel("Flux")
                    i_w = _MODELS_ELINE_PAR['central_wavelength']
                    config = self.config
                    par_to_get_wave = config.guess
                    if par_to_get_wave[0][i_w] == 0:
                        par_to_get_wave = config.guess
                    waves = [par_to_get_wave[i_m][i_w] for i_m in range(config.n_models) if config.model_types[i_m] == 'eline']
                    wlrange_str = ''
                    if len(waves):
                        wlrange_str = f'{int(np.min(waves))}_{int(np.max(waves))}_'
                    f.savefig(f'{wlrange_str}fit_elines_rnd_output.png')

    def output_to_screen(self, spec_id=None):
        if spec_id is not None:
            if isinstance(spec_id, tuple):
                ix, iy = spec_id
                print(f'-> ID {ix},{iy}')
            else:
                ix = spec_id
                print(f'-> ID {ix}')
        print(f'-> number of models = {self.config.n_models} | chi^2 = {self.final_chi_sq}')
        max_model_name = np.max([len(x) for x in self.config.model_types])
        names_parse = {'central_wavelength': 'wave', 'v0': 'vel'}
        max_par_name = np.max([
            len(names_parse.get(x, x)) for x in list(itertools.chain.from_iterable([v.keys() for v in _EL_MODELS.values()]))
        ])
        # max_par_name = np.max([
        #     len(names_parse.get(x, x)) for x in list(itertools.chain.from_iterable([v.keys() for v in self.config._EL_MODELS.values()]))
        # ])
        # for model, _MODELS_PAR in self.config._EL_MODELS.items():
        msg = ''
        bar = ''
        last_model = ''
        for i_m, model in enumerate(self.config.model_types):
            last_msg = msg
            _MODELS_PAR = _EL_MODELS[model]
            # _MODELS_PAR = self.config._EL_MODELS[model]
            msg = ('| {:>' + str(max_model_name) + 's} ').format(model)
            space_after_model = ' '*len(msg)
            msg_pars = ''
            # 2021-06-25: bugfix output poly1d coefficients
            if model == 'poly1d':
                ncoeffs = self.config.to_fit[i_m].astype(int).sum()
                _MODELS_PAR = {f'coeff{i}': i for i in range(ncoeffs)}
            for par_name, i_p in _MODELS_PAR.items():
                par_name = names_parse.get(par_name, par_name)
                val = self.final_fit_params_mean[i_m][i_p]
                e_val = self.final_fit_params_std[i_m][i_p]
                # msg += ('| {:>' + str(max_par_name) + 's} = ').format(par_name)
                _msg = f'| {val:9.4f} +/- {e_val:7.4f} '
                if not i_p:
                    msg_pars += str(space_after_model)
                msg_pars += ('| {:^' + str(len(_msg) - 3) + 's} ').format(par_name)
                msg += _msg
            msg_pars += '|'
            msg += '|'
            if not i_m:
                bar = space_after_model + '-'*(len(msg_pars)-len(space_after_model))
                print(bar)
                print(msg_pars)
                bar = '-'*len(msg)
                print(bar)
            else:
                if last_model != model:
                    print(bar)
                    # bar = space_after_model + '-'*(len(msg_pars)-len(space_after_model))
                    # print(bar)
                    print(msg_pars)
                    bar = '-'*len(msg)
                    print(bar)
            print(msg)
            bar = '-'*len(msg)
            last_model = model
        print(bar)

    def output(self, filename_output, filename_spectra=None, filename_config=None,
               append_output=False, spec_id=None):
        """ Outputs the final data from the rnd method.

        Parameters
        ----------
        filename_output : str
            The output filename for the final result.

        filename_spectra : str or None
            The output filename for the input spectra, output model and output residuals.

        filename_config : str or None
            The output filename for the final configuration of the emission lines system.
            This file will be formatted like the input config file. If None does not print
            the config file.

        append_output : bool
            When True, open `filename_output` for append. Otherwise, rewrite the file.

        spec_id : int or tuple
            The spec coordinate id. For cubes is a tuple with (ix, iy).
        """
        self.update_config(self.final_fit_params_mean)

        output_config_final_fit(self.config, filename_output, self.final_chi_sq,
                                e_parameters=self.final_fit_params_std,
                                append_output=append_output, spec_id=spec_id)

        if filename_spectra is not None:
            output_spectra(self.wavelength, [self.flux, self.model, self.flux - self.model],
                           filename_spectra)

        if filename_config is not None:
            self.config.print(filename=filename_config)

class EmissionLinesLM(EmissionLines):
    def __init__(self, wavelength, flux, sigma_flux, config, n_MC, n_loops, scale_ini, plot):

        EmissionLines.__init__(self, wavelength, flux, sigma_flux, config)

        self.__script_name__ = sys.argv[0]
        self.__script_name__ = self.__script_name__.strip(".py").strip("/")

        self.n_loops = n_loops
        self.n_MC = n_MC
        self.n_loops_real = 0

        self.scale_ini = scale_ini
        self.plot = plot

        self.model_configurations = []

    def FITfunc(self, x, config, wavelength, flux, sigma_flux):
        """Return the models evaluated at the current state of the parameters

        This function deals with parameter linkings & fixed parameters

        Parameters
        ----------
        x : array like
            The list of parameters to fit

        all_parameters : array like
            The list of *all* parameters (fixed and not fixed)

        is_fixed : array like
            A boolean mask for the fixed parameters so that::

                x = map(lambda ar: ar[is_fixed])

        wavelength : array like
            The wavelength array in which the model will be evaluated

        flux : array like
            The observed fluxes

        sigma_flux : array like
            The standard deviation on the observed fluxes

        Returns
        -------
        array like
            The array of residuals resulting from evaluating the model in at the
            given parameters and the observed independent variable, i.e., the wavelength
        """

        # TODO: handle current state of the parameters:
        #       split the parameters into fixed and to be fitted
        #       replace current state of the parameters into the fitted ones
        #       join back in the parameters and evaluate the model

        # reconstruct original structure of the fitting pars
        updated_pars = config.list_pars(x)
        # update the parameters to be fitted with the new parameters
        self.update_fitting(updated_pars, inplace=True)
        # evaluate the new model
        model = self.create_system_model()
        # compute the residual weighted by the flux standard deviation
        weighted_residuals = np.divide(model - flux, sigma_flux, where=sigma_flux != 0.0, out=np.zeros_like(sigma_flux))
        # update the current chi square chain with the latest model
        self.update_chi_square(residuals=weighted_residuals, inplace=True)

        return weighted_residuals

    def fit(self, check_stats=True):
        # create guess model
        guess_model = self.create_system_model()
        # initialize iteration conditions
        success_count, perturbate_guess = 0, False
        i_iter = 0

        if self.plot:
            if 'matplotlib.pyplot' not in sys.modules:
                from matplotlib import pyplot as plt
            else:
                plt = sys.modules['matplotlib.pyplot']
            if 'seaborn' not in sys.modules:
                import seaborn as sns
            else:
                sns = sys.modules['seaborn']

            sns.set(context="paper",
                    style="ticks",
                    palette="colorblind",
                    color_codes=True,
                    font_scale=1.5,
                    rc={'figure.figsize':(latex_text_width, latex_text_width/golden_mean)})
            # fig, ax = plt.subplots()
            # fig = plt.gcf()
            # ax = plt.gca()

            sns.despine()
            # configure axes
            # ax.set_xlabel("Wavelength")
            # ax.set_ylabel("Flux")
            # data_colors = sns.color_palette(n_colors=self.n_loops, palette="Greys")
            # fits_colors = sns.color_palette(n_colors=self.n_loops, palette="Blues")

        self.best_fits_loc = []

        stop = 0
        st_flux = pdl_stats(self.flux)
        mean_st_flux = st_flux[_STATS_POS['mean']]
        pRMS_st_flux = st_flux[_STATS_POS['pRMS']]
        if check_stats:
            if not ((mean_st_flux != pRMS_st_flux) | (mean_st_flux != 0)):
                stop = 1

        if not stop and not (np.isnan(self.flux).all()) and (self.config.n_free_param > 0):
            if self.plot:
                # display original data
                # ax.plot(self.wavelength, self.flux, "-", lw=1.5, color="0.4", label=r"$F_\mathrm{observed}$")
                # ax.plot(self.wavelength, self.sigma_flux, "b-", label=r"$\sigma_\mathrm{observed}$")
                # display initial guess model rescaled to data flux units
                # ax.plot(self.wavelength, guess_model, "--", lw=1, color="0.7", label=r"$F_\mathrm{guess}$")
                title = r'$\chi^2=$-- [{:.1f} $\AA$ - {:.1f} $\AA$]'.format(self.wavelength[0], self.wavelength[-1])
                wave = [self.wavelength]
                spectra = [self.flux, self.sigma_flux]
                labels = ['observed', r'$\sigma$(flux)']
                colors = ["0.4", "0.8"]
                if self.plot == 1:
                    ax = plt.gca()
                    plt.cla()
                    plot_spectra_ax(ax, wave_list=wave, spectra_list=spectra, labels_list=labels, title=title)
                    plt.pause(0.001)
                    ylim = ax.get_ylim()
                    ylim = list(ylim)
                elif self.plot == 2:
                    f, ax = plt.subplots()
                    plot_spectra_ax(ax, wave_list=wave, spectra_list=spectra, labels_list=labels, title=title)
                    ylim = ax.get_ylim()
                    ylim = list(ylim)
                    ax.set_xlabel("Wavelength")
                    ax.set_ylabel("Flux")
                    config = self.config
                    par_to_get_wave = config.guess
                    i_w = _MODELS_ELINE_PAR['central_wavelength']
                    if par_to_get_wave[0][i_w] == 0:
                        par_to_get_wave = config.guess
                    waves = [par_to_get_wave[i_m][i_w] for i_m in range(config.n_models) if config.model_types[i_m] == 'eline']
                    wlrange_str = ''
                    if len(waves):
                        wlrange_str = f'{int(np.min(waves))}_{int(np.max(waves))}_'
                    f.savefig(f'{wlrange_str}fit_elines_LM_input.png')
                    plt.close(f)

            while success_count < self.n_loops:
                # increase the counter of iterations (including failed ones) and
                # check if the current iteration is not equal to the maximum
                # number of iterations requested by the user.
                i_iter += 1

                if i_iter > self.n_MC:
                    # print(f"[{self.__script_name__}] maximum number of MC realisations reached without convergence.")
                    break
                # if requested by a failed iteration, randomly perturbate the initial
                # guess. This comes useful when the fitting drowns in a deep minimum
                # avoiding convergence.
                current_cf = copy(self.config)
                if perturbate_guess:
                    current_cf.guess = current_cf.randomize_guess()
                    perturbate_guess = False

                # randomly draw a sample of the observed spectrum
                # assuming a Gaussian distribution of the errors
                flux_realisation = self.flux + np.random.randn(self.flux.size) * self.sigma_flux

                try:
                    # FITfunc returns the current residuals between the observed SED
                    # and the proposed model determined by a set of parameters defined
                    # and updated in the configuration object (self.config).
                    # The least_squares function then computes the minimization of the
                    # cost function:
                    #
                    #    F(pars, *args) = 0.5 * sum_i( rho(FITfunc_i(pars, *args)**2) )
                    #
                    # for i = 0,..., N_wl and where args is a tuple containing the
                    # configuration object, and the wavelength and the fluxes arrays of
                    # N_wl wavelength points. The rho function is the so called loss
                    # function, by default:
                    #
                    #   rho(FITfunc(pars, *args)**2) = FITfunc(pars, *args)**2
                    #
                    # is exactly the traditional minimum least squares. The default
                    # minimization algorithm is the Trust Region Reflective (TRF)
                    # which allows for boundaries on the fitting parameters to be set.
                    # See least_squares documentation for more details on this.
                    fit = least_squares(
                        fun=self.FITfunc,
                        args=(current_cf, self.wavelength, flux_realisation, self.sigma_flux),
                        x0=current_cf.vectorize_pars(self.latest_fit, only_tofit=True),
                        bounds=current_cf.boundaries,
                        verbose=0
                    )

                    if not fit.success:
                        raise Exception(f"Status: {fit.status} message: {fit.message}")

                    self.best_fits_loc.append(len(self.fitting_chain)-1)
                    # display the fitted model to the current MC realisation
                    if self.plot == 1:
                        plt.cla()
                        title = r'$\chi^2=${:.3f} [{:.1f} $\AA$ - {:.1f} $\AA$]'.format(
                            self.latest_chi_sq,
                            self.wavelength[0],
                            self.wavelength[-1]
                        )
                        wave = [self.wavelength]
                        model = self.create_system_model()
                        spectra = [self.flux, model, self.flux-model]
                        labels = ['observed', 'latest best fit', 'residual']
                        colors = ["0.4", "b", "r"]
                        ax = plt.gca()
                        plot_spectra_ax(ax, wave_list=wave, spectra_list=spectra, labels_list=labels, color=colors, title=title)
                        ylimtmp = ax.get_ylim()
                        ylim = ax.get_ylim()
                        if ylimtmp[0] < ylim[0]:
                            ylim[0] = ylimtmp[0]
                        if ylimtmp[1] > ylim[1]:
                            ylim[1] = ylimtmp[1]
                        ax.set_ylim(ylim)
                        plt.pause(0.001)

                    success_count += 1
                except Exception as ex:
                    # print(f"[{self.__script_name__}] {ex}")
                    # print(f"[{self.__script_name__}] Running perturbation of the initial guess")
                    perturbate_guess = True

                self.model_configurations.append(current_cf)

            # compute the best (averaged) model
            self.model = np.zeros(self.wavelength.size)
            for self.config in self.model_configurations:
                self.model += self.create_system_model()
            self.model /= len(self.model_configurations)

        if stop or (len(self.best_fits_loc) == 0):
            self.final_chi_sq = np.nan
            self.model = guess_model
        else:
            best_fits = np.asarray(self.fitting_chain)[self.best_fits_loc]
            final_fit_params_mean = best_fits.mean(axis=0)
            cf = self.model_configurations[-1]
            self.final_fit_params_mean = self.update_fitting(parameters=cf.get_tofit(final_fit_params_mean), inplace=True)
            self.final_fit_params_std = best_fits.std(axis=0, ddof=1)
            self.model = self.create_system_model(parameters=self.final_fit_params_mean)
            self.final_chi_sq, _ = calc_chi_sq(self.flux, self.model, self.sigma_flux,
                                               self.config.n_free_param + 1)
            # self.final_chi_sq = self.chi_sq_chain[self.best_fits_loc[-1]]

        if self.plot:
            plt.cla()
            title = r'$\chi^2=${:.3f} [{:.1f} $\AA$ - {:.1f} $\AA$]'.format(self.final_chi_sq, self.wavelength[0], self.wavelength[-1])
            wave = [self.wavelength]
            spectra = [self.flux, self.sigma_flux, self.model, self.flux-self.model]
            labels = ['observed', r'$\sigma$(flux)', 'latest best fit', 'residual']
            colors = ["0.4", "0.8", "b", "r"]

            if self.plot == 1:
                ax = plt.gca()
                plot_spectra_ax(ax, wave_list=wave, spectra_list=spectra, labels_list=labels, color=colors, title=title)
                ylimtmp = ax.get_ylim()
                ylim = ax.get_ylim()
                if ylimtmp[0] < ylim[0]:
                    ylim[0] = ylimtmp[0]
                if ylimtmp[1] > ylim[1]:
                    ylim[1] = ylimtmp[1]
                ax.set_ylim(ylim)
            elif self.plot == 2:
                f, ax = plt.subplots()
                plot_spectra_ax(ax, wave_list=wave, spectra_list=spectra, labels_list=labels, color=colors, title=title)
                ylimtmp = ax.get_ylim()
                ylim = ax.get_ylim()
                if ylimtmp[0] < ylim[0]:
                    ylim[0] = ylimtmp[0]
                if ylimtmp[1] > ylim[1]:
                    ylim[1] = ylimtmp[1]
                ax.set_ylim(ylim)
                config = self.config
                par_to_get_wave = config.guess
                i_w = _MODELS_ELINE_PAR['central_wavelength']
                if par_to_get_wave[0][i_w] == 0:
                    par_to_get_wave = config.guess
                waves = [par_to_get_wave[i_m][i_w] for i_m in range(config.n_models) if config.model_types[i_m] == 'eline']
                wlrange_str = ''
                if len(waves):
                    wlrange_str = f'{int(np.min(waves))}_{int(np.max(waves))}_'
                f.savefig(f'{wlrange_str}fit_elines_LM_output.png')
                plt.close(f)

    def output_to_screen(self, spec_id=None):
        if spec_id is not None:
            if isinstance(spec_id, tuple):
                ix, iy = spec_id
                print(f'-> ID {ix},{iy}')
            else:
                ix = spec_id
                print(f'-> ID {ix}')
        print(f'-> number of models = {self.config.n_models} | chi^2 = {self.final_chi_sq}')
        max_model_name = np.max([len(x) for x in self.config.model_types])
        names_parse = {'central_wavelength': 'wave', 'v0': 'vel'}
        max_par_name = np.max([
            len(names_parse.get(x, x)) for x in list(itertools.chain.from_iterable([v.keys() for v in _EL_MODELS.values()]))
        ])
        # for model, _MODELS_PAR in _EL_MODELS.items():
        n_models = self.config.model_types
        msg = ''
        bar = ''
        last_model = ''
        for i_m, model in enumerate(self.config.model_types):
            last_msg = msg
            _MODELS_PAR = _EL_MODELS[model]
            # _MODELS_PAR = self.config._EL_MODELS[model]
            msg = ('| {:>' + str(max_model_name) + 's} ').format(model)
            space_after_model = ' '*len(msg)
            msg_pars = ''
            if model == 'poly1d':
                ncoeffs = self.config.to_fit[i_m].astype(int).sum()
                _MODELS_PAR = {f'coeff{i}': i for i in range(ncoeffs)}
            for par_name, i_p in _MODELS_PAR.items():
                par_name = names_parse.get(par_name, par_name)
                val = self.final_fit_params_mean[i_m][i_p]
                e_val = self.final_fit_params_std[i_m][i_p]
                # msg += ('| {:>' + str(max_par_name) + 's} = ').format(par_name)
                _msg = f'| {val:9.4f} +/- {e_val:7.4f} '
                if not i_p:
                    msg_pars += str(space_after_model)
                msg_pars += ('| {:^' + str(len(_msg) - 3) + 's} ').format(par_name)
                msg += _msg
            msg_pars += '|'
            msg += '|'
            if not i_m:
                bar = space_after_model + '-'*(len(msg_pars)-len(space_after_model))
                print(bar)
                print(msg_pars)
                bar = '-'*len(msg)
                print(bar)
            else:
                if last_model != model:
                    print(bar)
                    # bar = space_after_model + '-'*(len(msg_pars)-len(space_after_model))
                    # print(bar)
                    print(msg_pars)
                    bar = '-'*len(msg)
                    print(bar)
            print(msg)
            bar = '-'*len(msg)
            last_model = model
        print(bar)

    def output(self, filename_output, filename_spectra=None, filename_config=None,
               append_output=False, spec_id=None):
        """ Outputs the final data from the LM method.

        Parameters
        ----------
        filename_spectra : str
            The output filename for the input spectra, output model and output residuals.

        filename_output : str or None
            The output filename for the final result.

        filename_config : str or None
            The output filename for the final configuration of the emission lines system.
            This file will be formatted like the input config file.

        append_output : bool
            When True, open `filename_output` for append. Otherwise, rewrite the file. It
            defaults to rewrite the file.

        spec_id : int or tuple
            The spec coordinate id. For cubes is a tuple with (ix, iy).
        """
        self.update_config(self.final_fit_params_mean)

        output_config_final_fit(self.config, filename_output, self.final_chi_sq,
                                e_parameters=self.final_fit_params_std,
                                append_output=append_output, spec_id=spec_id)

        if filename_spectra is not None:
            output_spectra(self.wavelength, [self.flux, self.model, self.flux - self.model],
                           filename_spectra)

        if filename_config is not None:
            self.config.print(filename=filename_config)
