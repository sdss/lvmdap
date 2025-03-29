import os
import sys
import itertools
import numpy as np
from os import getcwd
from astropy.io import fits
from os.path import basename, isfile, join, abspath
from copy import deepcopy as copy

from .tools import peak_finder, list_eml_compare
from .stats import pdl_stats, _STATS_POS, calc_chi_sq
from .io import get_data_from_fits, get_wave_from_header, array_to_fits
from .io import print_time, read_masks_file, remove_isfile, print_verbose
from .io import trim_waves, sel_waves, read_spectra, output_spectra, write_img_header
from pyFIT3D.modelling.gas import EmissionLines, EmissionLinesLM, EmissionLinesRND

from .constants import __sigma_to_FWHM__, __FWHM_to_sigma__, __c__
from .constants import __selected_extlaw__, __selected_R_V__, __n_models_params__
from .constants import __ELRND_fine_search_option__, _MODELS_ELINE_PAR, _EL_MODELS

# _EL_MODELS

# TODO: implement attribute `self.fitting_pars_chain` where all the iterations
#       over the parameters made during the fitting procedure will be stored
#       In this attribute, the last item will correspond to the latest successful
#       fitting during the fitting procedure.
# TODO: in line with the above TODO, implement a method to get the last state of
#       of the fitting parameters.
# TODO: implement a method to calculate the reduced chi square for
#       `self.fitting_pars_chain`.
# TODO: implement error estimation method for the fitting parameters chain
class ConfigEmissionModel(object):
    """
    Constructor for a configuration of one emission line system.

    This class takes one config filename compliant with the FIT3D
    formating and builds an object containing

    Attributes
    ----------
    n_models : int
        Number of models found in the given config file.

    chi_goal : float
        The maximum chi square to admit during the fit.

    chi_step : float
        The size of the step in the chi square.

    model_types : array like
        A list of the types of models to apply during the emission line fitting.

    guess : array like
        A list of initial values for the parameters.

    to_fit : boolean array like
        Whether the given parameter will be fitted (1) or left fixed at initial value (0).

    ea : array like
        MISSING DOC

    ranges : tuple of array like
        The ranges to explore in the parameter space during the fitting

    links : array like
        The list of link types in config file::

            -1: no linking
             0: the guess value of the current parameter should be *added*
                by the corresponding fitted parameter in the previous model
             1: the guess value of the current parameter should be *multiplied*
                by the corresponding fitted parameter in the previous model

    fitting_chain : array like
        The most important attribute. This is a list of all parameters stages during
        the fitting procedure. This attribute is meant to be updated using the
        `self.updated_pars` method, taking into account the fixed parameters and the
        links from the configuration of the system.

    Methods
    -------
    _load :
        Loads the configuration for an emission line system from a config file.

    _correct_guess :
        Fixes the messed up guess parameters out of the given ranges.

    _set_linkings :
        Return a copy of the given parameters with links parsed
        also updating the boundadies accordingly.

    get_tofit :
        Gets a filtered version of the parameters to fit.

    randomize_guess :
        Returns a random perturbation of the initial guess parameters.

    update_ranges :
        Updates the values of the parameters ranges also updating boundaries.

    get_updated_guess :
        Returns a version of `self.guess` updated by `new_guess` fixed by masks and links.

    vectorize_pars :
        Returns a vectorized (flattened np.array) version of the given parameters list.

    list_pars :
        Returns a list version of the given parameter vector

    check_par_fit :
        Returns True if a choosen parameter should be fitted.

    print :
        Produces a config file using the present config.
    """

    def __init__(self, filename=None, chi_goal=0.5, chi_step=0.1, verbose=False):
        self._filename = filename
        self._verbose = verbose
        self.chi_goal = chi_goal
        self.chi_step = chi_step
        self.n_models = 0
        self.n_free_param = 0
        self.model_types = []
        self.guess, self.to_fit, self.ea, self.pars_0, self.pars_1, self.links = [], [], [], [], [], []

        self._EL_MODELS = {}

        # load configuration file
        if filename is not None:
            # self._new_load()
            self._load()
            # fix guess parameters in case
            # users mess up (they will, always!)
            self._correct_guess()

            # create boundaries attribute
            self.boundaries = (
                self.vectorize_pars(self.pars_0),
                self.vectorize_pars(self.pars_1)
            )

    def _new_load(self):
        with open(self._filename, 'r') as f:
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)

            l = f.readline()
            _, n_models, chi_goal, chi_step = l.split()

            # self.n_models = int(n_models)
            self.chi_goal = float(chi_goal)
            self.chi_step = float(chi_step)
            self.n_free_param = 0

            model_names = np.unique(list(_EL_MODELS.keys()))
            stop = False
            while l:
                l = f.readline()
                model_name = l.strip()
                if model_name in model_names:
                    pos = f.tell()
                    l = f.readline()
                    n_pars = 0
                    while (l.strip() not in model_names) and (f.tell() != file_size):
                        n_pars += 1
                        _pos = f.tell()
                        l = f.readline()
                    if (f.tell() == file_size):
                        n_pars += 1
                        stop = True
                    f.seek(pos)
                    _m = np.loadtxt(
                        fname=f,
                        max_rows=n_pars,
                        dtype=np.dtype([
                            ("guess", np.float),
                            ("to_fit", np.bool),
                            ("pars_0", np.float),
                            ("pars_1", np.float),
                            ("links", np.int)
                        ]),
                        unpack=True
                    )
                    if not stop:
                        f.seek(_pos)
                    self.add_model(
                        model_name, guess=_m[0], to_fit=_m[1],
                        pars_0=_m[2], pars_1=_m[3], links=_m[4],
                        n_pars=n_pars,
                    )

            if self._verbose:
                print(f'{n_models} models to fit')
                print(f'{self.n_free_param} free parameters')

        return None

    def _load(self):
        """Set the configuration parameters from the given config filename

        Each column in the config file corresponding to a specific model is
        stored in a list, such that there is one element in the list per model.
        The global parameters in the config file are also stored, such as the
        number of models (n_models), the maximum chi square allowed (chi_goal)
        and the step in chi square (chi_step).

        n_models : int
            Number of models found

        chi_goal : float
            The maximum chi square to admit during the fit

        chi_step : float
            The size of the step in the chi square

        model_types : array like
            A list of the types of models to apply during the emission line fitting

        guess : array like
            A list of initial values for the parameters

        to_fit : array like
            Whether the given parameter will be fitted (1) or left fixed at initial value (0)

        ea : array like
            MISSING DOC

        {pars_0, pars_1} : arrays-like
            The range to explore in the parameter space during the fitting

        links : array like
            Type of link between the models

        n_free_param : array like
            Number of free parameters in the configuration for the fitting
        """

        with open(self._filename) as f:
            l = f.readline()
            _, n_models, chi_goal, chi_step = l.split()

            self.n_models = int(n_models)
            self.chi_goal = float(chi_goal)
            self.chi_step = float(chi_step)
            self.n_free_param = 0

            for i in range(self.n_models):
                l = f.readline()
                model_name = l.strip()
                self.model_types.append(model_name)
                l_skip = 2 + 10*i

                _m = np.loadtxt(
                    fname=f,
                    max_rows=__n_models_params__,
                    dtype=np.dtype([
                        ("guess", np.float),
                        ("to_fit", np.bool),
                        ("pars_0", np.float),
                        ("pars_1", np.float),
                        ("links", np.int)
                    ]),
                    unpack=True
                )

                # self.add_model(
                #     model_type, guess=_m[0], to_fit=_m[1],
                #     pars_0=_m[2], pars_1=_m[3], links=_m[4],
                # )
                if model_name not in self._EL_MODELS.keys():
                    _tmp = _EL_MODELS[model_name]
                    if model_name == 'poly1d':
                        _tmp = {f'coeff{i}':i for i in range(__n_models_params__)}
                    self._EL_MODELS[model_name] = copy(_tmp)

                self.guess.append(_m[0])
                self.to_fit.append(_m[1])
                self.ea.append(np.zeros(__n_models_params__))
                self.pars_0.append(_m[2])
                self.pars_1.append(_m[3])
                self.links.append(_m[4])
                self.n_free_param += (self.to_fit[i] & (self.links[i] == -1)).sum()

            if self._verbose:
                print(f'{n_models} models to fit')
                print(f'{self.n_free_param} free parameters')

        return None

    def add_model(self, model_type, guess, to_fit, pars_0, pars_1, links, n_pars=__n_models_params__):
        i = self.n_models
        self.n_models += 1
        self.model_types.append(model_type)
        self.to_fit.append(to_fit)
        self.guess.append(guess)
        self.pars_0.append(pars_0)
        self.pars_1.append(pars_1)
        self.links.append(links)

        # ea deprecated
        self.ea.append(np.zeros(n_pars))

        self.n_free_param += (self.to_fit[i] & (self.links[i] == -1)).sum()

        # _MODEL_PARS
        if model_type not in self._EL_MODELS.keys():
            _tmp = _EL_MODELS[model_type]
            if model_type == 'poly1d':
                _tmp = {f'coeff{i}':i for i in range(n_pars)}
            self._EL_MODELS[model_type] = copy(_tmp)

        self._correct_guess()

        # create boundaries attribute
        self.boundaries = (
            self.vectorize_pars(self.pars_0),
            self.vectorize_pars(self.pars_1)
        )

        if self._verbose:
            print(f'{self.n_models} models to fit')
            print(f'{self.n_free_param} free parameters')

    def _correct_guess(self):
        """Fix the out of bound guess values using the ranges in config file

        This method changes inplace the guess values to be within the ranges
        given in the config file.
        """

        fixed_guess = []
        for pars, fit_mask, links, pars_0, pars_1 in zip(self.guess, self.to_fit, self.links, self.pars_0, self.pars_1):
            # TODO: this limits should be inclusive (<= and >=)
            lower_mask = pars < pars_0
            upper_mask = pars > pars_1

            fit_link_mask = fit_mask & (links == -1)

            pars[fit_link_mask & lower_mask] = pars_0[fit_link_mask & lower_mask]
            pars[fit_link_mask & upper_mask] = pars_1[fit_link_mask & upper_mask]
            fixed_guess.append(pars)
        self.guess = self._set_linkings(fixed_guess)
        return None

    def _set_linkings(self, parameters=None):
        """Return a copy of the given parameters with links parsed
        also updating the boundadies accordingly.

        This method interprets the links column of the config file for
        emission line systems, also updating in the process the range
        of the linked parameters.

        Parameters
        ----------
        parameters : array like
            A list of the parameters to be filtered. Defaults to `self.guess`.

        Returns
        -------
        linked_parameters : array like
            An updated list of all the parameters with the links taken
            care of.
        """

        linked_pars = copy(self.guess) if parameters is None else copy(parameters)
        for i in range(self.n_models):
            if self.model_types[i] == 'poly1d':
                # linking is not implemented for poly1d
                continue
            linked_index = np.arange(len(self.links[i]))[self.links[i] != -1]
            # linked_index = np.arange(self.links[i].size)[self.links[i] != -1]
            for j in linked_index:
                value = self.pars_0[i][j]
                operator = self.pars_1[i][j]
                to_imodel = self.links[i][j] - 1
                if operator == 0:
                    linked_pars[i][j] = value + linked_pars[to_imodel][j]
                elif operator == 1:
                    linked_pars[i][j] = value * linked_pars[to_imodel][j]
                else:
                    raise ValueError((
                        f"[{__name__}]: links type '{operator}' are not implemented. Try"
                        f" one of the following options: -1 (no linking), 0 (additive linking) and"
                        f" 1 (multiplicative linking)."
                    ))
        return linked_pars

    def get_tofit(self, parameters=None):
        """Return a copy of the parameters to be fitted

        If a list of parameters are given, this method will
        filter the parameters to be fitted. If no parameters
        are given, returns a filtered copy of `self.latest_fit`.

        Parameters
        ----------
        parameters : array like
            A list of the parameters to be filtered. Defaults to
            `self.latest_fit`.

        Returns
        -------
        filtered_parameters : array like
            A copy of the original parameters containing only
            parameters to be fitted.
        """

        if parameters is None:
            parameters_ = copy(self.guess)
        else:
            parameters_ = parameters

        filtered_parameters = []
        for pars, fit_mask, links in zip(parameters_, self.to_fit, self.links):
            filtered_parameters.append(pars[fit_mask & (links == -1)])
        return filtered_parameters

    # TODO: implement this method as private and run it on object initialization
    def randomize_guess(self, parameters=None):
        """Return a copy of the initial guess parameters randomly perturbed"""
        # TODO: use a gaussian perturbation such that the 99% of the
        #       probability lies within the parameter valid range
        guess_ = copy(self.guess) if parameters is None else parameters
        randomized_pars = []
        for pars, pars_0, pars_1, fit_mask, links in zip(guess_, self.pars_0, self.pars_1, self.to_fit, self.links):
            to_fit = fit_mask & (links == -1)
            pars[to_fit] = pars_0[to_fit] + np.random.rand(to_fit.sum()) * (pars_1[to_fit] - pars_0[to_fit])
            randomized_pars.append(pars)
        return randomized_pars

    def update_ranges(self, min_values=None, max_values=None):
        """ Updates the values of the parameters ranges also updating boundaries.

        Parameters
        ----------
        min_values : array like, optional
            Array updating the values of `self.pars_0`. Should have the same
            dimension of it. If None it uses `self.pars_0`.

        max_values : array like, optional
            Array updating the values of `self.pars_1`. Should have the same
            dimension of it. If None it uses `self.pars_1`.
        """
        if min_values is None:
            min_values = self.pars_0
        if max_values is None:
            max_values = self.pars_1
        # updated_pars_0 = []
        # updated_pars_1 = []
        iter_update = zip(self.pars_0, self.pars_1, self.to_fit, self.links, min_values, max_values)
        for pars_0, pars_1, fit_mask, links, new_pars_0, new_pars_1 in iter_update:
            m = fit_mask & (links == -1)
            pars_0[m] = new_pars_0[m]
            pars_1[m] = new_pars_1[m]
        #     updated_pars_0.append(pars_0)
        #     updated_pars_1.append(pars_1)
        # self.pars_0 = updated_pars_0
        # self.pars_1 = updated_pars_1
        self.boundaries = (
            self.vectorize_pars(self.pars_0),
            self.vectorize_pars(self.pars_1)
        )

    def get_updated_guess(self, new_guess, placeholder=None, inplace=False):
        """
        Returns a version of `self.guess` updated by `new_guess` fixed by masks and links.

        Parameters
        ----------
        new_guess : array like
            A list of the parameter arrays to update the current values in `self.guess`
            parameters to be fitted (e.g. `self.get_tofit(self.guess)`).

        placeholder: array like, optional
            A list of parameters like `parameters` to be used as the initial
            values to the updated parameters instead `self.guess`.
            Default value is None.

        inplace : boolean
            if True updates `self.guess` with the updated values.

        Returns
        -------
        array like :
            A version of `self.guess` updated by `new_guess` fixed by masks and links.
        """
        updated_guess = []
        _tmp = self.guess if placeholder is None else placeholder
        old_guess = copy(_tmp)
        for pars, fit_mask, links, new_pars in zip(old_guess, self.to_fit, self.links, new_guess):
            pars[fit_mask & (links == -1)] = new_pars
            updated_guess.append(pars)
        updated_guess = self._set_linkings(updated_guess)
        if inplace:
            self.guess = updated_guess
        return updated_guess

    def vectorize_pars(self, list_parameters, only_tofit=True):
        """Return a copy of the parameters squeezed into a one-dimensional array

        This method makes a copy of the given list-formatted
        parameters and turns it into an array of one dimenssion
        taking into account the fixed parameters. By default,
        only fitting parameters are returned in the vector.

        Parameters
        ----------
        list_parameters : array like
            A list containing the parameters to be vectorized.

        only_tofit : boolean
            If True (default), only parameters to be fitted are
            returned. If False, all parameters are returned.

        Returns
        -------
        vectorized_pars : array like
            A vectorized array with the parameters to be fitted (default)
            or all the parameters in the given list.
        """

        pars_ = copy(list_parameters)
        if only_tofit:
            vectorized_pars = np.hstack(self.get_tofit(parameters=pars_))
        else:
            vectorized_pars = np.hstack(pars_)

        return vectorized_pars

    def list_pars(self, vector_parameters):
        """Return a list copy of the given vectorized parameters

        Parameters
        ----------
        vector_parameters : array like
            A vectorized version of the parameters. It can either
            contain all the parameters or only those meant to be
            fitted.

        Returns
        -------
        list_pars : array like
            A list of arrays in the same format of `self.guess`.
        """

        fitting_pars_count = list(map(lambda fit_mask, links: (fit_mask & (links == -1)).sum(), self.to_fit, self.links))
        all_pars_count = list(map(lambda fit_mask, links: (fit_mask & (links == -1)).size, self.to_fit, self.links))

        list_pars = []
        if vector_parameters.size == np.sum(fitting_pars_count):
            lower_ = 0
            for l in fitting_pars_count:
                list_pars.append(vector_parameters[lower_:lower_+l])
                lower_ += l
        elif vector_parameters.size == np.sum(all_pars_count):
            lower_ = 0
            for l in all_pars_count:
                list_pars.append(vector_parameters[lower_:lower_+l])
                lower_ += l
        else:
            # TODO: make this name to be the name of the calling script
            raise ValueError((
                f"[{__name__}]: It looks like either you tampered with the parameters vector"
                f" or you're passing other vector. The expected size is either"
                f" {np.sum(fitting_pars_count)} for only fitting parameters or"
                f" {np.sum(all_pars_count)} for all the parameters."
            ))

        return list_pars

    def print(self, list_parameters=None, filename=None, verbose=1):
        """ Produces a config file using the present config.

        Parameters
        ----------
        list_parameters : array like, optional
            A list of parameters to be printed. If None, uses `self.guess`
            as the parameters list.

        filename : str, optional
            Output filename. If None, the filename is /dev/stdout.

        verbose : int, optional
            Print information. Defaults to 1.
        """
        pars = list_parameters if list_parameters is not None else self.guess
        if filename is None:
            f = sys.stdout
        else:
            f = open(filename, 'w')
        print_verbose(f'OUT_CONFIG={f.name}', verbose=verbose)
        print(f'0 {self.n_models} {self.chi_goal} {self.chi_step}', file=f)
        for i in range(self.n_models):
            print(f'{self.model_types[i]}', file=f)
            for j in range(self.guess[i].size):
            # for j in range(len(self.guess[i])):
                line = '{:7.4f} {:d} {:.4g} {:.4g} {:d}'.format(pars[i][j],
                                                                self.to_fit[i][j],
                                                                self.pars_0[i][j],
                                                                self.pars_1[i][j],
                                                                self.links[i][j])
                print(line, file=f)
                    # print(f'{cf.guess[i][j]} {int(cf.to_fit[i][j])} {cf.pars_0[i][j]} {cf.pars_1[i][j]} {int(cf.links[i][j])}', file=f)
        if filename is not None:
            f.close()

    def check_par_fit(self, model_type, i_par):
        """
        Check if a parameter needs to be fitted.

        Parameters
        ----------
        model_type : str
            The model type of the checked parameter. E.g., `eline` or `poly1d`.

        i_par : int
            The parameter position in configuration file (a.k.a. the index of the model array).

        Returns
        -------
        boolean :
            True if parameter should be fitted. False otherwise.
        """
        check = [tofit[i_par] if ((link[i_par] == -1) and (model == model_type)) else 0
                 for model, tofit, link in zip(self.model_types, self.to_fit, self.links)]
        check = np.asarray(check).astype('int').sum()
        return check.astype('bool')

def create_ConfigEmissionModel(wave_guess_list, flux_guess_list,
                               sigma_guess_list, v0_guess_list,
                               flux_boundaries=None, sigma_boundaries=None,
                               v0_boundaries=None,
                               sort_by_flux=False, config_filename=None,
                               polynomial_order=None, polynomial_coeff_guess=None,
                               polynomial_coeff_boundaries=None,
                               output_path=None, verbose=0):
    """
    Creates a ConfigEmissionModel.

    Parameters
    ----------
    wave_guess_list : array like
        Wavelengths guesses list.

    flux_guess_list : array like
        Flux guesses list.

    sigma_guess_list : array like
        Sigma guesses list.

    v0_guess_list : array like
        Velocity guesses list.

    flux_boundaries : array like, optional
        Defaults to None.

    sigma_boundaries : array like, optional
        Defaults to None.

    v0_boundaries : array like, optional
        Defaults to None.

    sort_by_flux : bool, optional
        Defaults to False

    polynomial_order : int, optional
        Defaults to None.

    polynomial_coeff_guess : array like, optional
        Defaults to None.

    polynomial_coeff_boundaries : array like, optional
        Defaults to None.

    output_path : str, optional
        Defaults to os.getcwd().

    verbose : int, optional
        Defaults to 0.

    Returns
    -------
    array like
        List of config filenames generated.

    array like
        List of wavelength intervals of the configs.

    """
    # TODO: ADD LINKS SUPPORT.
    output_path = getcwd() if output_path is None else abspath(output_path)
    i_wave = _EL_MODELS['eline']['central_wavelength']
    i_flux = _EL_MODELS['eline']['flux']
    i_sigma = _EL_MODELS['eline']['sigma']
    i_v0 = _EL_MODELS['eline']['v0']

    # assure np.array
    _wl = np.asarray(wave_guess_list)
    _fl = np.asarray(flux_guess_list)
    _sl = np.asarray(sigma_guess_list)
    _vl = np.asarray(v0_guess_list)

    _f_bounds = np.zeros((_wl.size, 2), dtype='float')
    _s_bounds = np.zeros((_wl.size, 2), dtype='float')
    _v_bounds = np.zeros((_wl.size, 2), dtype='float')

    if flux_boundaries is None:
        flux_boundaries = _f_bounds
    else:
        flux_boundaries = np.asarray(flux_boundaries) + _f_bounds

    if sigma_boundaries is None:
        sigma_boundaries = _s_bounds
    else:
        sigma_boundaries = np.asarray(sigma_boundaries) + _s_bounds

    if v0_boundaries is None:
        v0_boundaries = _v_bounds
    else:
        v0_boundaries = np.asarray(v0_boundaries) + _v_bounds

    wl_chunk = f'{int(np.floor(_wl.min()))}_{int(np.ceil(_wl.max()))}'
    if config_filename is None:
        config_filename = join(output_path, f'{wl_chunk}.conf')
    print_verbose(f'Creating config {config_filename} ...', verbose=verbose)

    cf = ConfigEmissionModel(filename=None, chi_goal=0.5, chi_step=0.1)
    # final_eml_list should be redshift corrected and preferred to be sorted by flux (from higher to lower)
    iS = np.arange(_fl.size)
    if sort_by_flux:
        iS = np.argsort(_fl, kind='stable')[::-1]
    for i, wave_guess, flux_guess, sigma_guess, v0_guess in zip(iS, _wl[iS], _fl[iS], _sl[iS], _vl[iS]):
        guess = np.zeros((__n_models_params__), dtype=np.float)
        pars_0 = np.zeros((__n_models_params__), dtype=np.float)
        pars_1 = np.zeros((__n_models_params__), dtype=np.float)
        links = -1*np.ones((__n_models_params__), dtype=np.int)
        to_fit = np.zeros((__n_models_params__), dtype=np.bool)
        guess[i_wave] = wave_guess
        guess[i_flux] = flux_guess
        guess[i_sigma] = sigma_guess
        guess[i_v0] = v0_guess

        # default values
        pars_0[i_flux] = flux_boundaries[i, 0]
        pars_1[i_flux] = flux_boundaries[i, 1]
        to_fit[i_flux] = int(not np.all(flux_boundaries[i] == flux_boundaries[i, 0]))
        pars_0[i_sigma] = sigma_boundaries[i, 0]
        pars_1[i_sigma] = sigma_boundaries[i, 1]
        to_fit[i_sigma] = int(not np.all(sigma_boundaries == sigma_boundaries[i, 0]))
        pars_0[i_v0] = v0_boundaries[i, 0]
        pars_1[i_v0] = v0_boundaries[i, 1]
        to_fit[i_v0] = int(not np.all(v0_boundaries == v0_boundaries[i, 0]))
        cf.add_model(
            model_type='eline',
            guess=guess,
            to_fit=to_fit,
            pars_0=pars_0,
            pars_1=pars_1,
            links=links
        )

    if polynomial_order is not None:
        n = polynomial_order + 1
        guess = np.zeros((__n_models_params__), dtype=np.float)
        pars_0 = np.zeros((__n_models_params__), dtype=np.float)
        pars_1 = np.zeros((__n_models_params__), dtype=np.float)
        links = -1*np.ones((__n_models_params__), dtype=np.int)
        to_fit = np.zeros((__n_models_params__), dtype=np.bool)

        _poly_coeff_guess = np.zeros(shape=(n), dtype=float)
        _poly_bounds = np.zeros(shape=(n, 2), dtype=float)

        if n == 1:
            polynomial_coeff_guess = _poly_coeff_guess + polynomial_coeff_guess
        if polynomial_coeff_boundaries is None:
            polynomial_coeff_boundaries = _poly_bounds
        else:
            polynomial_coeff_boundaries = np.asarray(polynomial_coeff_boundaries) + _poly_bounds
        for i, (coeff_guess, coeff_boundaries) in enumerate(zip(polynomial_coeff_guess, polynomial_coeff_boundaries)):
            guess[i] = coeff_guess
            pars_0[i] = coeff_boundaries[0]
            pars_1[i] = coeff_boundaries[1]
            to_fit[i] = int(not np.all(coeff_boundaries == coeff_boundaries[0]))
        cf.add_model(
            model_type='poly1d',
            guess=guess, to_fit=to_fit, pars_0=pars_0, pars_1=pars_1, links=links
        )
    cf.print(filename=config_filename, verbose=verbose)

    return config_filename, wl_chunk.replace('_', ' ')

def detect_create_ConfigEmissionModel(wave, flux,
                                      sigma_guess=2.5,
                                      redshift=0,
                                      chunks=5,
                                      flux_boundaries_fact=None,
                                      sigma_boundaries_fact=None,
                                      v0_boundaries_add=None,
                                      polynomial_order=None,
                                      polynomial_coeff_guess=None,
                                      polynomial_coeff_boundaries=None,
                                      peak_find_nsearch=1,
                                      peak_find_threshold=2,
                                      peak_find_dmin=1,
                                      plot=0,
                                      output_path=None,
                                      label=None,
                                      crossmatch_list_filename=None,
                                      crossmatch_absdmax_AA=4,
                                      crossmatch_redshift_search_boundaries=None,
                                      crossmatch_redshift_search_step=0.001,
                                      sort_by_flux=False, verbose=0):
    """
    Detect peaks from a spectrum and creates a ConfigEmissionModel file.

    Parameters
    ----------
    wave : array like
        Wavelengths array.

    flux : array like
        Flux array.

    sigma_guess : float, optional
        Sigma guess. Defaults to 2.5.

    redshift : float, optional
        Input redshift. Defaults to 0.s

    chunks : int or sequence of pairs of scalars of floats, optional
        If ``chunks`` is an int, it defines the number of divisions of the input
        spectrum to be analyzed. If ``chunks`` is a sequence of pairs of scalars,
        each pair defines a chunk interval to be analyzed. Defaults to 1.

    flux_boundaries_fact : pair of scalars, optional
        Mutiplyer factor of the flux guess to define the flux boundaries.
        Defaults to (0.001, 1000).

    sigma_boundaries_fact : pair of scalars, optional
        Mutiplyer factor of ``sigma_guess`` to define the flux boundaries.
        Defaults to (0.1, 2)

    v0_boundaries_add : pair of scalars, optional
        Additive factor of the v0 guess to define the v0 boundaries.
        Defaults to (-50, 50).

    polynomial_order : int, optional
        If it is different than None adds a polynomial model of order ``polynomial_order``
        to all generated ConfigEmissionModel. Defaults to None.

    polynomial_coeff_guess : scalar or a sequence of scalars, optional
        If is a scalar defines the guess of the coefficient guess of the polynomial
        model added to all generated ConfigEmissionModel files. If is a sequence
        defines the guesses of the ``polynomial_order`` coefficients. Should be
        of dimension ``polynomial_order``. Defaults to None.

    polynomial_coeff_boundaries : pair of scalars or sequence of pairs or scalars, optional
        Defines the boundaries of the coefficients of the polynomial model. Should
        be a pair of scalar or a sequence of pairs of scalars. Defaults to None.

    peak_find_nsearch : int, optional
        Defaults to 1.

    peak_find_threshold : float, optional
        Defaults to 2.0.

    peak_find_dmin : int, optional
        Defaults to 1.

    plot : int, optional
        Defaults to 0.

    output_path : str, optional
        Defaults to os.getcwd().

    label: str, optional
        label to prepend to config file names. Defaults to None.

    crossmatch_list_filename : str, optional
        Defaults to None.

    crossmatch_absdmax_AA : float, optional
        Defaults to 4 AA.

    crossmatch_redshift_search_boundaries : pair of scalars, optional
        Defaults to None.

    crossmatch_redshift_search_step : float, optional
        Defaults to 0.001.

    sort_by_flux : bool, optional
        Sort emission lines to be fitted decreasingly by flux at each chunk config. Defaults to False.

    verbose : int, optional
        Defaults to 0.

    Returns
    -------
    array like
        List of config filenames generated.

    array like
        List of wavelength intervals of the configs.

    array like
        List of wavelengths peaks detected.

    array like
        List of wavelengths peaks detected and corrected by ``redshift``.

    See also
    --------
    `create_ConfigEmissionModel`, `pyFIT3D.common.tools.list_eml_compare`, `pyFIT3D.common.tools.peak_finder`
    """
    def chunk_pair_n(x, y, n):
        for i in range(0, len(x), n):
            _ii = i
            _if = i + n
            if (_if >= len(x)):
                _if = len(x) - 1
            _x = x[_ii:_if]
            _y = y[_ii:_if]
            if len(_x) > 0:
                yield _x, _y

    def chunk_pair_list(x, y, chunks):
        for (l, r) in chunks:
            m = (x > l) & (x < r)
            if m.astype(int).sum() > 0:
                _x = x[m]
                _y = y[m]
                yield _x, _y

    output_path = getcwd() if output_path is None else abspath(output_path)
    flux_boundaries_fact = np.asarray([0.001, 1000]) if flux_boundaries_fact is None else np.asarray(flux_boundaries_fact)
    sigma_boundaries_fact = np.asarray([.1, 2]) if sigma_boundaries_fact is None else np.asarray(sigma_boundaries_fact)
    v0_boundaries_add = np.asarray([-50, 50]) if v0_boundaries_add is None else np.asarray(v0_boundaries_add)

    wl_chunks = []
    wl_eff_chunks = []
    config_filenames = []
    eml_masks, auto_ssp = [], []
    wave_peaks_tot, wave_peaks_tot_corr = [], []

    if isinstance(chunks, int):
        chunk_iter = chunk_pair_n(wave, flux, int(len(wave)/chunks))
    elif isinstance(chunks, list) or (isinstance(chunks, np.ndarray) and (chunks.shape[-1] == 2)):
        chunk_iter = chunk_pair_list(wave, flux, chunks)
    else:
        print('[detect_create_ConfigEmissionModel()]: unable to identify chunks')
        return None

    for i, (_wl, _fl) in enumerate(chunk_iter):
        wl_chunk = f'{int(np.floor(_wl[0]))}_{int(np.ceil(_wl[-1]))}'
        config_filename = join(output_path, (f'autodetect.{wl_chunk}.conf' if label is None else f'{label}.autodetect.{wl_chunk}.conf'))
        print_verbose(f"Analyzing chunk {wl_chunk.replace('_', ' ')} ...", verbose=verbose)

        # search for peak
        r = peak_finder(_wl, _fl, nsearch=peak_find_nsearch,
                        peak_threshold=peak_find_threshold,
                        dmin=peak_find_dmin, plot=plot, verbose=verbose)
        ind_peaks, wave_peaks, wave_hyperbolicfit_peaks, flux_peaks = r
        # redshift corrected peaks
        final_eml_list = wave_hyperbolicfit_peaks
        final_eml_list_corr = wave_hyperbolicfit_peaks/(1 + redshift)  #[x/(1 + redshift) for x in wave_hyperbolicfit_peaks]
        final_flux_peaks = flux_peaks

        # Crossmatch wave list
        crossmatch_v0_guess = None
        if (crossmatch_list_filename is not None) and (len(final_eml_list_corr) > 0):
            print_verbose('-- List crossmatch currently under develpment.', verbose=True)
            if crossmatch_redshift_search_boundaries is not None:
                zmin, zmax = crossmatch_redshift_search_boundaries
                print_verbose('-- List crossmatch: looking for best redshift...', verbose=True)
                redshift__z = np.arange(zmin, zmax, crossmatch_redshift_search_step)
                _n_match__z = []
                for z in redshift__z:
                    r = list_eml_compare(wave_hyperbolicfit_peaks, crossmatch_list_filename,
                                         redshift=z, abs_max_dist_AA=crossmatch_absdmax_AA,
                                         verbose=0)
                    _n_match__z.append(len(r[-1]))
                inmax = np.argmax(_n_match__z)
                best_redshift = redshift__z[inmax]
                print_verbose(f'-- List crossmatch: best redshift for the emission lines: {best_redshift}', verbose=True)
                redshift = best_redshift
            r = list_eml_compare(wave_hyperbolicfit_peaks, crossmatch_list_filename,
                                 redshift=redshift, abs_max_dist_AA=crossmatch_absdmax_AA,
                                 verbose=verbose)
            final_eml_list_corr, ind_list, names_eml, assoc_peaks = r
            final_eml_list = np.asarray([wave_hyperbolicfit_peaks[i] for i in ind_list])
            final_flux_peaks = np.asarray([flux_peaks[i] for i in ind_list])
            # TODO: fix v0 guess
            crossmatch_v0_guess = __c__*(final_eml_list/final_eml_list_corr - 1)

            if plot:
                if 'matplotlib.pyplot' not in sys.modules:
                    from matplotlib import pyplot as plt
                else:
                    plt = sys.modules['matplotlib.pyplot']
                plt.gcf().set_size_inches(15, 5)
                plt.cla()
                ax = plt.gca()
                ax.set_ylabel(r'Flux', fontsize=18)
                ax.set_xlabel(r'wavelength [$\AA$]', fontsize=18)
                ax.plot(_wl, _fl, '-', alpha=0.5, color='k')
                for (x, y) in zip(final_eml_list, final_flux_peaks):
                    ax.axvline(x, ls='--', lw=1, color='k')
                if plot > 1:
                    plot_filename = f'autodetect.peaks_{wl_chunk}_crossmatch.png'
                    plt.savefig(plot_filename)
                    plt.close()
                else:
                    plt.show()

        # If any peak found -> create config.
        if len(final_eml_list_corr) > 0:
            config_filenames.append(config_filename)
            left_wl = int(final_eml_list_corr[0] - 10)
            right_wl = int(final_eml_list_corr[-1] + 10)
            effective_chunk = f'{left_wl} {right_wl}'
            wl_chunks.append(wl_chunk.replace('_', ' '))
            wl_eff_chunks.append(effective_chunk)
            wave_peaks_tot += final_eml_list.tolist()
            wave_peaks_tot_corr += final_eml_list_corr.tolist()

            pars = {
                'wave_guess': [],
                'sigma_guess': [],
                'flux_guess': [],
                'v0_guess': [],
                'sigma_boundaries': [],
                'flux_boundaries': [],
                'v0_boundaries': [],
            }
            for i, (wave_peak, flux_peak) in enumerate(zip(final_eml_list_corr, final_flux_peaks)):
                flux_guess = flux_peak*sigma_guess*(2*np.pi)**0.5
                v0_guess = redshift*__c__ if crossmatch_v0_guess is None else crossmatch_v0_guess[i]
                pars['wave_guess'].append(wave_peak)
                pars['sigma_guess'].append(sigma_guess)
                pars['flux_guess'].append(flux_guess)
                pars['v0_guess'].append(v0_guess)
                pars['sigma_boundaries'].append(sigma_guess*sigma_boundaries_fact)
                pars['flux_boundaries'].append(flux_guess*flux_boundaries_fact)
                pars['v0_boundaries'].append(v0_guess+v0_boundaries_add)

            for k in pars.keys():
                pars[k] = np.asarray(pars[k])

            iS = np.arange(final_flux_peaks.size)
            if sort_by_flux:
                iS = np.argsort(final_flux_peaks, kind='stable')[::-1]

            create_ConfigEmissionModel(wave_guess_list=pars['wave_guess'],
                                       flux_guess_list=pars['flux_guess'],
                                       v0_guess_list=pars['v0_guess'],
                                       sigma_guess_list=pars['sigma_guess'],
                                       flux_boundaries=pars['flux_boundaries'],
                                       sigma_boundaries=pars['sigma_boundaries'],
                                       v0_boundaries=pars['v0_boundaries'],
                                       sort_by_flux=sort_by_flux,
                                       config_filename=config_filename,
                                       polynomial_order=polynomial_order,
                                       polynomial_coeff_guess=polynomial_coeff_guess,
                                       polynomial_coeff_boundaries=polynomial_coeff_boundaries,
                                       output_path=output_path,
                                       verbose=verbose)

    return config_filenames, wl_eff_chunks, wave_peaks_tot, wave_peaks_tot_corr

def guide_vel(config, vel, mask, fit, half_range=30):
    """
    Guide the v0 parameter of the eline model in a `ConfigEmissionModel` instance.

    Parameters
    ----------
    config : :class:`ConfigEmissionModel`
        `ConfigEmissionModel` instance.

    vel : float
        The guided velocity (v0 parameter in eline model).

    mask : int or boolean
        The input mask value of the velocity.

    fit: int or boolean
        It will set if `config.to_fit` parameters. If True or 1 it will search for the best
        v0 inside the defined range (see `half_range`). If False or 0 it will set the v0
        parameter fixed.

    half_range : float
        If fit is 1 or True, it will set the v0 parameter fit range (+/- half_range).
    """
    if vel is None:
        return config
    if isinstance(fit, bool):
        fit = int(fit)
    if isinstance(mask, bool):
        mask = int(mask)
    print('guided...')
    # print(vel, mask)
    new_guess = copy(config.guess)
    i_m = 0
    i_v0 = _MODELS_ELINE_PAR['v0']
    i_wl = _MODELS_ELINE_PAR['central_wavelength']
    new_pars_0 = copy(config.pars_0)
    new_pars_1 = copy(config.pars_1)
    upd = False
    update_iter = zip(config.model_types, config.guess, config.to_fit, config.links, config.pars_0, config.pars_1)
    for model, guess, fit_mask, links, pars_0, pars_1 in update_iter:
        # m = fit_mask & (links == -1)
        m = (links == -1)
        # if i_m: print('')
        if m[i_v0] and model == 'eline' and (mask == 1):
            new_guess[i_m][i_v0] = vel

            print(f'{new_guess[i_m][i_wl]} - guided vel = {vel}', end=' ')
            upd = True
            if fit is not None:
                config.to_fit[i_m][i_v0] = fit
                if fit == 1:
                    new_pars_0[i_m][i_v0] = vel - half_range
                    new_pars_1[i_m][i_v0] = vel + half_range
                    print(f'({new_pars_0[i_m][i_v0]:.4f}, {new_pars_1[i_m][i_v0]:.4f})', end='')
                else:
                    print(' fixed', end='')
        i_m += 1
    if upd: print('')
    config.update_ranges(max_values=new_pars_1, min_values=new_pars_0)
    config.guess = config._set_linkings(new_guess)

def guide_sigma(config, sigma, mask, fit, half_range_frac=0.2):
    """
    Guide the sigma parameter of the eline model in a `ConfigEmissionModel` instance.

    Parameters
    ----------
    config : :class:`ConfigEmissionModel`
        `ConfigEmissionModel` instance.

    sigma : float
        The guided sigma (sigma parameter in eline model).

    mask : int or boolean
        The input mask value of the sigma.

    fit : int or boolean
        It will set if `config.to_fit` parameters. If True or 1 it will search for the best
        sigma inside the defined range (see `half_range_frac`). If False or 0 it will set
        the sigma parameter fixed.

    half_range_frac : float
        If fit is 1 or True, it will set the sigma parameter fit range:
        [sigma*(1 - half_range_frac), sigma*(1 + half_range_frac)]
    """
    if sigma is None or (sigma <= 0):
        return config
    if isinstance(fit, bool):
        fit = int(fit)
    if isinstance(mask, bool):
        mask = int(mask)
    print('guided disp...')
    new_guess = copy(config.guess)
    i_m = 0
    i_sigma = _MODELS_ELINE_PAR['sigma']
    new_pars_0 = copy(config.pars_0)
    new_pars_1 = copy(config.pars_1)
    upd = False
    update_iter = zip(config.model_types, config.to_fit, config.links, config.pars_0, config.pars_1)
    for model, fit_mask, links, pars_0, pars_1 in update_iter:
        # m = fit_mask & (links == -1)
        m = (links == -1)
        # if i_m: print('')
        if m[i_sigma] and model == 'eline' and (mask == 1):
            new_guess[i_m][i_sigma] = sigma
            print(f'guided disp = {sigma}', end=' ')
            upd = True
            if fit is not None:
                config.to_fit[i_m][i_sigma] = fit
                if fit == 1:
                    new_pars_0[i_m][i_sigma] = sigma*(1 - half_range_frac)
                    new_pars_1[i_m][i_sigma] = sigma*(1 + half_range_frac)
                    print(f'({new_pars_0[i_m][i_sigma]:.4f}, {new_pars_1[i_m][i_sigma]:.4f})', end='')
                else:
                    print(' fixed', end='')
        i_m += 1
    if upd: print('')
    config.update_ranges(max_values=new_pars_1, min_values=new_pars_0)
    config.guess = config._set_linkings(new_guess)

def fit_elines_main(
    wavelength,
    flux,
    sigma_flux,
    config,
    run_mode='RND',
    n_MC=50,
    n_loops=15,
    scale_ini=0.15,
    fine_search=None,
    redefine_max=False,
    flux_max=None,
    max_factor=2,
    redefine_min=False,
    min_factor=0.012,
    plot=0,
    randomize_flux=False,
    vel_guide=None,
    vel_mask=None,
    vel_fixed=2,
    vel_guide_half_range=30,
    sigma_guide_half_range_frac=0.2,
    sigma_guide=None,
    sigma_fixed=2,
    check_stats=True,
    rnd_redshift_flux_threshold=True,
    rnd_sigma_flux_threshold=False,
    oversize_chi=True,
    update_config_frac_range=0.25,
    ):
    """
    The main function of the fit_elines script. It will run the fit of the emission models
    defined by the `config`. At the end, outputs the results to the screen.

    Parameters
    ----------
    wavelength : array like
        The observed wavelengths.

    flux : array like
        The observed flux.

    sigma_flux :
        The error in `flux`.

    config : :class:`ConfigEmissionModel`
        The :class:`ConfigEmissionModel` instance that will configure the fit.

    run_mode : str {'RND', 'LM', 'both'}, optional
        It will configure which :class:`EmissionLines` it will run.
        'RND': The RND algorithm of fit.
        'LM': Levemberg-Marquadt algorithm of fit.
        'both': It will run first the 'RND' method and inputs the `final_fit_params` of
        'RND' at the 'LM' method.
        Defaults to the RND method.

    n_MC: int, optional
        Number of Monte-Carlo iterations in the fit. Defaults to 50.

    n_loops : int, optional
        Number of loops of the fit. Defaults to 15.

    scale_ini : float, optional
        The scale_ini parameter of fit. Defaults to 0.15

    fine_search : boolean, optional
        The fine_search parameter of RND method. Defaults to
        `__ELRND_fine_search_option__` defined in pyFIT3D.common.constants.

    redefine_max : int, boolean, optional
        The redefine_max parameter of the fit. Defaults to 0.

    flux_max : float, optional
        If redefine_max is True or 1 it will set force the flux_max on redefine_max.

    max_factor : float, optional
        Set up the max flux factor when redefine_max is on. Defaults to 2.

    redefine_min : boolean, optional
        If redefine_max is on, it will also redefine the min flux. Defaults
        to False.

    min_factor : float, optional
        Set up the min flux factor when redefine_max is on. Defaults to 0.012.

    plot : int, optional
        If 1 it will plot fit. If 2 it will plot to a file. None it will be set as 0, i.e.
        do not plot.

    randomize_flux : boolean, optional.
        If true it will randomize the input flux of RND method::

            flux_fit = flux + rand()*sqrt(sigma_flux)

        if None it defaults to False.

    vel_guide : float, optional
        Sets up the guide velocity of the fit run. (see: `guide_vel`).

    vel_mask : int, boolean.
        Sets up the mask of the guide velocity of the fit run. (see: `guide_vel`).

    vel_fixed : int, boolean, optional
        Sets up the to_fit parameter of the guide velocity of the fit run. (see: `guide_vel`).

    vel_guide_half_range : float, optional
        Sets up the range of the guided velocity of the fit run. (see: `guide_vel`).

    sigma_guide : float, optional
        Sets up the guide sigma of the fit run. (see: `guide_sigma`).

    sigma_fixed : int, optional
        Sets up the to_fit parameter of the guide sigma of the fit run. (see: `guide_sigma`).

    sigma_guide_half_range_frac : float, optional
        Sets up the range of the guided sigma of the fit run. (see: `guide_sigma`).

    check_stats : bool, optional
        Sets up the stats check in EmissionLinesRND fit. Defaults to True.

    rnd_redshift_flux_threshold : bool, optional
        Keep flux inside limits during redshift search in RND mode. Defaults to True.

    rnd_sigma_flux_threshold : bool, optional
        Keep flux inside limits during sigma search in RND mode. Defaults to False.

    oversize_chi : bool, optional
        if True, will triplify the first chi of each MC round in RND mode. Defaults to True.

    update_config_frac_range : float, optional
        Configures the range of the free parameters in when `run_mode` = 'BOTH'.
        Defaults to 0.25.

    Returns
    -------
    :class:EmissionLines class like
        if run_mode is 'RND' it will return :class:`EmissionLinesRND` instance.
        if run_mode is 'LM' it will return :class:`EmissionLinesLM` instance.
        if run_mode is 'BOTH' it will return :class:`EmissionLinesRND` or
        :class:`EmissionLinesLM` instance depending if the
        :class:`EmissionLinesLM` run find a best fit.
    """
    n_MC = 50 if n_MC is None else n_MC
    n_loops = 15 if n_loops is None else n_loops
    scale_ini = 0.15 if scale_ini is None else scale_ini
    fine_search = __ELRND_fine_search_option__ if fine_search is None else fine_search
    redefine_max = False if redefine_max is None else bool(redefine_max)
    max_factor = 2 if max_factor is None else max_factor
    redefine_min = False if redefine_min is None else bool(redefine_min)
    min_factor = 0.012 if min_factor is None else min_factor
    plot = 0 if plot is None else plot
    vel_fixed = 2 if vel_fixed is None else vel_fixed
    sigma_fixed = 2 if sigma_fixed is None else sigma_fixed
    _runs = ['RND', 'LM', 'BOTH']
    run_mode = 'RND' if run_mode is None else run_mode.upper()
    if run_mode not in _runs:
        run_mode = 'RND'

    if run_mode == 'RND' or run_mode == 'BOTH':
        EL = EmissionLinesRND(wavelength=wavelength, flux=flux, sigma_flux=sigma_flux,
                              config=config, n_MC=n_MC, plot=plot, n_loops=n_loops,
                              scale_ini=scale_ini, fine_search=fine_search)
        guide_vel(EL.config, vel_guide, vel_mask, vel_fixed, vel_guide_half_range)
        guide_sigma(EL.config, sigma_guide, vel_mask, sigma_fixed, sigma_guide_half_range_frac)
        EL.latest_fit = copy(EL.config.guess)
        if redefine_max:
            EL.redefine_max_flux(flux_max=flux_max, max_factor=max_factor,
                                 redefine_min=redefine_min, min_factor=min_factor)
        EL.fit(vel_fixed=vel_fixed, sigma_fixed=sigma_fixed, randomize_flux=randomize_flux,
               check_stats=check_stats, oversize_chi=oversize_chi,
               redshift_flux_threshold=rnd_redshift_flux_threshold,
               sigma_flux_threshold=rnd_sigma_flux_threshold)

    if (run_mode == 'LM') or (run_mode == 'BOTH'):
        if run_mode == 'BOTH':
            RND_last_chisq = EL.final_chi_sq
            _EL = copy(EL)
            _EL.update_config(EL.final_fit_params_mean, update_ranges=True,
                              frac_range=update_config_frac_range)
            _conf = _EL.config
        else:
            RND_last_chisq = 1e12
            _conf = config
        _EL = EmissionLinesLM(wavelength=wavelength, flux=flux, sigma_flux=sigma_flux,
                              config=_conf, n_MC=n_MC, plot=plot, n_loops=n_loops,
                              scale_ini=scale_ini)
        if run_mode == 'LM':
            guide_vel(_EL.config, vel_guide, vel_mask, vel_fixed, vel_guide_half_range)
            guide_sigma(_EL.config, sigma_guide, vel_mask, sigma_fixed, sigma_guide_half_range_frac)
            _EL.latest_fit = copy(_EL.config.guess)
            if redefine_max:
                _EL.redefine_max_flux(flux_max=flux_max, max_factor=max_factor,
                                      redefine_min=redefine_min, min_factor=min_factor)
        _EL.fit()

        # This IF makes that a bad fit do not replaces a good fit
        if run_mode == 'BOTH':
            # print(f'EL chi_sq - BOTH[RND+LM]:{_EL.final_chi_sq}, RND:{RND_last_chisq}')
            if (len(_EL.best_fits_loc) > 0):  # and (_EL.final_chi_sq < RND_last_chisq):
                # errors are estimated in RND round
                _EL.final_fit_params_std = EL.final_fit_params_std
                EL = _EL
        if run_mode == 'LM':
            EL = _EL
    EL.output_to_screen()
    return EL

def fit_elines(
    spec_file,
    config_file,
    w_min,
    w_max,
    out_file=None,
    mask_file=None,
    run_mode='RND',
    n_MC=50,
    n_loops=15,
    scale_ini=0.15,
    fine_search=None,
    redefine_max=False,
    plot=0,
    out_mod_res_final=None,
    seed=None,
    check_stats=True,
    ):
    """
    The fit_elines script (i.e. a well prepared wrap of `fit_elines_main`).

    It will run the `run_mode` fit the the emission models defined by the `config_file`
    on the spectrum in `spec_file` masked by `mask_file`, inside the define wavelength
    range (`w_min`, `w_max`).

    If `run_mode` is::

        'RND': mimics fit_elines_rnd script.
        'LM': mimics fit_elines_LM script.
        'both': It will run the 'RND' first and inputs the results of the fit in
            the 'LM' fit.

    Parameters
    ----------
    spec_file : str
        The filename of the file containing the wavelengths, observed flux and optionally the
        error in observed flux.

    config_file : str
        The filename of the :class:`ConfigEmissionModel` input file.

    w_min : int
        The minimun (bluer) wavelength which defines the fitted wavelength range.

    w_max : int
        The maximum (redder) wavelength which defines the fitted wavelength range.

    out_file : str, optional
        The name of the output results of the fit. If None it will be
        defined by basename(sys.argv[0]).

    mask_file : str, optional
        If mask_file is a valid file it will mask the wavelengths.

    run_mode : str {'RND', 'LM', 'both'}, optional
        It will configure which :class:`EmissionLines` it will run.
        'RND': The RND algorithm of fit.
        'LM': Levemberg-Marquadt algorithm of fit.
        'both': It will run first the 'RND' method and inputs the `final_fit_params` of
        'RND' at the 'LM' method.
        Defaults to the RND method.

    n_MC : int, optional
        Number of Monte-Carlo iterations in the fit. Defaults to 50.

    n_loops : int, optional
        Number of loops of the fit. Defaults to 15.

    scale_ini : float, optional
        The scale_ini parameter of fit. Defaults to 0.15

    fine_search : boolean, optional
        The fine_search parameter of RND method. Defaults to
        `__ELRND_fine_search_option__` defined in `pyFIT3D.common.constants`.

    redefine_max : int or boolean, optional
        The redefine_max parameter of the fit. Defaults to 0.

    plot : int, optional
        If 1 it will plot fit. If 2 it will plot to a file. None it will be set as 0, i.e.
        do not plot.

    out_mod_res_final : str, optional
        The filename of output the result spectra of the fit. Defaults to
        out.fit_spectra.

    seed : int, optional
        It will define the input seed. Defaults to the ``int(time.time())``.

    check_stats : bool, optional
        Sets up the stats check in EmissionLinesRND fit. Defaults to True.

    See also
    --------
    :class:`ConfigEmissionModel` and `fit_elines_main`.
    """
    out_file = basename(sys.argv[0]).replace('.py', '.out') if out_file is None else out_file
    out_mod_res_final = 'out.fit_spectra' if out_mod_res_final is None else out_mod_res_final
    seed = print_time() if seed is None else print_time(time_ini=seed)
    np.random.seed(seed)

    time_ini_run = print_time(print_seed=False, get_time_only=True)

    # read masks file
    masks, n_masks = read_masks_file(mask_file)
    print(f'{n_masks} regions to mask')
    # read config FOR TESTING PURPOSES -------------------------------------------------------------
    cf = ConfigEmissionModel(config_file)
    # read seds
    wave__w, flux__w, eflux__w = read_spectra(spec_file, f_error=lambda f: 0.2*np.sqrt(np.abs(f)),
                                              variance_column=False)

    st_eflux = pdl_stats(eflux__w)
    eflux__w = 0.5*(eflux__w + st_eflux[_STATS_POS['median']])

    # trim & apply masks
    sel_wl_range = trim_waves(wave__w, [w_min, w_max])
    sel_wl_goods = sel_waves(masks, wave__w)
    sel_wl = sel_wl_range & sel_wl_goods
    wave_msk__w = wave__w[sel_wl]
    flux_msk__w = flux__w[sel_wl]
    eflux_msk__w = eflux__w[sel_wl]

    # output the valid input part of the spectra
    output_spectra(wave_msk__w, [flux_msk__w, eflux_msk__w], 'fit_spectra.input')
    # eflux__w = 0.5*(eflux_msk__w + np.median(eflux_msk__w))

    EL = fit_elines_main(wavelength=wave_msk__w, flux=flux_msk__w, sigma_flux=eflux_msk__w,
                        config=cf, n_MC=n_MC, n_loops=n_loops, scale_ini=scale_ini,
                        fine_search=fine_search, redefine_max=redefine_max,
                        max_factor=1.2, redefine_min=False,
                        plot=plot, run_mode=run_mode, check_stats=check_stats)

    EL.output(filename_spectra=out_mod_res_final,
              filename_output=out_file,
              filename_config='out_config.fit_spectra')

    time_end = print_time(print_seed=False)
    time_total = time_end - time_ini_run
    print(f'# SECONDS = {time_total}')

def load_spec(filename, wave_axis=3, error_filename=None, error_extension=0):
    """
    Loads the rss or the cube with an entire galaxy.

    Parameters
    ----------
    filename : str
        The filename of the fits file with the cube or row-stacked spectra.

    wave_axis : int
        The axis of the wavelengths in the FITS file. It defaults to 3.

    error_filename : str, optional
        The filename of the fits file with the errors in observed flux.

    error_extension : int, optional
        The number of the extesion where the error is in
        ``error_filename`` FITS file.

    Returns
    -------
    tuple of array like
        array like
            The wavelengths.

        array like
            The flux

        :class:`astropy.io.fits.header.Header` instance
            The header of the cube or RSS fits file.

        array like:
            The error of the flux.
    """
    # if f_error is None:
    #     f_error = lambda x: 0.2*np.sqrt(np.abs(x))
    flux, h = get_data_from_fits(filename, header=True)
    wave = get_wave_from_header(h, wave_axis=wave_axis)
    eflux = None
    if error_filename is not None:
        eflux = get_data_from_fits(error_filename, extension=error_extension)

    return wave, flux, h, eflux

def create_emission_lines_parameters(config, shape):
    """
    Creates the output maps and associated metadata for a cube or RSS
    run of `fit_elines_main`. Used in `kin_rss_elines_main` and
    `kin_cube_elines_main`.

    Parameters
    ----------
    config : :class:`ConfigEmissionModel`
        Configuration file of the Emission Line system.

    shape : int or tuple
        The shape of the input spectra (RSS or cube). If the run is over a RSS
        `shape` should be an integer ``(i_spec)``. In the case of a cube, should be a
        tuple ``(iy, ix)``.

    Returns
    -------
    output_models : dict
        The keys depends on the models parameters present in the config
        (`_EL_MODELS` dict is defined in `pyFIT3D.common.constants`). Each parameter
        has a e_par key also.

        Example:
            A config has 2 eline models and 1 poly1d model:

            `output_models` it will have keys ``_EL_MODELS['eline'].keys()`` and
            ``_EL_MODELS['poly1d'].keys()`` with their respective errors hashed as
            ``e_{key}``. The output_models parameters from eline models will have
            shape `(2, shape)`. The poly1d parameters have `(1, shape)`.

            E.g.:
                ``output_models['v0'][0]`` is the map of the v0 parameter of the first eline
                model.

    See also
    --------
    `kin_rss_elines_main`, `kin_cube_elines_main`, `append_emission_lines_parameters`
    """
    def flatten(T):
        if not isinstance(T, tuple): return (T,)
        elif len(T) == 0: return ()
        else: return flatten(T[0]) + flatten(T[1:])
    output_models = {}
    # for k, v in config._EL_MODELS.items():
    for k, v in _EL_MODELS.items():
        models_index = np.arange(config.n_models)[np.array(config.model_types) == k]
        n = len(models_index)
        for k2 in v.keys():
            _shape = tuple([n] + list(flatten(shape)))
            output_models[k2] = np.zeros(_shape, dtype='float')
            output_models[f'e{k2}'] = np.zeros(_shape, dtype='float')
    return output_models

def append_emission_lines_parameters(EL, output_models, current_spaxel):
    """
    Appends the result parameters of a fit to the output_models structure.

    Parameters
    ----------
    EL : EmissionLines class
        The instance of the `EmissionLines` class with the fit results.

    output_models : dict
        The dict outputted by `create_emission_lines_parameters`.

    curret_spaxel : int or tuple
        If int it will be consider as the spectra id of a RSS run (see
        kin_rss_elines_main). If a tuple, it will be consider as (ix, iy),
        the coordinates of the fitted spectrum in the cube run (see
        kin_cube_elines_main).

    See also
    --------
    `kin_rss_elines_main`, `kin_cube_elines_main`, `create_emission_lines_parameters`
    """
    # config, wavelength, pars, std_pars, current_spaxel, output_models):
    config = EL.config
    wavelength = EL.wavelength
    pars = EL.final_fit_params_mean
    std_pars = EL.final_fit_params_std
    if isinstance(current_spaxel, tuple):
        # if current spaxel is a tuple, it comes in the shape (ix, iy)
        # and we need (iy, ix) in order to get the right coordinates of
        # the desired spaxel.
        current_spaxel = (current_spaxel[1], current_spaxel[0])
    i_m_out = {'eline': 0, 'poly1d': 0}
    # for model, _MODELS_PAR in _EL_MODELS.items():
    for model, _MODELS_PAR in _EL_MODELS.items():
        models_index = np.arange(config.n_models)[np.array(config.model_types) == model]
        for i_m in models_index:
            _i = i_m_out[model]
            # Different from other models, poly1d needs to be integrated.
            # TODO:
            # We should design an output FITS for the continuum coefficients.
            # The present version only saves the integrated continuum in this
            # structure.
            if (model == 'poly1d'):
                sum, esum = 0, 0
                k = 'cont'
                ek = f'e{k}'
                for i_p, (par, sigma_par) in enumerate(zip(pars[i_m], std_pars[i_m])):
                    _int_mod = np.array([l**i_p for l in wavelength]).sum()
                    sum += par*_int_mod
                    esum += sigma_par*_int_mod
                output_models[k][_i][current_spaxel] = sum
                output_models[ek][_i][current_spaxel] = esum
            else:
                for k, i_p in _MODELS_PAR.items():
                    ek = f'e{k}'
                    med = pars[i_m][i_p]
                    stddev = std_pars[i_m][i_p]
                    output_models[k][_i][current_spaxel] = med
                    output_models[ek][_i][current_spaxel] = stddev
            i_m_out[model] += 1

def output_emission_lines_parameters(prefix, config, output_models):
    """
    Outputs the maps of parameters fit by `kin_cube_elines_main` or
    `kin_rss_elines_main`.

    Parameters
    ----------
    prefix : str
        The output prefix of the map files.

    config : :class:`ConfigEmissionModel`
        Config used in the run.

    output_models : dict
        The dictionary created by `create_emission_lines_parameters`.

    See also
    --------
    `kin_cube_elines_main`, `kin_rss_elines_main`, `create_emission_lines_parameters`,
    `append_emission_lines_parameters`, `fit_elines_main`
    """
    cf = config
    names_parse = {'central_wavelength': 'wave', 'v0': 'vel'}
    for i_m in range(len(cf.model_types)):
        model = cf.model_types[i_m]
        _MODELS_PAR = _EL_MODELS[model]
        for k, i_p in _MODELS_PAR.items():
            ek = f'e{k}'
            k_name = names_parse.get(k, k)
            ek_name = f'e{k_name}'
            if model == 'poly1d':
                i = 0
                filename = f'{prefix}_{k_name}.fits'
                efilename = f'{prefix}_{ek_name}.fits'
            else:
                i = i_m
                filename = f'{prefix}_{k_name}_{i:02d}.fits'
                efilename = f'{prefix}_{ek_name}_{i:02d}.fits'
            array_to_fits(filename, output_models[k][i], overwrite=True)
            array_to_fits(efilename, output_models[ek][i], overwrite=True)
            if i_p == _MODELS_ELINE_PAR['sigma']:
                k_name = 'disp'
                ek_name = f'e{k_name}'
                filename = f'{prefix}_{k_name}_{i:02d}.fits'
                efilename = f'{prefix}_{ek_name}_{i:02d}.fits'
                array_to_fits(filename, __sigma_to_FWHM__*output_models[k][i], overwrite=True)
                array_to_fits(efilename, __sigma_to_FWHM__*output_models[ek][i], overwrite=True)

def output_emission_lines_spectra(wave, spectra, header, prefix='kin_back_cube', wave_axis=3):
    """
    Outputs the spectra from the fit by `kin_cube_elines_main` or
    `kin_rss_elines_main` to 3 FITS files (flux, model flux and residuals).

    Parameters
    ----------
    wave : array like
        Wavelengths.

    spectra : 3 elements list
        Observed flux, modeled flux and residual flux of the fit.

    header : dict or :class:`astropy.io.fits.header.Header`
        Header to be recorded in output FITS file.

    prefix : str, optional
        The output prefix of the spectra files.

    wave_axis : int, optional
        The wavelength axis which will be recorded in FITS files.

    See also
    --------
    `kin_cube_elines_main`, `kin_rss_elines_main`, `fit_elines_main`
    """
    flux, model, res = spectra
    output_org_filename = f'{prefix}_org.fits'
    output_mod_filename = f'{prefix}_mod.fits'
    output_res_filename = f'{prefix}_res.fits'
    array_to_fits(output_org_filename, flux, header=header, overwrite=True)
    array_to_fits(output_mod_filename, model, header=header, overwrite=True)
    array_to_fits(output_res_filename, res, header=header, overwrite=True)
    for f in [output_org_filename, output_mod_filename, output_res_filename]:
        h_set = {
            f'CRPIX{wave_axis}': 1,
            f'CRVAL{wave_axis}': wave[0],
            f'CDELT{wave_axis}': wave[1] - wave[0],
            f'NAXIS{wave_axis}': wave.size,
        }
        write_img_header(f, list(h_set.keys()), list(h_set.values()))

def kin_cube_elines_main(
    wavelength,
    cube_flux,
    config,
    out_file=None,
    cube_eflux=None,
    run_mode='RND',
    vel_map=None,
    vel_mask_map=None,
    vel_fixed=2,
    sigma_map=None,
    sigma_fixed=2,
    memo=False,
    n_MC=50,
    n_loops=15,
    scale_ini=0.15,
    redefine_max=True,
    max_factor=2,
    redefine_min=True,
    min_factor=0.012,
    plot=0,
    oversize_chi=False,
    ):
    """
    The main function of the kin_cube_elines script. It will run the `fit_elines_main`
    over all observed spectra `cube_flux`, with shape ``(NW, NY, NX)``.
    All spectra has to be sampled over the same `wavelength` with shape ``(NW)``.

    Parameters
    ----------
    wavelength : array
        The observed wavelengths.

    cube_flux : array like
        The observed flux spectra with shape ``(NW, NY, NX)``.

    config : :class:`ConfigEmissionModel`
        The :class:`ConfigEmissionModel` instance that will configure the fit.

    out_file : str, optional
        The filename of the file where the output of :class:`EmissionLines` result will be recorded.

    cube_eflux : array like, optional
        The error in `flux` with shape ``(NW, NY, NX)``.

    run_mode : 'RND', 'LM', 'both', optional
        It will configure which :class:`EmissionLines` it will run.
        'RND': The RND algorithm of fit.
        'LM': Levemberg-Marquadt algorithm of fit.
        'both': It will run first the 'RND' method and inputs the `final_fit_params` of 'RND' at the 'LM' method.
        Defaults to RND method.

    vel_map : array like, optional
        Sets up the guide velocity map. (see: `guide_vel`).

    vel_mask_map : array like, optional
        Sets up the mask map of the guide velocity. (see: `guide_vel`).

    vel_fixed : int or boolean, optional
        Sets up the to_fit parameter of the guide velocity. (see: `guide_vel`).

    sigma_map : array like, optional
        Sets up the guide sigma map. (see: `guide_sigma`).

    sigma_fixed : int or boolean
        Sets up the to_fit parameter of the guide sigma map. (see: `guide_sigma`).

    memo : boolean, optional
        While fitting the cube, memorizes the last result and inputs to the next spectrum
        in the loop.

    n_MC : int, optional
        Number of Monte-Carlo iterations in the fit. Defaults to 50.

    n_loops : int, optional
        Number of loops of the fit. Defaults to 15.

    scale_ini : float, optional
        The scale_ini parameter of fit. Defaults to 0.15

    redefine_max : int or boolean, optional
        The redefine_max parameter of the fit. Defaults to True.

    max_factor : float, optional
        Set up the max flux factor when redefine_max is on. Defaults to 2.

    redefine_min : boolean, optional
        If redefine_max is on, it will also redefine the min flux. Defaults
        to True.

    min_factor : float, optional
        Set up the min flux factor when redefine_max is on. Defaults to 0.012.

    plot : int, optional
        If 1 it will plot fit. If 2 it will plot to a file. None it will be set as 0, i.e.
        do not plot.

    oversize_chi : bool, optional
        if True, will triplify the first chi of each MC round in RND mode. Defaults to False.

    Returns
    -------
    tuple
        Constructed as::

            list of array like
                Observed spectra, model spectra and residual spectra, all with same shape as `cube_flux`.

            dict
                Output maps of fitted parameters
                (see: `create_emission_lines_parameters`, `append_emission_lines_parameters`
                and `output_emission_lines_parameters`).

    """
    nw, ny, nx = cube_flux.shape
    out_file = basename(sys.argv[0]).replace('.py', '.out') if out_file is None else out_file

    # create models output
    output_el_models = create_emission_lines_parameters(config, shape=(ny, nx))
    # create spectra output
    org__wyx = np.zeros_like(cube_flux)
    mod__wyx = np.zeros_like(cube_flux)
    res__wyx = np.zeros_like(cube_flux)

    for ixy in itertools.product(range(nx), range(ny)):
        ix, iy = ixy
        print(f'# ID {ix}/{nx - 1},{iy}/{ny - 1}')
        current_spaxel = ixy
        current_guide_vel = None if vel_map is None else vel_map[iy, ix]
        current_guide_vel_mask = None if vel_mask_map is None else vel_mask_map[iy, ix]
        current_guide_sigma = None if sigma_map is None else sigma_map[iy, ix]
        flux = cube_flux[:, iy, ix]
        # setbadtoval(0)
        flux[~np.isfinite(flux)] = 0
        st_f = pdl_stats(flux)
        st_f_median = np.abs(st_f[_STATS_POS['median']])
        st_f_sigma = np.abs(st_f[_STATS_POS['pRMS']])
        if cube_eflux is not None:
            eflux = cube_eflux[:, iy, ix]
            eflux[~np.isfinite(eflux)] = 0
        else:
            eflux = np.ones_like(flux)*0.1
            if (st_f_median != 0) and (st_f_sigma != 0):
                eflux *= (st_f_median + st_f_sigma**2)
            # setvaltobad(0)
            eflux[eflux == 0] = np.nan
            if st_f_median != 0:
                eflux[np.isnan(eflux)] = st_f_median
            else:
                eflux[np.isnan(eflux)] = 0.001
        # enter if all flux is zero or if is in the first spaxel
        if not ((ix or iy) and flux.any()):
            cf = copy(config)
        else:
            last_chi_sq, _ = calc_chi_sq(EL.flux, EL.model, EL.sigma_flux,
                                         EL.config.n_models + 1)
            # Memorize last result to input the next spectrum
            if memo and (last_chi_sq < 2*EL.config.chi_goal):
                print('update')
                memo_cf = EL.update_config()
            else:
                cf = copy(config)
        flux_max = np.abs(st_f[_STATS_POS['max']] - st_f[_STATS_POS['median']])
        EL = fit_elines_main(wavelength, flux, eflux, cf,
                             vel_guide=current_guide_vel, sigma_guide=current_guide_sigma,
                             vel_fixed=vel_fixed, sigma_fixed=sigma_fixed,
                             vel_mask=current_guide_vel_mask,
                             n_MC=n_MC, plot=plot, n_loops=n_loops,
                             scale_ini=scale_ini, fine_search=True,
                             redefine_max=redefine_max, redefine_min=redefine_min,
                             max_factor=max_factor, min_factor=min_factor, flux_max=flux_max,
                             run_mode=run_mode, randomize_flux=True, oversize_chi=oversize_chi)
        EL.output(filename_output=out_file, append_output=True, spec_id=ixy)
        org__wyx[:, iy, ix] = EL.flux
        mod__wyx[:, iy, ix] = EL.model
        res__wyx[:, iy, ix] = EL.flux - EL.model
        append_emission_lines_parameters(EL, output_el_models, ixy)
        # ixy, output_el_models)
    output_el_spectra = np.array([org__wyx, mod__wyx, res__wyx])
    return output_el_spectra, output_el_models

def kin_cube_elines(
    spec_file,
    config_file,
    w_min,
    w_max,
    out_file=None,
    mask_file=None,
    error_file=None,
    prefix=None,
    run_mode='RND',
    memo=False,
    vel_map_file=None,
    vel_mask_file=None,
    vel_fixed=2,
    sigma_map_file=None,
    sigma_fixed=2,
    sigma_val=None,
    n_MC=50,
    n_loops=15,
    scale_ini=0.15,
    redefine_max=True,
    max_factor=2,
    redefine_min=False,
    min_factor=0.012,
    plot=0,
    seed=None,
    ):
    """
    The kin_cube_elines script (i.e. a well prepared wrap of `kin_cube_elines_main`).

    It will run the `run_mode` fit the the emission models defined by the `config_file`
    on the cube of spectra in `spec_file` masked by `mask_file`, inside the define wavelength
    range (`w_min`, `w_max`).

    If `run_mode` is::

        'RND': mimics fit_elines_rnd script.
        'LM': mimics fit_elines_LM script.
        'both': It will run the 'RND' first and inputs the results of the fit in the 'LM' fit.


    Parameters
    ----------
    spec_file : str
        The filename of the FITS file containing the wavelengths and observed flux spectra.

    config_file : str
        The filename of the :class:`ConfigEmissionModel` input file.

    w_min : int
        The minimun (bluer) wavelength which defines the fitted wavelength range.

    w_max : int
        The maximum (redder) wavelength which defines the fitted wavelength range.

    out_file : str, optional
        The name of the output results of the fit. Defaults to
        ``basename(sys.argv[0]).replace('py', 'out')``.

    mask_file : str, optional
        If mask_file is a valid file it will mask the wavelengths.

    error_file : str
        The filename of the FITS file containing the errors in observed flux spectra.

    prefix : str, optional
        It will define the prefix used in `output_emission_lines_parameters` and
        `output_emission_lines_spectra`. Defaults to basename of `out_file`.

    run_mode : str {'RND', 'LM', 'both'}, optional
        It will configure which :class:`EmissionLines` it will run.
        'RND': The RND algorithm of fit.
        'LM': Levemberg-Marquadt algorithm of fit.
        'both': It will run first the 'RND' method and inputs the `final_fit_params` of
        'RND' at the 'LM' method.
        Defaults to the RND method.

    memo : boolean, optional
        While fitting the cube, memorizes the last result and inputs to the next spectrum
        in the loop.

    vel_map_file : str, optional
        The filename of the guide velocity map. (see: `guide_vel`).

    vel_mask_file : array like, optional
        The filename of the mask map of the guide velocity. (see: `guide_vel`).

    vel_fixed : int or boolean, optional
        Sets up the to_fit parameter of the guide velocity. (see: `guide_vel`).

    sigma_map_file : array like, optional
        The filename of the guide sigma map. (see: `guide_sigma`).

    sigma_fixed : int or boolean
        Sets up the to_fit parameter of the guide sigma map. (see: `guide_sigma`).

    sigma_val : float, optional
        Rewrites the sigma_map using a single value for the guide sigma map. (see: `guide_sigma`)

    n_MC : int, optional
        Number of Monte-Carlo iterations in the fit. Defaults to 50.

    n_loops : int, optional
        Number of loops of the fit. Defaults to 15.

    scale_ini : float, optional
        The scale_ini parameter of fit. Defaults to 0.15

    redefine_max : int or boolean, optional
        The redefine_max parameter of the fit. Defaults to True.

    max_factor : float, optional
        Set up the max flux factor when redefine_max is on. Defaults to 2.

    redefine_min : boolean, optional
        If redefine_max is on, it will also redefine the min flux. Defaults
        to False.

    min_factor : float, optional
        Set up the min flux factor when redefine_max is on. Defaults to 0.012.

    plot : int, optional
        If 1 it will plot fit. If 2 it will plot to a file. None it will be set as 0, i.e.
        do not plot.

    seed : int, optional
        It will define the input seed. Defaults to the ``int(time.time())``.


    See also
    --------
    :class:`ConfigEmissionModel`, `fit_elines_main`, `kin_cube_elines_main`, `output_emission_lines_parameters`, `output_emission_lines_spectra`

    """
    seed = print_time() if seed is None else print_time(time_ini=seed)
    np.random.seed(seed)
    time_ini_run = print_time(print_seed=False, get_time_only=True)
    memo = False if memo is None else bool(memo)
    vel_fixed = 2 if vel_fixed is None else vel_fixed
    sigma_fixed = 2 if sigma_fixed is None else sigma_fixed
    out_file = basename(sys.argv[0]).replace('.py', '.out') if out_file is None else out_file
    if prefix is None:
        _s = out_file.split('.')
        prefix = '.'.join(_s[0:-1]) if len(_s) > 1 else f'{out_file}'
    remove_isfile(out_file)
    _runs = ['RND', 'LM', 'both']
    if (run_mode is None) or (run_mode not in _runs):
        run_mode = 'RND'
    if vel_map_file is None or not isfile(vel_map_file):
        vel_fixed = 2
    if sigma_map_file is None or not isfile(sigma_map_file):
        sigma_fixed = 2
    # read masks file
    masks, n_masks = read_masks_file(mask_file)
    print(f'{n_masks} regions to mask')
    error_extension = 0
    if error_file is not None:
        _tmp = error_file.split('[')
        if len(_tmp) > 1:
            try:
                error_file = _tmp[0]
                error_extension = int(_tmp[-1].split(']')[0])
            except:
                print(f'kin_cube_elines: error extension not found, using {error_extension}')
    wave__w, flux__wyx, header, eflux__wyx = load_spec(filename=spec_file,
                                                       error_filename=error_file,
                                                       error_extension=error_extension)

    nW, ny, nx = flux__wyx.shape
    sel_wl_range = trim_waves(wave__w, [w_min, w_max])
    sel_wl_goods = sel_waves(masks, wave__w)
    sel_wl = sel_wl_range & sel_wl_goods
    org_cf = ConfigEmissionModel(config_file)

    # EL: 2020-09-04: bugfix - If there is no avaible spectral information for the fit
    # This 0 could be changed to another value for a better fit.
    if sel_wl.astype('int').sum() < 2:
        print('[kin_cube_elines]: n_wavelength < 2: No avaible spectra to perform the configured analysis.')
        output_el_models = create_emission_lines_parameters(org_cf, shape=(ny, nx))
    else:
        wave_msk__w = wave__w[sel_wl]
        flux_msk__wyx = flux__wyx[sel_wl]
        if eflux__wyx is not None:
            eflux_msk__wyx = eflux__wyx[sel_wl]
        else:
            eflux_msk__wyx = None
        nw, ny, nx = flux_msk__wyx.shape
        # read config FOR TESTING PURPOSES -------------------------------------------------------------
        vel_map__yx = None
        if (vel_map_file is not None) and (os.path.isfile(vel_map_file)):
            vel_map__yx = get_data_from_fits(vel_map_file)
        vel_mask_map__yx = None
        if (vel_mask_file is not None) and (os.path.isfile(vel_mask_file)):
            vel_mask_map__yx = get_data_from_fits(vel_mask_file)
        sigma_map__yx = None
        if (sigma_map_file is not None) and (os.path.isfile(sigma_map_file)):
            sigma_map__yx = get_data_from_fits(sigma_map_file)*__FWHM_to_sigma__
        if sigma_val is not None:
            sigma_map__yx = np.ones((ny, nx), dtype='float')*sigma_val
        r = kin_cube_elines_main(
            wave_msk__w, flux_msk__wyx, org_cf, out_file,
            cube_eflux=eflux_msk__wyx,
            vel_map=vel_map__yx, vel_mask_map=vel_mask_map__yx, vel_fixed=vel_fixed,
            sigma_map=sigma_map__yx, sigma_fixed=sigma_fixed,
            memo=memo, n_MC=n_MC, n_loops=n_loops, redefine_max=redefine_max,
            scale_ini=scale_ini, run_mode=run_mode,
            plot=plot,
        )
        output_el_spectra, output_el_models = r
        output_emission_lines_spectra(wave_msk__w, output_el_spectra, header, prefix)
    output_emission_lines_parameters(prefix, org_cf, output_el_models)
    time_end = print_time(print_seed=False)
    time_total = time_end - time_ini_run
    print(f'# SECONDS = {time_total}')

def kin_rss_elines_main(
    wavelength,
    rss_flux,
    config,
    out_file=None,
    rss_eflux=None,
    run_mode='RND',
    guided=False,
    memo=False,
    vel_map=None,
    vel_mask_map=None,
    vel_fixed=2,
    sigma_map=None,
    sigma_fixed=2,
    n_MC=50,
    n_loops=15,
    scale_ini=0.15,
    fine_search=None,
    redefine_max=False,
    max_factor=2,
    redefine_min=False,
    min_factor=0.012,
    plot=0,
    ):
    """
    The main function of the kin_rss_elines script. It will run the `fit_elines_main`
    over all observed spectra `rss_flux`, with shape ``(NW, NS)``. All spectra
    has to be sampled over the same `wavelength` with shape ``(NW)``.

    Parameters
    ----------
    wavelength : array
        The observed wavelengths.

    rss_flux : array like
        The observed flux spectra with shape (NW, NS).

    config : :class:`ConfigEmissionModel`
        The :class:`ConfigEmissionModel` instance that will configure the fit.

    out_file : str, optional
        The filename of the file where the output of :class:`EmissionLines` result
        will be recorded. Defaults to ``basename(sys.argv[0]).replace('py', 'out')``.

    rss_eflux : array like, optional
        The error in `flux` with shape ``(NW, NY, NX)``.

    run_mode : 'RND', 'LM', 'both', optional
        It will configure which :class:`EmissionLines` it will run.
        'RND': The RND algorithm of fit.
        'LM': Levemberg-Marquadt algorithm of fit.
        'both': It will run first the 'RND' method and inputs the `final_fit_params` of 'RND' at the 'LM' method.
        Defaults to RND method.

    guided : boolean, optional
        It will redefine the max and the min flux parameter of the eline models.

    memo : boolean, optional
        While fitting the rss, memorizes the last result and inputs to the next spectrum
        in the loop.

    vel_map : array like, optional
        Sets up the guide velocity map. (see: `guide_vel`).

    vel_mask_map : array like, optional
        Sets up the mask map of the guide velocity. (see: `guide_vel`).

    vel_fixed : int or boolean, optional
        Sets up the to_fit parameter of the guide velocity. (see: `guide_vel`).

    sigma_map : array like, optional
        Sets up the guide sigma map. (see: `guide_sigma`).

    sigma_fixed : int or boolean
        Sets up the to_fit parameter of the guide sigma map. (see: `guide_sigma`).

    n_MC : int, optional
        Number of Monte-Carlo iterations in the fit. Defaults to 50.

    n_loops : int, optional
        Number of loops of the fit. Defaults to 15.

    scale_ini : float, optional
        The scale_ini parameter of fit. Defaults to 0.15

    fine_search : boolean, optional
        The fine_search parameter of RND method. Defaults to
        `__ELRND_fine_search_option__` defined in pyFIT3D.common.constants.

    redefine_max : int or boolean, optional
        The redefine_max parameter of the fit. Defaults to 0.

    max_factor : float, optional
        Set up the max flux factor when redefine_max is on. Defaults to 2.

    redefine_min : boolean, optional
        If redefine_max is on, it will also redefine the min flux. Defaults
        to False.

    min_factor : float, optional
        Set up the min flux factor when redefine_max is on. Defaults to 0.012.

    plot : int, optional
        If 1 it will plot fit. If 2 it will plot to a file. None it will be set as 0, i.e.
        do not plot.

    Returns
    -------
    list of array like
        Observed spectra, model spectra and residual spectra, all with same shape as `rss_flux`.

    dict
        Output maps of fitted parameters (see: `create_emission_lines_parameters`,
        `append_emission_lines_parameters` and `output_emission_lines_parameters`).
    """
    guided = False if guided is None else bool(guided)
    memo = False if memo is None else bool(memo)
    out_file = basename(sys.argv[0]).replace('.py', '.out') if out_file is None else out_file
    ns, nw = rss_flux.shape

    # create models output
    output_el_models = create_emission_lines_parameters(config, shape=(ns))
    # create spectra output
    org__sw = np.zeros_like(rss_flux)
    mod__sw = np.zeros_like(rss_flux)
    res__sw = np.zeros_like(rss_flux)

    for i_s in range(ns):
        print(f'# ID {i_s}/{ns - 1}')
        current_spaxel = i_s
        current_guide_vel = None if vel_map is None else vel_map[i_s]
        current_guide_vel_mask = None if vel_mask_map is None else vel_mask_map[i_s]
        current_guide_sigma = None if sigma_map is None else sigma_map[i_s]
        flux = rss_flux[i_s, :]
        flux[~np.isfinite(flux)] = 0
        st_f = pdl_stats(flux)
        st_f_median = st_f[_STATS_POS['median']]
        st_f_sigma = st_f[_STATS_POS['pRMS']]
        if rss_eflux is not None:
            eflux = rss_eflux[i_s, :]
            eflux[~np.isfinite(eflux)] = 0
        else:
            eflux = np.ones_like(flux)*0.1
            if (st_f_median != 0) and (st_f_sigma != 0):
                eflux *= (st_f_median + st_f_sigma**2)
            # setvaltobad(0)
            eflux[eflux == 0] = np.nan
            if st_f_median != 0:
                eflux[np.isnan(eflux)] = st_f_median
            else:
                eflux[np.isnan(eflux)] = 0.001
        # enter if all flux is zero or if is in the first spaxel
        if not bool(i_s) and flux.any():
            cf = copy(config)
        else:
            last_chi_sq, _ = calc_chi_sq(EL.flux, EL.model, EL.sigma_flux,
                                         EL.config.n_models + 1)
            # Memorize last result to input the next spectrum
            if memo and (last_chi_sq < 2*EL.config.chi_goal):
                print('update')
                memo_cf = EL.update_config()
            else:
                cf = copy(config)
        model = np.zeros_like(flux)
        flux_max = np.abs(st_f[_STATS_POS['max']] - st_f[_STATS_POS['median']])
        if guided:
            redefine_min = True
            redefine_max = True
        EL = fit_elines_main(wavelength, flux, eflux, cf,
                             vel_guide=current_guide_vel, sigma_guide=current_guide_sigma,
                             vel_fixed=vel_fixed, sigma_fixed=sigma_fixed,
                             vel_mask=current_guide_vel_mask,
                             n_MC=n_MC, plot=plot, n_loops=n_loops,
                             scale_ini=scale_ini, fine_search=True,
                             redefine_max=redefine_max, redefine_min=redefine_min,
                             max_factor=max_factor, min_factor=min_factor, flux_max=flux_max,
                             run_mode=run_mode, randomize_flux=False)
        EL.output(filename_output=out_file, append_output=True, spec_id=i_s)
        org__sw[i_s, :] = EL.flux
        mod__sw[i_s, :] = EL.model
        res__sw[i_s, :] = EL.flux - EL.model
        append_emission_lines_parameters(EL, output_el_models, i_s)
    output_el_spectra = np.array([org__sw, mod__sw, res__sw])
    return output_el_spectra, output_el_models

def kin_rss_elines(
    spec_file,
    config_file,
    w_min,
    w_max,
    out_file=None,
    mask_file=None,
    error_file=None,
    prefix=None,
    run_mode='RND',
    guided=False,
    memo=False,
    vel_map_file=None,
    vel_mask_file=None,
    vel_fixed=None,
    sigma_map_file=None,
    sigma_fixed=None,
    n_MC=50,
    n_loops=15,
    scale_ini=0.15,
    fine_search=None,
    redefine_max=False,
    plot=0,
    seed=None,
    ):
    """
    The kin_rss_elines script (i.e. a well prepared wrap of `kin_rss_elines_main`)

    It will run the `run_mode` fit the the emission models defined by the `config_file`
    on the rss of spectra in `spec_file` masked by `mask_file`, inside the define wavelength
    range (`w_min`, `w_max`).

    If `run_mode` is::

        'RND': mimics fit_elines_rnd script.
        'LM': mimics fit_elines_LM script.
        'both': It will run the 'RND' first and inputs the results of the fit in the 'LM' fit.

    Parameters
    ----------
    spec_file : str
        The filename of the FITS file containing the wavelengths and observed flux spectra.

    config_file : str
        The filename of the :class:`ConfigEmissionModel` input file.

    w_min : int
        The minimun (bluer) wavelength which defines the fitted wavelength range.

    w_max : int
        The maximum (redder) wavelength which defines the fitted wavelength range.

    out_file : str, optional
        The name of the output results of the fit. Defaults to
        ``basename(sys.argv[0]).replace('py', 'out')``.

    mask_file : str, optional
        If mask_file is a valid file it will mask the wavelengths.

    error_file : str
        The filename of the FITS file containing the errors in observed flux spectra.

    prefix : str, optional
        It will define the prefix used in `output_emission_lines_parameters` and
        `output_emission_lines_spectra`. Defaults to basename of `out_file`.

    run_mode : str {'RND', 'LM', 'both'}, optional
        It will configure which :class:`EmissionLines` it will run.
        'RND': The RND algorithm of fit.
        'LM': Levemberg-Marquadt algorithm of fit.
        'both': It will run first the 'RND' method and inputs the `final_fit_params` of 'RND' at the 'LM' method.
        Defaults to the RND method.

    guided : boolean, optional
        It will redefine the max and the min flux parameter of the eline models.

    memo : boolean, optional
        While fitting the rss, memorizes the last result and inputs to the next spectrum
        in the loop.

    vel_map_file : str, optional
        The filename of the guide velocity map. (see: `guide_vel`).

    vel_mask_file : array like, optional
        The filename of the mask map of the guide velocity. (see: `guide_vel`).

    vel_fixed : int or boolean, optional
        Sets up the to_fit parameter of the guide velocity. (see: `guide_vel`).

    sigma_map_file : array like, optional
        The filename of the guide sigma map. (see: `guide_sigma`).

    sigma_fixed : int or boolean
        Sets up the to_fit parameter of the guide sigma map. (see: `guide_sigma`).

    n_MC : int, optional
        Number of Monte-Carlo iterations in the fit. Defaults to 50.

    n_loops : int, optional
        Number of loops of the fit. Defaults to 15.

    scale_ini : float, optional
        The scale_ini parameter of fit. Defaults to 0.15

    fine_search : boolean, optional
        The fine_search parameter of RND method. Defaults to
        `__ELRND_fine_search_option__` defined in `pyFIT3D.common.constants`.

    redefine_max : int or boolean, optional
        The redefine_max parameter of the fit. Defaults to 0.

    plot : int, optional
        If 1 it will plot fit. If 2 it will plot to a file. None it will be set as 0, i.e.
        do not plot.

    seed : int, optional
        It will define the input seed. Defaults to the ``int(time.time())``.

    See also
    --------
    :class:`ConfigEmissionModel`, `fit_elines_main`, `kin_rss_elines_main`, `output_emission_lines_parameters`, `output_emission_lines_spectra`

    """

    seed = print_time() if seed is None else print_time(time_ini=seed)
    np.random.seed(seed)
    time_ini_run = print_time(print_seed=False, get_time_only=True)
    guided = False if guided is None else bool(guided)
    memo = False if memo is None else bool(memo)
    vel_fixed = 2 if vel_fixed is None else vel_fixed
    sigma_fixed = 2 if sigma_fixed is None else sigma_fixed
    plot = 0 if plot is None else plot
    fine_search = __ELRND_fine_search_option__ if fine_search is None else fine_search
    out_file = basename(sys.argv[0]).replace('.py', '.out') if out_file is None else out_file
    if prefix is None:
        _s = out_file.split('.')
        prefix = '.'.join(_s[0:-1]) if len(_s) > 1 else f'{out_file}'
    remove_isfile(out_file)
    _runs = ['RND', 'LM', 'both']
    if (run_mode is None) or (run_mode not in _runs):
        run_mode = 'RND'
    if vel_map_file is None or not isfile(vel_map_file):
        vel_fixed = 0
    if sigma_map_file is None or not isfile(sigma_map_file):
        sigma_fixed = 0
    if sigma_val is not None:
        sigma_val = sigma_val * __FWHM_to_sigma__
    # read masks file
    masks, n_masks = read_masks_file(mask_file)
    print(f'{n_masks} regions to mask')
    error_extension = 0
    if error_file is not None:
        _tmp = error_file.split('[')
        if len(_tmp) > 1:
            try:
                error_file = _tmp[0]
                error_extension = int(_tmp[-1].split(']')[0])
            except:
                print(f'kin_rss_elines: error extension not found, using {error_extension}')
    wave__w, flux__sw, header, eflux__sw = load_spec(filename=spec_file, wave_axis=1,
                                                     error_filename=error_file,
                                                     error_extension=error_extension)
    sel_wl_range = trim_waves(wave__w, [w_min, w_max])
    sel_wl_goods = sel_waves(masks, wave__w)
    sel_wl = sel_wl_range & sel_wl_goods
    wave_msk__w = wave__w[sel_wl]
    flux_msk__sw = flux__sw[:, sel_wl]
    if eflux__sw is not None:
        eflux_msk__sw = eflux__sw[:, sel_wl]
    else:
        eflux_msk__sw = None
    # read config FOR TESTING PURPOSES -------------------------------------------------------------
    org_cf = ConfigEmissionModel(config_file)
    vel_map__s = None
    if (vel_map_file is not None) and (os.path.isfile(vel_map_file)):
        vel_map__s = get_data_from_fits(vel_map_file)
    vel_mask_map__s = None
    if (vel_mask_file is not None) and (os.path.isfile(vel_mask_file)):
        vel_mask_map__s = get_data_from_fits(vel_mask_file)
    sigma_map__s = None
    if (sigma_map_file is not None) and (os.path.isfile(sigma_map_file)):
        sigma_map__s = get_data_from_fits(sigma_map_file)
    r = kin_rss_elines_main(
        wave_msk__w, flux_msk__sw, org_cf, out_file,
        rss_eflux=eflux_msk__sw,
        vel_map=vel_map__s, vel_mask_map=vel_mask_map__s, vel_fixed=vel_fixed,
        sigma_map=sigma_map__s, sigma_fixed=sigma_fixed, memo=memo,
        n_MC=n_MC, n_loops=n_loops, redefine_max=redefine_max,
        scale_ini=scale_ini, fine_search=fine_search, run_mode=run_mode,
        plot=plot, guided=guided,
    )
    output_el_spectra, output_el_models = r
    output_emission_lines_parameters(prefix, org_cf, output_el_models)
    output_emission_lines_spectra(wave_msk__w, output_el_spectra, header, prefix, wave_axis=1)
    time_end = print_time(print_seed=False)
    time_total = time_end - time_ini_run
    print(f'# SECONDS = {time_total}')

# TODO
# def read_fit_elines_with_config(filename, filename_config, verbose=0):

def read_fit_elines_output(filename, verbose=0):
    """
    Reads the output file of a fit_elines_main run. Outputs a dict with the fitted
    parameters.

    Parameters
    ----------
    filename : str
        The filename of the output file of a fit_elines_main run.

    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    array like :
        number of models of each emission lines system fit

    array like :
        Resultant chi-sq (goodness) of the fit of each emission lines system.

    array of dicts :
        An array of ``output_models`` dicts.
        The keys of the dicts depends on the models parameters present in the config
        (`_EL_MODELS` dict is defined in `pyFIT3D.common.constants`). Each parameter
        has a e_par key also.

        Example with a config containing 2 eline models and 1 poly1d model:

        `output_models` it will have keys ``_EL_MODELS['eline'].keys()`` and
        ``_EL_MODELS['poly1d'].keys()`` with their respective errors hashed as
        ``e_{key}``.

        The output_models parameters from eline models will have shape `(2, shape)`.

        The poly1d parameters have `(1, shape)`

        E.g, ``output_models['v0'][0]`` is  the map of the v0 parameter of the first
        eline).

    See also
    --------
    `fit_elines_main`, `create_emission_lines_parameters`.
    """
    # TODO: rebuild function using config._EL_MODELS dictionary structure.
    systems = []
    n_models = []
    chi_sq = []
    with open(filename, 'r') as f:
        for l in f:
            _tmp = l.split()
            n_models.append(eval(_tmp[0]))
            chi_sq.append(eval(_tmp[1]))
            models = {}
            for i in range(n_models[-1]):
                l = f.readline()
                _s = l.split()
                model = _s[0]
                if not (model in models.keys()):
                    models[model] = []
                if model == 'eline':
                    models[model].append({
                        'central_wavelength': eval(_s[1]),
                        'e_central_wavelength': eval(_s[2]),
                        'flux': eval(_s[3]),
                        'e_flux': eval(_s[4]),
                        'sigma': eval(_s[5]),
                        'e_sigma': eval(_s[6]),
                        'v0': eval(_s[7]),
                        'e_v0': eval(_s[8]),
                        'disp':  __sigma_to_FWHM__*eval(_s[5]),
                        'e_disp':  __sigma_to_FWHM__*eval(_s[6]),
                    })
                elif model == 'poly1d':
                    models[model].append({
                        'cont': eval(_s[1]),
                        'e_cont': eval(_s[2])
                    })
                    ##############################################
                    # TODO: Treat poly1d coefficients correctly.
                    #   The coefficients should be returned.
                    ##############################################
                    # NOTE:
                    #   Missing wavelength to integrate the
                    #   coefficients.
                    ##############################################
                    #   e.g.:
                    #       # ncoeffs = self.config.to_fit[i_m].astype(int).sum()
                    #       # v = {f'coeff{i}': i for i in range(ncoeffs)}
                    #       _sum, _esum = 0, 0
                    #       k = 'cont'
                    #       ek = f'e_{k}'
                    #       coeffs = {}
                    #       for i in range(__n_models_params__):
                    #           i_p = 2*i
                    #           i_ep = i_p + 1
                    #           _par = eval(_s[i_p])
                    #           _std_par = eval(_s[i_p])
                    ############ SAVE COEFFICIENTS
                    #           coeffs['coeff{i}'] = _par
                    #           coeffs['e_coeff{i}'] = _std_par
                    #       models[model].append(coeffs)
                    ############ ANOTHER OPTION IS TO INTEGRATE
                    ############ missing wavelengths
                    #           _int_mod = np.array([l**i_p for l in wavelength]).sum()
                    #           _sum += _par*_int_mod
                    #           _esum += sigma_par*_int_mod
                    #       models[model].append({k: sum, ek: esum})
                    ##############################################
            systems.append(models)

    output_systems = []
    for _s in systems:
        models = _s
        n_eline_models = 0 if 'eline' not in models.keys() else len(models['eline'])
        n_poly1d_models = 0 if 'poly1d' not in models.keys() else len(models['poly1d'])
        output_models = {}
        print_verbose(f'Number of eline models = {n_eline_models}', verbose=verbose)
        print_verbose(f'Number of poly1d models = {n_poly1d_models}', verbose=verbose)
        if n_eline_models > 0:
            for par_key in models['eline'][0].keys():
                _arr = np.empty((n_eline_models), dtype='float')
                for i in range(n_eline_models):
                    _arr[i] = models['eline'][i][par_key]
                output_models[par_key] = _arr
        if n_poly1d_models > 0:
            for par_key in models['poly1d'][0].keys():
                _arr = np.empty((n_poly1d_models), dtype='float')
                for i in range(n_poly1d_models):
                    _arr[i] = models['poly1d'][i][par_key]
                output_models[par_key] = _arr
        output_systems.append(output_models)
    return n_models, chi_sq, output_systems
