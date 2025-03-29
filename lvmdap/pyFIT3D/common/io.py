#### System imports
import os
import io
import sys
import time
import itertools
import subprocess
import numpy as np
from os import getcwd
import argparse as ap
from astropy.io import fits
from datetime import datetime
from copy import deepcopy as copy
from os.path import basename, isfile, join, abspath

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(context="talk", style="ticks", palette="colorblind", color_codes=True)

class readFileArgumentParser(ap.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(readFileArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg

#### Local imports
from .constants import __n_models_params__, __FWHM_to_sigma__, __version__

class ReadArguments(object):
    """
    Argument parser for the FIT3D echo-system
    """
    # class static configuration:
    # arguments names and conversion string to number functions

    def __init__(self, args_list=None, verbose=False):
        self.verbose = verbose
        """Initialize the argument list received and other helper variables"""
        self.args_list = args_list
        # if no list is in args_list, reads sys.argv, without the script pathname (first position of the list)
        if self.args_list is None:
            self.args_list = sys.argv[1:]
        self.N = len(self.args_list)
        # Error: less args than needed
        if (self.N < len(self.__mandatory__)) | (self.N > self.__N_tot_args__):
            print(self.__usage_msg__)
            sys.exit()
        # attributes values to arguments
        for i in range(self.N):
            arg_name = self.__arg_names__[i]
            # arguments values passed by the user
            arg_value = self.args_list[i]
            print_verbose(f'i) arg_name --------- {arg_name}', verbose=self.verbose)
            print_verbose(f'i) arg_value -------- {arg_value}', verbose=self.verbose)
            # None values
            if arg_value not in ['none', 'None']:
                # print(f'{arg_name}:{arg_value}')
                _cf = self.__conv_func__.get(arg_name, eval)
                print_verbose(f'i) conversion_func -- {_cf}', verbose=self.verbose)
                setattr(self, arg_name, _cf(arg_value))
        # attributes a default value to arguments that were not defined the by user
        i += 1
        while i < self.__N_tot_args__:
            arg_name = self.__arg_names__[i]
            try:
                arg_value = self.__def_optional__.get(arg_name, None)
            except AttributeError:
                arg_value = None
            # print(f'{arg_name}:{arg_value}')
            print_verbose(f'ii) arg_name --------- {arg_name}', verbose=self.verbose)
            print_verbose(f'ii) arg_value -------- {arg_value}', verbose=self.verbose)
            if arg_value is not None:
                if not isinstance(arg_value, str):
                    _cf = lambda x: x
                else:
                    try:
                        _cf = self.__conv_func__.get(arg_name, eval)
                    except AttributeError:
                        _cf = eval
                print_verbose(f'ii) conversion_func -- {_cf}', verbose=self.verbose)
                setattr(self, arg_name, _cf(arg_value))
            i += 1

    def __getattr__(self, attr):
        """Return None if some inexistent argument is accessed"""
        r = self.__dict__.get(attr, None)
        return r

def read_first_line(filename):
    """Return the first record of the given filename

    Parameters
    ----------
    filename : str
        The name of the file for which to get the first record

    Returns
    -------
    str
        The first not commented record found in the given filename
    """
    with open(filename) as f:
        l = f.readline()
        # jump comments
        while l.startswith('#'):
            l = f.readline()
    return l

def print_time(print_seed=True, time_ini=None, get_time_only=False):
    """ Return the local timestamp

    Parameters
    ----------
    print_seed: boolean
        Whether to print or not a formatted version of the local time

    Returns
    -------
    int
        The rounded current timestamp
    """
    time_ini = time.time() if time_ini is None else time_ini
    tepoch = int(time_ini)
    seedstr = ''
    if print_seed:
        seedstr = f' (random number generator seed: {tepoch})'
    if not get_time_only:
        print(f"# TIME {datetime.fromtimestamp(tepoch)}{seedstr}\n")
    return tepoch

# this assumes that the number of columns does not vary within the file
def get_num_col_file(filename, sep=None, maxsplit=-1):
    """ Reads the number of columns of file using `str.split`.

    Parameters
    ----------
    filename : str
        Filename of the inspected file.
    sep : None or char, optional
        If `sep` is given uses as the column delimiter. Default is None.
    maxsplit : int, optional
        If maxsplit is given, at most maxsplit splits are done (thus, the list
        will have at most maxsplit+1 elements). If maxsplit is not specified or
        -1, then there is no limit on the number of splits (all possible splits
        are made).

    Returns
    -------
    int
        Number of columns in `filename` separated by `sep`.

    """
    l = read_first_line(filename)
    return len(l.split(sep=sep, maxsplit=maxsplit))

def read_spectra(filename, f_error=None, variance_column=True):
    """Return the wavelength, and the flux and error spectra from a given filename

    If the given filename contains only 2 columns or less an error will be raised.
    If there are only 3 columns, the error will be computed as follow:

      0.002 * abs(flux)

    Parameters
    ----------
    filename : str
        Filename of the spectra.

    f_error : function, optional
        Function that defines the error when it is not present in the spectrum file.
        It defaults to 0.002 * abs(flux).

    variance_column : bool, optional
        When True treats the error column as variance and takes the square-root of the
        values.

    Returns
    -------
    {wave, flux, eflux} : array like
       The wavelength, and the flux and the error flux stored in the given file.
    """
    if f_error is None:
        f_error = lambda x: 0.002*np.abs(x)

    N_col_spec_file = get_num_col_file(filename)
    if N_col_spec_file < 2:
        raise IOError(f'read_spectra: the file "{filename}" has missing data (columns: {N_col_spec_file}), expected at least 3')
    if N_col_spec_file == 4:
        wave, flux, eflux = np.loadtxt(filename, unpack=True, usecols=(1,2,3))
        eflux = np.abs(eflux)
        if variance_column:
            eflux = np.sqrt(eflux)
    else:  # no error avaible.
        usecols = (1, 2)
        if N_col_spec_file == 2: # missing index column in file
            usecols = (0, 1)
        wave, flux = np.loadtxt(filename, unpack=True, usecols=usecols)
        eflux = f_error(flux)
    return wave, flux, eflux

def output_spectra(w, v, filename):
    """
    A wrapper to np.savetxt() creating a column with an ID.
    """

    _out = list(zip(list(range(1, w.size+1)), w, *v))
    # np.savetxt(filename, _out, fmt='%s')
    np.savetxt(filename, _out, fmt=['%d'] + (len(v) + 1)*['%.18g'])

def probe_method_time_consumption(**kwargs):
    """Function created as a wrapper to cProfile Python Profilers, a set of statistics that describes how often and how long various parts of the program (or any call of method/functions) executed.

    Example
    -------

    .. code-block:: python

        def f_sum_xy_ntimes(x, y, n=100):
            for i in range(n):
                s = x + y
            return s

        a = np.random.normal(10, 0.3, 100)
        b = np.random.normal(3, 0.1, 100)
        a_plus_b = probe_method_time_consumption(f=f_sum_xy_ntimes, f_args=dict(x=a, y=b, n=10000))

        #         2 function calls in 0.011 seconds
        #
        #   Ordered by: cumulative time
        #
        #   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        #        1    0.011    0.011    0.011    0.011 <ipython-input-32-389c091e3289>:1(f_sum_xy_ntimes)
        #        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        # ...

    Parameters
    ----------
    f : function
        Function that will be probed.

    f_args : dict
        A dictionary with the arguments for `f` call.

    Returns
    -------
    ret_f
        The return from `f`.
    """
    import cProfile, pstats, io
    f = kwargs.get('f')
    f_args = kwargs.get('f_args')
    pr = cProfile.Profile()
    pr.enable()
    f_ret = f(**f_args)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    return f_ret

def remove_isfile(filename):
    if os.path.isfile(filename):
        os.remove(filename)

# EL: deprecated
def clean_preview_results(args):
    """Remove output files from older runs

    Parameters
    ----------
        args : class `ReadArguments`
    """

    if os.path.isfile(args.out_file):
        os.remove(args.out_file)
    if os.path.isfile(args.out_file_elines):
        os.remove(args.out_file_elines)
    if os.path.isfile(args.out_file_single):
        os.remove(args.out_file_single)
    if os.path.isfile(args.out_file_coeffs):
        os.remove(args.out_file_coeffs)
    if os.path.isfile(args.out_file_fit):
        os.remove(args.out_file_fit)

def clean_preview_results_files(out_file, out_file_elines, out_file_single,
                                out_file_coeffs, out_file_fit):
    """Remove output files from older runs

    Parameters
    ----------
        args : class `ReadArguments`
    """
    remove_isfile(out_file)
    remove_isfile(out_file_elines)
    remove_isfile(out_file_single)
    remove_isfile(out_file_coeffs)
    remove_isfile(out_file_fit)

def trim_waves(wave, wave_range):
    """Return a mask for the trimmed version of the wavelength range

    For a given observed wavelength for which it is intended to extract
    physical information, this function returns a boolean selection of
    the wavelength range to consider during the fitting. The selection
    is computd such that:

    wave_range[0] <= wave <= wave_range[1]

    In the case where wave_range is None or wave_range[0] = wave_range[1],
    returns the entire wavelength range.

    Parameters
    ----------
    wave: array like
        A one-dimensional array of the observed wavelength
    """
    if (wave_range is None) or (wave_range[0] == wave_range[1]):
       mask = np.ones(wave.shape, dtype='bool')
    else:
       mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
    return mask

def sel_waves(masks, wave):
    """Return the mask for the given wavelength array

    For a given observed wavelength for which it is intended to extract
    physical information, this function returns a boolean selection of wavelength
    ranges to consider during the fitting. The selections are computed such that for
    any i-mask:

    masks[i, 0] <= wave <= masks[i, 1]

    Parameters
    ----------
    masks: array like
        A Nx2 array containing the masks for the emission lines
    wave: array like
        A one-dimensional array of the observed wavelength

    Returns
    -------
    array like :
        A boolean array with True for the non-masked wavelengths.

    See Also
    --------
    read_masks_file
    """
    selected_wavelengths = np.ones(wave.size, dtype='bool')
    if (masks is not None) and (len(masks) > 0):
        for left, right in masks:
            # limit to above left limit
            left_selection = wave >= left
            # limit to below right limit
            right_selection = wave <= right
            # selects all range
            range_selection = left_selection & right_selection
            # mask masked range
            selected_wavelengths[range_selection] = False
    return selected_wavelengths

def read_masks_file(filename):
    """
    Read the masks file and returns an array with the ranges to be masked.

    Parameters
    ----------
    filename : str
        Masks filename.

    Returns
    -------
    array like :
        An array that saves in each position the range to be masked.
    int :
        The number of intervals to be masked.

    Example
    -------
    A typical mask file `filename`:

    5800 6080
    5550 5610
    7100 15000

    should return:

    array([5800., 6808.],
          [5500., 5610.],
          [7100., 15000.]),
    """
    # read masks
    masks = None
    n_masks = 0
    if filename is not None:
        if isfile(filename):
            masks = np.loadtxt(filename)
            n_masks = len(masks)
        else:
            print(f'{basename(sys.argv[0])}: {filename}: mask list file not found')
    else:
        print(f'{basename(sys.argv[0])}: no mask list file')
    return masks, n_masks

def print_verbose(text, verbose=0, level=1):
    """ Print `text` if verbose is True.

    Parameters
    ----------
    text : str
        Text to be printed.
    verbose : int, bool, optional
        If `verbose` is greater than `level` prints `text`. If is True forces
        the print of `text`. Defaults to 0.
    level : int, optional
        Configures the print level of the verbosity. Default level is 1.
    """
    if (level if (0 if verbose is None else verbose) is True else verbose) >= level:
        print(text)

def plot_spectra_ax(ax, wave_list, spectra_list, title='', labels_list=None,
                    color=None, cmap=None, ylim=None, xlim=None, alpha=None, lw=None):
    """
    Print spectra provided by `wave_list` and `spectra_list`.

    Parameters
    ----------
    ax : matplotlib.axis
        Axis to plot spectra.
    wave_list : list
        List of `spectra_list` wavelengths.
    spectra_list : list
        List of spectra.
    title: str, optional
        Title to the axis. Default value is ''.
    labels_list : list or None
        List of labels, if None no labels are shown.
    color: str, list or None
        A color, a list of colors or None. If None automatically generates the colors
        for `spectra_list` by using the `cmap` colormap.
    cmap: str
        The colormap to generate the colors for spectra if `color` is None.
        Default colormap is viridis.
    xlim: list or None
        Plot boudaries of the x-axis. Default value is None (i.e., define automatically the
        x-limits).
    ylim: list or None
        Plot boudaries of the y-axis. Default value is None (i.e., define automatically the
        y-limits).
    alpha: float, list or None
        An alpha, a list of alphas or None. If None automatically uses alpha=1.
    lw: float, list or None
        A line width (lw), a list of lw or None. If None automatically uses defaul matplotlib lw.
    """
    if 'seaborn' not in sys.modules:
        import seaborn as sns
    else:
        sns = sys.modules['seaborn']

    N_wave = len(wave_list)
    N_spec = len(spectra_list)
    if (N_spec > 1) and (N_wave != N_spec):
        wave_list = [wave_list[0]]*N_spec
    ax.set_title(title)
    if alpha is None:
        alpha = [1]*N_spec
    if not isinstance(alpha, list):
        alpha = [alpha]*N_spec
    alpha_list = alpha
    if lw is None:
        lw = [1]*N_spec
    if not isinstance(lw, list):
        lw = [lw]*N_spec
    lw_list = lw
    if color is None:
        # cm = plt.cm.get_cmap(cmap if cmap is not None else "colorblind")
        # colors_list = [cm(i/N_spec) for i in range(N_spec)]
        colors_list = sns.color_palette(cmap if cmap is not None else "colorblind", n_colors=N_spec)
    else:
        if isinstance(color, str):
            colors_list = [color]*N_spec
        else:
            if len(color) < N_spec:
                colors_list = [color[0]]*N_spec
            else:
                colors_list = color
    legend = True
    if labels_list is None:
        legend = False
        labels_list = ['']*N_spec
    for w, f, c, l, a, lw in zip(wave_list, spectra_list, colors_list, labels_list, alpha_list, lw_list):
        ax.plot(w, f, color=c, label=l, alpha=a, lw=lw)
    # it is very expensive in time to call `ax.legend` if no label is set.
    if legend:
        ax.legend(loc=2, frameon=False)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

def get_data_from_fits(filename, extension=0, header=None, return_n_extensions=False):
    """ Retrieve the data from the HDU `extension` in a FITS file.
    If `header` is True, also, retrieves the header from the HDU
    `extension`.
    if `filename` is not a existent file or `astropy.io.fits.open(filename)`
    raises an `OSError`, returns the `filename`. if header is True,
    returns None.

    Parameters
    ----------
    filename: str
        FITS file name.

    extension: int
        The retrieved HDU extension. Default is 0.

    header: bool or None
        Also retrieves the HDU `extension` header.

    return_n_extensions: bool or None
        Also returns the number of `extensions` of the FITS file.

    Returns
    -------
    array like:
        The data from HDU `extension`.

    astropy.io.fits.header.Header:
        If `header` is True also returns the header from HDU `extension`.

    int:
        If `return_n_extensions` is True also returns the number of extensions.
    """
    data = filename
    h = None
    n_ext = 0
    if filename is not None and os.path.isfile(filename):
        try:
            with fits.open(filename) as t:
                hdu = t[extension]
                hdu.verify('silentfix')
                data = copy(hdu.data)
                h = copy(hdu.header)
                n_ext = len(t)
        except OSError:
            print(f'{sys.argv[0]}: {filename}: not a valid FITS file.')
            pass
    else:
        print(f'{sys.argv[0]}: {filename}: not a valid file. Returning the input value.')
    if header:
        if return_n_extensions:
            return data, h, n_ext
        else:
            return data, h
    else:
        if return_n_extensions:
            return data, n_ext
        else:
            return data

def get_wave_from_header(header, wave_axis=None):
    '''
    Generates a wavelength array using `header`, a :class:`astropy.io.fits.header.Header`
    instance, at axis `wave_axis`.

    wavelengths = CRVAL + CDELT*([0, 1, ..., NAXIS] + 1 - CRPIX)

    Parameters
    ----------
    header : :class:`astropy.io.fits.header.Header`
        FITS header with spectral data.

    wave_axis : int, optional
        The axis where the wavelength information is stored in `header`,
        (CRVAL, CDELT, NAXIS, CRPIX).
        Defaults to 1.

    Returns
    -------
    array like
        Wavelengths array.

        wavelengths = CRVAL + CDELT*([0, 1, ..., NAXIS] + 1 - CRPIX)
    '''
    if wave_axis is None:
        wave_axis = 1
    h = header
    crval = h[f'CRVAL{wave_axis}']
    cdelt = h[f'CDELT{wave_axis}']
    naxis = h[f'NAXIS{wave_axis}']
    crpix = h[f'CRPIX{wave_axis}']
    if not cdelt:
        cdelt = 1
    return crval + cdelt*(np.arange(naxis) + 1 - crpix)

def array_to_fits(filename, arr, new_card=False, header=None, overwrite=False, sort_dict_header=True):
    hdul = fits.HDUList()
    if new_card:
        hdul.append(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(data=arr))
    else:
        hdul.append(fits.PrimaryHDU(arr))
    hdu = hdul[0]
    # XXX version should be a global parameter
    if header is not None:
        if isinstance(header, dict):
            if sort_dict_header:
                header = dict(sorted(header.items()))
            for k, v in header.items():
                hdu.header.set(k, value=v)
        else:
            hdu.header = header
    hdu.header.set('PIPELINE', value=__version__)
    now = time.time()
    dt_now = datetime.fromtimestamp(now)
    hdu.header.set('UNIXTIME', value=int(now), comment=f'{dt_now}')
    hdul.writeto(filename, overwrite=overwrite)

def call_cmnd(cmnd, logfile=None, verbose=True, logfilemode='a', timeout=None):
    close = False
    if isinstance(logfile, str):
        close = True
        lfd = open(logfile, mode=logfilemode)
    elif isinstance(logfile, io.TextIOWrapper):
        lfd = logfile
    else:
        lfd = sys.stdout
    if verbose:
        print(cmnd, file=lfd)
    try:
        output = subprocess.check_output(cmnd, stderr=subprocess.STDOUT,
                                         shell=True, timeout=timeout, universal_newlines=True)
    except subprocess.CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output, file=lfd)
    else:
        print("Output: \n{}\n".format(output), file=lfd)

def print_done(time_ini, message=None):
    message = 'DONE!' if message is None else message
    time_end = print_time(print_seed=False, get_time_only=True)
    time_spent = time_end - time_ini
    print(f'# {message} - time spent: {time_spent} s\n####\n', flush=True)

def print_block_init(message, print_seed=False):
    print(f'####\n# {message}\n#', flush=True)
    return print_time(print_seed=print_seed)

def write_img_header(input_fits, keyword, value,
                     comment=None, before=None, after=None, output=None, overwrite=True):
    if output is None:
        output = input_fits
        overwrite = True
    t, h = get_data_from_fits(input_fits, header=True)
    if isinstance(keyword, list):
        for i, k in enumerate(keyword):
            if value is None:
                v = None
            else:
                v = value if not isinstance(value, list) else value[i]
            if comment is None:
                c = None
            else:
                c = comment if not isinstance(comment, list) else comment[i]
            if before is None:
                b = None
            else:
                b = before if not isinstance(before, list) else before[i]
            if after is None:
                a = None
            else:
                a = after if not isinstance(after, list) else after[i]
            h.set(k, value=v, comment=c, before=b, after=a)
    else:
        # set new value to header
        h.set(keyword, value=value, comment=comment, before=before, after=after)
    array_to_fits(output, t, header=h, overwrite=overwrite)

def create_emission_lines_mask_file_from_list(wave_list, eline_half_range=16, output_path=None, label=None):
    output_path = getcwd() if output_path is None else abspath(output_path)
    np.savetxt(join(output_path, ('autodetect.mask_elines.txt' if label is None else f'{label}.autodetect.mask_elines.txt')),
               [[int(wave_peak-eline_half_range), int(wave_peak+eline_half_range)] for wave_peak in wave_list],
               fmt='%5d')

def create_emission_lines_file_from_list(wave_list, output_path=None, label=None):
    output_path = getcwd() if output_path is None else abspath(output_path)
    np.savetxt(join(output_path, ('autodetect.emission_lines.txt' if label is None else f'{label}.autodetect.emission_lines.txt')), wave_list, fmt='%5.2f')

def create_ConfigAutoSSP_from_lists(list_chunks, list_systems_config, output_path=None, label=None):
    # TODO: ConfigAutoSSP should be created dynamically (pyFIT3D/common/auto_ssp_tools.py)
    #       This function should return the list of wavelengths and generated
    #       config filenames to the new function.
    output_path = getcwd() if output_path is None else abspath(output_path)
    ConfigAutoSSP_filename = join(output_path, ('autodetect.auto_ssp_several.config' if label is None else f'{label}.autodetect.auto_ssp_several.config'))
    n_systems = len(list_systems_config)
    eml_systems_lines = [f'{chunk} none {config_filename} {__n_models_params__} none 20 1950\n' for chunk, config_filename in zip(list_chunks, list_systems_config)]
    with open(ConfigAutoSSP_filename, 'w') as ConfigAutoSSP_file:
        ConfigAutoSSP_file.write('0.017 0.002 0.0001 0.027 10 100 0.1 0.5 3800 5500\n')
        ConfigAutoSSP_file.write('3.2  0.0    1.9    6.5\n')
        ConfigAutoSSP_file.write('0.4  0.1    0.0   2.5\n')
        ConfigAutoSSP_file.write(f'{n_systems}\n')
        ConfigAutoSSP_file.writelines(eml_systems_lines)
        ConfigAutoSSP_file.write('0.0001  1 0.00\n')
        ConfigAutoSSP_file.write('6588 6760\n')
