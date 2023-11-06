# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['lvmdap', 'lvmdap._cmdline', 'lvmdap.analysis', 'lvmdap.modelling']

package_data = \
{'': ['*']}

install_requires = \
[]

entry_points = \
{'console_scripts': ['clean-outputs = lvmdap._cmdline.clean_outputs:_main',
                     'lvm-dap = lvmdap._cmdline.dap:_main']}

setup_kwargs = {
    'name': 'lvm-dap',
    'version': '0.1.0',
    'description': 'LVM Data Analisys Pipeline',
    'long_description': "# Introduction\n\nThis is the SDSS-V Local Volume Mapper (LVM) Data Analysis Pipeline (DAP) official repository.\n\nThe main and only script, `lvm-dap`, implements the Resolved Stellar Population method (Mejia-Narvaez+, in prep.). Instructions on how to run this code below.\n\n# Usage\n\n```\nlvm-dap [-h] [--emission-lines-file EMISSION_LINES_FILE] [--rsp-nl-file RSP_NL_FILE]\n               [--plot PLOT] [--flux-scale min max] [--w-range wmin wmax] [--w-range-nl wmin2 wmax2]\n               [--redshift input_redshift delta_redshift min_redshift max_redshift]\n               [--sigma input_sigma delta_sigma min_sigma max_sigma]\n               [--AV input_AV delta_AV min_AV max_AV] [--ext-curve {CCM,CAL}] [--RV RV]\n               [--single-rsp] [--n-mc N_MC] [-o path] [-c] [-v] [-d]\n               spectrum-file rsp-file sigma-inst label mask-file config-file\n\nRun the spectral fitting procedure for the LVM\n\npositional arguments:\n  spectrum-file         input spectrum to fit\n  rsp-file              the resolved stellar population basis\n  sigma-inst            the standard deviation in wavelength of the Gaussian kernel to downgrade the\n                        resolution of the models to match the observed spectrum. This is:\n                        sigma_inst^2 = sigma_obs^2 - sigma_mod^2\n  label                 string to label the current run\n  mask-file             the file listing the wavelength ranges to exclude during the fitting\n  config-file           the configuration file used to set the parameters for the emission line\n                        fitting\n\noptional arguments:\n  -h, --help            show this help message and exit\n  --emission-lines-file EMISSION_LINES_FILE\n                        file containing emission lines list\n  --rsp-nl-file RSP_NL_FILE\n                        the resolved stellar population *reduced* basis, for non-linear fitting\n  --plot PLOT           whether to plot (1) or not (0, default) the fitting procedure. If 2, a plot\n                        of the result is store in a file without display on screen\n  --flux-scale min max  scale of the flux in the input spectrum\n  --w-range wmin wmax   the wavelength range for the fitting procedure\n  --w-range-nl wmin2 wmax2\n                        the wavelength range for the *non-linear* fitting procedure\n  --redshift input_redshift delta_redshift min_redshift max_redshift\n                        the guess, step, minimum and maximum value for the redshift during the\n                        fitting\n  --sigma input_sigma delta_sigma min_sigma max_sigma\n                        same as the redshift, but for the line-of-sight velocity dispersion\n  --AV input_AV delta_AV min_AV max_AV\n                        same as the redshift, but for the dust extinction in the V-band\n  --ext-curve {CCM,CAL}\n                        the extinction model to choose for the dust effects modelling. Choices are:\n                        ['CCM', 'CAL']\n  --RV RV               total to selective extinction defined as: A_V / E(B-V). Default to 3.1\n  --single-rsp          whether to fit a single stellar template to the target spectrum or not.\n                        Default to False\n  --n-mc N_MC           number of MC realisations for the spectral fitting\n  -o path, --output-path path\n                        path to the outputs. Defaults to '/disk-a/mejia/Research/UNAM/lvm-dap'\n  -c, --clear-outputs   whether to remove or not a previous run with the same label (if present).\n                        Defaults to false\n  -v, --verbose         if given, shows information about the progress of the script. Defaults to\n                        false.\n  -d, --debug           debugging mode. Defaults to false.\n```\n",
    'author': 'Alfredo Mejia-Narvaez',
    'author_email': 'amejia@astro.unam.mx',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://gitlab.com/chemical-evolution/lvm-dap',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.7,<4.0',
}


setup(**setup_kwargs)

