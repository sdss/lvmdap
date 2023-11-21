# Introduction

This is the SDSS-V Local Volume Mapper (LVM) Data Analysis Pipeline (DAP) official repository.

The main and only script, `lvm-dap`, implements the Resolved Stellar Population method (Mejia-Narvaez+, in prep.). Instructions on how to run this code below.

# Usage

```
usage: lvm-dap [-h] [--input-fmt INPUT_FMT] [--error-file ERROR_FILE] [--config-file CONFIG_FILE]
               [--emission-lines-file EMISSION_LINES_FILE] [--mask-file MASK_FILE] [--sigma-gas SIGMA_GAS] [--ignore-gas]
               [--rsp-nl-file RSP_NL_FILE] [--plot PLOT] [--flux-scale min max] [--w-range wmin wmax] [--w-range-nl wmin2 wmax2]
               [--redshift input_redshift delta_redshift min_redshift max_redshift]
               [--sigma input_sigma delta_sigma min_sigma max_sigma] [--AV input_AV delta_AV min_AV max_AV]
               [--ext-curve {CCM,CAL}] [--RV RV] [--single-rsp] [--n-mc N_MC] [-o path] [-c] [-v] [-d]
               spectrum-file rsp-file sigma-inst label

Run the spectral fitting procedure for the LVM

positional arguments:
  spectrum-file         input spectrum to fit
  rsp-file              the resolved stellar population basis
  sigma-inst            the standard deviation in wavelength of the Gaussian kernel to downgrade the resolution of the models to
                        match the observed spectrum. This is: sigma_inst^2 = sigma_obs^2 - sigma_mod^2
  label                 string to label the current run

optional arguments:
  -h, --help            show this help message and exit
  --input-fmt INPUT_FMT
                        the format of the input file. It can be either 'single' or 'rss'. Defaults to 'single'
  --error-file ERROR_FILE
                        the error file
  --config-file CONFIG_FILE
                        the configuration file used to set the parameters for the emission line fitting
  --emission-lines-file EMISSION_LINES_FILE
                        file containing emission lines list
  --mask-file MASK_FILE
                        the file listing the wavelength ranges to exclude during the fitting
  --sigma-gas SIGMA_GAS
                        the guess velocity dispersion of the gas
  --ignore-gas          whether to ignore gas during the fitting or not. Defaults to False
  --rsp-nl-file RSP_NL_FILE
                        the resolved stellar population *reduced* basis, for non-linear fitting
  --plot PLOT           whether to plot (1) or not (0, default) the fitting procedure. If 2, a plot of the result is store in a
                        file without display on screen
  --flux-scale min max  scale of the flux in the input spectrum
  --w-range wmin wmax   the wavelength range for the fitting procedure
  --w-range-nl wmin2 wmax2
                        the wavelength range for the *non-linear* fitting procedure
  --redshift input_redshift delta_redshift min_redshift max_redshift
                        the guess, step, minimum and maximum value for the redshift during the fitting
  --sigma input_sigma delta_sigma min_sigma max_sigma
                        same as the redshift, but for the line-of-sight velocity dispersion
  --AV input_AV delta_AV min_AV max_AV
                        same as the redshift, but for the dust extinction in the V-band
  --ext-curve {CCM,CAL}
                        the extinction model to choose for the dust effects modelling. Choices are: ['CCM', 'CAL']
  --RV RV               total to selective extinction defined as: A_V / E(B-V). Default to 3.1
  --single-rsp          whether to fit a single stellar template to the target spectrum or not. Default to False
  --n-mc N_MC           number of MC realisations for the spectral fitting
  -o path, --output-path path
                        path to the outputs. Defaults to '/disk-a/mejia/Research/UNAM/lvm-dap'
  -c, --clear-outputs   whether to remove or not a previous run with the same label (if present). Defaults to false
  -v, --verbose         if given, shows information about the progress of the script. Defaults to false.
  -d, --debug           debugging mode. Defaults to false.
```

# Installation

We recommend installing in a virtual environment to avoid dependencies crashing. Some popular options are [miniconda](https://docs.conda.io/en/latest/miniconda.html), [venv](https://docs.python.org/3.8/library/venv.html), [pipenv](https://pipenv.pypa.io/en/latest/). We recommend venv.

Once you have created a virtual environment (if you chose to do so), simply run the following commands:

git clone https://github.com/sdss/lvmdap

cd lvmdap

pip install .


Then you need to download the content of the following directory in your computer:

http://ifs.astroscu.unam.mx/LVM/lvmdap_fitting-data/

We recommend you to define three environmental variables:

LVM_DAP     :

The directory in which the DAP is installed

LVM_DAP_CFG :

The directory in which the configuration files are stored
               nominally ${LVM_DAP}/_legacy
	       
LVM_DAP_RSP :

The directory in which the RSP (stellar templates) are stored
               that would be the directory were is stored the content of the "lvmdap_fitting-data" URL
	       e.g., export LVM_DAP_RSP="_fitting_data";

We will assume hereafter that LVM_DAP_RSP corresponds to "_fitting_data" for simplicity.


If the installation went successfully (and you downloaded the data) your tree directory should look like:

    ├── dist
    ├── lvmdap
    ├── _examples
    ├── _fitting_data
    ├── _legacy    
    ├── notebooks
    ├── poetry.lock
    ├── pyproject.toml
    ├── README.md
    ├── README.rst
    ├── run-fsps
    ├── run-fsps-MaNGA
    ├── run-fsps-MaNGA-v2
    └── setup.py

and you should be able to run the following examples

# Tests

(1) Go to the _examples directory and run the script fit_6109_strong.sh. If everything runs ok,
the you have the required files to run the DAP.

(2) Go to the _examples directory and run the command line:

lvm-dap-conf data/lvmCFrame-00006109.fits.gz dap-4-00006109 ../_legacy/lvm-dap_fast.yaml



