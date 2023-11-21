# Introduction

This is the SDSS-V Local Volume Mapper (LVM) Data Analysis Pipeline (DAP) official repository.

It comprises a set of routines, and two main scripts:

lvm-dap      : Allows to fit individual and RSS spectra, using a set of command-line given entries

lvm-dap-conf : Allows to fit a fits-file in the LVM format as produced by the LVM-DRP, i.e., the RSS
	       corresponding to a single pointing, using a configuration file to define the parametes.


# Installation

We recommend installing in a virtual environment to avoid dependencies crashing. Some popular options are [miniconda](https://docs.conda.io/en/latest/miniconda.html), [venv](https://docs.python.org/3.8/library/venv.html), [pipenv](https://pipenv.pypa.io/en/latest/). We recommend venv.

The LVM-dap heavy depends on pyPipe3D (http://ifs.astroscu.unam.mx/pyPipe3D/ , Lacerda et al. 2022).
That software should be installed before installing the LVM-dap.

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

(2) Go to the _examples directory and run the command line. Download the lvmCFrame-00006109.fits.gz
file in the LVM format from the following location:

http://ifs.astroscu.unam.mx/LVM/lvmdap_fitting-data/data/ 

or from the official repositories of the LVM. Then, just run the following script (assuming
that you place the data in the "data" directory.


lvm-dap-conf data/lvmCFrame-00006109.fits.gz dap-4-00006109 ../_legacy/lvm-dap_fast.yaml

'''
Both two scripts, `lvm-dap` and 'lvm-dap-conf', implement the Resolved Stellar Population method (Mejia-Narvaez+, in prep.).

# Usage

```
usage: lvm-dap-conf [-h] [-d] lvm_file label config_yaml

lvm_file:  Fitsfile in the LVM format, comprising the following extensions:

No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU     301   () 
  1  FLUX          1 ImageHDU        22   (12401, 1944)   float32   
  2  ERROR         1 ImageHDU         8   (12401, 1944)   float32   
  3  MASK          1 ImageHDU         8   (12401, 1944)   uint8   
  4  WAVE          1 ImageHDU         7   (12401,)   float32   
  5  FWHM          1 ImageHDU         8   (12401, 1944)   float32   
  6  SKY           1 ImageHDU         8   (12401, 1944)   float32   
  7  SKY_ERROR     1 ImageHDU         8   (12401, 1944)   float32   
  8  SUPERSKY      1 BinTableHDU     24   1458345R x 6C   [E, E, E, J, J, 4A]   
  9  SLITMAP       1 BinTableHDU     43   1944R x 17C   [K, K, 3A, K, 8A, 5A, K, 4A, D, D, D, 6A, 8A, K, 17A, K, K]   

label: string to label the current run

config_ymal: config file including the entries required to fit the spectra

---
# output directory
 output_path: "/disk-a/sanchez/LVM/LVM/ver_231113/output_dap/"
# LVM-DAP software directory  
 lvmdap_dir: "/home/sanchez/sda2/code/python/lvmdap"
# rsp-file for the full stellar decomposition 
 rsp-file: "../_fitting-data/_basis-binned_mastar/stellar-basis-spectra-binned-4-mastar.fits.gz"
# rsp-file for the derivation of the non-linear (NL) parameters (vel, vel_disp, Av)
 rsp-nl-file: "../_fitting-data/_basis-binned_mastar/stellar-basis-spectra-binned-4-mastar.fits.gz"
# approximate instrumental resolution in AA
 sigma-inst: 1
# input format for fitting the data (deprecated)
input-fmt: rss
# redshift range (or velocity) to explore in the NL
 redshift:
  -  0.0      # guess
  -  0.0001   # delta
  -  -0.0003  # min
  -  0.0003   # max
# velocity dispersion in km/s range to explore in the NL
 sigma:
  - 1  # guess
  - 5  # delta
  - 1  # min
  - 15 # max
# Dust attenuation in the V-band
 AV:
  - 0    # guess
  - 0.3  # delta
  - 0.0  # min
  - 1.5  # max
# list of strong emission lines to analyze
 emission-lines-file: ../_legacy/emission_lines_strong.LVM 
# emission-lines-file-long: ../_legacy/emission_lines_long_list.LVM
# full list of emission lines to analyze using the non-parametric fitting procedure
emission-lines-file-long: ../_legacy/emission_lines_strong.LVM
# Wavelength range in which the full stellar decomposition is peformed
w-range:
  - 3700
  - 9500
# Wavelength range in which the kinematics parameters (vel,vel_disp) are fitted
w-range-nl:
  - 3800
  - 4200
# File comprising the wavelength bands to be masked (e.g., the bands between spectrographs)
 mask-file: ../_legacy/mask_bands_LVM.txt
# File with a set wavelengths to mask (narrow range around the designed wavelength)
 mask_file: none
# Configuration file defining the emission lines to fit during the RSP fitting.
 config-file: ../_legacy/auto_ssp_LVM.config
# Flag indicating if any previous file with the same label should be removed
 clear_outputs: True
# not used so far. Possible binning the wavelength range to speed-up the fitting procedure
 bin-nl: 3
# not used so far. Possible binning in the derivation of Av to speed-up the process
 bin-AV: 50
# Flag to ignore the emission line fitting during the RSP analysis
 ignore_gas: False
# Initial guess for the velocity dispersion for the emission lines analysis in AA
 sigma-gas: 0.8
# Flag to perform the emission line fitting one once during the RSP analysis
single-gas-fit: True
# Flag indicating if instead of a decomposition the code looks for the best RSP in the template that
# matchs with the observed spectra
single_rsp: False
# Flag indicating wether to generate plots or not along the fitting process
 do_plots: 1
# Flux range for plotting. When set ot [-1,1] it will automatically determine the required range
 flux-scale-org:
  - -1 # min
  - 1  # max
# Range of fibers to fit in case that you do not want to fit the full RSS
 ny_range:
  - 500
  - 503
# Not used so far: range of spectral pixels to perform the fitting
 nx_range:
  - 100
  - 3000

...
 	   
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


