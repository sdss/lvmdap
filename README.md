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

LVM_DAP     => The directory in which the DAP is installed
LVM_DAP_CFG => The directory in which the configuration files are stored
               nominally ${LVM_DAP}/_legacy
LVM_DAP_RSP => The directory in which the RSP (stellar templates) are stored
               that would be the directory were is stored the content of the "lvmdap_fitting-data" URL
	       e.g., export LVM_DAP_RSP="_fitting_data";

If you want to run the notebooks in the [testing notebooks](https://gitlab.com/chemical-evolution/lvm-dap/-/tree/master/notebooks/dap-testing) section, you will need also to download the required data stored in [google drive](https://drive.google.com/drive/folders/1FwEGhTxnAyM7ld6nsSorG15Dq3LVH1I9?usp=sharing) into the `lvm-dap` directory. Ask for access to amejia@astro.unam.mx.

If the installation went successfully (and you downloaded the data) your tree directory should look like:

    ├── dist
    ├── lvmdap
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

and you should be able to run the following example:

lvm-dap _fitting-data/simulations/ssps/fsps-ssp-mist-miles-1p00000_0p00100gyr.txt _fitting-data/_basis_mastar_v2/stellar-basis-spectra-100.fits.gz 0.33283937056926377 1p00000_0p00100gyr --mask-file _fitting-data/_configs/MaNGA/mask_elines.txt --emission-lines-file _fitting-data/_configs/MaNGA/emission_lines_long_list.MaNGA --w-range 3600 10000 --w-range-nl 3600 4700 --redshift 0 0 0 0 --sigma 0 0 0 0 --AV 0 0 0 0

lvm-dap _fitting-data/simulations/ssps/fsps-ssp-mist-miles-1p00000_0p00100gyr.txt _fitting-data/_basis_mastar_v2/stellar-basis-spectra-100.fits.gz 0.33283937056926377 1p00000_0p00100gyr --mask-file _fitting-data/_configs/MaNGA/mask_elines.txt --emission-lines-file _fitting-data/_configs/MaNGA/emission_lines_long_list.MaNGA --w-range 3600 10000 --w-range-nl 3600 4700 --redshift 0 0 0 0 --sigma 0 0 0 0 --AV 0 0 0 0

which will produce the following output files:

    1p00000_0p00100gyr                            1p00000_0p00100gyr.autodetect.8400_9999.conf           coeffs_1p00000_0p00100gyr
    1p00000_0p00100gyr.autodetect.3600_5199.conf  1p00000_0p00100gyr.autodetect.auto_ssp_several.config  elines_1p00000_0p00100gyr
    1p00000_0p00100gyr.autodetect.5200_6799.conf  1p00000_0p00100gyr.autodetect.emission_lines.txt       output.1p00000_0p00100gyr.fits.gz
    1p00000_0p00100gyr.autodetect.6800_8399.conf  1p00000_0p00100gyr.autodetect.mask_elines.txt

# Examples

You can get familiar with the full spectral analysis implemented in `lvm-dap` either running the notebooks in the `notebooks` folder or running the following example in the console:

    lvm-dap CS.LMC_043.RSS.fits.gz _fitting-data/_basis_mastar_v2/stellar-basis-spectra-100.fits.gz 2.31 test --input-fmt rss --error-file e_CS.LMC_043.RSS.fits.gz --rsp-nl-file _fitting-data/_basis_mastar_v2/stellar-basis-spectra-5.fits.gz --w-range 4800 8000 --w-range-nl 4800 6000 --redshift 0.000875 0 -0.5 0.5 --sigma 0 0 0 350 --AV 0 0.01 0 1.6 --sigma-gas 3.7 --emission-lines-file _fitting-data/_configs/MaNGA/emission_lines_long_list.txt -c

This will analyse the MUSE-LMC pointing 43 in RSS format and produce the outputs in the same format as `pyFIT3D`.

