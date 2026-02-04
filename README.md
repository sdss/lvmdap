# Local Volume Mapper Data Analysis Pipeline (LVM-DAP)

## Introduction
The **LVM-DAP** is the official data analysis pipeline for the **SDSS-V Local Volume Mapper (LVM)**. It provides routines and scripts to analyze and fit spectral data from LVM observations.

### Main Scripts:
- **`lvm-dap-conf`**: Fits FITS files in LVM format using a predefined configuration file.
- **`lvm-dap-gen-out-mod`**: Generates output models from spectral data.
- **`lvm-dap-sim`**: Simulates spectral data based on predefined parameters.
- **`lvm-dap`**: Runs spectral fitting for LVM data.

---
## Installation

### Recommended Setup
It is recommended to install **LVM-DAP** in a virtual environment to avoid dependency issues. Options include:
- [Conda/Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Recommended)
- [venv](https://docs.python.org/3.8/library/venv.html) 
- [pipenv](https://pipenv.pypa.io/en/latest/)

### Prerequisites
**LVM-DAP** depends on **pyPipe3D** ([pyPipe3D](http://ifs.astroscu.unam.mx/pyPipe3D/), Lacerda et al. 2022). Install pyPipe3D before proceeding.

### Installation Steps
```bash
# Create and activate a Conda environment
conda create --name lvmdap python=3.11
conda activate lvmdap

# In some cases it is needed to install the following version of 
# matplotlib
pip install matplotlib==3.7.3


# Clone LVM-DAP
# For users who only need to run the pipeline:
git clone https://github.com/sdss/lvmdap.git

# For developers who will push changes back to the repository:
git clone git@github.com:sdss/lvmdap.git

# Install LVM-DAP
cd lvmdap
pip install . --user




```

### Additional Data
Download and store the required fitting data from:
[Download LVM fitting data](https://tinyurl.com/mudr6yw7)

### Environmental Variables
Not needed

### Directory Structure
After a successful installation, the directory should be structured as follows:
```
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
└── setup.py
```

---
## Running Tests

### Test 1: Running DAP main script on a FITS File

1. Download the example FITS file:
   - [Download example FITS](https://tinyurl.com/mudr6yw7)

2. Place it inside `_examples/data/`

3. Edit the file '_legacy/lvm-dap_fast.yaml' chaning the local directories to match the ones in your computer

4. Run the following command:

```bash

lvm-dap-conf _examples/data/lvmSFrame-example.fits.gz dap-4-example _legacy/lvm-dap_fast.yaml

```

### Test 2: Running the notebook lvm-dap-conf.ipynb located in the notebook
director.


1. Download the example FITS file:
   - [Download example FITS](https://tinyurl.com/mudr6yw7)

2. Place it inside a local directory

3. Copy the notebook into that directory.

4: Copy the file: _examples/data/lvmSFrame-example.fits.gz dap-4-example _legacy/lvm-dap_fast.yaml
    into that directory

3. Edit the notebook and change the 'lvm_file', 'label' and 'yaml_path' variables.

4. Run the notebook cell by cell



---
## Usage

### `lvm-dap-conf` Command

#### Example YAML Configuration
```yaml
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
  out-plot-format: "pdf"
# Format of the output plot format : "png"
 only-integrated: True
# Fit only the integrated spectrum or not
 auto-redshift: True
# Find the central redshift automatically using the integrated spectrum
auto_z_min: -0.005
auto_z_max: 0.01
auto_z_del: 0.0001
# Parameters that define the range min,max and delta
auto_z_bg: True
# Subtract the background to detect Ha+NII in the integrated spectrum to find the central redshift
auto_z_sc: True
# Correct the residual background for the integrated spectrum
sky-hack: False
# Appply a correction of the sky spectrum (deprecated, useful for DRP<1.0.3)
SN_CUT: 20
# SN cut for the multi-RSP analysis
SN_CUT_INT: 0.1
# SN cut for the single-RSP analysis
```

```bash
lvm-dap-conf [-h] [-d] lvm_file label config_yaml
```
- **`lvm_file`**: FITS file in LVM format
- **`label`**: Label for the current run
- **`config_yaml`**: Configuration file for fitting parameters

### `lvm-dap-gen-out-mod` Command
```bash
lvm-dap-gen-out-mod [-h] [-d] [-output_path OUTPUT_PATH] [--plot PLOT] [--flux-scale min max] lvm_file DAP_table_in label config_yaml
```
- **`lvm_file`**: Input LVM spectrum that was fitted
- **`DAP_table_in`**: DAP file generated by the LVM-DAP fitting
- **`label`**: Label for the output file (e.g., LABEL.output.fits.gz)
- **`config_yaml`**: Configuration file with fitting parameters

**Options:**
- `-d, --debug` : Enable debugging mode (default: False)
- `-output_path OUTPUT_PATH` : Directory to store the results
- `--plot PLOT` : Whether to plot (1) or not (0, default). If 2, saves a plot without displaying on screen
- `--flux-scale min max` : Scale of the flux in the input spectrum

### `lvm-dap-sim` Command
```bash
lvm-dap-sim [-h] [-n_sim n_sim] [-n_st n_st] [-f_st f_scale_st] [-f_el f_scale_el] [-dap_fwhm dap_fwhm] [-d] lvm_file label config_yaml
```
- **`lvm_file`**: Input LVM spectrum for the simulation
- **`label`**: Label for the current run
- **`config_yaml`**: Configuration file with fitting parameters

**Options:**
- `-h, --help` : Show help message and exit
- `-n_sim n_sim` : Number of simulated spectra (default: 10)
- `-n_st n_st` : Number of stars included in the model (default: 10)
- `-f_st f_scale_st` : Scaling factor applied to stellar population spectra (~S/N level, default: 1.0)
- `-f_el f_scale_el` : Scaling factor applied to emission lines relative to reference (default: 1.0)
- `-dap_fwhm dap_fwhm` : Scaling factor applied to DAP non-parametric dispersion for emission lines (default: 2.354)
- `-d, --debug` : Enable debugging mode (default: False)

### `lvm-dap` Command (Deprecated: Do not use it)

`lvm-dap` is now deprecated. Users are encouraged to use `lvm-dap-conf`, `lvm-dap-gen-out-mod`, or `lvm-dap-sim` instead.
```bash
lvm-dap [-h] [--config-file CONFIG_FILE] spectrum-file rsp-file sigma-inst label
```
- **`spectrum-file`**: Input spectrum to fit
- **`rsp-file`**: The resolved stellar population basis
- **`sigma-inst`**: The resolution downgrade parameter
- **`label`**: Label for the run

---
## Troubleshooting & Additional Notes
- If you encounter installation issues, check **numpy version** compatibility.
- Ensure **pyPipe3D** is installed **before** LVM-DAP.
- For support, refer to the **[official repository](https://github.com/sdss/lvmdap)**.

---
## References
- SDSS-V Local Volume Mapper: [LVM Project](https://www.sdss.org/sdss5/instruments/lvm/)
- [pyPipe3D Documentation](http://ifs.astroscu.unam.mx/pyPipe3D/)

