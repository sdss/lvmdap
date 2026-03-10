# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LVM-DAP (Local Volume Mapper Data Analysis Pipeline) is the official SDSS-V data analysis pipeline for spectral fitting of LVM observations. It implements Resolved Stellar Population (RSP) fitting using stellar population synthesis, emission line fitting, and non-linear parameter optimization (redshift, velocity dispersion, dust extinction).

**Key dependency**: Requires pyPipe3D (http://ifs.astroscu.unam.mx/pyPipe3D/) to be installed first.

## Build and Installation

```bash
# Create conda environment
conda create --name lvmdap python=3.11
conda activate lvmdap

# Install (matplotlib version is critical for compatibility)
pip install matplotlib==3.7.3
pip install .

# Developer install
pip install -e .
```

External data required: Download stellar templates from https://tinyurl.com/mudr6yw7 and place in `_fitting_data/`.

**There is no test suite.** Verify installation by running:
```bash
lvm-dap-conf _examples/data/lvmSFrame-example.fits.gz test-run _legacy/lvm-dap_fast.yaml
```

## CLI Commands

| Command | Purpose |
|---------|---------|
| `lvm-dap-conf <fits_file> <label> <config.yaml>` | Main production workflow - fits spectra using YAML config |
| `lvm-dap-gen-out-mod <fits_file> <dap_table> <label> <config.yaml>` | Generate output models from fitted data |
| `lvm-dap-sim <fits_file> <label> <config.yaml>` | Simulate spectra based on fitted parameters |
| `cube2map <fits_file> ...` | Convert cube to 2D map |
| `coadd-cubes ...` | Coadd spectral cubes |
| `clean-outputs ...` | Cleanup output files |
| `preprocess-muse ...` | Preprocess MUSE data to LVM format |
| `preprocess-manga ...` | Preprocess MaNGA data to LVM format |
| `gas-cube-extractor ...` | Extract gas cube |
| `mwm-dap ...` | MWM survey processing |
| `lvm-dap` | Deprecated direct fitting command |

## Building Documentation

```bash
cd docs/sphinx
make html          # Build HTML docs to docs/sphinx/html/
make livehtml      # Live-reload server for doc development
make clean         # Remove built docs
```

## Architecture

### Core Modules

```
lvmdap/
├── _cmdline/           # CLI entry points
│   ├── dap.py          # Main pipeline orchestration (_dap_yaml, _main)
│   ├── gen_output_model.py
│   ├── sim_spec_rsp.py
│   ├── cube2map.py
│   ├── coadd_cubes.py
│   ├── preprocess_muse.py / preprocess_manga.py
│   ├── gas_cube_extractor.py
│   └── mwm_dap.py
├── modelling/          # Spectral fitting core
│   ├── synthesis.py    # StellarSynthesis - main fitting class
│   ├── auto_rsp_tools.py # RSP fitting functions (auto_rsp_elines_rnd)
│   └── ingredients.py  # StellarModels - template management
├── analysis/           # Post-processing utilities
│   ├── plotting.py
│   ├── stats.py        # PDF calculations, statistical moments
│   └── img_scale.py    # Image scaling utilities
├── pyFIT3D/            # Embedded fitting library (pyFIT3D common + modelling)
├── dap_tools.py        # Main utilities (129KB) - RSS loading, plotting, I/O
├── flux_elines_tools.py # Emission line flux handling
├── io.py               # I/O utilities
└── config.py           # ConfigRSP - YAML configuration class
```

**Note**: `_cmdline/` contains date-stamped backup files (e.g., `dap.231118.py`, `dap.back.py`) — these are snapshots, not active code.

### Data Flow

1. **Input**: RSS/FITS spectral data + YAML configuration
2. **Processing**: `dap.py` → `auto_rsp_elines_rnd()` → `StellarSynthesis.fit()`
3. **Output**: DAP FITS files, coefficient tables, emission line tables, diagnostic plots

### Key Classes

- **StellarSynthesis** (`modelling/synthesis.py`): Main fitting class, inherits from StPopSynt. Handles spectral decomposition with kinematics and dust extinction.
- **StellarModels** (`modelling/ingredients.py`): Manages stellar template basis (SSP models). Methods: `get_model_from_coeffs()`, `moments_from_coeffs()`.
- **ConfigRSP** (`config.py`): Loads YAML configuration, inherits from ConfigAutoSSP.

### Important Functions

- `dap_tools.load_LVM_rss()` - Load LVM RSS format data
- `dap_tools.read_rsp()` - Read RSP basis files
- `dap_tools.read_coeffs_RSP()` - Read fitting coefficients
- `auto_rsp_tools.auto_rsp_elines_rnd()` - Core fitting routine

## Configuration

YAML config files control all fitting parameters. Example configs in `_legacy/`. Key sections:

```yaml
output_path: "/path/to/output/"
rsp-file: "path/to/stellar-templates.fits.gz"
rsp-nl-file: "path/to/stellar-templates.fits.gz"
sigma-inst: 1           # Instrumental resolution in Angstroms

# Non-linear parameters: [guess, delta, min, max]
redshift: [0.0, 0.0001, -0.0003, 0.0003]
sigma: [1, 5, 1, 15]   # Velocity dispersion in km/s
AV: [0, 0.3, 0.0, 1.5] # Dust extinction (V-band)

w-range: [3700, 9500]
w-range-nl: [3800, 4200]
emission-lines-file: path/to/emission_lines.LVM
mask-file: path/to/mask_bands.txt
do_plots: 1
only-integrated: True
auto-redshift: True
```

## Environment Variables

```bash
export LVM_DAP="/path/to/lvmdap"
export LVM_DAP_CFG="$LVM_DAP/_legacy"
export LVM_DAP_RSP="_fitting_data"
```

## Data Formats

- **Input**: FITS files in LVM RSS format (also supports MaNGA, MUSE, LVMSIM)
- **Templates**: FITS files with stellar basis spectra (in `_fitting_data/`)
- **Output**: DAP FITS files with fitted parameters, emission line tables, coefficient tables

## Interactive Notebooks

`notebooks/` contains Jupyter notebooks for interactive exploration:
- `lvm-dap-conf.ipynb` — interactive pipeline run
- `lvm-dap-gen-out-mod.ipynb` — output model generation
- `plot_fitting_output_spectra.ipynb` — visualize fitting results
- `Plot_PDF_from_DAP.ipynb` / `Plot_PDF_from_coeffs.ipynb` — PDF analysis
- `MaStar_Clustering.ipynb` — stellar library clustering
