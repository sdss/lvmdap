Introduction
============

The **LVM-DAP** (Local Volume Mapper Data Analysis Pipeline) is the official data analysis pipeline for the **SDSS-V Local Volume Mapper (LVM)**. It provides routines and scripts to analyze and fit spectral data from LVM observations.

Purpose
-------

LVM-DAP implements the Resolved Stellar Population (RSP) method for spectral analysis, combining:

* **Stellar population synthesis** - Decomposition of observed spectra into stellar template components
* **Emission line fitting** - Detection and measurement of nebular emission lines
* **Non-linear parameter optimization** - Fitting of redshift, velocity dispersion, and dust extinction

Key Features
------------

* YAML-based configuration for reproducible analysis
* Support for multiple input formats (LVM, MaNGA, MUSE, LVMSIM)
* Automated redshift detection
* Diagnostic plotting and output generation
* Parallel processing support via joblib

Main Scripts
------------

* **lvm-dap-conf** - Fits FITS files in LVM format using a predefined configuration file
* **lvm-dap-gen-out-mod** - Generates output models from spectral data
* **lvm-dap-sim** - Simulates spectral data based on predefined parameters

References
----------

* SDSS-V Local Volume Mapper: https://www.sdss.org/sdss5/instruments/lvm/
* pyPipe3D Documentation: http://ifs.astroscu.unam.mx/pyPipe3D/
