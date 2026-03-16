Directory Structure
===================

After a successful installation, the LVM-DAP directory should be structured as follows:

Top-Level Structure
-------------------

.. code-block:: text

    lvmdap/
    ├── dist/               # Distribution files
    ├── lvmdap/             # Main package source code
    ├── _examples/          # Example data and scripts
    ├── _fitting_data/      # Stellar templates (download separately)
    ├── _legacy/            # Configuration file examples
    ├── notebooks/          # Jupyter notebooks for interactive use
    ├── docs/               # Documentation
    ├── poetry.lock         # Poetry dependency lock file
    ├── pyproject.toml      # Project configuration
    ├── README.md           # Project readme
    └── setup.py            # Setup script

Source Code Structure
---------------------

.. code-block:: text

    lvmdap/
    ├── _cmdline/              # CLI entry points
    │   ├── dap.py             # Main DAP pipeline
    │   ├── gen_output_model.py # Output model generation
    │   ├── sim_spec_rsp.py    # Spectral simulation
    │   ├── clean_outputs.py   # Cleanup utility
    │   ├── cube2map.py        # Cube to map conversion
    │   ├── coadd_cubes.py     # Cube coaddition
    │   ├── preprocess_muse.py # MUSE preprocessing
    │   ├── preprocess_manga.py # MaNGA preprocessing
    │   ├── gas_cube_extractor.py # Gas cube extraction
    │   └── mwm_dap.py         # MWM survey processing
    ├── modelling/             # Core fitting/synthesis logic
    │   ├── synthesis.py       # StellarSynthesis class
    │   ├── auto_rsp_tools.py  # RSP fitting tools
    │   └── ingredients.py     # StellarModels class
    ├── analysis/              # Analysis utilities
    │   ├── plotting.py        # Visualization functions
    │   ├── stats.py           # Statistical utilities
    │   └── img_scale.py       # Image scaling utilities
    ├── pyFIT3D/               # Embedded pyFIT3D library
    │   ├── common/            # Shared utilities
    │   └── modelling/         # pyFIT3D modeling core
    ├── dap_tools.py           # Main utility module
    ├── flux_elines_tools.py   # Emission line flux tools
    ├── config.py              # Configuration management
    └── io.py                  # I/O utilities

Configuration Files
-------------------

The ``_legacy/`` directory contains example configuration files:

* ``lvm-dap_fast.yaml`` - Fast configuration example
* ``emission_lines_strong.LVM`` - Emission line lists
* ``auto_ssp_LVM.config`` - SSP fitting configuration
* ``mask_bands_LVM.txt`` - Wavelength masking bands

Example Data
------------

The ``_examples/data/`` directory contains example FITS files for testing.
Download additional example data from `<https://tinyurl.com/mudr6yw7>`_.
