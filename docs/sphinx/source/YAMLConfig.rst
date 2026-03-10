YAML Configuration File
=======================

The ``lvm-dap-conf`` command is driven entirely by a YAML configuration file.
All fitting parameters, file paths, and processing options are set here.
Example configuration files can be found in the ``_legacy/`` directory.

Path Resolution
---------------

All string values that begin with ``../`` are automatically replaced at runtime
with the value of ``lvmdap_dir``. This means template and auxiliary file paths
can be written relative to the installation directory regardless of where the
pipeline is run from.

.. code-block:: yaml

   lvmdap_dir: "/home/user/lvmdap"
   rsp-file: "../_fitting_data/stellar-basis.fits.gz"
   # rsp-file is resolved to: /home/user/lvmdap/_fitting_data/stellar-basis.fits.gz


Required Keys
-------------

Paths
~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``output_path``
     - string
     - Directory where all output files are written.
   * - ``lvmdap_dir``
     - string
     - Absolute path to the LVM-DAP installation directory. Used to resolve
       all ``../`` paths in other keys.
   * - ``rsp-file``
     - string
     - Path to the full stellar template basis (FITS). Used for the main
       linear RSP decomposition.
   * - ``rsp-nl-file``
     - string
     - Path to the reduced stellar template basis (FITS). Used for
       non-linear fitting of redshift, velocity dispersion, and dust
       extinction. Typically a smaller/faster subset of ``rsp-file``.
   * - ``emission-lines-file``
     - string
     - Path to the emission line list (``.LVM`` format) used to mask
       gas lines during stellar continuum fitting.
   * - ``config-file``
     - string
     - Path to the pyPipe3D emission line system configuration file
       (``.config`` format). Defines which emission line systems are
       fitted after the stellar continuum is subtracted.
   * - ``mask_file``
     - string
     - Path to a file listing wavelength bands to exclude from fitting
       entirely (e.g. telluric features). Set to ``none`` to disable.

Instrumental Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``sigma-inst``
     - float
     - Instrumental spectral resolution in Angstroms (Gaussian sigma).
       The model spectra are convolved to match: :math:`\sigma_\mathrm{obs}^2 = \sigma_\mathrm{mod}^2 + \sigma_\mathrm{inst}^2`.
       Typical value for LVM is ``1``.
   * - ``input-fmt``
     - string
     - Input data format. Currently only ``rss`` is supported (LVM SFrame RSS format).

Non-linear Fitting Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each of the three non-linear parameters is specified as a 4-element list:
``[guess, delta, min, max]``.

* **guess** — initial value passed to the fitter
* **delta** — search step size
* **min / max** — allowed parameter boundaries

.. code-block:: yaml

   redshift:
     - 0.0       # guess
     - 0.00005   # delta
     - -0.0003   # min
     - 0.0003    # max

   sigma:         # velocity dispersion in km/s
     - 1
     - 5
     - 0.1
     - 30

   AV:            # V-band dust extinction (magnitudes)
     - 0
     - 0.3
     - 0.0
     - 2.5

Wavelength Ranges
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``w-range``
     - [float, float]
     - Wavelength window (Å) for the full RSP stellar decomposition:
       ``[w_min, w_max]``.
   * - ``w-range-nl``
     - [float, float]
     - Wavelength window (Å) for non-linear parameter fitting.
       Should be narrower than ``w-range``, centred on absorption features
       sensitive to kinematics (e.g. Balmer lines: ``[3800, 4300]``).

Fitting Control
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``bin-nl``
     - int
     - Spectral pixel binning applied during non-linear fitting.
       ``1`` = no binning. Higher values speed up the kinematic/extinction
       search at the cost of resolution.
   * - ``bin-AV``
     - int
     - Spectral pixel binning used specifically when fitting dust extinction.
       Typical value: ``50``.
   * - ``ignore_gas``
     - bool
     - If ``True``, skip emission line fitting entirely (stellar continuum only).
   * - ``single-gas-fit``
     - bool
     - If ``True``, fit emission lines once. If ``False``, detect and refine
       iteratively.
   * - ``sigma-gas``
     - float
     - Initial guess for the gas velocity dispersion in Angstroms.
       Used as the starting width for emission line detection.
   * - ``single_rsp``
     - bool
     - If ``True``, fit only the single best-matching stellar template
       (fast mode). If ``False``, perform the full linear decomposition
       over the complete template basis.
   * - ``clear_outputs``
     - bool
     - If ``True``, remove any existing output files with the same label
       before starting.

Output Control
~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``do_plots``
     - int
     - Generate diagnostic summary plots. ``0`` = no plots, ``1`` = save plots.
   * - ``flux-scale-org``
     - [float, float]
     - Flux scale range ``[min, max]`` for output plot colour maps.
       Set to ``[-1, 1]`` to trigger auto-scaling.


Optional Keys
-------------

Emission Line Files
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``emission-lines-file-long``
     - string
     - Extended emission line list used for the refinement pass after
       the initial stellar fit. Defaults to the same file as
       ``emission-lines-file`` if omitted.
   * - ``emission-lines-file-sky``
     - string
     - Sky emission line list. Used to mask sky lines before
       auto-redshift detection (see ``auto_z_sc``).
   * - ``mask-file``
     - string
     - Alternative spelling of ``mask_file`` (hyphenated). Both forms
       are accepted; ``mask_file`` takes precedence.
   * - ``ask-file``
     - string
     - Legacy alias for the mask file. Use ``mask_file`` or ``mask-file``
       in new configurations.

Spatial / Spectral Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``ny_range``
     - [int, int]
     - Subset of fibers (rows) to process: ``[start, end]``. If omitted,
       all fibers are analysed.
   * - ``nx_range``
     - [int, int]
     - Subset of spectral pixels (columns) to keep: ``[start, end]``.
       If omitted, the full wavelength range is used.
   * - ``only-integrated``
     - bool
     - If ``True``, fit only the spatially integrated spectrum and skip
       individual fiber fitting. Default: ``False``.

Auto-Redshift Detection
~~~~~~~~~~~~~~~~~~~~~~~

When ``auto-redshift: True``, the pipeline cross-correlates the integrated
spectrum against an Hα + [N II] template to obtain a refined redshift
before per-fiber fitting begins.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``auto-redshift``
     - bool
     - Enable automatic redshift estimation. Default: ``False``.
   * - ``auto_z_min``
     - float
     - Minimum redshift for the auto-redshift search. Default: ``-0.003``.
   * - ``auto_z_max``
     - float
     - Maximum redshift for the auto-redshift search. Default: ``0.005``.
   * - ``auto_z_del``
     - float
     - Redshift step size for the search grid. Finer steps are more
       accurate but slower. Default: ``0.00001``.
   * - ``auto_z_bg``
     - bool
     - Subtract a continuum estimate before cross-correlation.
       Default: ``True``.
   * - ``auto_z_sc``
     - bool
     - Mask sky emission lines before cross-correlation (requires
       ``emission-lines-file-sky``). Default: ``True``.

Quality and Fitting Refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``SN_CUT``
     - float
     - Signal-to-noise threshold per fiber. Fibers below this value
       are fitted with ``single_rsp=True`` regardless of the global
       setting. Default: ``3``.
   * - ``SN_CUT_INT``
     - float
     - S/N threshold for the integrated spectrum. If below this value,
       the stellar component is skipped. Default: ``3``.
   * - ``kin_fixed``
     - int
     - Kinematic fixing mode when propagating kinematics from the
       integrated to individual fiber fits.
       ``0`` = all free, ``1`` = fix velocity, ``2`` = fix sigma.
       Default: ``2``.
   * - ``N_MC``
     - int
     - Number of Monte Carlo realisations for error estimation.
       Default: ``20``.
   * - ``n_loops``
     - int
     - Number of fitting iterations. More loops improve convergence
       at the cost of runtime. Default: ``5``.
   * - ``smooth_size``
     - int
     - Kernel width (pixels, must be odd) for median-filter smoothing
       of emission line residuals. Default: ``21``.
   * - ``n_leg``
     - int
     - Order of the Legendre polynomial used for continuum subtraction
       in emission line analysis. Default: ``11``.

Output Options
~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``out-plot-format``
     - string
     - File format for saved plots: ``"pdf"`` or ``"png"``.
       Default: ``"pdf"``.
   * - ``dump_model``
     - bool
     - Save the fitted model spectrum arrays to output files.
       Default: ``False``.

Sky Subtraction
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``sky-hack``
     - bool
     - Apply a custom sky re-evaluation step using the ``SKY`` extension
       of the input RSS file. Deprecated for DRP reductions newer than
       v1.1; set to ``False`` for current LVM data. Default: ``False``.


Full Example
------------

The following is representative of a production configuration for LVM SFrame data:

.. code-block:: yaml

   ---
    output_path: "/home/user/output_dap/"
    lvmdap_dir:  "/home/user/lvmdap"

    # Stellar template bases
    rsp-file:    "../_fitting_data/mstar-stlib-cl-108.fits.gz"
    rsp-nl-file: "../_fitting_data/mstar-stlib-cl-12.fits.gz"

    sigma-inst: 1
    input-fmt: rss

    # Non-linear parameter ranges: [guess, delta, min, max]
    redshift: [0.0, 0.00005, -0.0003, 0.0003]
    sigma:    [1,   5,       0.1,     30   ]   # km/s
    AV:       [0,   0.3,     0.0,     2.5  ]   # mag

    # Wavelength ranges (Angstroms)
    w-range:    [3700, 6900]
    w-range-nl: [3890, 4050]

    # Auxiliary files
    emission-lines-file:     "../_legacy/emission_lines_long_list.LVM"
    emission-lines-file-long: "../_legacy/emission_lines_long_list.LVM"
    emission-lines-file-sky: "../_legacy/emission_lines_sky.LVM"
    config-file: "../_legacy/auto_ssp_LVM.config"
    mask-file:   "../_legacy/mask_bands_LVM.txt"
    mask_file:   none

    # Fitting control
    bin-nl: 1
    bin-AV: 50
    ignore_gas: False
    single-gas-fit: True
    sigma-gas: 0.8
    single_rsp: False
    clear_outputs: True
    kin_fixed: 0
    N_MC: 30
    n_loops: 10
    SN_CUT: 20
    SN_CUT_INT: 0.1
    smooth_size: 21
    n_leg: 21

    # Auto-redshift
    auto-redshift: True
    auto_z_min: -0.005
    auto_z_max: 0.01
    auto_z_del: 0.0001
    auto_z_bg: True
    auto_z_sc: True

    # Output
    do_plots: 1
    out-plot-format: "png"
    flux-scale-org: [-1, 1]
    dump_model: False

    # Sky
    sky-hack: False

    # Spatial/spectral selection (commented out = process all)
    # ny_range: [0, 100]
    # nx_range: [100, 5000]
    only-integrated: False
