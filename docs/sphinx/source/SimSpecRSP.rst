Pipeline Workflow: lvm-dap-sim
================================

``lvm-dap-sim`` generates realistic simulated LVM spectra with known input
parameters. It uses a reference model output (from ``lvm-dap-gen-out-mod``)
and a DAP results table (from ``lvm-dap-conf``) to produce ``n_sim``
synthetic spectra, each with randomly drawn stellar populations, kinematics,
and noise realisations. The output is a valid LVM RSS FITS file that can be
fed back into ``lvm-dap-conf`` to test and characterise pipeline performance.

Entry point: ``lvmdap/_cmdline/sim_spec_rsp.py``, function ``_main``.

Syntax
------

.. code-block:: bash

   lvm-dap-sim [-h] [-n_sim N] [-n_st N] [-f_st F] [-f_el F]
               [-dap_fwhm F] [-d] [--plot PLOT] [--flux-scale min max]
               lvm_file label config_yaml

Arguments
---------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Argument
     - Type
     - Description
   * - ``lvm_file``
     - string
     - Original LVM FITS file (the same one used for the reference fit).
       Provides the RSS structure — dimensions, fiber table, wavelength grid,
       error array — that is cloned into the simulated output.
   * - ``label``
     - string
     - Run label. The actual output label is automatically expanded to
       ``sim_{label}_{n_sim}_{n_st}_{f_st}_{f_el}``.
   * - ``config_yaml``
     - string
     - YAML configuration file. Must additionally contain:

       * ``DAP_output_in`` — path to the ``{label}.output.fits.gz`` from
         ``lvm-dap-gen-out-mod``
       * ``DAP_table_in`` — path to the ``{label}.dap.fits.gz`` from
         ``lvm-dap-conf``

Options
~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``-n_sim``
     - ``10``
     - Number of simulated spectra to produce. The first ``n_sim`` good
       science fibers from ``lvm_file`` are used as the carrier structure.
   * - ``-n_st``
     - ``10``
     - Number of stellar templates drawn randomly from the RSP basis to
       build each simulated stellar population.
   * - ``-f_st``
     - ``10.0``
     - Stellar flux scaling factor. Higher values raise the S/N of the
       stellar continuum in the output.
   * - ``-f_el``
     - ``1.0``
     - Emission line scaling factor relative to the reference emission
       line model.
   * - ``-dap_fwhm``
     - ``2.354``
     - Conversion factor applied as :math:`\sigma = D / \mathrm{dap\_fwhm}`
       when synthesising emission line Gaussians from DAP dispersion values.
   * - ``--plot``
     - ``0``
     - ``0`` = no plot, ``1`` = display, ``2`` = save.
   * - ``--flux-scale min max``
     - ``-1 1``
     - Flux scale range for plots.
   * - ``-d / --debug``
     -
     - Print full traceback on error.

Processing Flow
---------------

.. _sim-step1:

Step 1 — Load reference files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three files are opened simultaneously:

1. **``lvm_file``** — the original LVM RSS FITS. This is the *carrier* for
   the simulation: its full HDU structure (dimensions, header, fiber table)
   is cloned into the output. The following extensions are read and filtered
   to good science fibers (via ``read_PT()`` / ``SLITMAP`` mask), then
   further trimmed to the first ``n_sim`` rows:

   * ``FLUX``, ``ERROR``, ``MASK``, ``FWHM``, ``SKY``, ``SKY_ERROR``,
     ``SLITMAP``

2. **``DAP_output_in``** — the 9-plane model FITS from ``lvm-dap-gen-out-mod``.
   Plane 4 (``res_joint_spec``, the full residual) is used as the noise
   template for all ``n_sim`` realisations.

3. **``DAP_table_in``** — the DAP FITS from ``lvm-dap-conf``. Emission line
   parameters (flux, dispersion, velocity) are read from this table to
   construct the reference emission line model.

.. _sim-step2:

Step 2 — Build the reference emission line model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A 2D emission line spectrum (shape ``n_sim × n_wavelengths``) is synthesised
once from the DAP table before the per-simulation loop. For each non-parametric
emission line (columns matching ``e_flux_*`` without ``_pe_``), the flux,
dispersion, and velocity of each fiber are used to evaluate a Gaussian profile
via the local ``eline()`` function:

.. math::

   f(\lambda) = \frac{F}{\sigma\sqrt{2\pi}}
       \exp\!\left[-\frac{1}{2}
           \left(\frac{\lambda - \lambda_\mathrm{rest}(1+V/c)}{\sigma}\right)^2
       \right], \quad \sigma = D / \mathrm{dap\_fwhm}

The resulting ``spec2D_elines`` array is held in memory and added to all
simulated spectra at the end of the loop (see Step 4).

.. _sim-step3:

Step 3 — Per-simulation loop (×n_sim)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each simulation index ``i_sim``:

**Random parameter generation**
  Non-linear parameters are drawn independently from normal distributions
  centred on the midpoint of the allowed range, with standard deviation
  equal to one-third of the full range:

  .. code-block:: python

     z     = normal(0.5*(z_max + z_min),     0.33*(z_max - z_min))
     sigma = normal(0.5*(sig_max + sig_min), 0.33*(sig_max - sig_min))
     AV    = normal(0.5*(AV_max + AV_min),   0.33*(AV_max - AV_min))

  where ``[z_min, z_max]``, ``[sig_min, sig_max]``, and ``[AV_min, AV_max]``
  come from the ``redshift``, ``sigma``, and ``AV`` entries in the YAML config.

**Random stellar population**
  ``n_st`` template indices are chosen without replacement from the RSP basis
  (``rsp-file``). Random weights are assigned, normalised to sum to one:

  .. code-block:: python

     choice = random.choice(n_templates, n_st, replace=False)
     coeffs[choice] = random.rand(n_st)
     coeffs /= sum(coeffs)

**Model fitting via** ``model_rsp_elines_single_main()``
  The randomly drawn parameters and coefficients are passed as fixed inputs to
  ``model_rsp_elines_single_main()`` (``modelling/auto_rsp_tools.py``). This
  function performs a *single* fitting pass with the supplied starting values
  rather than a full non-linear search, producing a best-fit model spectrum
  ``SPS.spectra['model_min']``.

**Noise realisation**
  A noise spectrum is constructed by randomly sampling rows of the reference
  residual cube (plane 4 of ``DAP_output_in``):

  .. code-block:: python

     random_index = (n_fibers * rand(n_wavelengths)).astype(int)
     err_spec_t[i] = residual_cube[random_index[i], i]

  The simulated spectrum for this realisation is:

  .. code-block:: python

     FLUX[i_sim] = SPS.spectra['model_min'] + err_spec_t / f_st

  where ``f_st`` is the ``-f_st`` scaling factor.

The per-realisation fitting outputs (emission lines, coefficients, RSP
summary) are appended to intermediate text files.

.. _sim-step4:

Step 4 — Apply emission lines, rescale, and update errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the loop, the emission line model and flux scaling are applied to
the entire simulated flux array in one step:

.. code-block:: python

   FLUX = (f_st * FLUX + f_el * spec2D_elines) / f_scale

where ``f_scale = 1e16``. This combines the stellar signal (amplified by
``f_st``) with the emission line model (amplified by ``f_el``) and rescales
to physical units.

The ``ERROR`` array is updated to be consistent with the new flux level:

.. code-block:: python

   ERROR = nan_to_num(ERROR, nan=mean_error)
   ERROR += 0.7 * |FLUX| + 0.3 * mean_flux
   ERROR *= mean_flux / mean_error / f_st

This gives errors that scale with both the observed flux (Poisson-like) and
a floor set by the mean background level.

.. _sim-step5:

Step 5 — Write simulated RSS FITS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The modified HDU (with updated ``FLUX``, ``ERROR``, ``MASK``, ``FWHM``,
``SKY``, ``SKY_ERROR``, and ``SLITMAP`` entries trimmed to ``n_sim`` fibers)
is written to ``sim_{label}.fits`` using ``hdu_org.writeto()``. Because the
HDU structure of the original ``lvm_file`` is preserved, this file can be
passed directly to ``lvm-dap-conf`` as a normal LVM input.

.. _sim-step6:

Step 6 — Assemble output DAP table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The intermediate text output files written during the loop are read back,
columns are renamed to match DAP conventions (e.g. ``Av`` → ``Av_st``,
``disp`` → ``disp_st``), and the RSP, coefficient, and emission line tables
are merged into a single ``{label}.dap.ecsv`` file. This records the *true*
input parameters for each simulation, enabling recovery accuracy tests.

FITS Extension Usage
---------------------

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Extension (from ``lvm_file``)
     - Used?
     - How
   * - ``FLUX``
     - Yes
     - Rows filtered to science fibers then trimmed to ``n_sim``. Each
       row is overwritten with the simulated spectrum in Step 4.
   * - ``ERROR``
     - Yes
     - Rows filtered and trimmed. Replaced with a physically motivated
       error array in Step 4.
   * - ``MASK``
     - Yes
     - Rows filtered and trimmed. Carried through into the output FITS
       unchanged (preserves the fiber quality flags).
   * - ``FWHM``
     - Yes
     - Rows filtered and trimmed. Written to output as-is; provides the
       LSF information for downstream analysis.
   * - ``SKY``
     - Yes
     - Rows filtered and trimmed. Written to output unchanged.
   * - ``SKY_ERROR``
     - Yes
     - Rows filtered and trimmed. Written to output unchanged.
   * - ``SLITMAP``
     - Yes
     - Read by ``read_PT()`` to derive the science-fiber mask; rows
       filtered and trimmed. Written to output so the simulated file is
       a valid LVM RSS file.
   * - ``IVAR``
     - No
     - Not referenced. Errors are taken from ``ERROR`` and re-synthesised.
   * - ``LSF``
     - No
     - Instrumental broadening is handled by ``sigma_inst`` during
       ``model_rsp_elines_single_main()``.

Output Files
------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Content
   * - ``sim_{label}.fits``
     - LVM RSS FITS with ``n_sim`` simulated spectra. Valid input for
       ``lvm-dap-conf``.
   * - ``{label}.dap.ecsv``
     - Combined DAP table with the *true* RSP parameters, SSP coefficients,
       and emission line values used to generate each spectrum. Used to
       assess parameter recovery accuracy.
   * - ``elines_{label}``
     - Intermediate emission line table (text).
   * - ``coeffs_{label}``
     - Intermediate SSP coefficient table (text).
   * - ``{label}`` (RSP text file)
     - Intermediate stellar population summary (text).

Data Flow Diagram
-----------------

.. code-block:: text

   lvm_file (FITS)
   ├─ FLUX, ERROR, MASK, FWHM, SKY, SKY_ERROR, SLITMAP
   └─ SLITMAP → read_PT() → science-fiber mask
      → trim all arrays to n_sim fibers

   DAP_output_in (9-plane FITS from lvm-dap-gen-out-mod)
   └─ plane 4 (res_joint_spec) → noise template

   DAP_table_in (dap.fits.gz from lvm-dap-conf)
   └─ flux/disp/vel (NP elines) → spec2D_elines

   rsp-file (FITS, SPECTRA HDU)
   └─ n_templates stellar spectra

   For each i_sim in 0…n_sim:
     random z, sigma, AV  ← normal(mid, 0.33*range)
     random coeffs         ← n_st templates from RSP basis
     model_rsp_elines_single_main() → SPS.spectra['model_min']
     noise ← random rows of plane 4 residuals
     FLUX[i_sim] = model_min + noise / f_st

   After loop:
   FLUX = (f_st * FLUX + f_el * spec2D_elines) / f_scale
   ERROR rescaled to match new FLUX level

   → sim_{label}.fits       (simulated LVM RSS, valid lvm-dap-conf input)
   → {label}.dap.ecsv       (true input parameters for recovery tests)
