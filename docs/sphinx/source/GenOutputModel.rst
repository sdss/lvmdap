Pipeline Workflow: lvm-dap-gen-out-mod
========================================

``lvm-dap-gen-out-mod`` reconstructs spectral model components from the results
of a previous ``lvm-dap-conf`` run. It reads the fitted coefficients, kinematics,
and emission line parameters stored in a DAP FITS file and re-synthesises each
spectral component so that the full observed-minus-model decomposition can be
inspected or replotted.

Entry point: ``lvmdap/_cmdline/gen_output_model.py``, function ``_main``.

Syntax
------

.. code-block:: bash

   lvm-dap-gen-out-mod [-h] [-d] [-output_path OUTPUT_PATH]
                       [--plot PLOT] [--flux-scale min max]
                       lvm_file DAP_table_in label config_yaml

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
     - Original input LVM FITS file that was fitted by ``lvm-dap-conf``.
       Must contain a ``FLUX`` and ``WAVE`` extension plus a ``SLITMAP``
       table.
   * - ``DAP_table_in``
     - string
     - DAP FITS file produced by ``lvm-dap-conf`` (``{label}.dap.fits.gz``).
       Provides the per-fiber fitted parameters.
   * - ``label``
     - string
     - Output file label. The result is written to
       ``{output_path}/{label}.output.fits.gz``.
   * - ``config_yaml``
     - string
     - YAML configuration file (same format as for ``lvm-dap-conf``).
       Used here mainly for ``rsp_file``, ``sigma_inst``, and flux-scale
       settings.

Options
~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``-output_path``
     - ``./``
     - Directory where the output file is written.
   * - ``--plot``
     - ``0``
     - ``0`` = no plot, ``1`` = display interactively, ``2`` = save to file.
   * - ``--flux-scale min max``
     - ``-1 1``
     - Flux display range for plots. Overridden by ``flux_scale_org`` from
       the YAML config when left at the default.
   * - ``-d / --debug``
     -
     - Print full Python traceback on error instead of a short message.

Processing Flow
---------------

.. _genmod-step1:

Step 1 — Load input files
~~~~~~~~~~~~~~~~~~~~~~~~~~

The original LVM FITS file is opened via ``astropy.io.fits.open()``.
The DAP results table is loaded by ``read_DAP_file()`` (``dap_tools.py``) and
sorted by ``fiberid``.

The ``SLITMAP`` table is read by ``read_PT()`` to reconstruct the science-fiber
mask. Only fibers with ``targettype == 'science'`` and ``fibstatus == 0`` are
retained. The ``FLUX`` array is then trimmed to those fibers and any NaN or Inf
values are replaced by the local interpolated average
(``replace_nan_inf_with_adjacent_avg()``).

.. note::

   The wavelength axis is read from the ``WAVE`` extension of the input FITS
   file (not reconstructed from header keywords as in ``lvm-dap-conf``). This
   gives the pixel-exact wavelength grid used during the original fit.

.. _genmod-step2:

Step 2 — Read fitted parameters from the DAP table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each fiber the following columns are extracted from the DAP table:

* ``redshift_st`` — best-fit stellar redshift
* ``disp_st`` — best-fit stellar velocity dispersion (km/s)
* ``Av_st`` — best-fit dust extinction A\ :sub:`V`
* ``med_flux_st`` — median fitted flux (used as overall normalisation)
* ``W_rsp_0 … W_rsp_N`` — SSP coefficient vector (one entry per template)

.. _genmod-step3:

Step 3 — Build emission line model planes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two emission line model cubes are synthesised from the DAP table using the
local ``eline()`` function, which evaluates a normalised Gaussian:

.. math::

   f(\lambda) = \frac{F}{\sigma\sqrt{2\pi}}
       \exp\!\left[-\frac{1}{2}\left(\frac{\lambda - \lambda_0}{\sigma}\right)^2\right]

where :math:`\lambda_0 = \lambda_\mathrm{rest}(1 + V/c)` and
:math:`\sigma = D / 2.354` (``D`` being the dispersion in Å as stored in the
DAP table).

* **Non-parametric (NP) model** — built from DAP columns whose names match
  ``e_flux_*`` but do *not* contain ``_pe_`` or ``_pek_``. Stored in plane 6.
* **Parametric (PE) model** — built from columns matching ``e_flux_*_pe_*``.
  Stored in plane 7.

.. _genmod-step4:

Step 4 — Reconstruct the stellar continuum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each fiber the stellar model spectrum is computed from the RSP template
basis (``rsp-file`` from the YAML config, ``SPECTRA`` HDU):

.. code-block:: python

   model_rsp = nansum(rsp_data.T * coeffs, axis=1)
   model_rsp = shift_convolve(wave, rsp_wave, model_rsp,
                              redshift, sigma, sigma_inst)
   model_st  = spec_apply_dust(wave, model_rsp * med_flux / f_scale,
                               AV, R_V=3.1, extlaw='CCM')

The three operations are:

1. **Linear combination** of templates weighted by the fitted coefficient vector.
2. **Kinematic broadening and redshift** via ``shift_convolve()``, which
   convolves the model with a Gaussian of width
   :math:`\sigma_\mathrm{tot} = \sqrt{\sigma_\mathrm{disp}^2 + \sigma_\mathrm{inst}^2}`
   and shifts it to the best-fit redshift.
3. **Dust reddening** via the Cardelli–Clayton–Mathis (CCM) extinction law
   with :math:`R_V = 3.1`.

The result is stored in plane 1 of the output cube.

.. _genmod-step5:

Step 5 — Assemble and refine the combined model planes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A residual spectrum is formed from the difference between the observed flux
and the sum of the stellar and NP emission models. A smooth continuum
correction is derived from this residual by:

1. Median-filtering with a window of ``smooth_size`` pixels (default 21).
2. Fitting a Legendre polynomial of degree ``n_leg`` (default 11) to the
   median-filtered residual.

The nine output planes are then assembled:

.. list-table::
   :widths: 10 20 70
   :header-rows: 1

   * - Plane
     - ``NAME`` keyword
     - Content
   * - 0
     - ``org_spec``
     - Original observed spectrum
   * - 1
     - ``model_spec``
     - Stellar continuum model (after model-patch refinement; see below)
   * - 2
     - ``mod_joint_spec``
     - Stellar + NP emission + smooth continuum correction
   * - 3
     - ``gas_spec``
     - Observed minus stellar model minus smooth correction
       (net gas emission)
   * - 4
     - ``res_joint_spec``
     - Observed minus combined model (full residual)
   * - 5
     - ``no_gas_spec``
     - Observed minus NP emission model
   * - 6
     - ``gas_model_NP``
     - Non-parametric emission line model (positive component only,
       after patch)
   * - 7
     - ``gas_model_PEK``
     - Parametric emission line model
   * - 8
     - ``mod_joint_spec_PEK``
     - Stellar + PE emission + smooth continuum correction

**Model-patch step** — after the initial assembly, a second median filter
pass is run on the residual (plane 0 − plane 8). Negative components of the
NP emission model (plane 6) are moved into the stellar model (plane 1) and
the joint model (plane 2) so that plane 6 contains only positive emission.
This prevents negative residuals from being mistaken for absorption features.

.. _genmod-step6:

Step 6 — Write output
~~~~~~~~~~~~~~~~~~~~~~

The 9-plane cube is written to ``{output_path}/{label}.output.fits.gz`` via
``array_to_fits()``. The header is populated with wavelength calibration
keywords (``CRPIX1``, ``CRVAL1``, ``CDELT1``) and ``NAME0``–``NAME8``
descriptors for each plane.

FITS Extension Usage
---------------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Extension
     - Used?
     - How
   * - ``FLUX``
     - Yes
     - Stored as plane 0; science-fiber rows selected via ``SLITMAP`` mask.
   * - ``WAVE``
     - Yes
     - Provides the pixel-exact wavelength array for model synthesis.
   * - ``SLITMAP``
     - Yes
     - Science-fiber mask applied to ``FLUX`` before processing.
   * - ``IVAR`` / ``ERR`` / ``ERROR``
     - No
     - Error information is not needed for deterministic model reconstruction.
   * - ``LSF``
     - No
     - Instrumental broadening is applied via ``sigma_inst`` from the YAML
       config, not from the per-pixel LSF array.
   * - ``SKY``
     - No
     - Sky subtraction was already applied by ``lvm-dap-conf``; the DAP
       table holds the sky-subtracted fitted results.
   * - ``MASK``
     - No
     - Pixel masks are not re-applied during reconstruction.

Output Files
------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - File
     - Content
   * - ``{label}.output.fits.gz``
     - 9-plane FITS cube of shape ``(9, n_fibers, n_wavelengths)`` containing
       the full spectral decomposition (planes described in Step 5 above).

Data Flow Diagram
-----------------

.. code-block:: text

   lvm_file (FITS)
   ├─ FLUX    ──→ plane 0 (observed spectra)
   └─ WAVE    ──→ wavelength array
   SLITMAP    ──→ read_PT() → science-fiber mask

   DAP table (dap.fits.gz)
   ├─ redshift_st, disp_st, Av_st, med_flux_st  → per-fiber kinematics
   ├─ W_rsp_0 … W_rsp_N                         → SSP coefficients
   ├─ flux/disp/vel (NP elines)                  → NP emission model → plane 6
   └─ flux/disp/vel (PE elines, _pe_ columns)    → PE emission model → plane 7

   rsp-file (FITS, SPECTRA HDU)
   └─ template basis
      → nansum(templates * coeffs)
      → shift_convolve (redshift + sigma broadening)
      → spec_apply_dust (CCM, R_V=3.1)
      → plane 1 (stellar model)

   Residual (plane 0 − plane 1 − plane 6)
   → median filter + Legendre polynomial
   → smooth continuum correction
   → planes 2, 3, 4, 5, 8

   → {label}.output.fits.gz
