Pipeline Workflow: lvm-dap-conf
================================

This page describes the end-to-end processing performed by the ``lvm-dap-conf``
command (entry point: ``lvmdap/_cmdline/dap.py``, function ``_dap_yaml``).

Overview
--------

The pipeline runs in five sequential stages:

1. :ref:`workflow-load` — open the FITS file and extract all spectral arrays
2. :ref:`workflow-integrated` — build a single co-added spectrum from all good fibers
3. :ref:`workflow-autozoom` — (optional) refine the redshift guess before fitting
4. :ref:`workflow-intfit` — fully fit the integrated spectrum to establish best-fit kinematics
5. :ref:`workflow-perfiber` — fit each fiber individually, seeded from the integrated result

.. _workflow-load:

Stage 1 — Data Loading
-----------------------

The input FITS file is opened with ``load_LVM_rss()`` (``dap_tools.py``, line 626).
All extensions are read at this point. The sections below explain exactly what is
done with each extension.

Required extensions
~~~~~~~~~~~~~~~~~~~

``FLUX``
  Read unconditionally as ``rss_flux_org``. This is the primary spectral array
  of shape ``(n_fibers, n_wavelengths)``.

  The wavelength axis is reconstructed from the header keywords ``CRVAL1`` and
  ``CDELT1``. If those keywords are missing from the ``FLUX`` header the pipeline
  falls back to the primary HDU (index 0) header.

  The pipeline then checks whether the flux is already in units of
  :math:`10^{-16}` erg/s/cm²/Å. If the median flux in the 5000–6000 Å window
  exceeds ``1e-3`` no scaling is applied; otherwise the array is multiplied by
  the default scale factor (``flux_scale = 1e16``).

``MASK``
  Read unconditionally. Bad pixels (value ``1``) are not explicitly set to NaN
  at load time. Instead, when ``mask_to_val = True`` (the default), any NaN or
  Inf values in both ``FLUX`` and the error array are replaced by the local
  interpolated average of their neighbours before fitting begins (line 818).

``SLITMAP``
  Read separately by ``read_PT()`` (``dap_tools.py``, line 1209) immediately
  after ``load_LVM_rss()`` returns. This table drives two things:

  * **Fiber selection** — only fibers where ``targettype == 'science'`` and
    ``fibstatus == 0`` are retained. All subsequent arrays (flux, error, LSF)
    are filtered to this subset before any fitting takes place (lines 835–838).
  * **Spatial coordinates** — the ``ra`` / ``dec`` columns (or ``xpmm`` /
    ``ypmm`` converted via the field-centre header keywords) are written into
    the output position table (``PT`` extension of the final DAP FITS file).

Optional extensions
~~~~~~~~~~~~~~~~~~~

``IVAR``, ``ERR``, ``ERROR`` — error spectrum
  The pipeline searches for an error spectrum in priority order:

  1. ``IVAR`` — if present, converted to :math:`\sigma = 1/\sqrt{\mathrm{IVAR}}`
     and appended to the HDU list as ``ERROR``.
  2. ``ERR`` — used directly as the error array and also re-appended as ``ERROR``.
  3. ``ERROR`` — used directly.
  4. **None found** — errors are auto-estimated fiber by fiber (lines 667–670):

     .. code-block:: python

        std = 0.1 * nanstd(flux - median_filter(flux, size=(1,51)), axis=1)

     Each fiber's error spectrum is set to a flat value equal to 10 % of the
     high-frequency spectral noise of that fiber.

  After loading, any zero-valued error pixels are replaced by the array median
  (line 820) to avoid division-by-zero during weighted fitting.

``LSF`` — Line Spread Function
  The LSF array is loaded with ``get_lsf=True`` alongside the flux (line 768),
  giving a per-fiber, per-pixel FWHM array (in Å). It is immediately divided by
  2.354 to convert FWHM to Gaussian :math:`\sigma` (``dap_tools.py``, line 634).
  If the ``LSF`` extension is absent the array is set to zero for all fibers and
  pixels (line 780 / 807).

  A separate step (lines 810–814) also computes a *median LSF spectrum*
  (``LSF_mean``) by taking the median of the per-fiber LSF array across all
  fibers. If the extension is missing, ``LSF_mean`` falls back to a flat array
  at the value of ``sigma-gas``.

  The LSF is used in two places:

  * **Emission line table** (lines 1091–1106) — before fitting begins, the LSF
    value at each emission line's rest wavelength is read from ``LSF_mean`` and
    stored in the line table (``tab_el['lsf']``). This is passed to the emission
    line fitting routines as the instrumental broadening at each line.
  * **Intrinsic velocity dispersion** (lines 1754–1765) — after all fitting is
    done, the observed line-of-sight velocity dispersion :math:`\sigma_\mathrm{obs}`
    (in km/s) is corrected for instrumental broadening to yield the intrinsic
    dispersion:

    .. math::

       \sigma_\mathrm{int} = \frac{c}{\lambda} \sqrt{
           \left(\frac{\sigma_\mathrm{obs,\,Å}}{2.354}\right)^2
           - \sigma_\mathrm{LSF}^2
       }

    where :math:`\sigma_\mathrm{LSF}` is the per-fiber LSF at the line
    wavelength. If :math:`\sigma_\mathrm{obs} < \sigma_\mathrm{LSF}` the LSF
    is rescaled downward to avoid a negative argument under the square root.
    This corrected value is written to the ``sigma_kms_{name}`` columns of the
    final DAP table.

``SKY`` — sky background
  When ``sky-hack: True`` is set and the ``SKY`` extension is present, the
  function ``sky_hack_f()`` (``dap_tools.py``, line 590) subtracts a
  fiber-dependent fraction of the sky spectrum from each flux fiber *before*
  any fitting. The algorithm:

  1. Defines three narrow wavelength bands:

     * **Target band** B: 7238–7242 Å (contains a sky line)
     * **Continuum band 0**: 7074–7084 Å
     * **Continuum band 1**: 7194–7265 Å

  2. Computes a continuum-subtracted sky signal in each fiber:

     .. math::

        \Delta f_i = \bar{f}_i^B - \tfrac{1}{2}\!\left(\bar{f}_i^0 + \bar{f}_i^1\right)
        \qquad
        \Delta s_i = \bar{s}_i^B - \tfrac{1}{2}\!\left(\bar{s}_i^0 + \bar{s}_i^1\right)

     where :math:`\bar{f}_i^X` is the mean flux in band X for fiber :math:`i`,
     and :math:`\bar{s}_i^X` is the corresponding value from the ``SKY``
     extension.

  3. Derives a per-fiber sky scale factor:
     :math:`\alpha_i = \Delta f_i / \Delta s_i`

  4. Subtracts the scaled sky:
     :math:`f_i^\mathrm{corr}(\lambda) = f_i(\lambda) - \alpha_i \, s_i(\lambda)`

  .. note::

     ``sky-hack`` is deprecated for LVM DRP reductions newer than v1.1.
     Set ``sky-hack: False`` (the default) for all current LVM data, where
     sky subtraction is already performed by the DRP.

.. _workflow-integrated:

Stage 2 — Integrated Spectrum
------------------------------

An inverse-variance-weighted co-add of all good-fiber spectra is computed
(lines 862–883). Before co-adding, the S/N of each fiber is evaluated in the
Hα window (6530–6650 Å, shifted by ``auto_z_min`` / ``auto_z_max``):

* If **three or more** fibers have peak S/N > 3 in this window, only those
  high-S/N fibers contribute to the co-add.
* Otherwise, all fibers are included.

The weighting in both cases is :math:`w_i = 1/\sigma_i^2(\lambda)`, using the
loaded (or auto-estimated) error spectrum.

The resulting integrated spectrum ``m_flux`` and its error ``e_flux`` are used
for all subsequent integrated-spectrum operations.

.. _workflow-autozoom:

Stage 3 — Auto-Redshift (optional)
------------------------------------

When ``auto-redshift: True``, the pipeline estimates the systemic redshift from
the integrated spectrum before fitting (lines 929–1025), using
``find_redshift_spec()`` (``dap_tools.py``, line 2626). Two optional
pre-processing steps are applied first:

1. **Continuum removal** (``auto_z_bg: True``) — ``find_continuum()`` subtracts
   a smooth continuum from ``m_flux`` using 15 iterations of clipped median
   filtering (box sizes 5–75 pixels, clip threshold 0.8). The continuum-
   subtracted spectrum ``m_flux_bgs`` highlights emission peaks.

2. **Sky line subtraction** (``auto_z_sc: True``, requires
   ``emission-lines-file-sky``) — each sky line in the sky emission line table
   is modelled as a Gaussian and subtracted from ``m_flux`` (not from
   ``m_flux_bgs``). This prevents sky features from being mis-identified as
   galaxy emission lines.

``find_redshift_spec()`` then searches for peaks in ``m_flux_bgs`` near the
reference wavelengths 6548.05, 6562.85, and 6583.45 Å ([N II] + Hα) over the
grid ``[auto_z_min, auto_z_max]`` with step ``auto_z_del``.

If a peak is found **and** the Hα-window S/N exceeds 10, the pipeline updates
the redshift guess and search bounds:

.. code-block:: python

   args.redshift[0] = auto_z                            # new guess
   args.redshift[2] = args.redshift[2]*(1+auto_z) + auto_z  # new min
   args.redshift[3] = args.redshift[3]*(1+auto_z) + auto_z  # new max
   args.w_range_nl  = [w*(1+auto_z) for w in args.w_range_nl]

If no peak is found but the integrated S/N exceeds 20, the full
``[auto_z_min, auto_z_max]`` range is used as the search bounds instead.

.. _workflow-intfit:

Stage 4 — Integrated Spectrum Fitting
---------------------------------------

The integrated spectrum is fitted by a single call to
``auto_rsp_elines_rnd()`` (``modelling/auto_rsp_tools.py``, line 94).
This function performs the full RSP decomposition in the following order:

1. **Non-linear parameter fitting** — a grid search over redshift, velocity
   dispersion (σ), and dust extinction (A\ :sub:`V`) using the reduced template
   basis ``rsp-nl-file`` in the wavelength window ``w-range-nl``:

   * *Redshift*: two-stage broad + fine grid search minimising χ².
   * *Velocity dispersion*: grid search with random perturbations; merit
     function is χ² or median residual depending on configuration.
   * *A*\ :sub:`V`: grid search followed by a hyperbolic-parabola fit to
     locate the χ² minimum to sub-step accuracy.

2. **Linear decomposition** — weighted least-squares (WLS) fit of the full
   template basis ``rsp-file`` in the wavelength window ``w-range`` at the
   best-fit kinematics and extinction. Repeated ``N_MC`` times (default 20)
   with noise realisations drawn from the error spectrum to estimate
   coefficient uncertainties.

3. **Emission line fitting** — gas emission lines from ``emission-lines-file``
   (and ``emission-lines-file-long`` for the refinement pass) are fitted as
   Gaussians in the stellar-continuum-subtracted residual spectrum. The per-line
   LSF values from ``tab_el['lsf']`` (populated in Stage 1) are applied as
   the fixed instrumental broadening for each line.

   If ``single-gas-fit: False``, this step is repeated on the residual after
   the first pass to detect weaker lines missed in the initial detection.

The ``SN_CUT_INT`` threshold applies here: if the integrated spectrum S/N falls
below this value the stellar component is skipped and only emission lines are
fitted.

**Outputs written after this stage** (prefix ``m_{label}`` in ``output_path``):

* ``m_{label}.output.fits`` — observed, model, and residual spectra
* ``m_{label}.coeffs.txt`` — SSP coefficients and Monte-Carlo uncertainties
* ``m_{label}.elines.txt`` — emission line fluxes, equivalent widths, kinematics
* ``m_{label}.rsp.txt`` — integrated stellar population summary (age, metallicity,
  T\ :sub:`eff`, log g, A\ :sub:`V`)

.. _workflow-perfiber:

Stage 5 — Per-Fiber Fitting
-----------------------------

Each good science fiber is fitted in sequence (lines 1385–1432). The loop
structure is:

**Seeding from the integrated fit**
  At the start of every fiber's fit, the best-fit parameters from the
  *previous* fit (or the integrated fit for fiber 0) are used as the new
  starting guess:

  .. code-block:: python

     args.redshift[0] = SPS.best_redshift
     args.sigma[0]    = SPS.best_sigma
     args.AV[0]       = SPS.best_AV

  This propagates the kinematic solution smoothly across the field of view.

**S/N gating via SN_CUT**
  Inside ``auto_rsp_elines_rnd()``, the S/N of each fiber is evaluated in
  the ``w-range-nl`` window. If S/N < ``SN_CUT`` (default 3), the full
  non-linear + Monte-Carlo fitting is skipped and only a single best-matching
  SSP template is fitted (``single_rsp`` mode). This greatly speeds up
  processing of faint or sky-dominated fibers.

**Outputs written per fiber** (appended after each fiber, prefix ``{label}``):

* ``{label}.coeffs.txt`` — one row of SSP coefficients per fiber
* ``{label}.elines.txt`` — one block of emission line measurements per fiber
* ``{label}.rsp.txt`` — one row of stellar population parameters per fiber
* ``{label}.output.fits.gz`` — per-fiber model spectra array

**Final assembly** — after all fibers are processed, the per-fiber text tables
and the position table from ``SLITMAP`` are assembled into the master output:

* ``{label}.dap.fits.gz`` — multi-extension FITS containing:

  .. list-table::
     :widths: 20 80
     :header-rows: 1

     * - Extension
       - Contents
     * - ``PT``
       - Fiber position table (ra, dec, fiberid) from ``SLITMAP``
     * - ``RSP``
       - Stellar population parameters per fiber (age, Z, T\ :sub:`eff`, log g,
         A\ :sub:`V`)
     * - ``COEFFS``
       - SSP coefficient vectors per fiber
     * - ``PM_ELINES``
       - Parametric emission line measurements (flux, EW, velocity, σ)
     * - ``NP_ELINES``
       - Non-parametric emission line measurements
     * - ``PM_KEL``
       - Kinematic emission line measurements
     * - ``ELINES_SIGMA_CHI``
       - Intrinsic velocity dispersions (LSF-corrected) and fit quality

Data Flow Diagram
-----------------

.. code-block:: text

   FITS Input
   ├─ FLUX       ──→ rss_flux     ─┐
   ├─ MASK       ──→ bad-pixel map  │ load_LVM_rss()
   ├─ IVAR/ERR   ──→ rss_eflux   ─┘
   ├─ LSF        ──→ LSF_rss, LSF_mean  (FWHM/2.354)
   ├─ SKY        ──→ sky_hack_f()  [if sky-hack: True]
   └─ SLITMAP    ──→ read_PT()  → fiber mask + ra/dec

       ↓ filter to science fibers with fibstatus==0

   Integrated spectrum
   (inverse-variance weighted co-add of high-S/N fibers)

       ↓ [if auto-redshift: True]

   find_redshift_spec()
   (continuum-subtract, sky-subtract, cross-correlate Hα+[NII])
   → update redshift guess & w-range-nl

       ↓

   auto_rsp_elines_rnd()  [integrated spectrum, SN_CUT_INT]
   ├─ non_linear_fit_kin()  (redshift, σ grid search, rsp-nl-file, w-range-nl)
   ├─ _fit_AV()             (A_V grid search + parabola fit)
   ├─ rsp_fit() × N_MC      (WLS decomposition, rsp-file, w-range)
   └─ emission line fit     (LSF_mean per line, emission-lines-file-long)
   → m_{label}.* outputs

       ↓

   Per-fiber loop (seeded from previous best-fit)
   └─ auto_rsp_elines_rnd()  [per fiber, SN_CUT]
      ├─ S/N < SN_CUT  →  single SSP fit only
      └─ S/N ≥ SN_CUT  →  full non-linear + MC fit
   → {label}.* outputs  →  {label}.dap.fits.gz
