Input RSS File Format
=====================

LVM-DAP expects input spectra in FITS Row-Stacked Spectra (RSS) format. The pipeline reads
this file via ``load_LVM_rss()`` in ``lvmdap/dap_tools.py``. Below are the required and
optional FITS extensions (HDUs).

Required Extensions
-------------------

``FLUX``
~~~~~~~~

The primary science data extension.

* **Type**: ImageHDU, 2D float array of shape ``(n_fibers, n_wavelengths)``
* **Units**: Expected in units of :math:`10^{-16}` erg/s/cm²/Å (flux scale auto-detected)
* **Required header keywords**:

  * ``CRVAL1`` — wavelength of the first pixel (Å)
  * ``CDELT1`` — wavelength step per pixel (Å/pixel)

  .. note::

     If ``CRVAL1`` is missing from the ``FLUX`` header, the pipeline falls back to
     reading these keywords from the primary HDU (index 0).

``MASK``
~~~~~~~~

Bad-pixel / quality mask.

* **Type**: ImageHDU, 2D integer array of same shape as ``FLUX``
* **Convention**: ``1`` = bad/masked pixel, ``0`` = good pixel

``SLITMAP``
~~~~~~~~~~~

Fiber position and status table. Required for spatial mapping of fibers.

* **Type**: BinTableHDU
* **Required columns**:

  * ``targettype`` — fiber target classification (science fibers identified by value ``'science'``)
  * ``fibstatus`` — fiber status flag (``0`` = good science fiber)
  * ``fiberid`` — unique fiber identifier

* **Positional columns** (one set required):

  +---------------------+-------------------------------------------+
  | Columns             | Description                               |
  +=====================+===========================================+
  | ``ra``, ``dec``     | Fiber sky coordinates (degrees)           |
  +---------------------+-------------------------------------------+
  | ``fib_ra``,         | Alternative fiber sky coordinates         |
  | ``fib_dec``         |                                           |
  +---------------------+-------------------------------------------+
  | ``xpmm``, ``ypmm`` | Focal-plane positions (mm); RA/Dec        |
  |                     | computed from field centre keywords       |
  +---------------------+-------------------------------------------+

  When ``xpmm``/``ypmm`` are used, the primary header must contain ``POSCIRA`` and
  ``POSCIDE`` (field centre RA/Dec in degrees). ``POSCIPA`` (position angle) is
  optional and defaults to ``0.0`` if absent.

Optional Extensions
-------------------

Error Arrays
~~~~~~~~~~~~

The pipeline searches for an error spectrum in the following priority order.
Only one is needed; if none is present, errors are auto-estimated from the
spectral noise.

+-------------+------------------------------------------------------+
| Extension   | Description                                          |
+=============+======================================================+
| ``IVAR``    | Inverse variance. Converted internally to            |
|             | :math:`\sigma = 1/\sqrt{\mathrm{IVAR}}`.             |
+-------------+------------------------------------------------------+
| ``ERR``     | Direct per-pixel error (same units as ``FLUX``).     |
+-------------+------------------------------------------------------+
| ``ERROR``   | Alternative name for direct error array.             |
+-------------+------------------------------------------------------+

.. note::

   If no error extension is found, the pipeline auto-computes per-fiber noise from
   the high-frequency residuals of a median-filtered version of each spectrum (kernel
   width 51 pixels).

``LSF``
~~~~~~~

Line Spread Function (LSF) profile widths.

* **Type**: ImageHDU, 2D float array of same shape as ``FLUX``
* **Units**: FWHM in Å (divided by 2.354 internally to convert to Gaussian :math:`\sigma`)
* **Usage**: Only loaded when ``get_lsf=True`` is passed to ``load_LVM_rss()``.
  If absent, LSF corrections are disabled (zero LSF assumed).

``SKY``
~~~~~~~

Sky background spectrum.

* **Type**: ImageHDU, 2D float array of same shape as ``FLUX``
* **Usage**: When present and ``sky_hack=True`` (default), a sky re-evaluation
  step is applied to the flux before fitting. If absent, sky refinement is
  silently skipped.

Primary HDU Header Keywords
----------------------------

The primary HDU (index 0) header is read for the following keywords:

+-------------+-----------------------------------------------------------+
| Keyword     | Description                                               |
+=============+===========================================================+
| ``POSCIRA`` | Field centre Right Ascension (degrees). Used when         |
|             | ``SLITMAP`` lacks ``ra``/``dec`` columns.                 |
+-------------+-----------------------------------------------------------+
| ``POSCIDE`` | Field centre Declination (degrees).                       |
+-------------+-----------------------------------------------------------+
| ``POSCIPA`` | Position angle (degrees). Defaults to ``0.0`` if absent.  |
+-------------+-----------------------------------------------------------+
| ``CRVAL1``  | Fallback wavelength origin if missing from ``FLUX`` HDU.  |
+-------------+-----------------------------------------------------------+
| ``CDELT1``  | Fallback wavelength step if missing from ``FLUX`` HDU.    |
+-------------+-----------------------------------------------------------+
| ``exposure``| Exposure identifier used for fiber ID labelling.          |
+-------------+-----------------------------------------------------------+

Minimal Valid File
------------------

The minimum set of extensions required to run ``lvm-dap-conf`` is:

.. code-block:: text

    HDU 0  — Primary (header with POSCIRA/POSCIDE if xpmm/ypmm used for positions)
    HDU 1  — FLUX    [ImageHDU, (n_fibers, n_wavelengths), headers: CRVAL1, CDELT1]
    HDU 2  — MASK    [ImageHDU, (n_fibers, n_wavelengths)]
    HDU 3  — SLITMAP [BinTableHDU, columns: targettype, fibstatus, fiberid, ra, dec]

A typical full LVM SFrame file additionally includes ``IVAR``, ``SKY``, and ``LSF``.

Supported Input Formats
-----------------------

The pipeline natively supports several survey formats by selecting the appropriate
pre-processing script:

+------------+----------------------------+----------------------------+
| Format     | Loader / preprocessor      | Notes                      |
+============+============================+============================+
| LVM SFrame | ``load_LVM_rss()``         | Default format             |
+------------+----------------------------+----------------------------+
| LVMSIM     | ``load_LVMSim_RSS()``      | Uses ``TARGET`` instead of |
|            |                            | ``FLUX``                   |
+------------+----------------------------+----------------------------+
| MaNGA      | ``preprocess-manga``       | Converts to LVM RSS first  |
+------------+----------------------------+----------------------------+
| MUSE       | ``preprocess-muse``        | Converts to LVM RSS first  |
+------------+----------------------------+----------------------------+
