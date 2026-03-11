Changelog
=========

Version 1.1.0
-------------

Current release.

Recent Changes
~~~~~~~~~~~~~~

* Bug fixes in the ``get-mod`` script
* New X-squared derivation for average spectrum
* Corrected error in creating emission lines model (np.abs)
* Changed model output to float32 for efficiency
* RSP PDF plotting enhancements

Version 1.0.0
-------------

Initial release of LVM-DAP.

Features
~~~~~~~~

* YAML-based configuration system
* Support for LVM RSS format
* Stellar population synthesis with RSP method
* Emission line detection and fitting
* Non-linear parameter optimization (redshift, velocity dispersion, AV)
* Integration with pyPipe3D
* Multiple output formats (DAP FITS, coefficient tables, emission line tables)
* Diagnostic plotting capabilities
