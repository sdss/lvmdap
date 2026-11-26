import unittest

from astropy.table import Table
import numpy as np

from lvmdap.dap_core import estimate_gas_redshift, gas_velocity_reference_lines


class TestPeakVelocityCatalog(unittest.TestCase):
    def test_exclusions_do_not_mutate_the_fitting_catalog(self):
        lines = Table({'id': ['Halpha', 'HeII', '[OI]', 'HeII'],
                       'wl': [6562.85, 6560.10, 6300.30, 4685.68]})
        self.assertEqual(list(gas_velocity_reference_lines(lines)['wl']), [6562.85, 4685.68])
        self.assertEqual(len(lines), 4)

    def test_direct_peak_fallback_uses_supplied_catalog(self):
        wave = np.arange(6530., 6600., .25)
        center = 6562.85 * 1.0001
        flux = 20 * np.exp(-.5 * ((wave-center)/.8)**2)
        kwargs = dict(z_min=-.0002, z_max=.0002, min_matches=1, return_details=True)
        catalog = Table({'id': ['Halpha'], 'wl': [6562.85]})
        result = estimate_gas_redshift(wave, flux, np.ones_like(wave),
                                       reference_catalog=catalog, **kwargs)
        self.assertEqual(result['n_matches'], 1)
        self.assertAlmostEqual(result['z'], .0001, delta=2e-5)
        catalog['wl'] = [6560.10]
        result = estimate_gas_redshift(wave, flux, np.ones_like(wave),
                                       reference_catalog=catalog, **kwargs)
        self.assertEqual(result['n_matches'], 0)
