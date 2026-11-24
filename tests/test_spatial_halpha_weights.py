import unittest

from astropy.table import Table
import numpy as np

from lvmdap.spatial_gas_seed import C_KMS, _measured_peaks, integrated_seed_weights, measure_halpha_weighted_seed, measure_peak_seed


class TestWeightedHalpha(unittest.TestCase):
    def setUp(self):
        self.ha, self.nii, self.he = 6562.85, 6583.45, 6560.10
        self.catalog = Table({'id': ['Halpha', '[NII]', 'HeII'],
                              'wl': np.array([self.ha, self.nii, self.he], dtype=np.float32)})
        self.wave = np.arange(6535., 6605., .25)
        self.error = np.ones_like(self.wave)
        self.lsf = self.error * .8

    def profile(self, rest, velocity, flux):
        dx = (self.wave - rest * (1 + velocity/C_KMS))/.8
        return flux * np.exp(-.5*dx**2) / (.8*np.sqrt(2*np.pi))

    def seed(self, gas, weights, good=None):
        if good is None:
            good = np.ones(self.wave.shape, bool)
        return measure_halpha_weighted_seed(self.wave, gas, self.error, self.lsf, good,
                                             -250/C_KMS, 250/C_KMS, self.ha, weights, 5., 2.)

    def test_detected_nitrogen_can_outweigh_halpha(self):
        table = Table({'flux_Halpha_6562.85_best': [50.], 'e_flux_Halpha_6562.85_best': [1.],
                       'flux_[NII]_6583.45_best': [75.], 'e_flux_[NII]_6583.45_best': [1.],
                       'flux_HeII_6560.10_best': [75.], 'e_flux_HeII_6560.10_best': [1.]})
        weights = integrated_seed_weights(table, self.catalog, self.ha, 5., -250/C_KMS, 250/C_KMS)
        self.assertEqual(weights, {'Halpha_6562.85': 1., '[NII]_6583.45': 1.5})
        gas = (self.profile(self.ha, 0, 75) + self.profile(self.ha, 200, 60) +
               self.profile(self.nii, 200, 95))
        self.assertAlmostEqual(self.seed(gas, {'Halpha_6562.85': 1.})[0]*C_KMS, 0.)
        self.assertAlmostEqual(self.seed(gas, weights)[0]*C_KMS, 200.)

    def test_support_alone_cannot_make_a_seed(self):
        gas = self.profile(self.nii, 200, 100)
        self.assertTrue(np.isnan(self.seed(gas, {'Halpha_6562.85': 1., '[NII]_6583.45': 10.})[0]))
        gas += self.profile(self.ha, 200, 75)
        good = (self.wave < 6556) | (self.wave > 6572)
        self.assertTrue(np.isnan(self.seed(gas, {'Halpha_6562.85': 1., '[NII]_6583.45': 10.}, good)[0]))

    def test_one_halpha_peak_cannot_be_relabelled_as_support(self):
        gas = self.profile(self.ha, 0, 300)
        z = self.seed(gas, {'Halpha_6562.85': 1., 'HeII_6560.10': 1.5})[0]
        self.assertAlmostEqual(z*C_KMS, 0.)

    def test_two_peak_heii_alias_cannot_vote(self):
        velocity_alias = (self.ha/self.he-1)*C_KMS
        gas = (self.profile(self.ha, 0., 300.) +
               self.profile(self.ha, velocity_alias, 120.))
        peaks, _ = _measured_peaks(self.wave, gas, np.ones(self.wave.size, bool), 2.)
        self.assertEqual(len(peaks), 2)
        self.assertAlmostEqual(self.seed(gas, {'Halpha_6562.85': 1., 'HeII_6560.10': 1.5})[0]*C_KMS, 0.)

    def test_legacy_peak_route_still_returns_redshift(self):
        gas = self.profile(self.ha, 0., 300.)
        z, _, _, _ = measure_peak_seed(
            self.wave, gas, self.error, np.ones(self.wave.shape, bool),
            np.array([self.ha, self.nii]), -250/C_KMS, 250/C_KMS, self.ha, 2.)
        self.assertAlmostEqual(z*C_KMS, 0.)

    def test_weight_requires_matching_integrated_identity(self):
        catalog = Table({'id': ['Halpha', 'WrongID'], 'wl': [self.ha, self.nii]})
        table = Table({'flux_Halpha_6562.85_best': [50.], 'e_flux_Halpha_6562.85_best': [1.],
                       'flux_[NII]_6583.45_best': [150.], 'e_flux_[NII]_6583.45_best': [1.]})
        self.assertEqual(integrated_seed_weights(table, catalog, self.ha, 5., -250/C_KMS, 250/C_KMS),
                         {'Halpha_6562.85': 1.})

    def test_detected_catalogue_lines_define_support(self):
        catalog = Table({'id': ['Halpha', 'HeI', '[NII]'],
                         'wl': [self.ha, 6678.15, 6583.46]})
        table = Table({'flux_Halpha_6562.85_best': [50.], 'e_flux_Halpha_6562.85_best': [1.],
                       'flux_HeI_6678.15_best': [9.], 'e_flux_HeI_6678.15_best': [1.],
                       'flux_[NII]_6583.46_best': [75.], 'e_flux_[NII]_6583.46_best': [1.]})
        self.assertEqual(integrated_seed_weights(
            table, catalog, self.ha, 5., -250/C_KMS, 250/C_KMS),
            {'Halpha_6562.85': 1., 'HeI_6678.15': .18, '[NII]_6583.46': 1.5})
        table['e_flux_HeI_6678.15_best'][0] = 3.
        weights = integrated_seed_weights(table, catalog, self.ha, 5., -250/C_KMS, 250/C_KMS)
        self.assertNotIn('HeI_6678.15', weights)

    def test_no_integrated_halpha_uses_halpha_only(self):
        self.assertEqual(integrated_seed_weights(None, self.catalog, self.ha, 5., -250/C_KMS, 250/C_KMS),
                         {'Halpha_6562.85': 1.})


if __name__ == '__main__':
    unittest.main()
