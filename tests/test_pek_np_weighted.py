import unittest
from unittest.mock import patch

import numpy as np
from astropy.table import Table

from lvmdap import pek_tools


class TestNPWeightedPEK(unittest.TestCase):
    def setUp(self):
        self.names = ['OII_3726.03', 'OII_3728.82']
        self.rest = np.array([3726.03, 3728.82])
        self.catalog = Table({'id': ['OII', 'OII'], 'wl': self.rest})
        self.wave = np.arange(3700., 3755., .3)
        self.lsf = np.full_like(self.wave, .62)
        center = self.rest * (1 + 12. / 299792.458)
        sigma = np.hypot(.62, center * 14. / 299792.458)
        flux = np.array([100., 125.])
        self.data = np.sum(flux / (np.sqrt(2*np.pi)*sigma) * np.exp(
            -.5*((self.wave[:, None]-center)/sigma)**2), axis=1)
        self.pm = Table({'id_fib': [0]})
        self.np = Table({'id_fib': [0]})
        for name, velocity in zip(self.names, (62., -38.)):
            for field, value in [('flux', 90.), ('vel', 70.), ('e_vel', 2.),
                                 ('disp', .7), ('e_disp', .1)]:
                self.pm[f'{field}_{name}_pm'] = [value]
            for field, value in [('flux', 100.), ('e_flux', 10.),
                                 ('vel', velocity), ('e_vel', 2.)]:
                self.np[f'{field}_{name}'] = [value]

    def fit(self, **kwargs):
        with patch.object(pek_tools, 'fit_linked_lines',
                          wraps=pek_tools.fit_linked_lines) as solve:
            result, _ = pek_tools.fit_pek_rss(
                self.data, np.ones_like(self.wave), 0., .7,
                self.catalog, self.pm, self.np, wl=self.wave,
                lsf_spectrum=self.lsf, velocity_tolerance=150.,
                dispersion_tolerance=3., **kwargs)
        return result, solve.call_args.kwargs

    def test_doublet_recovery(self):
        result, call = self.fit()
        self.assertEqual(call['initial_velocity'], 12.)
        self.assertEqual(call['velocity_bounds'], (-8., 32.))
        velocities = [result[f'vel_{name}_pek'][0] for name in self.names]
        self.assertEqual(*velocities)
        self.assertAlmostEqual(velocities[0], 12., delta=3.)
        a, b = [result[f'flux_{name}_pek'][0] for name in self.names]
        self.assertAlmostEqual(np.log10(b/a), np.log10(1.25), delta=.03)

    def test_map_intersection_conflict_and_fixed(self):
        _, call = self.fit(map_velocity_bounds=(15., 45.))
        self.assertEqual(call['initial_velocity'], 15.)
        self.assertEqual(call['velocity_bounds'], (15., 32.))
        result, call = self.fit(map_velocity_bounds=(40., 80.))
        self.assertEqual(call['velocity_bounds'], (40., 80.))
        self.assertTrue(all(result[f'vel_w_conflict_{name}_pek'][0]
                            for name in self.names))
        _, call = self.fit(fixed_velocity=7., map_velocity_bounds=(40., 80.))
        self.assertEqual(call['fixed_velocities'], {0: 7.})

    def test_diagnostics_preserve_columns_and_pm_fallback(self):
        original = self.np.copy()
        pek_tools.add_weighted_np_velocities(self.np, self.catalog, 75., 25.)
        for col in original.colnames:
            self.assertEqual(self.np[col].tobytes(), original[col].tobytes())
        for name in self.names:
            self.assertEqual(self.np[f'vel_w_{name}'][0], 12.)
            self.assertAlmostEqual(self.np[f'e_vel_w_{name}'][0], np.sqrt(2.))
            self.np[f'e_flux_{name}'] = [100.]
        _, fallback = self.fit()
        _, pm = self.fit(velocity_start='pm')
        self.assertEqual(fallback['initial_velocity'], pm['initial_velocity'])
        self.assertEqual(fallback['velocity_bounds'], pm['velocity_bounds'])

    def test_eligibility_and_missing_velocity_error(self):
        a, b = self.names
        self.np[f'flux_{a}'] = [30.]
        self.np[f'flux_{b}'] = [-100.]
        self.np.remove_column(f'e_vel_{a}')
        mean, error = pek_tools.weighted_np_velocity(self.np, self.names)
        self.assertEqual(mean[0], 62.)
        self.assertTrue(np.isnan(error[0]))
        self.assertEqual(pek_tools._weighted_velocity_bounds(mean[0], error[0], 20., None),
                         ((42., 82.), False))
        self.assertEqual(pek_tools._weighted_velocity_bounds(0., 10., 20., None),
                         ((-30., 30.), False))

    def test_weighted_bounds_can_recover_outside_legacy_union(self):
        for name in self.names:
            self.np[f'vel_{name}'] = [0.]
            self.pm[f'vel_{name}_pm'] = [0.]
        weighted, _ = self.fit()
        legacy, call = self.fit(velocity_start='pm')
        self.assertEqual(call['velocity_bounds'], (-2., 2.))
        key = f'vel_{self.names[0]}_pek'
        self.assertAlmostEqual(weighted[key][0], 12., delta=3.)
        self.assertLessEqual(abs(legacy[key][0]), 2.)


if __name__ == '__main__':
    unittest.main()
