import numpy as np
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from astropy.table import Table

from lvmdap import pek_tools
from lvmdap.linked_line_fitter import fit_linked_lines


class TestHalphaRestart(unittest.TestCase):
    def test_halpha_restart(self):
        for behavior in ('accepted', 'worse', 'failed', 'fixed', 'same_start', 'disabled'):
            with self.subTest(behavior=behavior):
                check_halpha_restart('Halpha', 6562.85, behavior)

    def test_hbeta_does_not_restart(self):
        check_halpha_restart('Hbeta', 4861.36, 'excluded')

    def test_linked_restart_window(self):
        check_linked_restart_window()


    def test_halpha_guard_keeps_fit_when_only_neighbor_improves(self):
        rest = np.array([6562.85, 6576.49])
        wave = np.arange(6535., 6600., .5)
        sigma = 1.2
        flux = np.array([10., 100.])

        def profiles(v):
            centers = rest * (1 + v / 299792.458)
            return flux / (np.sqrt(2*np.pi)*sigma) * np.exp(
                -.5 * ((wave[:, None] - centers)/sigma)**2)

        original = profiles(65.)
        alternate = profiles(0.)
        data = original[:, 0] + alternate[:, 1]
        catalog = Table({'id': ['Halpha', 'Other'], 'wl': rest})
        tags = ['Halpha_6562.85', 'Other_6576.49']
        pm = Table({f'{field}_{tag}_pm': [value] for tag, strength in zip(tags, flux)
                    for field, value in [('flux', strength), ('vel', 65.),
                                         ('disp', sigma), ('e_disp', .2)]})
        fits = []
        for velocity, components in [(65., original), (0., alternate)]:
            fits.append(dict(success=True, flux=flux, vel=np.full(2, velocity),
                             disp=np.full(2, sigma), model=components.sum(axis=1)))
        assert np.sum((data-fits[1]['model'])**2) < np.sum((data-fits[0]['model'])**2)
        with patch.object(pek_tools, 'fit_linked_lines', side_effect=fits) as solve:
            table, _ = pek_tools.fit_pek_rss(
                data, np.ones_like(wave), 0., sigma, catalog, pm, Table({'id_fib': [0]}),
                wl=wave, lsf_spectrum=np.ones_like(wave), velocity_tolerance=150.,
                dispersion_tolerance=3., map_velocity_bounds=(-75., 75.),
                restart_velocity=0.)
        self.assertEqual(solve.call_count, 2)
        self.assertEqual(table['vel_Halpha_6562.85_pek'][0], 65.)


class TestAdoptedContinuum(unittest.TestCase):
    def test_continuum_uses_current_fiber_with_reused_synthesis(self):
        from lvmdap import dap_core

        class ReachedContinuum(Exception):
            pass

        wave = np.arange(4920., 5081.)
        flux = 3. + (wave-5000.)/100.
        pt = Table({'id': ['fiber-0', 'fiber-1']})
        rsp = Table({'redshift': [0., 0.], 'disp': [0., 0.], 'Av': [0., 0.],
                     'med_flux': [0., 0.], 'e_med_flux': [0., 0.], 'flux': [0., 0.]})
        coeffs = Table({'N.Coeff': [1., 1.]})
        sps = SimpleNamespace(models=SimpleNamespace(wavenorm=5000., wavelength=wave,
                                                     flux_models=np.ones((1, len(wave)))),
                              spectra={})
        args = SimpleNamespace(redshift=[0., 0., -.001, .001], w_range=[4920., 5080.])
        residuals = []

        def capture(wl, residual, *args, **kwargs):
            residuals.append(residual.copy())
            raise ReachedContinuum

        with (patch.object(dap_core, 'interpolate_emission_line_regions',
                           return_value=(flux, np.zeros(wave.size, dtype=bool))),
              patch.object(dap_core, 'shift_convolve', return_value=np.ones(wave.size)),
              patch.object(dap_core, 'spec_apply_dust', return_value=np.ones(wave.size)),
              patch.object(dap_core, 'fit_masked_residual_continuum', side_effect=capture)):
            for predecessor in (10., -20.):
                sps.spectra['model_joint'] = np.full(wave.size, predecessor)
                with self.assertRaises(ReachedContinuum):
                    dap_core.fit_individual_rss_spectrum_or_adopt_result(
                        wave, flux, np.ones_like(wave), np.ones_like(wave),
                        pt, 1, Table(), [], None, args, SPS_master=sps,
                        tab_rsp_adopt=rsp, tab_coeffs_adopt=coeffs)
        np.testing.assert_array_equal(residuals[0], flux-3.)
        np.testing.assert_array_equal(residuals[0], residuals[1])


def check_halpha_restart(label, rest, behavior):
    wave = rest + np.arange(-50., 50.1, .5)
    error = lsf = np.ones(wave.size)
    bad = np.zeros(wave.size, dtype=bool)
    bad[97] = True
    tag = f'{label}_{rest:.2f}'
    start = 5. if behavior == 'same_start' else 65.
    pm = Table({f'{key}_{tag}_pm': [val] for key, val in
                [('flux', 10.), ('vel', start), ('disp', 1.2), ('e_disp', .2)]})
    catalog = Table({'id': [label], 'wl': [rest]})

    def profile(velocity):
        center = rest * (1 + velocity / 299792.458)
        return 10 / (np.sqrt(2*np.pi)*1.2) * np.exp(-.5*((wave-center)/1.2)**2)

    calls = []

    def solve(*args, **kw):
        calls.append(kw)
        second = len(calls) == 2
        velocity = kw['initial_velocity']
        model = profile(velocity)
        if second and behavior == 'worse':
            model = -model
        return {'success': not (second and behavior == 'failed'),
                'flux': np.array([10.]), 'e_flux': np.array([1.]),
                'vel': np.array([velocity]), 'e_vel': np.array([1.]),
                'disp': np.array([1.2]), 'e_disp': np.array([.1]), 'model': model}

    with patch.object(pek_tools, 'fit_linked_lines', solve):
        table, model = pek_tools.fit_pek_rss(
            profile(0.), error, 0., 1.2, catalog, pm, Table({'id_fib': [0]}),
            wl=wave, lsf_spectrum=lsf, bad_pixel_mask=bad, velocity_tolerance=150.,
            dispersion_tolerance=3., map_velocity_bounds=(-75., 75.),
            restart_velocity=None if behavior == 'disabled' else 0.,
            fixed_velocity=start if behavior == 'fixed' else None)
    if behavior in ('accepted', 'worse', 'failed'):
        assert len(calls) == 2
        assert calls[1]['fit_window_velocity'] == start
        assert calls[1]['initial_velocity'] == 0.
        for key in ('velocity_bounds', 'sigma_kms_bounds', 'initial_sigma_kms'):
            assert calls[1][key] == calls[0][key]
        assert np.array_equal(calls[1]['bad_pixel_mask'], bad)
    else:
        assert len(calls) == 1
    assert table[f'vel_{tag}_pek'][0] == (0. if behavior == 'accepted' else start)
    assert np.isnan(model[bad]).all()


def check_linked_restart_window():
    wave = np.arange(4840., 4880., .5)
    flux = np.exp(-.5 * ((wave - 4861.36)/1.2)**2)
    common = dict(initial_sigma_kms=20., velocity_bounds=(-75., 75.), fit_padding=4.)
    reference = fit_linked_lines(wave, flux, np.ones_like(wave), [4861.36],
                                 np.ones_like(wave), initial_velocity=60., **common)
    alternate = fit_linked_lines(wave, flux, np.ones_like(wave), [4861.36],
                                 np.ones_like(wave), initial_velocity=0.,
                                 fit_window_velocity=60., **common)
    assert reference['success'] and alternate['success']
    assert reference['n_pixels'] == alternate['n_pixels']
    np.testing.assert_allclose(reference['chi2'], alternate['chi2'], atol=1e-8)
