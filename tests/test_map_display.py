from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import numpy as np

from lvmdap import dap_tools


class TestMapDisplay(unittest.TestCase):
    def setUp(self):
        flux = np.linspace(10., 20., 101)
        flux[:2] = [-1e6, 1e6]
        self.tab = Table({'ra': np.linspace(67., 68.3, len(flux)),
                          'dec': np.linspace(-69.2, -68.8, len(flux)),
                          'flux_Halpha_6562.85': flux,
                          'r': flux, 'g': flux/2., 'b': flux/4.})

    def test_scalar_masks_negative_and_limits_outlier(self):
        with tempfile.TemporaryDirectory() as out, patch.object(
                dap_tools, 'scatter', wraps=dap_tools.scatter) as points:
            dap_tools.map_plot_DAP(self.tab, figs_dir=out)
            args, kwargs = points.call_args
            self.assertEqual(len(args[0]), 100)
            self.assertLess(kwargs['norm'].vmax, 21.)
            self.assertGreater(kwargs['norm'].vmin, 0.)
            self.assertAlmostEqual(args[2].get_aspect(), 1 / np.cos(np.deg2rad(69)))
            self.assertTrue((Path(out)/'junk.png').is_file())

    def test_rgb_preserves_color_order_and_sky_projection(self):
        with tempfile.TemporaryDirectory() as out, patch.object(
                dap_tools, 'scatter', wraps=dap_tools.scatter) as points:
            dap_tools.map_plot_DAP_rgb(self.tab, rgb_key=('r','g','b'), figs_dir=out)
            args, kwargs = points.call_args
            colors = kwargs['color']
            self.assertEqual(len(colors), 100)
            self.assertGreater(colors[50, 0], colors[50, 1])
            self.assertGreater(colors[50, 1], colors[50, 2])
            self.assertGreater(colors[50, 0], .5)
            self.assertAlmostEqual(args[2].get_aspect(), 1 / np.cos(np.deg2rad(69)))
            self.assertTrue((Path(out)/'junk.png').is_file())
