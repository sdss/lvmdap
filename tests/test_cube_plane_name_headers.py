"""Regression tests for NAME_<plane> FITS header naming.

Pure header tests run without Astropy; FITS roundtrip requires Astropy.
"""
import ast
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
CUBE_SOURCE = (ROOT / 'lvmdap/cube_io.py').read_text()
RETROFIT_SOURCE = (ROOT / 'tools/add_dap_cube_plane_names.py').read_text()


def _load_helpers():
    tree = ast.parse(RETROFIT_SOURCE)
    keep = {'_text', '_plane_names', '_edit_header'}
    body = [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in keep]
    import_module = ast.Module(body=body, type_ignores=[])
    namespace = {}
    exec(compile(import_module, 'helpers', 'exec'), namespace)
    return namespace


def test_normal_column_names_preserved_as_is():
    helpers = _load_helpers()
    hdr = {'NAXIS3': 2, 'CRVAL1': 83.8, 'CTYPE3': 'PARAM', 'EXTNAME': 'BEST_ELINES_R'}
    edited = helpers['_edit_header'](hdr, 'BEST_ELINES_R', {
        0: ('flux_Halpha_6562.85_best', None),
        1: ('vel_Halpha_6562.85_best', None),
    })
    assert edited['NAME_0'][0] == 'flux_Halpha_6562.85_best'
    assert edited['NAME_1'][0] == 'vel_Halpha_6562.85_best'
    assert edited['CTYPE3'] == 'PARAM'
    assert edited['CRVAL1'] == 83.8
    assert 'NAME_0' not in hdr


def test_coefficient_plane_has_original_name_and_template_id():
    helpers = _load_helpers()
    class Index(list):
        names = ('HDU', 'PLANE', 'PARAMETER')
    rows = Index([
        {'HDU': 'COEFFS', 'PLANE': 1, 'PARAMETER': 'weight@rsp=8'},
        {'HDU': 'COEFFS', 'PLANE': 0, 'PARAMETER': 'weight@rsp=7'},
    ])
    mapping = helpers['_plane_names'](rows)['COEFFS']
    hdr = helpers['_edit_header']({'NAXIS3': 2, 'CHECKSUM': 'outdated'}, 'COEFFS', mapping)
    assert hdr['NAME_0'][0] == 'weight' and hdr['RSP_0'][0] == 7
    assert hdr['NAME_1'][0] == 'weight' and hdr['RSP_1'][0] == 8
    assert 'CHECKSUM' not in hdr


def test_fitting_code_is_not_part_of_patch():
    # The automatic writer now attaches names at creation time; no spectral
    # data / optimizer / selection parameter is part of this change.
    assert "out_header[f'NAME_{k}']" in CUBE_SOURCE
    assert "out_header[f'RSP_{k}']" in CUBE_SOURCE
    assert "result[k, ypix, xpix] = values" in CUBE_SOURCE


def test_fits_roundtrip_preserves_raw_data(tmp_path):
    np = pytest.importorskip('numpy')
    fits = pytest.importorskip('astropy.io.fits')
    table = pytest.importorskip('astropy.table')
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'retrofit', ROOT / 'tools/add_dap_cube_plane_names.py')
    retrofit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retrofit)

    src = tmp_path / 'orion.cube.dap.fits.gz'
    dst = tmp_path / 'orion.named.fits.gz'
    image = np.array([[[1., np.nan]], [[-2., 5.]]], dtype=np.float64)
    hdr = fits.Header()
    hdr['CTYPE1'] = 'RA---TAN'
    hdr['CTYPE2'] = 'DEC--TAN'
    hdr['CTYPE3'] = 'PARAM'
    hdr['CRVAL1'] = 83.8
    hdr['NPLANE'] = 2
    names = table.Table(rows=[
        ('BEST_ELINES_R', 0, 'flux_Halpha_6562.85_best', '', 'NUMERIC'),
        ('BEST_ELINES_R', 1, 'vel_Halpha_6562.85_best', '', 'NUMERIC'),
    ], names=('HDU','PLANE','PARAMETER','UNIT','ENCODING'))
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(image, header=hdr, name='BEST_ELINES_R'),
        fits.BinTableHDU(names, name='PLANE_INDEX'),
    ]).writeto(src)
    n = retrofit.add_names(src, dst)
    assert n == 2
    with fits.open(src) as before, fits.open(dst) as after:
        assert before['BEST_ELINES_R'].header.get('NAME_0') is None
        assert after['BEST_ELINES_R'].header['NAME_0'] == 'flux_Halpha_6562.85_best'
        assert after['BEST_ELINES_R'].header['NAME_1'] == 'vel_Halpha_6562.85_best'
        assert after['BEST_ELINES_R'].header['CTYPE3'] == 'PARAM'
        np.testing.assert_equal(before['BEST_ELINES_R'].data,
                                after['BEST_ELINES_R'].data)
        np.testing.assert_equal(before['PLANE_INDEX'].data['PARAMETER'],
                                after['PLANE_INDEX'].data['PARAMETER'])
