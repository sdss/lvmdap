#!/usr/bin/env python3
"""Add Pipe3D-style NAME_<plane> FITS headers to an existing LVM-DAP cube.

Uses the existing PLANE_INDEX table, without re-fitting or modifying the
science data. Each HDU's binary payload is copied *byte-for-byte* with fixed
memory usage. This also works for large .fits.gz cube products.

Examples
--------
python tools/add_dap_cube_plane_names.py orion_cube_test.cube.dap.fits.gz
python tools/add_dap_cube_plane_names.py orion_cube_test.cube.dap.fits.gz --in-place
"""
from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
import shutil
import tempfile

from astropy.io import fits

_CHUNK_SIZE = 8 * 1024 * 1024


def _text(value):
    return value.decode('utf-8').strip() if isinstance(value, bytes) else str(value).strip()


def _destination_for(source: Path) -> Path:
    name = source.name
    if name.endswith('.fits.gz'):
        return source.with_name(name[:-8] + '.named.fits.gz')
    if name.endswith('.fits'):
        return source.with_name(name[:-5] + '.named.fits')
    raise ValueError('Input filename must end in .fits or .fits.gz')


def _copy_n_bytes(source, target, nbytes: int):
    while nbytes:
        data = source.read(min(_CHUNK_SIZE, nbytes))
        if not data:
            raise IOError('Unexpected end of FITS input while copying image/table bytes')
        target.write(data)
        nbytes -= len(data)


def _plane_names(index):
    required = {'HDU', 'PLANE', 'PARAMETER'}
    if not required.issubset(index.names):
        raise ValueError(f'PLANE_INDEX must have {sorted(required)} columns')
    grouped = {}
    for row in index:
        name, plane, label = (_text(row['HDU']), int(row['PLANE']),
                              _text(row['PARAMETER']))
        if plane < 0:
            raise ValueError(f'Negative plane number for {name}: {plane}')
        entry = grouped.setdefault(name.upper(), {})
        if plane in entry:
            raise ValueError(f'Duplicate PLANE_INDEX {name} plane {plane}')
        if name.upper() == 'COEFFS':
            try:
                column, rsp = label.rsplit('@rsp=', 1)
                entry[plane] = (column, int(rsp))
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Malformed COEFFS PLANE_INDEX label {label!r}') from exc
        else:
            entry[plane] = (label, None)
    return grouped


def _edit_header(header, name, mapping):
    if mapping is None:
        return header
    nplanes = int(header['NAXIS3'])
    if set(mapping) != set(range(nplanes)):
        raise ValueError(
            f'{name}: PLANE_INDEX describes {len(mapping)} planes but image has {nplanes}; '
            'indexes must be contiguous and zero-based')
    result = header.copy()
    for plane, (column, template_id) in sorted(mapping.items()):
        result[f'NAME_{plane}'] = (column, 'Original DAP table column')
        if template_id is not None:
            result[f'RSP_{plane}'] = (template_id, 'RSP template index')
    # An existing HDU CHECKSUM covers header bytes, which have changed.
    # Keep DATASUM: the entire image data/padding is copied without change.
    if 'CHECKSUM' in result:
        del result['CHECKSUM']
    return result


def add_names(source, destination, *, overwrite=False):
    """Rewrite only FITS headers; leave each HDU's data bytes unchanged."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source != destination and destination.exists() and not overwrite:
        raise FileExistsError(f'{destination} already exists (use --overwrite)')
    if source.suffix not in ('.fits', '.gz'):
        raise ValueError('Source must be .fits or .fits.gz')
    if destination.suffix not in ('.fits', '.gz'):
        raise ValueError('Destination must be .fits or .fits.gz')

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.lvmdap_plane_names_',
                                     dir=destination.parent) as tmpdir:
        plain_input = Path(tmpdir) / 'input.fits'
        plain_output = Path(tmpdir) / 'output.fits'
        if source.suffix == '.gz':
            with gzip.open(source, 'rb') as input_stream, plain_input.open('wb') as out:
                shutil.copyfileobj(input_stream, out, length=_CHUNK_SIZE)
        else:
            # Separate source and destination means input may be updated atomically.
            shutil.copyfile(source, plain_input)

        n_named = 0
        with fits.open(plain_input, memmap=True, do_not_scale_image_data=True,
                       lazy_load_hdus=False) as hdus:
            if 'PLANE_INDEX' not in hdus:
                raise ValueError('No PLANE_INDEX table: this is not a cube-mode DAP product')
            groups = _plane_names(hdus['PLANE_INDEX'].data)
            seen = set()
            with plain_input.open('rb') as fin, plain_output.open('wb') as fout:
                for hdu in hdus:
                    name = _text(hdu.header.get('EXTNAME', 'PRIMARY')).upper()
                    mapping = groups.get(name)
                    if mapping is not None:
                        if not isinstance(hdu, (fits.ImageHDU, fits.PrimaryHDU)):
                            raise ValueError(f'{name}: plane catalogue refers to non-image HDU')
                        seen.add(name)
                        n_named += len(mapping)
                    header = _edit_header(hdu.header, name, mapping)
                    # FITS headers are padded to 2880 bytes. Image/table data
                    # spans are copied raw, including their existing padding.
                    fout.write(header.tostring(sep='', endcard=True,
                                                padding=True).encode('ascii'))
                    info = hdu.fileinfo()
                    fin.seek(info['datLoc'])
                    _copy_n_bytes(fin, fout, int(info['datSpan']))
            if seen != set(groups):
                raise ValueError(f'No matching image HDU for: {sorted(set(groups) - seen)}')

        # Validate structural integrity before publishing the output.
        with fits.open(plain_output, memmap=True,
                       do_not_scale_image_data=True) as output_hdus:
            for name, mapping in groups.items():
                hdr = output_hdus[name].header
                for plane, (column, rsp_id) in mapping.items():
                    if hdr[f'NAME_{plane}'] != column:
                        raise RuntimeError(f'{name}: NAME_{plane} verification failed')
                    if rsp_id is not None and hdr[f'RSP_{plane}'] != rsp_id:
                        raise RuntimeError(f'{name}: RSP_{plane} verification failed')

        if destination.suffix == '.gz':
            final_temp = Path(tmpdir) / 'final.fits.gz'
            with plain_output.open('rb') as fin, gzip.open(final_temp, 'wb',
                                                           compresslevel=4) as out:
                shutil.copyfileobj(fin, out, length=_CHUNK_SIZE)
        else:
            final_temp = plain_output
        # Preserve the original access permissions, including in --in-place mode.
        os.chmod(final_temp, source.stat().st_mode & 0o7777)
        # Final destination appears only after the entire file is complete.
        os.replace(final_temp, destination)
    return n_named


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    target = parser.add_mutually_exclusive_group()
    target.add_argument('--output', type=Path,
                        help='Output path (default: add .named before .fits[.gz])')
    target.add_argument('--in-place', action='store_true',
                        help='Replace original file atomically, after validation')
    parser.add_argument('--overwrite', action='store_true',
                        help='Permit replacing an existing --output file')
    args = parser.parse_args(argv)
    if args.in_place:
        dest = args.source
    elif args.output is not None:
        dest = args.output
    else:
        dest = _destination_for(args.source)
    nplanes = add_names(args.source, dest, overwrite=args.overwrite)
    print(f'# Added NAME_<plane> labels to {nplanes} parameter planes')
    print(f'# Output: {dest}')


if __name__ == '__main__':
    main()
