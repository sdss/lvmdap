"""Opt-in, low-overhead DAP memory diagnostics (Linux /proc, no NumPy import).

Enable with LVMDAP_MEM_PROFILE=1; checkpoint spacing is LVMDAP_MEM_EVERY
(default 25 fibers). Does not modify arrays or call garbage collection.
"""
import os
import sys
import time

_ENABLED = os.environ.get('LVMDAP_MEM_PROFILE', '').strip().lower() in ('1','true','yes','on')
try:
    _EVERY = max(1, int(os.environ.get('LVMDAP_MEM_EVERY', '25')))
except (ValueError, TypeError):
    _EVERY = 25


def enabled():
    return _ENABLED


def every():
    return _EVERY


def _status_kib():
    out = {}
    try:
        with open('/proc/self/status', encoding='ascii') as f:
            for line in f:
                if line.startswith(('VmRSS:', 'VmHWM:', 'VmSize:', 'RssAnon:', 'RssFile:')):
                    key, rest = line.split(':', 1)
                    out[key] = int(rest.strip().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return out


def _uss_kib():
    try:
        with open('/proc/self/smaps_rollup', encoding='ascii') as f:
            values = {}
            for line in f:
                if line.startswith(('Private_Clean:', 'Private_Dirty:')):
                    key, rest = line.split(':', 1)
                    values[key] = int(rest.strip().split()[0])
        return values.get('Private_Clean', 0) + values.get('Private_Dirty', 0)
    except (OSError, ValueError, IndexError):
        return None


def _cgroup_mib():
    """Return cgroup v2 memory.current/max when visible in this namespace."""
    roots = ['/sys/fs/cgroup']
    try:
        with open('/proc/self/cgroup', encoding='ascii') as f:
            for line in f:
                fields = line.strip().split(':', 2)
                if len(fields) == 3 and fields[1] == '':
                    path = fields[2].lstrip('/')
                    if path:
                        roots.insert(0, os.path.join('/sys/fs/cgroup', path))
    except OSError:
        pass
    for base in roots:
        path = os.path.join(base, 'memory.current')
        if os.path.isfile(path):
            try:
                with open(path, encoding='ascii') as f:
                    current = int(f.read().strip()) / (1024**2)
                maxpath = os.path.join(base, 'memory.max')
                try:
                    with open(maxpath, encoding='ascii') as f:
                        raw = f.read().strip()
                    maximum = None if raw == 'max' else int(raw) / (1024**2)
                except (OSError, ValueError):
                    maximum = None
                return current, maximum
            except (OSError, ValueError):
                pass
    return None, None


def _nbytes(obj):
    """Count allocated element buffers, not Python-object overhead or alias-safe RSS."""
    try:
        n = obj.nbytes
        return int(n) if n is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _desc(obj):
    shape = getattr(obj, 'shape', None)
    dtype = getattr(obj, 'dtype', None)
    n = _nbytes(obj)
    if n is None:
        return None
    return f'shape={shape} dtype={dtype} nbytes_GiB={n/(1024**3):.4f}'


def _table_bytes(table):
    """Estimate column buffers for an Astropy Table without materializing it."""
    if table is None or not hasattr(table, 'columns'):
        return None
    try:
        return sum(_nbytes(column) or 0 for column in table.columns.values())
    except (AttributeError, TypeError):
        return None


def report(stage, arrays=None, lists=None, sps=None, max_arrays=16):
    if not _ENABLED:
        return
    status = _status_kib()
    uss = _uss_kib()
    cg, cgmax = _cgroup_mib()
    bits = [f'MEM {stage}', f't={time.monotonic():.1f}',
            f'RSS={status.get("VmRSS", 0)/1048576:.3f}GiB',
            f'HWM={status.get("VmHWM", 0)/1048576:.3f}GiB']
    if uss is not None:
        bits.append(f'USS={uss/1048576:.3f}GiB')
    pyplot = sys.modules.get('matplotlib.pyplot')
    if pyplot is not None:
        try:
            bits.append(f'FIGS_OPEN={len(pyplot.get_fignums())}')
        except Exception:
            pass
    if cg is not None:
        bits.append(f'CGROUP={cg/1024:.3f}GiB')
        if cgmax is not None:
            bits.append(f'CGROUP_LIMIT={cgmax/1024:.3f}GiB')
    print('# ' + ' '.join(bits), flush=True)
    if arrays:
        for name, obj in list(arrays.items())[:max_arrays]:
            msg = _desc(obj)
            if msg is not None:
                print(f'# MEM_ARRAY {stage} {name} {msg}', flush=True)
            elif hasattr(obj, 'columns'):
                n = _table_bytes(obj)
                if n is not None:
                    print(f'# MEM_TABLE {stage} {name} rows={len(obj)} '
                          f'ncols={len(obj.columns)} column_nbytes_GiB={n/(1024**3):.4f}', flush=True)
    if lists:
        for name, seq in lists.items():
            if seq is None:
                continue
            if hasattr(seq, 'memory_summary'):
                stats = seq.memory_summary()
                print(f'# MEM_ACC {stage} {name} rows={stats["rows"]} '
                      f'chunks={stats["chunks"]} '
                      f'pending_tables={stats["pending_tables"]} '
                      f'batch_size={stats["batch_size"]}', flush=True)
                continue
            try:
                length = len(seq)
                count = sum((_nbytes(v) or 0) for v in seq)
                last = seq[-1] if length else None
                last_tbl = _table_bytes(last)
                extra = (f' last_table_columns_MiB={last_tbl/(1024**2):.2f} last_ncols={len(last.columns)}'
                         if last_tbl is not None else '')
                print(f'# MEM_LIST {stage} {name} len={length} '
                      f'ndarray_buffers_GiB={count/(1024**3):.4f}{extra}',flush=True)
            except (TypeError, ValueError, IndexError):
                pass
    if sps is not None:
        spectra = getattr(sps, 'spectra', None)
        if isinstance(spectra, dict):
            counted = sorted(((name, _nbytes(val) or 0, val)
                              for name,val in spectra.items()),
                             key=lambda item:item[1], reverse=True)
            total = sum(item[1] for item in counted)
            print(f'# MEM_SPS {stage} spectra_arrays={len(counted)} '
                  f'element_buffers_GiB={total/(1024**3):.4f}',flush=True)
            for name, _, val in counted[:6]:
                msg = _desc(val)
                if msg is not None:
                    print(f'# MEM_SPS_ARRAY {stage} {name} {msg}',flush=True)
