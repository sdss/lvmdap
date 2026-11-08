"""Bounded-overhead batching of wide per-fiber Astropy Tables.

Stores at most ``batch_size`` small Tables at a time, compacting each
batch into one contiguous Table. The order and outer-join column semantics
are the same as a final ``astropy.table.vstack`` on the original sequence.
Only the representation in memory changes; no scientific values are edited.
"""

from astropy.table import vstack as tab_vstack


class TableAccumulator:
    """Accumulate Tables while avoiding thousands of tiny column objects.

    ``append`` accepts one Table (of any number of rows); ``finish`` returns
    the final joined Table. No on-disk intermediate or per-column dtype
    conversion is performed. Uses Astropy's normal outer-join and metadata
    conflict handling. Repeatedly stacking into an ever-growing result is
    intentionally avoided: each compact batch is joined only at the end.
    """

    def __init__(self, batch_size=64):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self._batch_size = batch_size
        self._pending = []
        self._chunks = []
        self._nrows = 0

    def append(self, table):
        self._pending.append(table)
        self._nrows += len(table)
        if len(self._pending) >= self._batch_size:
            # Release the original per-fiber Tables after one compact vstack.
            chunk = tab_vstack(self._pending, join_type='outer')
            self._chunks.append(chunk)
            self._pending.clear()

    def finish(self):
        # Calling Astropy on the full *sequence of compact chunks*, rather
        # than growing a master table at every iteration, avoids O(N^2)
        # reallocation. The final stack also handles the last partial batch.
        result = tab_vstack([*self._chunks, *self._pending], join_type='outer')
        self._pending.clear()
        self._chunks.clear()
        self._nrows = 0
        return result

    def memory_summary(self):
        """Constant-time counters for optional memory diagnostics."""
        return {'rows': self._nrows, 'chunks': len(self._chunks),
                'pending_tables': len(self._pending),
                'batch_size': self._batch_size}
