#!/usr/bin/env python3
"""Compare two LVM-DAP maps fiber-by-fiber, with marginal KDE distributions.

Usage:
    python LVMcompare_DAP.py first.dap.fits.gz second.dap.fits.gz \
        --column1 flux_pe_6562.85 --column2 flux_pe_6562.85

Requires: numpy, astropy, matplotlib, scipy, tkinter; lvmdap is optional.
The fiber match uses the PT 'id' column. Only IDs present in both files,
with valid coordinates, masks (when enabled), and finite parameter values,
contribute to the comparison.
"""
from __future__ import annotations

import argparse
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

try:
    from lvmdap.dap_tools import read_DAP_file
except ImportError:
    read_DAP_file = None

FIBER_DIAM_ARCSEC = 35.6
EXTENSIONS = ('RSP', 'NP_ELINES_B', 'NP_ELINES_R', 'NP_ELINES_I',
              'PM_ELINES_B', 'PM_ELINES_R', 'PM_ELINES_I',
              'PEK_ELINES_B', 'PEK_ELINES_R', 'PEK_ELINES_I',
              'ELINES_SIGMA_CHI')
DEFAULT_COLUMNS = ('flux_Halpha_6562.85', 'flux_pe_6562.85',
                   'flux_Halpha_6562.85_pm', 'med_flux_st')


def as_text(column):
    a = np.asarray(column)
    if a.dtype.kind == 'S':
        a = np.char.decode(a, 'utf-8')
    return np.char.strip(a.astype(str))


def read_dap_simple(path):
    """Standalone fallback following the wide-extension strategy of LVMview."""
    with fits.open(path, memmap=True) as hdul:
        tab = Table(hdul['PT'].data)
        ids = as_text(tab['id'])
        for ext in EXTENSIONS:
            if ext not in hdul:
                continue
            t = Table(hdul[ext].data)
            if 'id' not in t.colnames:
                continue
            pos = {key: i for i, key in enumerate(as_text(t['id']))}
            row = np.array([pos.get(key, -1) for key in ids])
            ok = row >= 0
            for col in t.colnames:
                if col == 'id' or t[col].ndim != 1 or t[col].dtype.kind not in 'fiub':
                    continue
                name = f'{col}_st' if ext == 'RSP' else col
                if name in tab.colnames:
                    continue
                vals = np.full(len(tab), np.nan)
                values = np.ma.filled(np.ma.asarray(t[col], dtype=float), np.nan)
                vals[ok] = values[row[ok]]
                tab[name] = vals
    return tab


def load_dap(path):
    pt = Table(fits.getdata(path, 'PT'))
    for required in ('id', 'ra', 'dec'):
        if required not in pt.colnames:
            raise ValueError(f'{os.path.basename(path)}: missing PT column {required!r}')
    tab = read_DAP_file(path) if read_DAP_file is not None else read_dap_simple(path)
    positions = {key: i for i, key in enumerate(as_text(tab['id']))}
    order = np.array([positions.get(key, -1) for key in as_text(pt['id'])])
    ok = order >= 0
    cols = {}
    for key in tab.colnames:
        col = tab[key]
        if col.ndim != 1 or col.dtype.kind not in 'fiub':
            continue
        values = np.ma.filled(np.ma.asarray(col, dtype=float), np.nan)
        out = np.full(len(pt), np.nan)
        out[ok] = values[order[ok]]
        cols[key] = out
    return {'path': path, 'pt': pt, 'ids': as_text(pt['id']),
            'ra': np.asarray(pt['ra'], dtype=float),
            'dec': np.asarray(pt['dec'], dtype=float),
            'good': (np.asarray(pt['mask'], dtype=bool) if 'mask' in pt.colnames
                     else np.ones(len(pt), dtype=bool)), 'cols': cols}


def match_fibers(a, b):
    """Return paired PT-row indices, in first-file order; duplicate IDs fail."""
    for data in (a, b):
        if len(set(data['ids'])) != len(data['ids']):
            raise ValueError(f"Duplicate PT fiber IDs in {data['path']}")
    where_b = {key: i for i, key in enumerate(b['ids'])}
    ia = np.array([i for i, key in enumerate(a['ids']) if key in where_b], dtype=int)
    ib = np.array([where_b[a['ids'][i]] for i in ia], dtype=int)
    return ia, ib


def safe_limits(values, scale):
    valid = values[np.isfinite(values)]
    if scale == 'log':
        valid = valid[valid > 0]
    if not valid.size:
        return (1., 10.) if scale == 'log' else (0., 1.)
    lo, hi = np.min(valid), np.max(valid)
    if lo == hi:
        return ((lo / 1.2, hi * 1.2) if scale == 'log'
                else (lo - max(abs(lo) * 0.05, 1.), hi + max(abs(hi) * 0.05, 1.)))
    return (float(lo), float(hi))


def marginal_kde(ax, values, scale, limit, vertical=False):
    """Histogram and Gaussian KDE, in linear or log10-coordinate space."""
    z = np.log10(values) if scale == 'log' else values
    low, high = (np.log10(limit) if scale == 'log' else np.asarray(limit, float))
    grid = np.linspace(low, high, 250)
    # Histogram normalized per linear / log10-coordinate unit, respectively.
    counts, edges = np.histogram(z, bins=35, range=(low, high), density=True)
    mid = 0.5 * (edges[:-1] + edges[1:])
    curve = None
    if len(z) >= 3 and np.ptp(z) > 0:
        try:
            curve = gaussian_kde(z)(grid)
        except (ValueError, np.linalg.LinAlgError):
            pass
    to_axis = (lambda t: 10.0 ** t) if scale == 'log' else (lambda t: t)
    if vertical:
        ax.plot(counts, to_axis(mid), drawstyle='steps-mid', color='0.65', lw=1)
        if curve is not None:
            ax.plot(curve, to_axis(grid), color='tab:blue', lw=1.8)
            ax.fill_betweenx(to_axis(grid), 0, curve, alpha=0.12, color='tab:blue')
        ax.set_ylim(limit)
        ax.set_yscale(scale)
        ax.set_xlabel('Density', fontsize=9)
        ax.tick_params(axis='y', labelleft=False)
    else:
        ax.plot(to_axis(mid), counts, drawstyle='steps-mid', color='0.65', lw=1)
        if curve is not None:
            ax.plot(to_axis(grid), curve, color='tab:blue', lw=1.8)
            ax.fill_between(to_axis(grid), 0, curve, alpha=0.12, color='tab:blue')
        ax.set_xlim(limit)
        ax.set_xscale(scale)
        ax.set_ylabel('Density', fontsize=9)
        ax.tick_params(axis='x', labelbottom=False)
    ax.grid(alpha=0.15)


class CompareApp(tk.Tk):
    def __init__(self, file1=None, file2=None, column1=None, column2=None):
        super().__init__()
        self.title('LVM-DAP: map-to-map comparison')
        self.geometry('1850x850')
        self.data = [None, None]
        self.matches = None
        self.columns = [tk.StringVar(value=column1 or ''), tk.StringVar(value=column2 or '')]
        self.paths = [tk.StringVar(value=file1 or ''), tk.StringVar(value=file2 or '')]
        self.filters = [tk.StringVar(value=''), tk.StringVar(value='')]
        self.only_good = tk.BooleanVar(value=True)
        self.same_cbar = tk.BooleanVar(value=False)
        self.cmap = tk.StringVar(value='Spectral_r')
        self.xscale = tk.StringVar(value='linear')
        self.yscale = tk.StringVar(value='linear')
        self.limits = {key: tk.StringVar(value='') for key in ('xmin', 'xmax', 'ymin', 'ymax')}
        self.status = tk.StringVar(value='Load two DAP files. Empty X/Y limits mean automatic limits.')
        self._build_ui()
        self.after(100, self._initial_load)

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(self, padding=8)
        sidebar.grid(row=0, column=0, sticky='ns')
        for k in range(2):
            frame = ttk.LabelFrame(sidebar, text=f'DAP {k + 1}', padding=6)
            frame.pack(fill='x', pady=(0, 7))
            ttk.Entry(frame, textvariable=self.paths[k], width=37).pack(fill='x')
            ttk.Button(frame, text='Open DAP…', command=lambda j=k: self.choose_file(j)).pack(fill='x')
            ttk.Label(frame, text='Filter columns (regex or substring)').pack(anchor='w')
            ttk.Entry(frame, textvariable=self.filters[k]).pack(fill='x')
            lbframe = ttk.Frame(frame)
            lbframe.pack(fill='x')
            lb = tk.Listbox(lbframe, height=8, width=35, exportselection=False)
            lb.pack(side='left', expand=True, fill='both')
            scrollbar = ttk.Scrollbar(lbframe, orient='vertical', command=lb.yview)
            scrollbar.pack(side='right', fill='y')
            lb.configure(yscrollcommand=scrollbar.set)
            lb.bind('<<ListboxSelect>>', lambda event, j=k: self.select_column(j))
            self.filters[k].trace_add('write', lambda *_args, j=k: self.fill_list(j))
            setattr(self, f'list{k}', lb)
        opt = ttk.LabelFrame(sidebar, text='Maps', padding=6)
        opt.pack(fill='x', pady=(0, 7))
        ttk.Checkbutton(opt, text='PT mask == True only', variable=self.only_good,
                        command=self.redraw).pack(anchor='w')
        ttk.Checkbutton(opt, text='Same color scale on both maps', variable=self.same_cbar,
                        command=self.redraw).pack(anchor='w')
        cm = ttk.Combobox(opt, textvariable=self.cmap, values=sorted(colormaps), width=25)
        cm.pack(fill='x')
        cm.bind('<<ComboboxSelected>>', lambda _event: self.redraw())
        savebar = ttk.LabelFrame(sidebar, text='Save individual panels', padding=6)
        savebar.pack(fill='x', pady=5)
        for idx, label in enumerate(('Save DAP 1…', 'Save DAP 2…', 'Save comparison…')):
            ttk.Button(savebar, text=label, command=lambda j=idx: self.save_figure(j)).pack(fill='x', pady=1)
        ttk.Label(sidebar, textvariable=self.status, wraplength=285,
                  justify='left').pack(fill='x', pady=8)

        # Three *independent* Tk frames, canvases and navigation toolbars.
        # Drag either sash to give the comparison (or either map) more space.
        right = ttk.Frame(self, padding=4)
        right.grid(row=0, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.panes = ttk.Panedwindow(right, orient=tk.HORIZONTAL)
        self.panes.grid(row=0, column=0, sticky='nsew')
        self.figures = []
        self.canvases = []
        for i, title in enumerate(('DAP 1 map', 'DAP 2 map', 'One-to-one comparison')):
            frame = ttk.Frame(self.panes, padding=2)
            # Each panel is independently resizable, with its own canvas and toolbar.
            frame.rowconfigure(2, weight=1)
            frame.columnconfigure(0, weight=1)
            self.panes.add(frame, weight=(1 if i < 2 else 2))
            ttk.Label(frame, text=title, anchor='center').grid(row=0, column=0, sticky='ew')
            if i == 2:
                self._build_comparison_controls(frame)
            else:
                # Keep the same grid layout for both map panels.
                ttk.Frame(frame).grid(row=1, column=0, sticky='ew')
            fig = Figure(figsize=(5 if i < 2 else 7, 6), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().grid(row=2, column=0, sticky='nsew')
            toolbar_frame = ttk.Frame(frame)
            toolbar_frame.grid(row=3, column=0, sticky='ew')
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            self.figures.append(fig)
            self.canvases.append(canvas)
            fig.text(.5, .5, 'Select two DAP files', ha='center', va='center')
            canvas.draw_idle()

        self._sashes_initialized = False
        self.panes.bind('<Configure>', self._configure_panes)

    def _build_comparison_controls(self, parent):
        """Keep the comparison axis controls directly above its own plot."""
        comp = ttk.LabelFrame(parent, text='One-to-one axes', padding=(5, 3))
        comp.grid(row=1, column=0, sticky='ew', pady=(2, 3))
        # Two compact rows fit even when the comparison pane is narrowed.
        for row_index, (dim, scale) in enumerate((('X', self.xscale), ('Y', self.yscale))):
            line = ttk.Frame(comp)
            line.grid(row=row_index, column=0, sticky='ew', pady=1)
            ttk.Label(line, text=dim + ':', width=2).pack(side='left')
            ttk.Label(line, text='min').pack(side='left')
            minimum = ttk.Entry(line, textvariable=self.limits[dim.lower() + 'min'], width=10)
            minimum.pack(side='left', padx=(2, 5))
            ttk.Label(line, text='max').pack(side='left')
            maximum = ttk.Entry(line, textvariable=self.limits[dim.lower() + 'max'], width=10)
            maximum.pack(side='left', padx=(2, 5))
            ttk.Label(line, text='scale').pack(side='left')
            cb = ttk.Combobox(line, textvariable=scale, values=('linear', 'log'),
                              state='readonly', width=7)
            cb.pack(side='left', padx=(3, 0))
            cb.bind('<<ComboboxSelected>>', lambda _event: self.redraw())
            for entry in (minimum, maximum):
                entry.bind('<Return>', lambda _event: self.redraw())
        buttons = ttk.Frame(comp)
        buttons.grid(row=2, column=0, sticky='w', pady=(3, 0))
        ttk.Button(buttons, text='Apply / Refresh', command=self.redraw).pack(side='left')
        ttk.Button(buttons, text='Auto ranges', command=self.auto_ranges).pack(side='left', padx=(6, 0))

    def _configure_panes(self, _event=None):
        if self._sashes_initialized:
            return
        width = self.panes.winfo_width()
        if width < 450:
            return
        # Initial proportions ~27% / 27% / 46%, subsequently user-resizable.
        self.panes.sashpos(0, int(width * .27))
        self.panes.sashpos(1, int(width * .54))
        self._sashes_initialized = True

    def _initial_load(self):
        for k in range(2):
            if self.paths[k].get():
                self.load_file(k, self.paths[k].get())

    def choose_file(self, k):
        path = filedialog.askopenfilename(title=f'Open DAP {k+1}',
                    filetypes=[('LVM DAP', '*.dap.fits.gz *.dap.fits'),
                               ('FITS', '*.fits *.fits.gz'), ('All', '*.*')])
        if path:
            self.load_file(k, path)

    def load_file(self, k, path):
        try:
            self.configure(cursor='watch')
            self.update_idletasks()
            data = load_dap(path)
            self.data[k] = data
            self.paths[k].set(path)
            current = self.columns[k].get()
            if current not in data['cols']:
                self.columns[k].set(next((c for c in DEFAULT_COLUMNS if c in data['cols']),
                                         next(iter(data['cols']), '')))
            self.fill_list(k)
            if all(d is not None for d in self.data):
                self.matches = match_fibers(*self.data)
            self.redraw()
        except Exception as exc:
            messagebox.showerror('DAP loading error', f'{type(exc).__name__}: {exc}')
        finally:
            self.configure(cursor='')

    def fill_list(self, k):
        lb = getattr(self, f'list{k}')
        lb.delete(0, tk.END)
        data = self.data[k]
        if data is None:
            return
        pat = self.filters[k].get()
        try:
            names = [c for c in data['cols'] if re.search(pat, c, flags=re.I)]
        except re.error:
            names = [c for c in data['cols'] if pat.lower() in c.lower()]
        setattr(self, f'shown{k}', names)
        for c in names:
            lb.insert(tk.END, c)
        if self.columns[k].get() in names:
            row = names.index(self.columns[k].get())
            lb.selection_set(row)
            lb.see(row)

    def select_column(self, k):
        lb = getattr(self, f'list{k}')
        sel = lb.curselection()
        if sel:
            self.columns[k].set(getattr(self, f'shown{k}')[sel[0]])
            self.redraw()

    def auto_ranges(self):
        for var in self.limits.values():
            var.set('')
        self.redraw()

    def _range(self, prefix, values, scale):
        lo, hi = safe_limits(values, scale)
        for key, default in (('min', lo), ('max', hi)):
            raw = self.limits[prefix + key].get().strip()
            if raw:
                val = float(raw)
                if not np.isfinite(val):
                    raise ValueError(f'{prefix}{key}: limit must be finite')
                if scale == 'log' and val <= 0:
                    raise ValueError(f'{prefix}{key}: log-axis limits must be > 0')
                if key == 'min':
                    lo = val
                else:
                    hi = val
        if hi <= lo:
            raise ValueError(f'{prefix.upper()} maximum must exceed minimum')
        return lo, hi

    def _draw_map(self, fig, ax, data, col, values, visible, norm):
        ra = data['ra'][visible]
        dec = data['dec'][visible]
        v = np.ma.masked_invalid(values[visible])
        ok = np.isfinite(ra) & np.isfinite(dec)
        ra, dec, v = ra[ok], dec[ok], v[ok]
        if not ra.size:
            ax.text(.5, .5, 'No visible fibers', transform=ax.transAxes,
                    ha='center', va='center')
            return
        dec0 = float(np.median(dec))
        cosd = max(np.cos(np.deg2rad(dec0)), 1e-3)
        diameter = FIBER_DIAM_ARCSEC / 3600.
        coll = EllipseCollection(widths=np.full(len(ra), diameter / cosd),
                                 heights=np.full(len(ra), diameter),
                                 angles=0, units='xy',
                                 offsets=np.column_stack((ra, dec)),
                                 offset_transform=ax.transData,
                                 cmap=self.cmap.get(), norm=norm)
        coll.set_array(v)
        ax.add_collection(coll)
        ax.set_aspect(1.0 / cosd)
        ax.set_xlim(ra.max() + diameter / cosd, ra.min() - diameter / cosd)
        ax.set_ylim(dec.min() - diameter, dec.max() + diameter)
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('Dec [deg]')
        ax.set_title(f'{os.path.basename(data["path"])}\n{col}', fontsize=10)
        fig.colorbar(coll, ax=ax, fraction=0.047, pad=0.025)

    def redraw(self):
        if any(d is None for d in self.data):
            return
        try:
            a, b = self.data
            ca, cb = (v.get() for v in self.columns)
            if ca not in a['cols'] or cb not in b['cols']:
                return
            aa, bb = a['cols'][ca], b['cols'][cb]
            va = (a['good'] if self.only_good.get() else np.ones(len(aa), bool))
            vb = (b['good'] if self.only_good.get() else np.ones(len(bb), bool))
            norms = []
            for values, vis in ((aa, va), (bb, vb)):
                finite = values[vis & np.isfinite(values)]
                if finite.size:
                    vmin, vmax = np.percentile(finite, (2, 98))
                    if vmax <= vmin:
                        vmin, vmax = safe_limits(finite, 'linear')
                else:
                    vmin, vmax = 0., 1.
                norms.append((vmin, vmax))
            if self.same_cbar.get():
                low = min(v[0] for v in norms)
                high = max(v[1] for v in norms)
                norms = [(low, high)] * 2

            ia, ib = self.matches if self.matches is not None else match_fibers(a, b)
            x, y = aa[ia], bb[ib]
            keep = va[ia] & vb[ib] & np.isfinite(x) & np.isfinite(y)
            keep &= (np.isfinite(a['ra'][ia]) & np.isfinite(a['dec'][ia]) &
                     np.isfinite(b['ra'][ib]) & np.isfinite(b['dec'][ib]))
            if self.xscale.get() == 'log':
                keep &= (x > 0)
            if self.yscale.get() == 'log':
                keep &= (y > 0)
            x, y = x[keep], y[keep]
            xlim = self._range('x', x, self.xscale.get())
            ylim = self._range('y', y, self.yscale.get())
            inside = ((x >= xlim[0]) & (x <= xlim[1]) &
                      (y >= ylim[0]) & (y <= ylim[1]))
            x, y = x[inside], y[inside]

            # Each panel owns its own Figure; redrawing the comparison does
            # not alter the map figures or their individual geometry.
            fig1, fig2, fig3 = self.figures
            for fig in self.figures:
                fig.clear()
            m1 = fig1.add_subplot(111)
            m2 = fig2.add_subplot(111)
            fig1.subplots_adjust(left=.17, right=.92, bottom=.11, top=.88)
            fig2.subplots_adjust(left=.17, right=.92, bottom=.11, top=.88)
            # Dedicated marginal axes remain proportional within panel 3.
            sg = fig3.add_gridspec(2, 2, height_ratios=[1, 4],
                                   width_ratios=[4, 1], hspace=.055, wspace=.055,
                                   left=.15, right=.93, bottom=.12, top=.91)
            top = fig3.add_subplot(sg[0, 0])
            central = fig3.add_subplot(sg[1, 0], sharex=top)
            side = fig3.add_subplot(sg[1, 1], sharey=central)
            self._draw_map(fig1, m1, a, ca, aa, va, Normalize(*norms[0]))
            self._draw_map(fig2, m2, b, cb, bb, vb, Normalize(*norms[1]))
            central.scatter(x, y, s=8, alpha=.45, color='tab:blue', rasterized=True)
            central.set_xscale(self.xscale.get())
            central.set_yscale(self.yscale.get())
            central.set_xlim(xlim)
            central.set_ylim(ylim)
            cross_lo = max(xlim[0], ylim[0])
            cross_hi = min(xlim[1], ylim[1])
            if cross_hi > cross_lo:
                central.plot([cross_lo, cross_hi], [cross_lo, cross_hi],
                             color='crimson', linestyle='--', lw=1.5, label='1:1')
                central.legend(loc='upper left', fontsize=9)
            central.set_xlabel(f'DAP 1: {ca}')
            central.set_ylabel(f'DAP 2: {cb}')
            central.grid(alpha=.2)
            if len(x):
                marginal_kde(top, x, self.xscale.get(), xlim)
                marginal_kde(side, y, self.yscale.get(), ylim, vertical=True)
            top.set_title(f'Common fibers: {len(ia)}; plotted: {len(x)}', fontsize=10)
            top.spines[['right', 'top']].set_visible(False)
            side.spines[['top', 'right']].set_visible(False)
            self.status.set(f'Matched by PT id: {len(ia)} fibers. '
                            f'Plotted: {len(x)} (valid mask, finite values, '
                            'axis domains and displayed ranges).\n'
                            'Blue marginal curves: Gaussian KDE; gray: histogram. '
                            'Log KDEs are densities per log10 unit.')
            for canvas in self.canvases:
                canvas.draw_idle()
        except Exception as exc:
            messagebox.showerror('Plot error', f'{type(exc).__name__}: {exc}')

    def save_figure(self, index):
        label = ('DAP_1_map', 'DAP_2_map', 'DAP_comparison')[index]
        path = filedialog.asksaveasfilename(title=f'Save {label}',
                      defaultextension='.png', initialfile=f'LVM_{label}.png',
                      filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'),
                                 ('SVG', '*.svg'), ('All', '*.*')])
        if path:
            try:
                self.figures[index].savefig(path, dpi=175, facecolor='white')
            except Exception as exc:
                messagebox.showerror('Save error', str(exc))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dap1', nargs='?', help='First .dap.fits[.gz]')
    p.add_argument('dap2', nargs='?', help='Second .dap.fits[.gz]')
    p.add_argument('--column1', help='First DAP map parameter')
    p.add_argument('--column2', help='Second DAP map parameter')
    args = p.parse_args()
    CompareApp(args.dap1, args.dap2, args.column1, args.column2).mainloop()


if __name__ == '__main__':
    main()
