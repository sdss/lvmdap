#!/usr/bin/env python3
"""
LVM DAP + MODEL viewer (stand-alone Tk application).

Reads an LVM-DAP product pair:
- DAP file   (*.dap.fits.gz)   : PT table (id, ra, dec, mask, ...) plus all DAP
                                 extensions, merged per fiber by
                                 lvmdap.dap_tools.read_DAP_file (as in
                                 LVMview_DAP_model.ipynb / plot_DAP_tests.ipynb)
- MODEL file (*.model.fits.gz) : primary cube (nchan, nfiber, nwave) with
                                 channel names in NAME0..NAMEn (org_spec,
                                 model_spec, mod_joint_spec, gas_spec, ...)

Features:
(1) Map frame: any numeric column of the DAP table (filterable list).
(2) Map frame: slice map built from the spectral window selected in the
    spectral frame (drag with the left button, middle-click to re-center),
    from any MODEL channel (mean / sum / median over the window).
(3) Spectral frame: every MODEL channel can be switched on/off with buttons.
    Optional PEK models: if the MODEL file has a PEK_MODEL extension and the
    DAP file has PEK tables (PEK_ELINES_* / PM_KEL), a "PEK models" toggle
    adds PEK_model, mod_pek_spec (model_spec + PEK) and res_pek_spec
    (org_spec - mod_pek_spec), also usable as slice channels.

Mouse on the map:
- Left click   : show the spectrum of the nearest fiber
- Middle drag  : browse spectra while moving
- Right click  : add/remove fiber to the selection (mean spectrum)
- "Rect. select" toggle: left-drag a rectangle to add fibers to the selection

Usage:
    python LVMview_DAP_model.py [file.dap.fits.gz] [file.model.fits.gz]
If only the DAP file is given, the MODEL file is guessed from its name.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from astropy.io import fits
from astropy.table import Table

import matplotlib as mpl
import matplotlib.colors as mpl_colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import EllipseCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector, SpanSelector

try:
    from lvmdap.dap_tools import read_DAP_file
except ImportError:  # lvmdap not installed: use the simplified reader below
    read_DAP_file = None


FIBER_DIAM_ARCSEC = 35.6

# Colors for the MODEL channels, in NAME0..NAMEn order
CHANNEL_COLORS = ['black', 'forestgreen', 'red', 'darkorange', 'steelblue',
                  'olive', 'purple', 'teal', 'grey', 'goldenrod']
PEK_COLORS = {'PEK_model': 'magenta', 'mod_pek_spec': 'deeppink', 'res_pek_spec': 'mediumpurple'}


def chan_color(i: int, name: str) -> str:
    return PEK_COLORS.get(name, CHANNEL_COLORS[i % len(CHANNEL_COLORS)])


# ------------------------- Data readers -------------------------

def _text(col) -> np.ndarray:
    arr = np.asarray(col)
    if arr.dtype.kind == 'S':
        arr = np.char.decode(arr, encoding='utf-8')
    return np.char.strip(arr.astype(str))


def _read_dap_simple(dap_file: str) -> Table:
    """Fallback reader: PT plus every wide per-fiber table, matched on 'id'."""
    with fits.open(dap_file, memmap=True) as hdul:
        tab = Table(hdul['PT'].data)
        ids = _text(tab['id'])
        for ext in ('RSP', 'NP_ELINES_B', 'NP_ELINES_R', 'NP_ELINES_I',
                    'PM_ELINES_B', 'PM_ELINES_R', 'PM_ELINES_I',
                    'PEK_ELINES_B', 'PEK_ELINES_R', 'PEK_ELINES_I', 'ELINES_SIGMA_CHI'):
            if ext not in hdul:
                continue
            t = Table(hdul[ext].data)
            if 'id' not in t.colnames:
                continue
            pos = {k: i for i, k in enumerate(_text(t['id']))}
            rows = np.array([pos.get(k, -1) for k in ids])
            for c in t.colnames:
                if c in tab.colnames or t[c].dtype.kind not in 'fiub':
                    continue
                v = np.full(len(tab), np.nan)
                ok = rows >= 0
                v[ok] = np.asarray(t[c], dtype=float)[rows[ok]]
                name = f'{c}_st' if ext == 'RSP' else c
                tab[name if name not in tab.colnames else c] = v
    return tab


def read_dap(dap_file: str) -> tuple[Table, dict[str, np.ndarray]]:
    """
    Return (PT table, {column: float array}) with rows in PT order.

    Why: read_DAP_file joins on 'id', which sorts the rows by the string id,
    while the MODEL cube rows follow the PT order.
    """
    pt = Table(fits.getdata(dap_file, 'PT'))
    tab = read_DAP_file(dap_file) if read_DAP_file is not None else _read_dap_simple(dap_file)

    pos = {k: i for i, k in enumerate(_text(tab['id']))}
    order = np.array([pos.get(k, -1) for k in _text(pt['id'])])
    ok = order >= 0

    cols: dict[str, np.ndarray] = {}
    for c in tab.colnames:
        col = tab[c]
        if col.ndim != 1 or col.dtype.kind not in 'fiub':
            continue
        v = np.ma.filled(np.ma.asarray(col).astype(float), np.nan)
        out = np.full(len(pt), np.nan)
        out[ok] = v[order[ok]]
        cols[c] = out
    return pt, cols


def read_model(model_file: str) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """
    Return (wave, cube[nchan, nfiber, nwave] float32, channel names, n_pek).

    If the file has a PEK_MODEL extension (PEK Gaussian emission lines only),
    n_pek channels are appended after the primary ones, built in the same way
    as mod_joint_spec = model_spec + gas_model_PM and
    res_joint_spec = org_spec - mod_joint_spec:
        PEK_model, mod_pek_spec = model_spec + PEK_model,
        res_pek_spec = org_spec - mod_pek_spec
    """
    with fits.open(model_file) as hdul:
        hdr = hdul[0].header
        cube = np.asarray(hdul[0].data, dtype=np.float32)
        pek = np.asarray(hdul['PEK_MODEL'].data, dtype=np.float32) if 'PEK_MODEL' in hdul else None
    if cube.ndim == 2:
        cube = cube[np.newaxis]
    nchan, _, nwave = cube.shape
    crpix = hdr.get('CRPIX1', 1.0)
    wave = hdr['CRVAL1'] + hdr['CDELT1'] * (np.arange(nwave) - (crpix - 1))
    names = [str(hdr.get(f'NAME{i}', f'chan{i}')) for i in range(nchan)]

    if pek is None or pek.shape != cube.shape[1:]:
        return wave, cube, names, 0
    extra, extra_names = [pek], ['PEK_model']
    if 'model_spec' in names:
        mod_pek = cube[names.index('model_spec')] + pek
        extra.append(mod_pek)
        extra_names.append('mod_pek_spec')
        if 'org_spec' in names:
            extra.append(cube[names.index('org_spec')] - mod_pek)
            extra_names.append('res_pek_spec')
    cube = np.concatenate([cube, np.stack(extra)], axis=0)
    return wave, cube, names + extra_names, len(extra)


def dap_has_pek(dap_file: str) -> bool:
    """True if the DAP file includes the PEK emission-line extensions."""
    with fits.open(dap_file) as hdul:
        return any(h.name.startswith('PEK_ELINES') or h.name == 'PM_KEL' for h in hdul)


def guess_model_file(dap_file: str) -> Optional[str]:
    model_suffixes = ('.model.fits', '.model.fits.gz')
    if dap_file.endswith('.gz'):
        model_suffixes = model_suffixes[::-1]
    for repl in (*model_suffixes, '.output.fits.gz'):
        cand = re.sub(r'\.dap\.fits(\.gz)?$', repl, dap_file)
        if cand != dap_file and os.path.exists(cand):
            return cand
    return None


# ------------------------- Application -------------------------

class DAPViewApp(tk.Tk):
    def __init__(self, dap_file: Optional[str] = None, model_file: Optional[str] = None) -> None:
        super().__init__()
        self.title("LVM DAP/MODEL viewer")
        self.geometry("2000x760")

        # ---- Data ----
        self.dap_path: Optional[str] = None
        self.model_path: Optional[str] = None
        self.pt: Optional[Table] = None
        self.cols: dict[str, np.ndarray] = {}
        self.col_names: list[str] = []
        self.ra: Optional[np.ndarray] = None
        self.dec: Optional[np.ndarray] = None
        self.good: Optional[np.ndarray] = None  # PT mask
        self.wave: Optional[np.ndarray] = None
        self.cube: Optional[np.ndarray] = None
        self.chan_names: list[str] = []

        self.map_vals: Optional[np.ndarray] = None
        self.map_label = ''
        self.dap_col = ''  # last DAP column shown
        self.current_fiber: Optional[int] = None
        self.selected: Optional[np.ndarray] = None

        # ---- UI state ----
        self.var_source = tk.StringVar(value='dap')  # dap | slice
        self.var_filter = tk.StringVar(value='')
        self.var_w1 = tk.DoubleVar(value=6555.0)
        self.var_w2 = tk.DoubleVar(value=6570.0)
        self.var_slice_chan = tk.StringVar(value='')
        self.var_slice_stat = tk.StringVar(value='mean')

        self.var_vmin = tk.DoubleVar(value=0.0)
        self.var_vmax = tk.DoubleVar(value=1.0)
        self.var_plo = tk.DoubleVar(value=2.0)
        self.var_phi = tk.DoubleVar(value=98.0)
        self.var_gamma = tk.DoubleVar(value=1.0)
        self.var_lock = tk.BooleanVar(value=False)
        self.var_cmap = tk.StringVar(value='Spectral_r')
        self.var_only_good = tk.BooleanVar(value=True)
        self.var_rectsel = tk.BooleanVar(value=False)

        self.var_spec_mode = tk.StringVar(value='single')  # single | mean
        self.var_fix_y = tk.BooleanVar(value=False)
        self.var_y1 = tk.DoubleVar(value=-1.0)
        self.var_y2 = tk.DoubleVar(value=1.0)
        self.chan_vars: list[tk.BooleanVar] = []
        self.n_pek = 0            # PEK channels appended to the cube (from PEK_MODEL)
        self.dap_pek = False      # DAP file includes PEK_ELINES_* / PM_KEL
        self.var_show_pek = tk.BooleanVar(value=False)

        # ---- Matplotlib artists ----
        self._coll: Optional[EllipseCollection] = None
        self._cbar = None
        self._sel_line = None
        self._cur_line = None
        self._rect_selector: Optional[RectangleSelector] = None
        self._spec_lines: list = []
        self._win_patch: Optional[Rectangle] = None
        self._span: Optional[SpanSelector] = None
        self._browse = False
        self._browse_after = None

        self._build_ui()

        if dap_file:
            self.after(100, lambda: self.load(dap_file, model_file or guess_model_file(dap_file)))

    # ------------------------- UI -------------------------

    def _build_ui(self) -> None:
        menubar = tk.Menu(self)
        fm = tk.Menu(menubar, tearoff=0)
        fm.add_command(label="Open DAP…", command=self.open_dap)
        fm.add_command(label="Open MODEL…", command=self.open_model)
        fm.add_separator()
        fm.add_command(label="Save map…", command=lambda: self.save_fig(self.fig_map, 'map'))
        fm.add_command(label="Save spectrum…", command=lambda: self.save_fig(self.fig_spec, 'spec'))
        fm.add_separator()
        fm.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=fm)
        self.config(menu=menubar)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.rowconfigure(2, weight=1)
        right = ttk.Frame(self, padding=4)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # ---- Files ----
        ff = ttk.LabelFrame(left, text="Files", padding=6)
        ff.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(ff, text="Open DAP…", command=self.open_dap).pack(side="left", expand=True, fill="x")
        ttk.Button(ff, text="Open MODEL…", command=self.open_model).pack(side="left", expand=True, fill="x")

        # ---- Map source ----
        sf = ttk.LabelFrame(left, text="Map source", padding=6)
        sf.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Radiobutton(sf, text="DAP column", variable=self.var_source, value='dap',
                        command=self.update_map_values).pack(side="left")
        ttk.Radiobutton(sf, text="Spectral slice", variable=self.var_source, value='slice',
                        command=self.update_map_values).pack(side="left", padx=(10, 0))

        # ---- DAP columns ----
        cf = ttk.LabelFrame(left, text="DAP columns", padding=6)
        cf.grid(row=2, column=0, sticky="nsew", pady=(0, 6))
        cf.rowconfigure(1, weight=1)
        cf.columnconfigure(0, weight=1)
        ent = ttk.Entry(cf, textvariable=self.var_filter)
        ent.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.var_filter.trace_add("write", lambda *_: self._fill_col_list())
        self.col_list = tk.Listbox(cf, width=36, height=12, exportselection=False)
        self.col_list.grid(row=1, column=0, sticky="nsew")
        sb = ttk.Scrollbar(cf, orient="vertical", command=self.col_list.yview)
        sb.grid(row=1, column=1, sticky="ns")
        self.col_list.configure(yscrollcommand=sb.set)
        self.col_list.bind("<<ListboxSelect>>", self._on_col_select)
        ttk.Label(cf, text="Filter: substring or regex (e.g. flux_.*6562)",
                  foreground="grey").grid(row=2, column=0, columnspan=2, sticky="w")

        # ---- Spectral slice ----
        wf = ttk.LabelFrame(left, text="Spectral slice (MODEL)", padding=6)
        wf.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        r = ttk.Frame(wf)
        r.pack(fill="x")
        ttk.Label(r, text="W1").pack(side="left")
        ttk.Entry(r, textvariable=self.var_w1, width=9).pack(side="left", padx=(2, 8))
        ttk.Label(r, text="W2").pack(side="left")
        ttk.Entry(r, textvariable=self.var_w2, width=9).pack(side="left", padx=(2, 0))
        r = ttk.Frame(wf)
        r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="Channel").pack(side="left")
        self.cb_slice_chan = ttk.Combobox(r, textvariable=self.var_slice_chan, state="readonly", width=16)
        self.cb_slice_chan.pack(side="left", padx=(2, 6))
        self.cb_slice_chan.bind("<<ComboboxSelected>>", lambda _e: self.set_slice())
        ttk.Combobox(r, textvariable=self.var_slice_stat, values=['mean', 'sum', 'median'],
                     state="readonly", width=7).pack(side="left")
        self.var_slice_stat.trace_add("write", lambda *_: self.set_slice())
        r = ttk.Frame(wf)
        r.pack(fill="x", pady=(4, 0))
        ttk.Button(r, text="SET", command=self.set_slice).pack(side="left", expand=True, fill="x")
        ttk.Button(r, text="<<", width=4, command=lambda: self.step_slice(-1)).pack(side="left", padx=4)
        ttk.Button(r, text=">>", width=4, command=lambda: self.step_slice(+1)).pack(side="left")
        ttk.Label(wf, text="Drag (left) on the spectrum to select;\nmiddle-click to re-center",
                  foreground="grey").pack(anchor="w", pady=(4, 0))

        self.status = ttk.Label(left, text="Open a DAP file to start.", wraplength=300, justify="left")
        self.status.grid(row=4, column=0, sticky="ew")

        # ---- Right: map | spectrum ----
        panes = ttk.Panedwindow(right, orient=tk.HORIZONTAL)
        panes.grid(row=0, column=0, sticky="nsew")
        map_frame = ttk.Frame(panes)
        spec_frame = ttk.Frame(panes)
        panes.add(map_frame, weight=1)
        panes.add(spec_frame, weight=2)
        # Set the sash only once the pane has a real width: before the window is
        # mapped winfo_width() is 1, which would collapse the map pane to zero.
        # Afterwards keep the map pane from being collapsed.
        self._sash_init = False

        def _on_panes_configure(_evt=None) -> None:
            total = panes.winfo_width()
            if total < 300:
                return
            if not self._sash_init:
                panes.sashpos(0, int(total * 0.36))
                self._sash_init = True
            elif panes.sashpos(0) < 150:
                panes.sashpos(0, min(int(total * 0.36), 150))

        panes.bind("<Configure>", _on_panes_configure)

        # Map scale bar
        mb = ttk.Frame(map_frame)
        mb.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(mb, text="min").pack(side="left")
        e = ttk.Entry(mb, textvariable=self.var_vmin, width=9)
        e.pack(side="left", padx=(2, 4))
        e.bind("<Return>", lambda _e: self.draw_map())
        ttk.Label(mb, text="max").pack(side="left")
        e = ttk.Entry(mb, textvariable=self.var_vmax, width=9)
        e.pack(side="left", padx=(2, 4))
        e.bind("<Return>", lambda _e: self.draw_map())
        ttk.Label(mb, text="γ").pack(side="left")
        e = ttk.Entry(mb, textvariable=self.var_gamma, width=4)
        e.pack(side="left", padx=(2, 4))
        e.bind("<Return>", lambda _e: self.draw_map())
        ttk.Button(mb, text="SET", width=4, command=self.draw_map).pack(side="left")
        ttk.Button(mb, text="Auto", width=5, command=lambda: self.auto_scale(redraw=True)).pack(side="left", padx=2)
        e = ttk.Entry(mb, textvariable=self.var_plo, width=4)
        e.pack(side="left")
        ttk.Label(mb, text="–").pack(side="left")
        e = ttk.Entry(mb, textvariable=self.var_phi, width=4)
        e.pack(side="left")
        ttk.Label(mb, text="%").pack(side="left")
        ttk.Checkbutton(mb, text="Lock", variable=self.var_lock).pack(side="left", padx=4)

        mb2 = ttk.Frame(map_frame)
        mb2.pack(side=tk.TOP, fill=tk.X)
        cmaps = sorted(mpl.colormaps, key=str.lower)
        ttk.Label(mb2, text="cmap").pack(side="left")
        cb = ttk.Combobox(mb2, textvariable=self.var_cmap, values=cmaps, width=14)
        cb.pack(side="left", padx=(2, 6))
        cb.bind("<<ComboboxSelected>>", lambda _e: self.draw_map())
        cb.bind("<Return>", lambda _e: self.draw_map())
        ttk.Checkbutton(mb2, text="mask==True only", variable=self.var_only_good,
                        command=lambda: self.draw_map(reset_view=True)).pack(side="left", padx=4)
        ttk.Checkbutton(mb2, text="Rect. select", variable=self.var_rectsel,
                        command=self._toggle_rect).pack(side="left", padx=4)
        ttk.Button(mb2, text="Clear sel.", command=self.clear_selection).pack(side="left", padx=4)

        self.fig_map = Figure(figsize=(6, 6), dpi=100)
        self.ax_map = self.fig_map.add_subplot(111)
        self.fig_map.subplots_adjust(left=0.17, right=0.86, bottom=0.10, top=0.93)
        self.cax_map = self.fig_map.add_axes([0.88, 0.10, 0.03, 0.83])
        self.cax_map.set_axis_off()
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=map_frame)
        NavigationToolbar2Tk(self.canvas_map, map_frame).pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_map.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_map.mpl_connect("button_press_event", self._on_map_press)
        self.canvas_map.mpl_connect("button_release_event", self._on_map_release)
        self.canvas_map.mpl_connect("motion_notify_event", self._on_map_motion)

        # Spectrum bar: channel buttons
        self.chan_bar = ttk.Frame(spec_frame)
        self.chan_bar.pack(side=tk.TOP, fill=tk.X)
        sb2 = ttk.Frame(spec_frame)
        sb2.pack(side=tk.TOP, fill=tk.X)
        ttk.Radiobutton(sb2, text="Single fiber", variable=self.var_spec_mode, value='single',
                        command=self.draw_spectrum).pack(side="left")
        ttk.Radiobutton(sb2, text="Mean of selection", variable=self.var_spec_mode, value='mean',
                        command=self.draw_spectrum).pack(side="left", padx=(6, 12))
        ttk.Checkbutton(sb2, text="Fix Y", variable=self.var_fix_y,
                        command=self.draw_spectrum).pack(side="left")
        ttk.Label(sb2, text="Y1").pack(side="left", padx=(6, 0))
        e = ttk.Entry(sb2, textvariable=self.var_y1, width=9)
        e.pack(side="left", padx=2)
        e.bind("<Return>", lambda _e: self.draw_spectrum())
        ttk.Label(sb2, text="Y2").pack(side="left")
        e = ttk.Entry(sb2, textvariable=self.var_y2, width=9)
        e.pack(side="left", padx=2)
        e.bind("<Return>", lambda _e: self.draw_spectrum())
        ttk.Button(sb2, text="Auto Y", command=self._auto_y_button).pack(side="left", padx=4)
        ttk.Button(sb2, text="Reset zoom", command=self.reset_spec_zoom).pack(side="left", padx=4)

        self.fig_spec = Figure(figsize=(10, 5), dpi=100)
        self.ax_spec = self.fig_spec.add_subplot(111)
        self.fig_spec.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.93)
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=spec_frame)
        NavigationToolbar2Tk(self.canvas_spec, spec_frame).pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_spec.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_spec.mpl_connect("button_press_event", self._on_spec_press)

        self._empty_message(self.ax_map, self.canvas_map, "Open a DAP file…")
        self._empty_message(self.ax_spec, self.canvas_spec, "Open a MODEL file…")

    def _empty_message(self, ax, canvas, msg: str) -> None:
        ax.clear()
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
        canvas.draw_idle()

    def _set_status(self, msg: str) -> None:
        self.status.config(text=msg)

    def _pek_available(self) -> bool:
        return self.n_pek > 0 and self.dap_pek

    def _is_pek_chan(self, i: int) -> bool:
        return i >= len(self.chan_names) - self.n_pek

    def _build_chan_buttons(self) -> None:
        old = {name: var.get() for name, var in zip(getattr(self, '_chan_var_names', []), self.chan_vars)}
        for w in self.chan_bar.winfo_children():
            w.destroy()
        row = ttk.Frame(self.chan_bar)
        row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(row, text="Channels:").pack(side="left")
        default_on = {'org_spec', 'mod_joint_spec', 'model_spec', 'res_joint_spec',
                      'mod_pek_spec', 'res_pek_spec'}
        self.chan_vars = []
        self._chan_var_names = list(self.chan_names)
        self._pek_buttons = []
        pek_row = None
        for i, name in enumerate(self.chan_names):
            var = tk.BooleanVar(value=old.get(name, (name in default_on) or i < 2))
            self.chan_vars.append(var)
            if self._is_pek_chan(i):
                if not self._pek_available():
                    continue
                if pek_row is None:
                    pek_row = ttk.Frame(self.chan_bar)
                    pek_row.pack(side=tk.TOP, fill=tk.X)
                    ttk.Checkbutton(pek_row, text="PEK models:", variable=self.var_show_pek,
                                    command=self._on_show_pek).pack(side="left")
                parent = pek_row
            else:
                parent = row
            color = chan_color(i, name)
            # tk (not ttk) Checkbutton so the label shows the line color
            b = tk.Checkbutton(parent, text=name, variable=var, fg=color, selectcolor='white',
                               activeforeground=color, command=self.draw_spectrum)
            b.pack(side="left", padx=2)
            if self._is_pek_chan(i):
                self._pek_buttons.append(b)
        ttk.Button(row, text="All", width=4,
                   command=lambda: self._set_all_chans(True)).pack(side="left", padx=(8, 2))
        ttk.Button(row, text="None", width=5,
                   command=lambda: self._set_all_chans(False)).pack(side="left")
        self._on_show_pek(redraw=False)

    def _slice_channels(self) -> list[str]:
        return [n for i, n in enumerate(self.chan_names) if not self._is_pek_chan(i) or self._pek_shown()]

    def _pek_shown(self) -> bool:
        return self._pek_available() and self.var_show_pek.get()

    def _on_show_pek(self, redraw: bool = True) -> None:
        state = 'normal' if self._pek_shown() else 'disabled'
        for b in getattr(self, '_pek_buttons', []):
            b.configure(state=state)
        self.cb_slice_chan.configure(values=self._slice_channels())
        if self.var_slice_chan.get() not in self._slice_channels():
            self.var_slice_chan.set(self.chan_names[0])
            if self.var_source.get() == 'slice' and redraw:
                self.update_map_values()
        if redraw:
            self.draw_spectrum()

    def _set_all_chans(self, on: bool) -> None:
        for i, v in enumerate(self.chan_vars):
            if not self._is_pek_chan(i) or self._pek_shown():
                v.set(on)
        self.draw_spectrum()

    # ------------------------- IO -------------------------

    def open_dap(self) -> None:
        path = filedialog.askopenfilename(
            title="Select DAP file",
            filetypes=[("DAP files", "*.dap.fits.gz *.dap.fits"), ("FITS", "*.fits *.fits.gz"), ("All", "*.*")])
        if path:
            model = guess_model_file(path)
            if model is None or not messagebox.askyesno("MODEL file", f"Load also\n{model}?"):
                model = None
            self.load(path, model)

    def open_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select MODEL file",
            filetypes=[("MODEL files", "*.model.fits *.model.fits.gz *.output.fits.gz"), ("FITS", "*.fits *.fits.gz"),
                       ("All", "*.*")])
        if path:
            self.load(None, path)

    def load(self, dap_file: Optional[str], model_file: Optional[str]) -> None:
        self.config(cursor="watch")
        self.update()
        try:
            if dap_file:
                self._set_status(f"Reading {os.path.basename(dap_file)} …")
                self.update()
                self.pt, self.cols = read_dap(dap_file)
                self.dap_path = dap_file
                self.dap_pek = dap_has_pek(dap_file)
                self.ra = np.asarray(self.pt['ra'], dtype=float)
                self.dec = np.asarray(self.pt['dec'], dtype=float)
                self.good = (np.asarray(self.pt['mask'], dtype=bool) if 'mask' in self.pt.colnames
                             else np.ones(len(self.pt), bool))
                self.col_names = list(self.cols.keys())
                self.selected = np.zeros(len(self.pt), bool)
                self.current_fiber = int(np.flatnonzero(self.good)[len(np.flatnonzero(self.good)) // 2]) \
                    if self.good.any() else 0
                self._coll = None
                self._fill_col_list()
            if model_file:
                self._set_status(f"Reading {os.path.basename(model_file)} …")
                self.update()
                self.wave, self.cube, self.chan_names, self.n_pek = read_model(model_file)
                self.model_path = model_file
                self.var_slice_chan.set(self.chan_names[0])
                self._spec_lines = []
                self.ax_spec.clear()
                self._win_patch = None
                self._span = None
            if self.cube is not None:
                # PEK buttons depend on both files: rebuild after either is (re)loaded
                self._build_chan_buttons()
        except Exception as e:
            self.config(cursor="")
            messagebox.showerror("Load error", f"{type(e).__name__}: {e}")
            return
        self.config(cursor="")

        if self.pt is not None and self.cube is not None and self.cube.shape[1] != len(self.pt):
            messagebox.showwarning(
                "Size mismatch",
                f"MODEL has {self.cube.shape[1]} fibers but DAP PT has {len(self.pt)} rows.\n"
                "Spectra/slices are disabled.")
            self.cube = None

        self._update_status()
        if self.pt is not None:
            if self.var_source.get() == 'slice' and self.cube is None:
                self.var_source.set('dap')
            if self.map_vals is None and self.var_source.get() == 'dap':
                self._select_default_column()
            self.update_map_values(reset_view=True)
        self.draw_spectrum(reset_x=True)

    def _update_status(self) -> None:
        lines = []
        if self.dap_path:
            lines.append(f"DAP: {os.path.basename(self.dap_path)}")
            lines.append(f"  fibers: {len(self.pt)} (mask: {int(self.good.sum())}), columns: {len(self.col_names)}")
        if self.model_path and self.cube is not None:
            lines.append(f"MODEL: {os.path.basename(self.model_path)}")
            lines.append(f"  {self.cube.shape[0]} ch × {self.cube.shape[2]} pix, "
                         f"{self.wave[0]:.1f}–{self.wave[-1]:.1f} Å")
            if self.n_pek and not self.dap_pek:
                lines.append("  PEK_MODEL present, but the DAP has no PEK tables: PEK hidden")
            elif self._pek_available():
                lines.append("  PEK models available (tick 'PEK models')")
        if self.current_fiber is not None and self.pt is not None:
            i = self.current_fiber
            v = self.map_vals[i] if self.map_vals is not None else np.nan
            lines.append(f"Fiber #{i} id={_text(self.pt['id'][i:i + 1])[0]}  "
                         f"ra={self.ra[i]:.5f} dec={self.dec[i]:.5f}")
            lines.append(f"  {self.map_label} = {v:.4g}")
        if self.selected is not None and self.selected.any():
            lines.append(f"Selection: {int(self.selected.sum())} fibers")
        self._set_status("\n".join(lines))

    # ------------------------- Column list -------------------------

    def _fill_col_list(self) -> None:
        pat = self.var_filter.get().strip()
        names = self.col_names
        if pat:
            try:
                rx = re.compile(pat, re.IGNORECASE)
                names = [n for n in names if rx.search(n)]
            except re.error:
                names = [n for n in names if pat.lower() in n.lower()]
        self._shown_cols = names
        self.col_list.delete(0, tk.END)
        for n in names:
            self.col_list.insert(tk.END, n)

    def _select_default_column(self) -> None:
        for cand in ('flux_Halpha_6562.85', 'flux_pe_6562.85', 'flux_Halpha_6562.85_pm', 'med_flux_st'):
            if cand in self.cols:
                self._set_dap_column(cand)
                return
        if self.col_names:
            self._set_dap_column(self.col_names[-1])

    def _set_dap_column(self, name: str) -> None:
        self.dap_col = name
        if name in self._shown_cols:
            i = self._shown_cols.index(name)
            self.col_list.selection_clear(0, tk.END)
            self.col_list.selection_set(i)
            self.col_list.see(i)

    def _on_col_select(self, _evt=None) -> None:
        sel = self.col_list.curselection()
        if not sel:
            return
        self.dap_col = self._shown_cols[int(sel[0])]
        self.var_source.set('dap')
        self.update_map_values()

    # ------------------------- Map values -------------------------

    def _window(self) -> tuple[float, float]:
        w1, w2 = float(self.var_w1.get()), float(self.var_w2.get())
        return (w1, w2) if w1 <= w2 else (w2, w1)

    def compute_slice(self) -> np.ndarray:
        w1, w2 = self._window()
        ch = self.chan_names.index(self.var_slice_chan.get()) if self.var_slice_chan.get() in self.chan_names else 0
        idx = np.flatnonzero((self.wave >= w1) & (self.wave <= w2))
        if idx.size == 0:  # narrower than one pixel: nearest pixel
            idx = np.array([int(np.argmin(np.abs(self.wave - 0.5 * (w1 + w2))))])
        data = self.cube[ch][:, idx[0]:idx[-1] + 1]
        stat = self.var_slice_stat.get()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if stat == 'sum':
                dw = float(np.median(np.diff(self.wave)))
                return np.nansum(data, axis=1) * dw
            if stat == 'median':
                return np.nanmedian(data, axis=1)
            return np.nanmean(data, axis=1)

    def update_map_values(self, reset_view: bool = False) -> None:
        if self.pt is None:
            return
        if self.var_source.get() == 'slice':
            if self.cube is None:
                messagebox.showinfo("Spectral slice", "Load a MODEL file first.")
                self.var_source.set('dap')
                return
            self.map_vals = self.compute_slice()
            w1, w2 = self._window()
            self.map_label = f"{self.var_slice_chan.get()} {self.var_slice_stat.get()} [{w1:.1f}–{w2:.1f}]"
        else:
            if self.dap_col not in self.cols:
                self._select_default_column()
            self.map_label = self.dap_col
            self.map_vals = self.cols.get(self.dap_col)
        if not self.var_lock.get():
            self.auto_scale(redraw=False)
        self.draw_map(reset_view=reset_view)
        self._update_status()

    def auto_scale(self, redraw: bool = True) -> None:
        if self.map_vals is None:
            return
        v = self.map_vals[self._visible()]
        v = v[np.isfinite(v)]
        if v.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = np.percentile(v, [float(self.var_plo.get()), float(self.var_phi.get())])
            if hi <= lo:
                lo, hi = float(v.min()), float(v.max())
                if hi <= lo:
                    lo, hi = lo - 1, hi + 1
        self.var_vmin.set(float(f"{lo:.4g}"))
        self.var_vmax.set(float(f"{hi:.4g}"))
        if redraw:
            self.draw_map()

    def set_slice(self) -> None:
        if self.cube is None:
            return
        self.var_source.set('slice')
        self.update_map_values()
        self._update_window_patch()
        self.canvas_spec.draw_idle()

    def step_slice(self, direction: int) -> None:
        if self.wave is None:
            return
        w1, w2 = self._window()
        step = max(w2 - w1, float(np.median(np.diff(self.wave))))
        if w1 + direction * step < self.wave[0] or w2 + direction * step > self.wave[-1]:
            return
        self.var_w1.set(round(w1 + direction * step, 2))
        self.var_w2.set(round(w2 + direction * step, 2))
        self.set_slice()

    # ------------------------- Map drawing -------------------------

    def _visible(self) -> np.ndarray:
        if self.good is None:
            return np.array([], bool)
        return self.good if self.var_only_good.get() else np.ones_like(self.good)

    def _norm(self):
        vmin, vmax = float(self.var_vmin.get()), float(self.var_vmax.get())
        if vmax <= vmin:
            vmax = vmin + 1e-6
        gamma = float(self.var_gamma.get())
        if gamma == 1.0:
            return mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        return mpl_colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    def _cmap(self):
        try:
            cmap = mpl.colormaps[self.var_cmap.get()].copy()
        except KeyError:
            cmap = mpl.colormaps['Spectral_r'].copy()
        cmap.set_bad('lightgrey')
        return cmap

    def draw_map(self, reset_view: bool = False) -> None:
        if self.pt is None or self.map_vals is None:
            return
        try:
            norm = self._norm()
        except (tk.TclError, ValueError):
            return
        vis = self._visible()
        ax = self.ax_map
        x, y = self.ra[vis], self.dec[vis]
        vals = np.ma.masked_invalid(self.map_vals[vis])

        if self._coll is None or reset_view:
            ax.clear()
            self._sel_line = self._cur_line = None
            self._rect_selector = None
            dec0 = float(np.nanmedian(self.dec[vis])) if vis.any() else 0.0
            cosd = max(np.cos(np.deg2rad(dec0)), 1e-3)
            d = FIBER_DIAM_ARCSEC / 3600.0
            # units='xy': widths in RA degrees, heights in Dec degrees -> round fibers on sky
            self._coll = EllipseCollection(widths=np.full(len(x), d / cosd), heights=np.full(len(x), d),
                                           angles=0, units='xy', offsets=np.column_stack([x, y]),
                                           offset_transform=ax.transData, cmap=self._cmap(), norm=norm)
            self._coll.set_array(vals)
            ax.add_collection(self._coll)
            ax.set_aspect(1.0 / cosd)
            pad = d
            if x.size:
                ax.set_xlim(np.nanmax(x) + pad / cosd, np.nanmin(x) - pad / cosd)  # RA increases to the left
                ax.set_ylim(np.nanmin(y) - pad, np.nanmax(y) + pad)
            ax.set_xlabel('RA [deg]')
            ax.set_ylabel('Dec [deg]')
            self.cax_map.clear()
            self.cax_map.set_axis_on()
            self._cbar = self.fig_map.colorbar(self._coll, cax=self.cax_map)
            self._ensure_rect_selector()
        else:
            self._coll.set_array(vals)
            self._coll.set_cmap(self._cmap())
            self._coll.set_norm(norm)
            self._cbar.update_normal(self._coll)
        ax.set_title(self.map_label, fontsize=10)
        self._draw_markers()
        self.canvas_map.draw_idle()

    def _draw_markers(self) -> None:
        ax = self.ax_map
        if self.selected is not None:
            s = self.selected
            if self._sel_line is None:
                (self._sel_line,) = ax.plot([], [], 'o', ms=7, mfc='none', mec='grey', mew=1.2)
            self._sel_line.set_data(self.ra[s], self.dec[s])
        if self.current_fiber is not None:
            i = self.current_fiber
            if self._cur_line is None:
                (self._cur_line,) = ax.plot([], [], 'o', ms=9, mfc='none', mec='orange', mew=2)
            self._cur_line.set_data([self.ra[i]], [self.dec[i]])

    # ------------------------- Map interaction -------------------------

    def _nearest_fiber(self, xd: float, yd: float) -> Optional[int]:
        vis = np.flatnonzero(self._visible())
        if vis.size == 0:
            return None
        cosd = np.cos(np.deg2rad(yd))
        d2 = ((self.ra[vis] - xd) * cosd) ** 2 + (self.dec[vis] - yd) ** 2
        return int(vis[np.argmin(d2)])

    def _map_toolbar_busy(self) -> bool:
        return bool(getattr(self.canvas_map.toolbar, 'mode', ''))

    def _on_map_press(self, event) -> None:
        if self.pt is None or event.inaxes != self.ax_map or event.xdata is None or self._map_toolbar_busy():
            return
        i = self._nearest_fiber(event.xdata, event.ydata)
        if i is None:
            return
        if event.button == 1 and not self.var_rectsel.get():
            self.current_fiber = i
            self.var_spec_mode.set('single')
        elif event.button == 2:
            self._browse = True
            self.current_fiber = i
            self.var_spec_mode.set('single')
        elif event.button == 3:
            self.selected[i] = not self.selected[i]
            self.var_spec_mode.set('mean')
        else:
            return
        self._after_fiber_change()

    def _on_map_release(self, event) -> None:
        if event.button == 2:
            self._browse = False

    def _on_map_motion(self, event) -> None:
        if not self._browse or event.inaxes != self.ax_map or event.xdata is None:
            return
        if self._browse_after is not None:  # throttle motion events
            return
        xd, yd = event.xdata, event.ydata

        def _do():
            self._browse_after = None
            i = self._nearest_fiber(xd, yd)
            if i is not None and i != self.current_fiber:
                self.current_fiber = i
                self._after_fiber_change()

        self._browse_after = self.after(50, _do)

    def _after_fiber_change(self) -> None:
        self._draw_markers()
        self.canvas_map.draw_idle()
        self._update_status()
        self.draw_spectrum()

    def _toggle_rect(self) -> None:
        self._ensure_rect_selector()

    def _ensure_rect_selector(self) -> None:
        if self.var_rectsel.get():
            if self._rect_selector is None:
                self._rect_selector = RectangleSelector(self.ax_map, self._on_rect, useblit=True, button=[1],
                                                        interactive=False, spancoords='data',
                                                        props=dict(facecolor='orange', alpha=0.2))
            self._rect_selector.set_active(True)
        elif self._rect_selector is not None:
            self._rect_selector.set_active(False)

    def _on_rect(self, eclick, erelease) -> None:
        if self.pt is None or None in (eclick.xdata, erelease.xdata):
            return
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        inside = (self.ra >= x0) & (self.ra <= x1) & (self.dec >= y0) & (self.dec <= y1) & self._visible()
        self.selected |= inside
        self.var_spec_mode.set('mean')
        self._after_fiber_change()

    def clear_selection(self) -> None:
        if self.selected is None:
            return
        self.selected[:] = False
        self.var_spec_mode.set('single')
        self._after_fiber_change()

    # ------------------------- Spectrum -------------------------

    def _spectra(self) -> tuple[Optional[np.ndarray], str]:
        """Return (spectra[nchan, nwave], title) for the current mode."""
        if self.var_spec_mode.get() == 'mean' and self.selected is not None and self.selected.any():
            idx = np.flatnonzero(self.selected)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                return np.nanmean(self.cube[:, idx, :], axis=1), f"Mean of {idx.size} fibers"
        if self.current_fiber is None:
            return None, ''
        i = self.current_fiber
        return self.cube[:, i, :], f"Fiber #{i}  id={_text(self.pt['id'][i:i + 1])[0]}" if self.pt is not None \
            else f"Fiber #{i}"

    def draw_spectrum(self, reset_x: bool = False) -> None:
        ax = self.ax_spec
        if self.cube is None:
            self._empty_message(ax, self.canvas_spec, "Open a MODEL file…")
            return
        if self.current_fiber is None:
            self.current_fiber = 0
        spec, title = self._spectra()
        if spec is None:
            return

        if not self._spec_lines:
            ax.clear()
            self._spec_lines = [ax.plot(self.wave, spec[i], lw=1.0+1.0/(1+i), color=chan_color(i, name),
                                        label=name)[0]
                                for i, name in enumerate(self.chan_names)]
            ax.axhline(0, color='grey', lw=0.5, ls=':')
            ax.set_xlabel('Wavelength [Å]')
            ax.set_ylabel('Flux')
            self._win_patch = Rectangle((0, 0), 1, 1, transform=ax.get_xaxis_transform(),
                                        color='orange', alpha=0.3, zorder=0)
            ax.add_patch(self._win_patch)
            self._span = SpanSelector(ax, self._on_span, 'horizontal', useblit=True, button=[1],
                                      minspan=float(np.median(np.diff(self.wave))),
                                      props=dict(facecolor='orange', alpha=0.3))
            reset_x = True
        else:
            for i, line in enumerate(self._spec_lines):
                line.set_ydata(spec[i])

        for i, line in enumerate(self._spec_lines):
            on = self.chan_vars[i].get() if i < len(self.chan_vars) else True
            line.set_visible(on and (not self._is_pek_chan(i) or self._pek_shown()))
        handles = [ln for ln in self._spec_lines if ln.get_visible()]
        if handles:
            ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=min(4, len(handles)))
        elif ax.get_legend() is not None:
            ax.get_legend().remove()

        self._update_window_patch()
        if reset_x:
            ax.set_xlim(self.wave[0], self.wave[-1])
        self._apply_ylim()
        ax.set_title(title, fontsize=10)
        self.canvas_spec.draw_idle()

    def _apply_ylim(self) -> None:
        ax = self.ax_spec
        if self.var_fix_y.get():
            try:
                y1, y2 = float(self.var_y1.get()), float(self.var_y2.get())
            except (tk.TclError, ValueError):
                return
            if y2 > y1:
                ax.set_ylim(y1, y2)
            return
        x0, x1 = sorted(ax.get_xlim())
        m = (self.wave >= x0) & (self.wave <= x1)
        vals = [ln.get_ydata()[m] for ln in self._spec_lines if ln.get_visible()]
        vals = np.concatenate(vals) if vals else np.array([])
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return
        lo, hi = np.percentile(vals, [0.5, 100.0])
        if hi <= lo:
            hi = lo + 1.0
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

    def _auto_y_button(self) -> None:
        self.var_fix_y.set(False)
        self._apply_ylim()
        self.canvas_spec.draw_idle()

    def reset_spec_zoom(self) -> None:
        if self.wave is None:
            return
        self.ax_spec.set_xlim(self.wave[0], self.wave[-1])
        self._apply_ylim()
        self.canvas_spec.draw_idle()

    def _update_window_patch(self) -> None:
        if self._win_patch is None:
            return
        w1, w2 = self._window()
        self._win_patch.set_x(w1)
        self._win_patch.set_width(max(w2 - w1, 1e-3))
        self._win_patch.set_visible(self.var_source.get() == 'slice')

    def _on_span(self, wmin: float, wmax: float) -> None:
        self.var_w1.set(round(wmin, 2))
        self.var_w2.set(round(wmax, 2))
        self.set_slice()

    def _on_spec_press(self, event) -> None:
        # Middle click: re-center the slice window keeping its width
        if event.button != 2 or event.inaxes != self.ax_spec or event.xdata is None or self.cube is None:
            return
        if getattr(self.canvas_spec.toolbar, 'mode', ''):
            return
        w1, w2 = self._window()
        half = 0.5 * (w2 - w1)
        self.var_w1.set(round(event.xdata - half, 2))
        self.var_w2.set(round(event.xdata + half, 2))
        self.set_slice()

    # ------------------------- Save -------------------------

    def save_fig(self, fig: Figure, kind: str) -> None:
        base = os.path.basename(self.dap_path or self.model_path or 'LVMview')
        base = re.sub(r'\.(dap|model)\.fits(\.gz)?$', '', base)
        path = filedialog.asksaveasfilename(
            title=f"Save {kind}", defaultextension=".png", initialfile=f"{base}_{kind}.png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("JPG", "*.jpg"), ("All", "*.*")])
        if not path:
            return
        try:
            fig.savefig(path, dpi=150, facecolor='white')
            self._set_status(self.status.cget("text") + f"\nSaved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="LVM DAP + MODEL interactive viewer")
    parser.add_argument('dap', nargs='?', help='DAP file (*.dap.fits.gz)')
    parser.add_argument('model', nargs='?', help='MODEL file (*.model.fits.gz); guessed from DAP name if omitted')
    args = parser.parse_args()
    app = DAPViewApp(args.dap, args.model)
    app.mainloop()


if __name__ == "__main__":
    main()
