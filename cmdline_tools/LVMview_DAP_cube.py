#!/usr/bin/env python3
"""
LVM DAP CUBE + MODEL viewer (stand-alone Tk application).

Reads the cube-mode LVM-DAP product pair:
- DAP file (*.cube.dap.fits.gz): spatial parameter image cubes, PLANE_INDEX,
                                  SPAXEL_INDEX; no per-fiber DAP tables.
  Each image plane is identified by NAME_# and/or PLANE_INDEX.
- MODEL file (*.model.fits.gz) : primary cube (nchan, nfiber, nwave) with
                                 channel names in NAME0..NAMEn (org_spec,
                                 model_spec, mod_joint_spec, gas_spec, ...)

Features:
(1) Map frame: any DAP parameter channel on the original WCS grid.
(2) Map frame: slice map built from the spectral window selected in the
    spectral frame (drag with the left button, middle-click to re-center),
    from any MODEL channel (mean / sum / median over the window).
(3) Spectral frame: every MODEL channel can be switched on/off with buttons.
    Optional PEK models: if the MODEL file has a PEK_MODEL extension and the
    DAP file has PEK tables (PEK_ELINES_* / PM_KEL), a "PEK models" toggle
    adds PEK_model, mod_pek_spec (model_spec + PEK) and res_pek_spec
    (org_spec - mod_pek_spec), also usable as slice channels.

Mouse on the map:
- Left click   : show the spectrum of the nearest spaxel
- Middle drag  : browse spectra while moving
- Right click  : add/remove fiber to the selection (mean spectrum)
- "Rect. select" toggle: left-drag a rectangle to add fibers to the selection

Usage:
    python LVMview_DAP_cube.py [label.cube.dap.fits.gz] [label.model.fits.gz]
If only the cube DAP file is given, the companion MODEL is guessed.
The MODEL is memory-mapped (gzip data is first extracted to a temporary file).
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import tempfile
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import matplotlib as mpl
import matplotlib.colors as mpl_colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import EllipseCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector, SpanSelector

# No lvmdap import is needed in cube mode. The FITS data model is read directly.
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
        self.title("LVM DAP CUBE/MODEL viewer")
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
            self.after(100, lambda: self.load(dap_file, model_file))

    # ------------------------- UI -------------------------

    def _build_ui(self) -> None:
        menubar = tk.Menu(self)
        fm = tk.Menu(menubar, tearoff=0)
        fm.add_command(label="Open cube DAP…", command=self.open_dap)
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
        ttk.Radiobutton(sf, text="Parameter plane", variable=self.var_source, value='dap',
                        command=self.update_map_values).pack(side="left")
        ttk.Radiobutton(sf, text="Spectral slice", variable=self.var_source, value='slice',
                        command=self.update_map_values).pack(side="left", padx=(10, 0))

        # ---- DAP columns ----
        cf = ttk.LabelFrame(left, text="DAP cube planes", padding=6)
        cf.grid(row=2, column=0, sticky="nsew", pady=(0, 6))
        cf.rowconfigure(1, weight=1)
        cf.columnconfigure(0, weight=1)
        ent = ttk.Entry(cf, textvariable=self.var_filter)
        ent.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.var_filter.trace_add("write", lambda *_: self._fill_col_list())
        self.col_list = tk.Listbox(cf, width=46, height=12, exportselection=False)
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

        self.status = ttk.Label(left, text="Open a cube-mode DAP file to start.", wraplength=300, justify="left")
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
        ttk.Checkbutton(mb2, text="Valid spaxels", variable=self.var_only_good,
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
        ttk.Radiobutton(sb2, text="Single spaxel", variable=self.var_spec_mode, value='single',
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

        self._empty_message(self.ax_map, self.canvas_map, "Open a cube-mode DAP file…")
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


# -----------------------------------------------------------------------------
# Cube-mode readers (independent of lvmdap installation)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PlaneInfo:
    extension: str
    plane: int
    parameter: str
    unit: str = ''
    encoding: str = 'NUMERIC'

    @property
    def label(self) -> str:
        return f'{self.extension} | {self.parameter}'


def _clean_fits_string(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8', 'replace').strip()
    return str(value).strip()


class CubeDAPReader:
    """Metadata-first cube-DAP reader; only decompress the requested map plane.

    3D parameter arrays have shape (n_parameter, n_y, n_x), while
    SPAXEL_INDEX records each fitted spaxel's xpix/ypix in MODEL row order.
    The original RA/DEC WCS is used to draw a genuine spatial pixel map.
    """

    def __init__(self, filename: str):
        self.filename = os.path.abspath(filename)
        self.planes: dict[str, PlaneInfo] = {}
        self.categories: dict[tuple[str, int], dict[int, str]] = {}
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

        with fits.open(self.filename, memmap=False, lazy_load_hdus=True) as hdul:
            if 'SPAXEL_INDEX' not in hdul:
                raise ValueError('Not a cube-mode DAP file: SPAXEL_INDEX is absent')
            pt = Table(hdul['SPAXEL_INDEX'].data)
            required = {'xpix', 'ypix', 'ra', 'dec', 'id'}
            if not required <= set(pt.colnames):
                raise ValueError(f'SPAXEL_INDEX lacks columns {sorted(required - set(pt.colnames))}')
            self.pt = pt
            self.pt['mask'] = np.ones(len(pt), dtype=bool)
            primary = hdul[0].header
            self.nx = int(primary.get('CUBENX', 0))
            self.ny = int(primary.get('CUBENY', 0))
            image_hdus = [h for h in hdul[1:] if isinstance(h, (fits.ImageHDU, fits.CompImageHDU))
                          and int(h.header.get('NAXIS', 0)) == 3]
            if not image_hdus:
                raise ValueError('No 3D parameter image extensions in cube-DAP')
            ref = image_hdus[0]
            if self.nx == 0:
                self.nx = int(ref.header['NAXIS1'])
            if self.ny == 0:
                self.ny = int(ref.header['NAXIS2'])
            self.wcs = WCS(ref.header).celestial
            self._ext_dims = {}
            for h in image_hdus:
                if (int(h.header['NAXIS1']), int(h.header['NAXIS2'])) != (self.nx, self.ny):
                    raise ValueError(f'{h.name}: inconsistent spatial dimensions')
                self._ext_dims[h.name] = int(h.header['NAXIS3'])

            if 'PLANE_INDEX' in hdul:
                catalogue = Table(hdul['PLANE_INDEX'].data)
                for row in catalogue:
                    ext = _clean_fits_string(row['HDU']).upper()
                    if ext not in self._ext_dims:
                        continue
                    k = int(row['PLANE'])
                    if not 0 <= k < self._ext_dims[ext]:
                        raise ValueError(f'PLANE_INDEX out of bounds: {ext} plane {k}')
                    full_param = _clean_fits_string(row['PARAMETER'])
                    unit = _clean_fits_string(row['UNIT']) if 'UNIT' in catalogue.colnames else ''
                    encoding = (_clean_fits_string(row['ENCODING']) if 'ENCODING' in catalogue.colnames
                                else 'NUMERIC')
                    info = PlaneInfo(ext, k, full_param, unit, encoding)
                    if info.label in self.planes:
                        raise ValueError(f'Duplicate plane catalogue entry: {info.label}')
                    self.planes[info.label] = info

            # Compatibility with early cube FITS files lacking PLANE_INDEX:
            # HIERARCH NAME_<k> cards were introduced after the first version.
            for h in image_hdus:
                for k in range(self._ext_dims[h.name]):
                    if any(v.extension == h.name and v.plane == k for v in self.planes.values()):
                        continue
                    param = str(h.header.get(f'NAME_{k}', f'plane_{k}')).strip()
                    if h.name == 'COEFFS' and f'RSP_{k}' in h.header:
                        param = f"{param}@rsp={h.header[f'RSP_{k}']}"
                    info = PlaneInfo(h.name, k, param)
                    self.planes[info.label] = info

            if 'CATEGORIES' in hdul:
                for row in hdul['CATEGORIES'].data:
                    key = (_clean_fits_string(row['HDU']).upper(), int(row['PLANE']))
                    self.categories.setdefault(key, {})[int(row['CODE'])] = _clean_fits_string(row['VALUE'])

        self.xpix = np.asarray(self.pt['xpix'], dtype=np.int64)
        self.ypix = np.asarray(self.pt['ypix'], dtype=np.int64)
        if (np.any(self.xpix < 0) or np.any(self.ypix < 0) or
                np.any(self.xpix >= self.nx) or np.any(self.ypix >= self.ny)):
            raise ValueError('SPAXEL_INDEX contains coordinates outside the image grid')
        addr = self.ypix * self.nx + self.xpix
        if len(np.unique(addr)) != len(addr):
            raise ValueError('Duplicate SPAXEL_INDEX spatial coordinates')
        self._corners = None

    @property
    def names(self) -> list[str]:
        return list(self.planes)

    def corners(self) -> tuple[np.ndarray, np.ndarray]:
        """Sky coordinates at all pixel corners, for a noninterpolated map."""
        if self._corners is None:
            xx, yy = np.meshgrid(np.arange(self.nx + 1) - 0.5,
                                 np.arange(self.ny + 1) - 0.5)
            self._corners = self.wcs.all_pix2world(xx, yy, 0)
        return self._corners

    def get_map(self, label: str) -> np.ndarray:
        """Read one (y,x) map only. Do not retain hundreds of large FITS cubes."""
        if label in self._cache:
            self._cache.move_to_end(label)
            return self._cache[label]
        info = self.planes[label]
        with fits.open(self.filename, memmap=False, lazy_load_hdus=True) as hdul:
            image = np.array(hdul[info.extension].data[info.plane],
                             dtype=np.float64, copy=True)
        if image.shape != (self.ny, self.nx):
            raise ValueError(f'{label} has incorrect shape {image.shape}')
        self._cache[label] = image
        if len(self._cache) > 5:
            self._cache.popitem(last=False)
        return image

    def decode_value(self, label: str, value: float) -> str:
        info = self.planes[label]
        if not np.isfinite(value):
            return 'NaN'
        cats = self.categories.get((info.extension, info.plane))
        if cats is not None:
            return f'{value:g} ({cats.get(int(round(value)), "unknown")})'
        return f'{value:.5g}' + (f' {info.unit}' if info.unit else '')


def guess_cube_model_file(dap_file: str) -> Optional[str]:
    """Output naming from lvmdap/_cmdline/dap.py (cube-mode)."""
    prefix = re.sub(r'\.cube\.dap\.fits(?:\.gz)?$', '', dap_file, flags=re.IGNORECASE)
    if prefix == dap_file:
        return None
    model_suffixes = ('.model.fits', '.model.fits.gz')
    if dap_file.lower().endswith('.gz'):
        model_suffixes = model_suffixes[::-1]
    for suffix in model_suffixes:
        candidate = prefix + suffix
        if os.path.isfile(candidate):
            return candidate
    return None


class LazyModel:
    """Spectral MODEL backed by FITS memmap with chunk-wise selection/slices.

    A .fits.gz MODEL is unpacked into a temporary disk file first; Astropy
    cannot memory-map a gzip stream directly. The temporary file is removed
    when the model or viewer is closed. The FITS data is never changed.
    """

    def __init__(self, filename: str):
        self.filename = os.path.abspath(filename)
        self._temp_path: Optional[str] = None
        self.hdul = None
        try:
            with open(filename, 'rb') as fd:
                compressed = fd.read(2) == b'\x1f\x8b'
            path = filename
            if compressed:
                tmp_dir = os.environ.get('LVMVIEW_TMPDIR')
                if tmp_dir is None:
                    source_dir = os.path.dirname(self.filename)
                    tmp_dir = source_dir if os.access(source_dir, os.W_OK) else None
                fd, path = tempfile.mkstemp(prefix='.lvmdap_view_model_',
                                             suffix='.fits', dir=tmp_dir)
                self._temp_path = path
                with os.fdopen(fd, 'wb') as out, gzip.open(filename, 'rb') as src:
                    shutil.copyfileobj(src, out, length=4 * 1024 * 1024)
            self.hdul = fits.open(path, memmap=True)
            hdr = self.hdul[0].header
            self.base = self.hdul[0].data
            if self.base.ndim == 2:
                self.base = self.base[np.newaxis]
            if self.base.ndim != 3:
                raise ValueError(f'MODEL must have (channel, row, wavelength): {self.base.shape}')
            self.names = [str(hdr.get(f'NAME{i}', f'chan{i}')).strip()
                          for i in range(self.base.shape[0])]
            self.pek = None
            self.extra_names = []
            if 'PEK_MODEL' in self.hdul:
                p = self.hdul['PEK_MODEL'].data
                if p.shape == self.base.shape[1:]:
                    self.pek = p
                    self.extra_names.append('PEK_model')
                    if 'model_spec' in self.names:
                        self.extra_names.append('mod_pek_spec')
                        if 'org_spec' in self.names:
                            self.extra_names.append('res_pek_spec')
            self.names += self.extra_names
            self.n_pek = len(self.extra_names)
            self.shape = (len(self.names), *self.base.shape[1:])
            crpix = float(hdr.get('CRPIX1', 1.0))
            self.wave = (float(hdr['CRVAL1']) + float(hdr['CDELT1']) *
                         (np.arange(self.shape[2]) + 1 - crpix))
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.hdul is not None:
            self.hdul.close()
            self.hdul = None
        self.base = None
        self.pek = None
        if self._temp_path and os.path.exists(self._temp_path):
            os.unlink(self._temp_path)
        self._temp_path = None

    def _channels(self, rows, wstart=None, wstop=None) -> np.ndarray:
        """Build the requested small chunk, including optional derived PEK planes."""
        wave = slice(None) if wstart is None else slice(wstart, wstop)
        base = np.asarray(self.base[:, rows, wave], dtype=np.float64)
        # Integer row selections return (n_chan, n_wave); vector row selections
        # return (n_chan, n_rows, n_wave).
        if self.pek is None:
            return base
        pek = np.asarray(self.pek[rows, wave], dtype=np.float64)
        extra = [pek]
        if 'model_spec' in self.names[:self.base.shape[0]]:
            mod_pek = base[self.names.index('model_spec')] + pek
            extra.append(mod_pek)
            if 'org_spec' in self.names[:self.base.shape[0]]:
                extra.append(base[self.names.index('org_spec')] - mod_pek)
        return np.concatenate([base, np.stack(extra)], axis=0)

    def spectrum(self, row: int) -> np.ndarray:
        return self._channels(int(row))

    def mean_spectrum(self, rows: np.ndarray, block: int = 32) -> np.ndarray:
        sums = np.zeros((self.shape[0], self.shape[2]), dtype=np.float64)
        counts = np.zeros_like(sums, dtype=np.int32)
        for start in range(0, len(rows), block):
            part = self._channels(rows[start:start + block])
            ok = np.isfinite(part)
            sums += np.sum(np.where(ok, part, 0.0), axis=1)
            counts += ok.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(counts != 0, sums / counts, np.nan)

    def slice_values(self, channel: int, start: int, stop: int,
                     operation: str, block: int = 64) -> np.ndarray:
        values = np.full(self.shape[1], np.nan, dtype=np.float64)
        dw = float(np.median(np.diff(self.wave)))
        for i in range(0, self.shape[1], block):
            rows = np.arange(i, min(i + block, self.shape[1]))
            spectra = self._channels(rows, start, stop)[channel]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                if operation == 'median':
                    result = np.nanmedian(spectra, axis=1)
                elif operation == 'sum':
                    result = np.nansum(spectra, axis=1) * dw
                else:
                    result = np.nanmean(spectra, axis=1)
            result[~np.isfinite(spectra).any(axis=1)] = np.nan
            values[rows] = result
        return values


class CubeViewApp(DAPViewApp):
    """Original LVMview_DAP_model GUI adapted for WCS-registered DAP cubes."""

    def __init__(self, dap_file: Optional[str] = None,
                 model_file: Optional[str] = None):
        super().__init__(dap_file, model_file)
        self.cube_reader: Optional[CubeDAPReader] = None
        self.model_reader: Optional[LazyModel] = None
        self.map_image: Optional[np.ndarray] = None
        self.title('LVMview DAP cube + MODEL')
        self.protocol('WM_DELETE_WINDOW', self._close_app)

    def _close_app(self):
        if self.model_reader is not None:
            self.model_reader.close()
            self.model_reader = None
        self.destroy()

    def open_dap(self) -> None:
        path = filedialog.askopenfilename(
            title='Select cube DAP FITS',
            filetypes=[('Cube DAP FITS', '*.cube.dap.fits.gz *.cube.dap.fits'),
                       ('FITS', '*.fits.gz *.fits'), ('All', '*.*')])
        if path:
            model = guess_cube_model_file(path)
            if model and not messagebox.askyesno('MODEL file',
                                                  f'Load companion spectral MODEL?\n{model}'):
                model = None
            self.load(path, model)

    def load(self, dap_file: Optional[str], model_file: Optional[str]) -> None:
        self.config(cursor='watch')
        self.update_idletasks()
        try:
            new_reader = None
            if dap_file:
                self._set_status(f'Inspecting {os.path.basename(dap_file)} ...')
                self.update_idletasks()
                new_reader = CubeDAPReader(dap_file)

            new_model = None
            if model_file:
                self._set_status(f'Opening MODEL {os.path.basename(model_file)} ...')
                self.update_idletasks()
                new_model = LazyModel(model_file)
                nspaxel = (len(new_reader.pt) if new_reader is not None
                           else len(self.pt) if self.pt is not None else None)
                if nspaxel is not None and new_model.shape[1] != nspaxel:
                    nmodel = new_model.shape[1]
                    new_model.close()
                    new_model = None
                    messagebox.showwarning(
                        'MODEL mapping mismatch',
                        f'MODEL contains {nmodel} RSS rows but SPAXEL_INDEX has {nspaxel}.\n'
                        'Parameter maps will still work; spectral plots are disabled.')

            if new_reader is not None:
                self.cube_reader = new_reader
                self.pt = new_reader.pt
                self.cols = {}  # Never expand the thousands of planes into columns.
                self.col_names = new_reader.names
                self.dap_path = new_reader.filename
                self.ra = np.asarray(self.pt['ra'], dtype=float)
                self.dec = np.asarray(self.pt['dec'], dtype=float)
                self.good = np.asarray(self.pt['mask'], dtype=bool)
                self.dap_pek = any(h.startswith('PEK_ELINES') for h in new_reader._ext_dims)
                self.selected = np.zeros(len(self.pt), dtype=bool)
                self.current_fiber = len(self.pt) // 2 if len(self.pt) else None
                self.map_image = None
                self.map_vals = None
                self._coll = None
                self._fill_col_list()
                self._select_default_column()

                # Never keep an old, potentially misaligned MODEL after changing DAP.
                if new_model is None and self.model_reader is not None:
                    self.model_reader.close()
                    self.model_reader = None
                    self.cube = None
                    self.model_path = None
                    self.wave = None
                    self.chan_names = []
                    self.n_pek = 0
                    self._spec_lines = []
                    self.ax_spec.clear()
                    self._span = self._win_patch = None
                    for child in self.chan_bar.winfo_children():
                        child.destroy()

            if new_model is not None:
                if self.model_reader is not None:
                    self.model_reader.close()
                self.model_reader = new_model
                self.cube = new_model  # Compatibility: original GUI checks cube.shape.
                self.wave = new_model.wave
                self.chan_names = new_model.names
                self.n_pek = new_model.n_pek
                self.model_path = new_model.filename
                self.var_slice_chan.set(self.chan_names[0])
                self._spec_lines = []
                self.ax_spec.clear()
                self._span = self._win_patch = None
            elif new_reader is not None and self.model_reader is None:
                self.cube = None

            if self.cube is not None:
                self._build_chan_buttons()
            if new_reader is not None:
                if self.var_source.get() == 'slice' and self.cube is None:
                    self.var_source.set('dap')
                self.update_map_values(reset_view=True)
            self._update_status()
            self.draw_spectrum(reset_x=True)
        except Exception as exc:
            messagebox.showerror('Cube load error', f'{type(exc).__name__}: {exc}')
        finally:
            self.config(cursor='')

    def _select_default_column(self) -> None:
        if self.cube_reader is None or not self.col_names:
            return
        # Default to H-alpha BEST flux; then PEK/PM/NP flux; then any RSP map.
        for preferred in ('BEST_ELINES_R | flux_Halpha_6562.85_best',
                          'BEST_ELINES_R | flux_Halpha_6562.85',
                          'PEK_ELINES_R | flux_Halpha_6562.85_pek',
                          'PM_ELINES_R | flux_Halpha_6562.85_pm',
                          'NP_ELINES_R | flux_Halpha_6562.85',
                          'RSP | med_flux'):
            if preferred in self.cube_reader.planes:
                self._set_dap_column(preferred)
                return
        for label in self.col_names:
            if 'flux_Halpha' in label and 'e_flux' not in label:
                self._set_dap_column(label)
                return
        self._set_dap_column(self.col_names[0])

    def compute_slice(self) -> np.ndarray:
        if self.model_reader is None:
            raise ValueError('Load the companion MODEL FITS for spectral slices')
        w1, w2 = self._window()
        chname = self.var_slice_chan.get()
        channel = self.chan_names.index(chname) if chname in self.chan_names else 0
        idx = np.flatnonzero((self.wave >= w1) & (self.wave <= w2))
        if not len(idx):
            idx = np.array([int(np.argmin(abs(self.wave - 0.5 * (w1 + w2))))])
        return self.model_reader.slice_values(
            channel, int(idx[0]), int(idx[-1]) + 1, self.var_slice_stat.get())

    def update_map_values(self, reset_view: bool = False) -> None:
        if self.cube_reader is None:
            return
        if self.var_source.get() == 'slice':
            if self.model_reader is None:
                messagebox.showinfo('Spectral slice', 'Load a MODEL FITS first.')
                self.var_source.set('dap')
                return
            values = self.compute_slice()
            self.map_image = np.full((self.cube_reader.ny, self.cube_reader.nx),
                                     np.nan, dtype=np.float64)
            self.map_image[self.cube_reader.ypix, self.cube_reader.xpix] = values
            self.map_vals = values
            w1, w2 = self._window()
            self.map_label = (f'{self.var_slice_chan.get()} '
                              f'{self.var_slice_stat.get()} [{w1:.1f}–{w2:.1f}] Å')
        else:
            if self.dap_col not in self.cube_reader.planes:
                self._select_default_column()
            if self.dap_col not in self.cube_reader.planes:
                return
            self.map_image = self.cube_reader.get_map(self.dap_col)
            self.map_vals = self.map_image[self.cube_reader.ypix,
                                           self.cube_reader.xpix]
            self.map_label = self.dap_col
        if not self.var_lock.get():
            self.auto_scale(redraw=False)
        self.draw_map(reset_view=reset_view)
        self._update_status()

    def draw_map(self, reset_view: bool = False) -> None:
        if self.cube_reader is None or self.map_image is None:
            return
        try:
            norm = self._norm()
        except (tk.TclError, ValueError):
            return
        ax = self.ax_map
        previous = (ax.get_xlim(), ax.get_ylim()) if ax.has_data() and not reset_view else None
        ax.clear()
        self.cax_map.clear()
        self.cax_map.set_axis_on()
        self._sel_line = self._cur_line = None
        if self._rect_selector is not None:
            self._rect_selector.set_active(False)
        self._rect_selector = None
        image = self.map_image
        if self.var_only_good.get() and self.good is not None:
            # The cube's unobserved pixels remain NaN. Hide any PT-flagged rows.
            image = image.copy()
            image[self.cube_reader.ypix[~self.good],
                  self.cube_reader.xpix[~self.good]] = np.nan
        ra_corner, dec_corner = self.cube_reader.corners()
        self._coll = ax.pcolormesh(
            ra_corner, dec_corner, np.ma.masked_invalid(image),
            cmap=self._cmap(), norm=norm, shading='flat', rasterized=True)
        self._cbar = self.fig_map.colorbar(self._coll, cax=self.cax_map)
        ax.set_title(self.map_label, fontsize=9)
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('Dec [deg]')
        dec0 = np.nanmedian(self.dec)
        cosd = max(np.cos(np.deg2rad(dec0)), 1e-4)
        ax.set_aspect(1 / cosd)
        if previous is not None:
            ax.set_xlim(previous[0])
            ax.set_ylim(previous[1])
        else:
            # Preserve the usual astronomical RA orientation.
            ax.set_xlim(float(np.nanmax(ra_corner)), float(np.nanmin(ra_corner)))
            ax.set_ylim(float(np.nanmin(dec_corner)), float(np.nanmax(dec_corner)))
        self._ensure_rect_selector()
        self._draw_markers()
        self.canvas_map.draw_idle()

    def _spectra(self) -> tuple[Optional[np.ndarray], str]:
        if self.model_reader is None:
            return None, ''
        if self.var_spec_mode.get() == 'mean' and self.selected is not None and self.selected.any():
            rows = np.flatnonzero(self.selected)
            return (self.model_reader.mean_spectrum(rows),
                    f'Mean spectrum of {len(rows)} selected spaxels')
        if self.current_fiber is None:
            return None, ''
        i = int(self.current_fiber)
        return (self.model_reader.spectrum(i),
                (f'Spaxel ({self.cube_reader.xpix[i]}, {self.cube_reader.ypix[i]}) RSS row {i}'
                 if self.cube_reader is not None else f'RSS row {i}'))

    def _update_status(self) -> None:
        lines = []
        if self.cube_reader is not None:
            rd = self.cube_reader
            lines.append(f'DAP cube: {os.path.basename(rd.filename)}')
            lines.append(f'Grid {rd.nx}×{rd.ny}; fitted {len(rd.pt)} spaxels; '
                         f'{len(rd.planes)} parameter planes')
        if self.model_reader is not None:
            model = self.model_reader
            lines.append(f'MODEL: {os.path.basename(model.filename)}')
            lines.append(f'{model.shape[0]} channels; {model.shape[2]} wavelengths '
                         f'[{model.wave[0]:.1f}, {model.wave[-1]:.1f}] Å')
            if self._pek_available():
                lines.append('PEK models available')
        if self.current_fiber is not None and self.cube_reader is not None:
            rd = self.cube_reader
            i = self.current_fiber
            lines.append(f'Spaxel x={rd.xpix[i]}, y={rd.ypix[i]} '
                         f'RSS row={i} id={_text(self.pt["id"][i:i+1])[0]}')
            lines.append(f'RA={self.ra[i]:.6f} Dec={self.dec[i]:.6f}')
            if self.map_vals is not None:
                value = float(self.map_vals[i])
                if self.var_source.get() == 'dap' and self.dap_col in rd.planes:
                    readable = rd.decode_value(self.dap_col, value)
                else:
                    readable = f'{value:.5g}'
                lines.append(f'{self.map_label}: {readable}')
        if self.selected is not None and self.selected.any():
            lines.append(f'Selected: {self.selected.sum()} spaxels')
        self._set_status('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Interactive LVM-DAP cube-mode parameter map + spectral MODEL viewer')
    parser.add_argument('dap', nargs='?', help='Cube DAP (*.cube.dap.fits.gz)')
    parser.add_argument('model', nargs='?', help='Companion MODEL (*.model.fits.gz)')
    parser.add_argument('--list-planes', action='store_true',
                        help='Print all cube parameter planes and exit without starting the GUI')
    parser.add_argument('--no-model', action='store_true',
                        help='Open parameter cubes only; do not auto-load companion MODEL')
    args = parser.parse_args()
    if args.list_planes:
        if not args.dap:
            parser.error('--list-planes requires a cube DAP FITS filename')
        rd = CubeDAPReader(args.dap)
        print(f'{rd.nx}x{rd.ny} spatial grid, {len(rd.pt)} fitted spaxels, '
              f'{len(rd.planes)} parameter planes')
        for plane in rd.planes.values():
            print(f'{plane.extension:18s} {plane.plane:5d} '
                  f'{plane.parameter} [{plane.unit}] ({plane.encoding})')
        return
    model = None if args.no_model else (args.model or (guess_cube_model_file(args.dap)
                                                       if args.dap else None))
    app = CubeViewApp(args.dap, model)
    app.mainloop()


if __name__ == '__main__':
    main()
