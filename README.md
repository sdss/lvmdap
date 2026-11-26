# Local Volume Mapper Data Analysis Pipeline (LVM-DAP)

LVM-DAP analyzes spectra from the SDSS-V Local Volume Mapper. It fits the integrated spectrum, models the stellar continuum, measures emission lines, and writes spectral models and fit diagnostics. The YAML-driven pipeline accepts LVM exposures, generic row-stacked spectra (RSS), LVM simulations, and reconstructed 3DCubeGen FITS cubes.

## Install from source

Use an isolated Python environment. The project's setup example uses Python 3.11. Install pyPipe3D if it is unavailable to your package installer.

```sh
conda create -n lvmdap python=3.11
conda activate lvmdap
git clone https://github.com/sdss/lvmdap.git
cd lvmdap
python -m pip install .
lvm-dap-conf --help
```

## Directory structure

| Path | Contents |
| --- | --- |
| [`lvmdap/`](lvmdap/) | Python package, fitting routines, and command entry points |
| [`_examples/`](_examples/) | Example scripts and analysis files |
| [`_legacy/`](_legacy/) | YAML configurations, emission-line catalogues, wavelength masks, and Gaia velocity map |
| [`_rsp/`](_rsp/) | Bundled DESI resolved stellar population (RSP) templates used in the quickstart |
| [`_fitting-data/`](_fitting-data/) | Bundled MaStar templates for earlier examples |
| [`cmdline_tools/`](cmdline_tools/) | Standalone viewers and comparison tools |
| [`notebooks/`](notebooks/) | Worked examples for fitting, template building, and product inspection |
| [`datamodel/`](datamodel/) | Repository data-model examples |
| [`docs/`](docs/) and [`tests/`](tests/) | Documentation source and regression tests |

## DR20 data access and visualization

**Use LVMvis first to explore the public DAP results.** It lets you select an exposure and fiber, inspect the observed spectrum, and compare the model components and residuals in the browser. For files produced by your own run, use the local viewer in the quickstart. [Open LVMvis](https://dr20.sdss.org/lvmvis/) · [LVMvis guide](https://www.sdss.org/dr20/data_access/lvmvis/)

- Start with the DR20 LVM data-access page for product descriptions, FITS data models, and links to the archive. Use `drpall` and `dapall` to identify exposures and products, and consult `qcall` for quality flags. [SDSS DR20](https://www.sdss.org/dr20/) · [DR20 LVM data access](https://www.sdss.org/dr20/lvm/data/)
- Download reduced spectra such as the quickstart `lvmSFrame` from the DRP archive. [DR20 LVM DRP archive](https://dr20.sdss.org/sas/dr20/spectro/lvm/redux/1.2.0/)
- Download the released DAP measurement tables and their companion spectral models from the DAP archive. The DR20 archive uses DAP version `1.2.0.251218`. [DR20 LVM DAP archive](https://dr20.sdss.org/sas/dr20/spectro/lvm/analysis/1.2.0/1.2.0.251218/)

## Quickstart with DR20 exposure 00004297

Run these commands from the root of the LVM-DAP source checkout after installation. Keep the checkout available because the fit uses its line lists and mask. The quickstart uses the DESI RSPs bundled in the checkout and fits the full exposure. Gaia matching is optional.

### 1. Prepare a work directory and download the exposure

```sh
lvmdap_checkout="$(pwd -P)"
mkdir -p ../lvmdap-fit-test
cp -R _examples _legacy _rsp ../lvmdap-fit-test/
cd ../lvmdap-fit-test
mkdir -p results

curl --fail --location --output lvmSFrame-00004297.fits \
  'https://dr20.sdss.org/sas/dr20/spectro/lvm/redux/1.2.0/0011XX/11111/60191/lvmSFrame-00004297.fits'
```

You can also download the exposure in a browser and save it as `lvmSFrame-00004297.fits` in the test directory. Allow about 489 MiB for the input and additional space for the fit products. [SDSS DR20 LVM data access](https://www.sdss.org/dr20/lvm/data/)

### 2. Create a local YAML from the SN5 configuration

The source configuration is [`_legacy/lvm-dap_SN5_general.yaml`](_legacy/lvm-dap_SN5_general.yaml). Its operational paths refer to a separate installation and template store. The following command writes a local `fit.yaml` with the results directory, the path to your source checkout, and the bundled templates. Run it in the same shell as step 1.

```sh
python - "$lvmdap_checkout" <<'PYTHON'
from pathlib import Path
import sys
import yaml

source = Path("_legacy/lvm-dap_SN5_general.yaml")
config = yaml.safe_load(source.read_text())
config.update({
    "output_path": str(Path("results").resolve()),
    "lvmdap_dir": sys.argv[1],
    "rsp-file": str(Path("_rsp/rsp-4.fits").resolve()),
    "rsp-nl-file": str(Path("_rsp/rsp-36.fits").resolve()),
    "use_gaia_vel": False,
    "clear_outputs": False,
})
Path("fit.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PYTHON
```

The example keeps `sigma-inst: 0.01` and the SN5 settings. It uses the bundled `rsp-4.fits` for stellar decomposition and `rsp-36.fits` for the nonlinear parameters. The 342-template production library in the source YAML is a separate asset. Run from the test directory so the component files referenced by the strong-line configuration are available under `_legacy/`.

### 3. Fit the exposure

```sh
lvm-dap-conf lvmSFrame-00004297.fits run-4297 fit.yaml
```

The three positional arguments are the input FITS file, an output label, and the YAML configuration. The example writes gzip-compressed products. In `results/`, look for `run-4297.dap.fits.gz`, `run-4297.model.fits.gz`, and their integrated and averaged counterparts prefixed with `m_` and `avg_`. The full fit can take substantially longer than a small subset. To test a subset, copy `fit.yaml`, add `ny_range: [0, 5]` to the copy, and fit with a fresh label.

Use `--in_rss` for generic RSS or `--lvmsim` for LVM simulator input.

### 4. Visualize the results

From the test directory, open the DAP table and its companion spectral model in the interactive viewer.

```sh
../lvmdap/cmdline_tools/LVMview_DAP_model.py results/run-4297.dap.fits.gz results/run-4297.model.fits.gz
```

### Common YAML settings

The following keys supplement a working fit configuration. Keep the existing template and wavelength settings for your dataset.

```yaml
output_path: /path/to/results
gzip: false
out-plot-format: png
flux-scale-org: [-1, 1]
only-integrated: false
```

| Key | Purpose |
| --- | --- |
| `rsp-file`, `rsp-nl-file` | Libraries for stellar decomposition and nonlinear fitting |
| `redshift`, `sigma`, `AV` | Four values in the order guess, step, lower bound, upper bound. Redshift is dimensionless, stellar dispersion is in km/s, and attenuation is in magnitudes. |
| `w-range`, `w-range-nl` | Wavelength ranges for stellar decomposition and nonlinear fitting |
| `emission-lines-file`, `emission-lines-file-long`, `config-file` | Line catalogues and strong-line fit configuration |
| `ny_range` | Optional RSS row selection. Review sample configurations before running a full exposure. |
| `only-integrated` | Stop after the integrated fit when true |
| `gzip` | Write compressed `.fits.gz` when true. Default is uncompressed `.fits`. You can also pass `--gzip`. |
| `out-plot-format` | PNG by default. Choose `pdf` for plots that support it. Plot generation is controlled separately. |
| `flux-scale-org` | Plot flux range `[min, max]`. Use `[-1, 1]` for automatic scaling. Supply this YAML key when `--flux-scale` is omitted or left at its default. |
| `remask` | Opt-in supplemental LSF-aware masking |
| `SN_CUT`, `SN_CUT_INT` | S/N settings for the fiber and integrated stellar analysis |

Keep hyphens and underscores as shown. Use YAML booleans such as `true` and `false`.

## Optional: match the pointing to a Gaia redshift prior

The direct quickstart runs with the redshift settings in the example YAML. For a Gaia-based prior, run these commands from the same work directory after preparing `fit.yaml`. `lvm-dap-yaml-set-z` finds the nearest populated cell in a Gaia radial-velocity map (FITS or ECSV). It uses the good-fiber pointing from the input LVM exposure unless RA and Dec are overridden. For standard LVM SFrames, the default conversion to the observed spectral frame is `v_Gaia - HVCOR`, with HVCOR read from the primary header keyword `HIERARCH WAVE HELIORV_SCI`. The command derives the redshift prior and bounds from the cell velocity and dispersion.

```sh
lvm-dap-yaml-set-z --dry-run lvmSFrame-00004297.fits _legacy/gaia_rv_map.fits fit.yaml fit-gaia.yaml
lvm-dap-yaml-set-z lvmSFrame-00004297.fits _legacy/gaia_rv_map.fits fit.yaml fit-gaia.yaml
lvm-dap-conf lvmSFrame-00004297.fits run-4297-gaia fit-gaia.yaml
```

This writes the four-element `redshift` array, sets `auto-redshift: true`, and updates `auto_z_min`, `auto_z_max` and `auto_z_del`. Use `--create-missing` if the base configuration lacks those entries. The helper computes a suggested step by dividing the range into 40 intervals. The default bounds span ±3 cell dispersions, with a dispersion floor of 20 km/s. Options include `--min-stars`, `--max-sep` (degrees), `--weighted`, `--nsigma`, and `--sigma-max`. Use `--hvcor` or `--hvcor-sign` only after verifying the wavelength frame.

**Grid-step compatibility:** This checkout's main DAP command reads `auto_z_d`, while the matching helper writes `auto_z_del`. The current integrated peak matcher uses the configured bounds and measured peak wavelengths. It does not scan a grid at the helper's suggested step. The Gaia match provides a prior for the spectral fit.

## Analyze a 3DCubeGen cube

```sh
lvm-dap-conf --in-cube reconstructed.fits run-cube cube-fit.yaml
```

Use a working fit YAML adjusted for the cube's wavelength coverage and templates. The input is a 3DCubeGen reconstruction with flux in HDU 0 and matching **1σ errors** in HDU 1, both in Python array order `(wavelength, y, x)`. The reader uses the spectral and celestial WCS and fits usable spaxels through the RSS workflow. Flux and errors use the same conversion factor derived from the flux HDU's `BUNIT` or an explicit override.

| Cube key | Default | Purpose |
| --- | --- | --- |
| `cube_x_range`, `cube_y_range` | Full axis | Optional crop as zero-based `[start, stop)` indices in the original cube |
| `create_cube_mask` | `true` | Expand invalid-sample masks over a 3×3 spatial footprint at each wavelength. With `false`, keep pointwise masking. |
| `cube_input_scale` | From flux `BUNIT` | Explicit positive multiplier into DAP fit units when needed |
| `cube_lsf_sigma` | Unset | Explicit instrumental Gaussian sigma in Å. The reconstructed cube has no preserved instrumental LSF, so supply a value only when independently justified. |

Cube mode uses its own input path. Keep `--in_rss` and `--lvmsim` off. The full cube product is `LABEL.cube.dap.fits` (or `.fits.gz`), with WCS-registered parameter planes, a `PLANE_INDEX` catalogue, and a `SPAXEL_INDEX` table. The companion `LABEL.model.fits` stores spectra indexed by the fitted RSS rows. The integrated and averaged products keep their usual layouts.

```sh
python cmdline_tools/LVMview_DAP_cube.py run-cube.cube.dap.fits --list-planes
```

## Gas discovery and emission-line fits

The gas velocity prepass seeds parametric (PM), nonparametric (NP), and constrained Gaussian (PEK) measurements. BEST records the selected method per line. The integrated redshift, per-fiber peak discovery and final fit width are separate stages.

| YAML key | Default | Role |
| --- | --- | --- |
| `gas_map_enabled` | `true` | Run the gas-velocity prepass |
| `gas_map_fixed_velocity` | `true` | Use the fixed-velocity gas fitting mode when the prepass supplies a velocity |
| `gas_map_wave_min`, `gas_map_wave_max`, `gas_map_padding` | `6548.05`, `6730.82`, `60.0` Å | Configure the prepass wavelength interval and padding |
| `gas_map_velocity_half_range_kms` | `300.0` | Discovery half-range in km/s |
| `gas_map_sn_cut`, `gas_map_min_matches`, `gas_map_peak_threshold` | `5.0`, `2`, `2.0` | Control matched-line significance, count, and peak finding |
| `gas_map_halpha_only_fallback` | `false` | Allow an Hα-only seed when the standard multiline prepass has no accepted solution |
| `gas_map_compare_halpha` | `true` | Run the fixed-versus-refined Hα diagnostic where applicable |
| `velocity_tolerance` | Stage-specific fallback | Explicit value in km/s. Several discovery and fitting stages read it, so changing it can affect more than the final fit window. |
| `pm_fit_engine` | `fit3d` | PM engine. Alternate engines are opt-in. |
| `pm_fit3d_recovery` | `false` | Compare a second FIT3D candidate fit for each strong-line cluster |
| `fit_PEK` | `true` | Run the PEK stage |
| `pek_velocity_tolerance` | Unset | Optional PEK velocity tolerance in km/s |
| `pek_fit_padding`, `pek_max_cluster_span`, `pek_max_adjacent_distance` | `20.0`, `75.0`, `25.0` Å | Control PEK cluster fitting windows and grouping |
| `pek_velocity_start` | `np_weighted` | Start per-fiber PEK at the flux-weighted NP cluster velocity. Use `pm` for the legacy PM start. |
| `pek_np_weighted_halfwidth` | 20 km/s | Minimum NP-centred half-width. PEK uses the larger of this value and three times the propagated uncertainty, intersected with the gas-map window when available. |
| `pek_halpha_restart` | `false` | Opt in to the Hα-seeded alternate PEK fit where eligible |

### Experimental spatial peak seeds

Enable `gas_map_spatial_peak_seed: true` with the gas map enabled. Supply a positive `gas_map_spatial_peak_sn_cut` and choose the spatial filter configuration for your test:

- **Velocity difference:** Set a positive `gas_map_spatial_difference_kms`.
- **Joint velocity and Hα flux:** Set both `gas_map_spatial_velocity_n_sigma` and `gas_map_spatial_flux_n_sigma`, or use `gas_map_spatial_n_sigma` as a common value. `gas_map_spatial_neighbors` defaults to 6 spatial neighbors, and `gas_map_spatial_iterations` defaults to 1.
- **Optional Hα identity guard:** `gas_map_spatial_halpha_identity_guard` defaults to `false`.
- **Optional weighted ranking:** `gas_map_halpha_weighted_seed` defaults to `false` and requires joint spatial settings. The peak finder supplies Hα candidates. Eligible supporting lines are ranked using flux ratios from the integrated BEST fit by line identity. A supporting line whose search window overlaps Hα is excluded. The weights are reported in the run log.
- **Fit window:** `gas_map_spatial_fit_fraction` defaults to `0.25` and must be greater than 0 and at most 1. It scales the configured `velocity_tolerance` for the spatial-seed fit half-width where used. This fraction does not change the prepass discovery half-range.

A narrow fit interval can limit recovery from an incorrect seed. Check the seed map and line residuals when testing these options.

## FITS products and utilities

NP emission-line bands include `vel_w_<line>` and `e_vel_w_<line>` in km/s.
They repeat the group's flux-weighted velocity and propagated uncertainty for
each member, using positive NP measurements with flux S/N ≥ 3. Lines
flagged as blended by the catalogue are excluded from the weighted seed.
A group with no eligible unblended measurement has NaN weighted NP fields
and PEK uses the PM start. The original
per-line NP measurements remain available. PEK marks disjoint or zero-width NP
and map intersections with `vel_w_conflict_<line>_pek` and uses the map
window in those cases. Explicitly fixed velocities take precedence. The
integrated PEK fit uses its PM start because integrated BEST can supply the
line weights used by the upstream gas prepass. BEST uses
the selected estimator's measurements and its existing common-pixel scoring.


- `LABEL.dap.fits` contains PT, stellar results, emission-line tables by method and arm, fit diagnostics and masks. Wide tables can span several extensions. When spatial-seed diagnostics are written in RSS mode, `GAS_SPATIAL_SEED` appears before the final `INFO` extension. Access extensions by name and use `lvmdap.dap_tools.read_DAP_file` for version-aware RSS table reading.
- `LABEL.model.fits` contains the observed spectra and model components. Inspect the plane metadata before indexing the arrays.
- `m_LABEL.dap.fits` contains integrated products. `avg_LABEL.dap.fits` contains averaged-fiber products from full runs. Each can have a companion model product.
- `lvm-dap-yaml-edit` edits configuration keys and can create missing keys with `--create-missing`. Example: `lvm-dap-yaml-edit --create-missing fit.yaml fit-gzip.yaml --gzip=true`.
- `lvm-dap-gen-out-mod` writes `LABEL.output.fits.gz`. `lvm-dap-sim` generates simulated spectra.
- For Gaia-assisted foreground cleaning, run `lvm-dap-conf` with `gzip: true`, then run `lvm-dap-gen-out-mod` with the same label. Place `LABEL.output.fits.gz` next to `LABEL.dap.fits.gz`. The cleaner infers the companion path from the DAP filename, including its compression suffix. Run `lvm-dap-clean-st-mw outputfile dap_file m_lim n_spec SN_spec`, where `SN_spec` is the minimum continuum S/N used to estimate the MW ISM contribution.
- The source checkout includes `cmdline_tools/LVMview_DAP_model.py` for model spectra and `cmdline_tools/LVMcompare_two_DAP_files.py` for product comparisons. Run each script with `--help` for usage.
- `lvm-dap-conf-v120` and `lvm-dap-outdate` are legacy entry points. Auxiliary commands include `lvm-dap-cube2map`, `lvm-dap-coadd-cubes`, `lvm-dap-gas-cube-extractor`, `lvm-dap-preprocess-muse`, `lvm-dap-preprocess-manga`, `lvm-dap-mwm-dap`, and `lvm-dap-clean-outputs`. See their command-specific help.

## References

Cite the pipeline paper when using LVM-DAP. The survey overview describes LVM's scientific motivation and design, and the Helix paper explains the public Helix data and DAP products.

- Sánchez et al. (2025), *The SDSS-V Local Volume Mapper: Data Analysis Pipeline*, AJ **169**, 52. [SciXplorer](https://scixplorer.org/abs/2025AJ....169...52S/abstract) · [Journal article](https://doi.org/10.3847/1538-3881/ad93bb)
- Drory et al. (2024), *The SDSS-V Local Volume Mapper (LVM): Scientific Motivation and Project Overview*, AJ **168**, 198. [SciXplorer](https://scixplorer.org/abs/2024AJ....168..198D/abstract) · [Journal article](https://doi.org/10.3847/1538-3881/ad6de9)
- Sánchez et al. (2026), *SDSS-V Local Volume Mapper (LVM): Helix Nebula public data, Data Analysis Pipeline data products*, RMxAA **62**, 87. [SciXplorer](https://scixplorer.org/abs/2026RMxAA..62a..87S/abstract) · [arXiv](https://arxiv.org/abs/2602.10072)

## LVM-DAP video tutorials

These YouTube tutorials cover the workflow and data products. Use the quickstart above for current commands and example settings.

1. [Introduction to LVM-DAP](https://youtu.be/vSBJbKHStoE)
2. [Installing LVM-DAP](https://www.youtube.com/watch?v=NfZtoQMA8xA)
3. [LVM SFrame format and visualization](https://www.youtube.com/watch?v=QaNu5cUktgg)
4. [Running LVM-DAP with scripts and notebooks](https://youtu.be/ZkkZU96XWbQ)
5. [Exploring DAP FITS tables with TOPCAT](https://www.youtube.com/watch?v=wikAIyVp2j8)
6. [Reading DAP products in Python](https://www.youtube.com/watch?v=phUuCq2syKM)
7. [LVM-DAP YAML and configuration files](https://www.youtube.com/watch?v=qoudb9Th7WA)
8. [Generating a spectral model from DAP outputs](https://www.youtube.com/watch?v=8QoLYO2F0VY)
9. [RSP template format, use, and construction](https://www.youtube.com/watch?v=d5CtkOs3-qE)
