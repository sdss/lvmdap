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

## Quickstart with DR20 exposure 00004297

Run these commands from the root of the source checkout after installation. The example uses the bundled MaStar templates, line lists and mask. Gaia matching is optional.

### 1. Prepare a work directory and download the exposure

```sh
mkdir -p ../lvmdap-quickstart
cp -R _examples _legacy _fitting-data ../lvmdap-quickstart/
cd ../lvmdap-quickstart
mkdir -p results

curl --fail --location --output lvmSFrame-00004297.fits \
  'https://dr20.sdss.org/sas/dr20/spectro/lvm/redux/1.2.0/0011XX/11111/60191/lvmSFrame-00004297.fits'
```

You can also download the file from the archive in a browser and save it as `lvmSFrame-00004297.fits` in the work directory. The file is about 489 MiB. [SDSS DR20 LVM data access](https://www.sdss.org/dr20/lvm/data/)

### 2. Fit a small subset as a smoke test

```sh
cp _examples/lvm-dap_SN5_general.yaml fit-smoke.yaml
printf '\nny_range: [0, 5]\n' >> fit-smoke.yaml
lvm-dap-conf lvmSFrame-00004297.fits run-4297-smoke fit-smoke.yaml
```

Run the fit from the work directory so the relative paths in the YAML resolve. `lvmdap_dir: .` points the loader at this directory, and the `../_fitting-data` and `../_legacy` prefixes in YAML values resolve to the copied assets. The SN5 example uses the bundled four-template RSP file for both template inputs, writes uncompressed FITS files to `results/`, and disables plots.

The slice selects the first five input rows. The pipeline applies its normal fiber selection within that slice. Look in `results/` for `run-4297-smoke.dap.fits`, `run-4297-smoke.model.fits`, and the integrated and averaged products with the `m_` and `avg_` prefixes. For a full exposure, pass the original example YAML and use a new label:

```sh
lvm-dap-conf lvmSFrame-00004297.fits run-4297 _examples/lvm-dap_SN5_general.yaml
```

The three positional arguments are the input FITS file, an output label, and the YAML configuration. Use `--in_rss` for generic RSS or `--lvmsim` for LVM simulator input.

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

The direct quickstart runs with the redshift settings in the example YAML. For a Gaia-based prior, run these commands from the same work directory after preparing `fit-smoke.yaml`. `lvm-dap-yaml-set-z` finds the nearest populated cell in a Gaia radial-velocity map (FITS or ECSV). It uses the good-fiber pointing from the input LVM exposure unless RA and Dec are overridden. For standard LVM SFrames, the default conversion to the observed spectral frame is `v_Gaia - HVCOR`, with HVCOR read from the primary header keyword `HIERARCH WAVE HELIORV_SCI`. The command derives the redshift prior and bounds from the cell velocity and dispersion.

```sh
lvm-dap-yaml-set-z --dry-run lvmSFrame-00004297.fits _legacy/gaia_rv_map.fits fit-smoke.yaml fit-smoke-gaia.yaml
lvm-dap-yaml-set-z lvmSFrame-00004297.fits _legacy/gaia_rv_map.fits fit-smoke.yaml fit-smoke-gaia.yaml
lvm-dap-conf lvmSFrame-00004297.fits run-4297-smoke-gaia fit-smoke-gaia.yaml
```

This writes the four-element `redshift` array, sets `auto-redshift: true`, and updates `auto_z_min`, `auto_z_max` and `auto_z_del`. Use `--create-missing` if the base configuration lacks those entries. The helper computes a suggested step by dividing the range into 40 intervals. The default bounds span ±3 cell dispersions, with a dispersion floor of 20 km/s. Options include `--min-stars`, `--max-sep` (degrees), `--weighted`, `--nsigma`, and `--sigma-max`. Use `--hvcor` or `--hvcor-sign` only after verifying the wavelength frame.

For a full exposure with Gaia matching, use `_examples/lvm-dap_SN5_general.yaml` as the helper input and save and fit a separate full-exposure YAML.

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

- `LABEL.dap.fits` contains PT, stellar results, emission-line tables by method and arm, fit diagnostics and masks. Wide tables can span several extensions. When spatial-seed diagnostics are written in RSS mode, `GAS_SPATIAL_SEED` appears before the final `INFO` extension. Access extensions by name and use `lvmdap.dap_tools.read_DAP_file` for version-aware RSS table reading.
- `LABEL.model.fits` contains the observed spectra and model components. Inspect the plane metadata before indexing the arrays.
- `m_LABEL.dap.fits` contains integrated products. `avg_LABEL.dap.fits` contains averaged-fiber products from full runs. Each can have a companion model product.
- `lvm-dap-yaml-edit` edits configuration keys and can create missing keys with `--create-missing`. Example: `lvm-dap-yaml-edit --create-missing fit.yaml fit-gzip.yaml --gzip=true`.
- `lvm-dap-gen-out-mod` writes `LABEL.output.fits.gz`. `lvm-dap-sim` generates simulated spectra.
- For Gaia-assisted foreground cleaning, run `lvm-dap-conf` with `gzip: true`, then run `lvm-dap-gen-out-mod` with the same label. Place `LABEL.output.fits.gz` next to `LABEL.dap.fits.gz`. The cleaner infers the companion path from the DAP filename, including its compression suffix. Run `lvm-dap-clean-st-mw outputfile dap_file m_lim n_spec SN_spec`, where `SN_spec` is the minimum continuum S/N used to estimate the MW ISM contribution.
- The source checkout includes `cmdline_tools/LVMview_DAP_model.py` for model spectra and `cmdline_tools/LVMcompare_two_DAP_files.py` for product comparisons. Run each script with `--help` for usage.
- `lvm-dap-conf-v120` and `lvm-dap-outdate` are legacy entry points. Auxiliary commands include `lvm-dap-cube2map`, `lvm-dap-coadd-cubes`, `lvm-dap-gas-cube-extractor`, `lvm-dap-preprocess-muse`, `lvm-dap-preprocess-manga`, `lvm-dap-mwm-dap`, and `lvm-dap-clean-outputs`. See their command-specific help.

For instrument context and software, see the [SDSS-V Local Volume Mapper](https://www.sdss.org/sdss5/instruments/lvm/) and the [LVM-DAP repository](https://github.com/sdss/lvmdap).
