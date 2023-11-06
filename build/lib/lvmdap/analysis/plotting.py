
import itertools as it
import numpy as np
import pandas as pd

import scipy.optimize as so
from scipy.interpolate import CloughTocher2DInterpolator

from matplotlib import rc
from cycler import cycler

from PIL import Image, ImageEnhance


clist = "#114477 #117755 #E8601C #771111 #771144 #4477AA #44AA88 #F1932D #AA4477 #774411 #777711 #AA4455".split()
ccycle = cycler("color", clist)

latex_preamble = "\n".join([
    r"\usepackage{helvet}",
    r"\usepackage{amsmath}",
    r"\usepackage[helvet]{sfmath}",
    r"\renewcommand{\familydefault}{\sfdefault}"
])

font = {"family":"sans-serif", "sans-serif":"Open Sans", "size":20, "weight":300}
text = {"usetex":True, "latex.preamble":latex_preamble, "hinting":"native"}

rc("figure", figsize=(10, 10))
rc("text", **text)
# rc("font", **font)
rc("axes", linewidth=1.0, labelsize="medium", titlesize="medium", labelweight=300)#, prop_cycle=ccycle)
rc("xtick", labelsize="x-small")
rc("xtick.major", width=1.0)
rc("ytick", labelsize="x-small")
rc("ytick.major", width=1.0)
rc("lines", linewidth=2.0, markeredgewidth=0.0, markersize=7)
rc("patch", linewidth=0.0)
rc("legend", numpoints=1, scatterpoints=1, fontsize="x-small", title_fontsize="small", handletextpad=0.4, handlelength=1, handleheight=1, frameon=False)
rc("savefig", dpi=92, format="pdf")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from pyFIT3D.common.io import get_wave_from_header
from lvmdap.analysis.stats import weighted_pdf
from lvmdap.analysis.stats import normalize_to_pdf, get_nth_moment, get_nth_percentile
from lvmdap.analysis import img_scale


LIGHT_COLOR = "#E3DEDE"
MED_COLOR = "#8284A5"
MASTAR_COLOR = "#574E6C"
MASTAR_CMAP = sns.color_palette(f"blend:{LIGHT_COLOR},{MASTAR_COLOR}")
QUARTILE_PALETTE = sns.color_palette(f"blend:{LIGHT_COLOR},{MASTAR_COLOR}", n_colors=3)

def find_confidence_interval(x, pdf, confidence_level):
    """Return the difference between the integrated PDF
    ABOVE a given threshold, and a target confidence level.

    Inspired by https://bit.ly/2A63f4A

    Parameters
    ----------
    x: float
        A threshold above which the PDF will be integrated
    percent_dist: array_like
        A PDF array
    percent: float
        Target confidence interval

    Returns
    -------
    residual: float
        integral(PDF > x) - confidence_level
    """
    return pdf[pdf > x].sum() - confidence_level

def contours_from_pdf(pdf_func, range_x, range_y, deltas=0.1, percentiles=[68,95,99], return_grid=False):
    """Return the contour levels that (nearly) represent the given
    confidence interval of the PDF.

    Inspired by: https://bit.ly/2A63f4A

    Parameters
    ----------
    pdf_func: function
        The PDF function from which to draw the confidence
        intervals
    range_y, range_y: tuple
        The ranges within which calculate the support of the
        PDF.
    deltas: float, tuple
        The step of the grid If float, the step will
        be the same in x and y. If tuple, the steps
        in x and the y directions, respectively
    percentiles: tuple
        The percentiles at which to compute the levels
    return_grid: boolean
        Whether to return also the grid X, Y, Z to draw the
        contours. Defaults to False

    Returns
    -------
    levels: array_like
        The sorted array of levels
    X, Y, Z: array_like, optional
        The arrays to draw the contours as in

        >>> plt.contour(X, Y, Z, levels=levels)
    """

    if not isinstance(deltas, str) and hasattr(deltas, "__getitem__"):
        delta_x, delta_y = deltas[:2]
    elif isinstance(deltas, (float, int)):
        delta_x, delta_y = deltas, deltas

    x_grid = np.arange(*range_x, delta_x)
    y_grid = np.arange(*range_y, delta_y)
    X, Y = np.meshgrid(x_grid, y_grid)

    Z = pdf_func(X.ravel(), Y.ravel())
    Z = Z.reshape(X.shape)

    prob = Z*delta_x*delta_y
    # in some cases the PDF will not be normalized to 1.0, fix this here...
    prob /= prob.sum()

    percentiles_ = np.asarray(sorted(percentiles, reverse=True))/100
    levels = np.asarray([so.brenth(find_confidence_interval, 0.0, prob.max(), args=(prob,p)) for p in percentiles_])
    levels = levels/delta_x/delta_y

    if return_grid:
        return np.asarray(levels), X, Y, Z

    return np.asarray(levels)

def plot_triang_pdfs(pdf_params_hdus, coeffs, cmap=None, axs=None):
    """Return the PDF triangular plots and the corresponding margins, given the PDF HDU list"""

    margins = {}
    colors = []
    labels = {
        "TEFF":r"$\log{T_\text{eff}}$",
        "LOGG":r"$\log{g}$",
        "MET":r"$[\text{Fe}/\text{H}]$",
        "ALPHAM":r"$[\alpha/\text{Fe}]$"
    }
    npars = len(labels)
    if axs is None:
        _, axs = plt.subplots(npars, npars, sharex="col", sharey=False, figsize=(12,12))

    for ihdu, (i,j) in zip(range(1,len(pdf_params_hdus)), it.combinations(range(npars),2)):

        wPDF, x_scale, y_scale = weighted_pdf(pdf_params_hdus, ihdu, coeffs=coeffs)

        if i not in margins:
            mPDF = (wPDF*pdf_params_hdus[ihdu].header["CDELT2"]).sum(axis=0)
            margins[i] = (x_scale, mPDF)
        if j not in margins:
            mPDF = (wPDF*pdf_params_hdus[ihdu].header["CDELT1"]).sum(axis=1)
            margins[j] = (y_scale, mPDF)

        X, Y = np.meshgrid(x_scale, y_scale)
        wPDF_func = CloughTocher2DInterpolator(np.column_stack((X.flatten(),Y.flatten())), wPDF.flatten())
        levels, X_, Y_, PDF_ = contours_from_pdf(
            lambda x, y: wPDF_func(np.column_stack((x,y))),
            range_x=x_scale[[0,-1]],
            range_y=y_scale[[0,-1]],
            deltas=0.05, return_grid=True
        )

        x_name = pdf_params_hdus[ihdu].header["CTYPE1"]
        y_name = pdf_params_hdus[ihdu].header["CTYPE2"]

        if cmap is None:
            cmap = sns.cubehelix_palette(start=ihdu-1, reverse=True, as_cmap=True)
            colors.append(cmap.colors[0])
        else:
            colors.append(sns.color_palette(cmap)[0])
        pcm = axs[j,i].pcolormesh(X, Y, wPDF, cmap=cmap, shading="auto")
        axs[j,i].contour(X_, Y_, PDF_, levels=levels, colors="w", linewidths=1)

        mask_x = np.any(~np.isclose(wPDF, 0, rtol=0.05), axis=0)
        mask_y = np.any(~np.isclose(wPDF, 0, rtol=0.05), axis=1)
        axs[j,i].set_xlim(X_.min(), X_.max())
        axs[j,i].set_ylim(Y_.min(), Y_.max())
        if axs[j,i].get_subplotspec().is_last_row():
            axs[j,i].set_xlabel(labels[x_name])
        if axs[j,i].get_subplotspec().is_first_col():
            axs[j,i].set_ylabel(labels[y_name])
        else:
            axs[j,i].tick_params(labelleft=False)

    for i in range(npars):
        x, pdf = margins[i]
        axs[i,i].plot(x, pdf/pdf.max(), "-", color=colors[i])
        axs[i,i].tick_params(left=False, labelleft=False)
        sns.despine(ax=axs[i,i], left=True)
    for i,j in zip(*np.triu_indices_from(axs, k=1)):
        axs[i,j].set_visible(False)

    xlim = axs[1,0].get_xlim()
    ylim = axs[1,0].get_ylim()
    axs[1,0].set_xlim(xlim[::-1])
    axs[1,0].set_ylim(ylim[::-1])
    axs[1,1].set_xlim(axs[1,1].get_xlim()[::-1])

    return axs

def plot_dap_fit(spec_hdu, weights, stellar_param, labels, cmap, color, true_param=None, rss_voxel=None):
    npars = len(labels)
    nproj = len(stellar_param) - 1

    fig = plt.figure(constrained_layout=True, figsize=(22,9))

    n = 10
    gs = GridSpec(npars, n, figure=fig)
    ax0 = fig.add_subplot(gs[:, :n-npars])

    wavelength = get_wave_from_header(spec_hdu.header)
    if rss_voxel is not None:
        ax0.step(wavelength, spec_hdu.data[0,rss_voxel], "-r", lw=1)
        ax0.step(wavelength, spec_hdu.data[2,rss_voxel], "-", color=color, lw=1)
    else:
        ax0.step(wavelength, spec_hdu.data[0], "-r", lw=1)
        ax0.step(wavelength, spec_hdu.data[2], "-", color=color, lw=1)

    ax0.set_xlabel(r"$\lambda$ (\AA)")
    ax0.set_ylabel(r"$f_\lambda$")
    sns.despine(ax=ax0)

    axs = []
    for i in range(npars):
        axs.append([fig.add_subplot(gs[i, j+(n-npars)], sharex=(axs[i-1][j] if i!=0 else None)) for j in range(npars)])
    axs = np.array(axs)
    for i,j in zip(*np.triu_indices_from(axs, k=1)):
            axs[i,j].set_visible(False)


    for ihdu, (i,j) in zip(range(1,nproj+1), it.combinations(range(npars),2)):
        if rss_voxel is not None:
            wPDF, x_scale, y_scale = weighted_pdf(
                stellar_param,
                ihdu,
                coeffs=weights[stellar_param[0].header["NCLUSTER"]*rss_voxel:stellar_param[0].header["NCLUSTER"]*(rss_voxel+1)]
            )
        else:
            wPDF, x_scale, y_scale = weighted_pdf(
                stellar_param,
                ihdu,
                coeffs=weights
            )
        X, Y = np.meshgrid(x_scale, y_scale)

        if not axs[i,i].lines:
            mPDF = (wPDF*stellar_param[ihdu].header["CDELT2"]).sum(axis=0)
            mPDF = normalize_to_pdf(mPDF, x_scale)
            axs[i,i].plot(x_scale, mPDF/mPDF.max(), "-", color=color)
            axs[i,i].axvline(get_nth_moment(x_scale, mPDF, nth=1), ls="-", lw=1.5, color=color)
            axs[i,i].axvspan(get_nth_percentile(x_scale, mPDF, percent=16), get_nth_percentile(x_scale, mPDF, percent=84), lw=0, color=color, alpha=0.5)
            if true_param is not None: axs[i,i].axvline(true_param[i], ls="-", lw=1.5, color="r")
            axs[i,i].tick_params(left=False, labelleft=False)
            sns.despine(ax=axs[i,i], left=True)
        if not axs[j,j].lines:
            mPDF = (wPDF*stellar_param[ihdu].header["CDELT1"]).sum(axis=1)
            mPDF = normalize_to_pdf(mPDF, y_scale)
            axs[j,j].plot(y_scale, mPDF/mPDF.max(), "-", color=color)
            axs[j,j].axvline(get_nth_moment(y_scale, mPDF, nth=1), ls="-", lw=1.5, color=color)
            axs[j,j].axvspan(get_nth_percentile(y_scale, mPDF, percent=16), get_nth_percentile(y_scale, mPDF, percent=84), lw=0, color=color, alpha=0.5)
            if true_param is not None: axs[j,j].axvline(true_param[j], ls="-", lw=1.5, color="r")
            axs[j,j].tick_params(left=False, labelleft=False)
            sns.despine(ax=axs[j,j], left=True)

        axs[j,i].pcolormesh(X, Y, wPDF, cmap=cmap, shading="auto")
        if true_param is not None:
            axs[j,i].axhline(true_param[j], ls="-", lw=1.5, color="r")
            axs[j,i].axvline(true_param[i], ls="-", lw=1.5, color="r")

        mask_x = np.any(~np.isclose(wPDF, 0, rtol=0.05), axis=0)
        mask_y = np.any(~np.isclose(wPDF, 0, rtol=0.05), axis=1)
        x_name = stellar_param[ihdu].header["CTYPE1"]
        y_name = stellar_param[ihdu].header["CTYPE2"]

        axs[j,i].set_xlim(X.min(), X.max())
        axs[j,i].set_ylim(Y.min(), Y.max())
        if axs[j,i].get_subplotspec().is_last_row():
            axs[j,i].set_xlabel(labels[x_name])
        if i == 0:
            axs[j,i].set_ylabel(labels[y_name])
        else:
            axs[j,i].tick_params(labelleft=False)

    xlim = axs[1,0].get_xlim()
    ylim = axs[1,0].get_ylim()
    axs[1,0].set_xlim(xlim[::-1])
    axs[1,0].set_ylim(ylim[::-1])
    axs[1,1].set_xlim(axs[1,1].get_xlim()[::-1])

    return fig, ax0, axs

def build_comparison_table(tablea, tableb, labela, labelb, columns, delta_prefix=r"$\Delta$"):
    tablea_ = tablea.filter(items=columns)
    tableb_ = tableb.filter(items=columns)

    comparison = pd.merge(tablea_.add_suffix(f" {labela}"), tableb_.add_suffix(f" {labelb}"), left_index=True, right_index=True, how="inner")
    residuals = comparison.filter(like=labela).rename(columns=lambda s: s.replace(f" {labela}",""))-comparison.filter(like=labelb).rename(columns=lambda s: s.replace(f" {labelb}",""))
    residuals = residuals.add_prefix(delta_prefix)

    comparison = pd.concat((comparison,residuals), axis="columns")
    return comparison

def consistency_plot(comparison_table, column, unit, is_logscale, labelx, labely, lims=None, filled_levels=(0.25,0.50,0.75,1.00), dashed_levels=(0.05,), filled_palette=QUARTILE_PALETTE, dashed_color=LIGHT_COLOR, guide_color="w", margins_color=MASTAR_COLOR):

    summary = comparison_table.describe(percentiles=(0.01,0.99))

    if lims is None:
        rangea = summary.loc[["1%","99%"],f"{column} {labelx}"].values
        rangeb = summary.loc[["1%","99%"],f"{column} {labely}"].values
        xrange = np.array([min(*rangea, *rangeb), max(*rangea, *rangeb)])
    else:
        xrange = lims[:2]
    mu_a, sigma_a = summary.loc[["mean","std"],f"{column} {labelx}"].values
    mu_b, sigma_b = summary.loc[["mean","std"],f"{column} {labely}"].values

    g = sns.jointplot(data=comparison_table, x=f"{column} {labelx}", y=f"{column} {labely}", kind="kde", dropna=True, height=7,
                      levels=filled_levels, marginal_kws=dict(alpha=1.0, color=margins_color),
                      joint_kws=dict(colors=filled_palette), fill=True, xlim=xrange, ylim=xrange)

    g.ax_joint.plot(xrange, xrange, "-", lw=1, color=guide_color)
    if is_logscale:
        xscale = np.log10(np.linspace(*(10**xrange)))
        g.ax_joint.plot(xscale, np.log10(10**xscale * 1.1), "--", lw=0.7, color=guide_color)
        g.ax_joint.plot(xscale, np.log10(10**xscale / 1.1), "--", lw=0.7, color=guide_color)
    else:
        g.ax_joint.plot(xrange, xrange+np.abs(0.1*xrange), "--", lw=0.7, color=guide_color)
        g.ax_joint.plot(xrange, xrange-np.abs(0.1*xrange), "--", lw=0.7, color=guide_color)

    g.ax_marg_x.axvline(mu_a, ls="-", lw=0.7, color=guide_color)
    g.ax_marg_x.axvline(mu_a-sigma_a, ls="--", lw=0.7, color=guide_color)
    g.ax_marg_x.axvline(mu_a+sigma_a, ls="--", lw=0.7, color=guide_color)
    g.ax_marg_y.axhline(mu_b, ls="-", lw=0.7, color=guide_color)
    g.ax_marg_y.axhline(mu_b-sigma_b, ls="--", lw=0.7, color=guide_color)
    g.ax_marg_y.axhline(mu_b+sigma_b, ls="--", lw=0.7, color=guide_color)

    g.ax_joint.text(0.95, 0.10, f"{labelx} $\mu,\sigma={mu_a:.2f},{sigma_a:.2f}\,${unit}", ha="right", size="small", transform=g.ax_joint.transAxes)
    g.ax_joint.text(0.95, 0.05, f"{labely} $\mu,\sigma={mu_b:.2f},{sigma_b:.2f}\,${unit}", ha="right", size="small", transform=g.ax_joint.transAxes)

    sns.kdeplot(data=comparison_table, x=f"{column} {labelx}", y=f"{column} {labely}", levels=dashed_levels, color=dashed_color, linestyles="--", linewidths=1.5, ax=g.ax_joint)

    return g

def consistency_plot_hist(comparison_table, column, unit, is_logscale, labelx, labely, lims=None, filled_color=MASTAR_COLOR, dashed_color=LIGHT_COLOR, guide_color="w", margins_color=MASTAR_COLOR):

    summary = comparison_table.describe(percentiles=(0.01,0.99))

    if lims is None:
        rangea = summary.loc[["1%","99%"],f"{column} {labelx}"].values
        rangeb = summary.loc[["1%","99%"],f"{column} {labely}"].values
        xrange = np.array([min(*rangea, *rangeb), max(*rangea, *rangeb)])
    else:
        xrange = lims[:2]
    mu_a, sigma_a = summary.loc[["mean","std"],f"{column} {labelx}"].values
    mu_b, sigma_b = summary.loc[["mean","std"],f"{column} {labely}"].values

    g = sns.jointplot(data=comparison_table, x=f"{column} {labelx}", y=f"{column} {labely}", kind="hist", stat="probability", dropna=True, height=7,
                      marginal_kws=dict(alpha=1.0, color=margins_color),
                      joint_kws=dict(color=filled_color), xlim=xrange, ylim=xrange)

    g.ax_joint.plot(xrange, xrange, "-", lw=1, color=guide_color)
    if is_logscale:
        xscale = np.log10(np.linspace(*(10**xrange)))
        g.ax_joint.plot(xscale, np.log10(10**xscale * 1.1), "--", lw=0.7, color=guide_color)
        g.ax_joint.plot(xscale, np.log10(10**xscale / 1.1), "--", lw=0.7, color=guide_color)
    else:
        g.ax_joint.plot(xrange, xrange+np.abs(0.1*xrange), "--", lw=0.7, color=guide_color)
        g.ax_joint.plot(xrange, xrange-np.abs(0.1*xrange), "--", lw=0.7, color=guide_color)

    g.ax_marg_x.axvline(mu_a, ls="-", lw=0.7, color=guide_color)
    g.ax_marg_x.axvline(mu_a-sigma_a, ls="--", lw=0.7, color=guide_color)
    g.ax_marg_x.axvline(mu_a+sigma_a, ls="--", lw=0.7, color=guide_color)
    g.ax_marg_y.axhline(mu_b, ls="-", lw=0.7, color=guide_color)
    g.ax_marg_y.axhline(mu_b-sigma_b, ls="--", lw=0.7, color=guide_color)
    g.ax_marg_y.axhline(mu_b+sigma_b, ls="--", lw=0.7, color=guide_color)

    g.ax_joint.text(0.95, 0.10, f"{labelx} $\mu,\sigma={mu_a:.2f},{sigma_a:.2f}\,${unit}", ha="right", size="small", transform=g.ax_joint.transAxes)
    g.ax_joint.text(0.95, 0.05, f"{labely} $\mu,\sigma={mu_b:.2f},{sigma_b:.2f}\,${unit}", ha="right", size="small", transform=g.ax_joint.transAxes)

    return g

def delta_plot(comparison_table, column, unit, labelx, delta_prefix=r"$\delta$", limx=None, limy=None, filled_color=MASTAR_COLOR, dashed_color=LIGHT_COLOR, guide_color="w", margins_color=MASTAR_COLOR):
    summary = comparison_table.describe(percentiles=(0.01,0.99))

    if limx is None:
        rangea = summary.loc[["1%","99%"],f"{column} {labelx}"].values
        rangeb = summary.loc[["1%","99%"],f"{delta_prefix}{column}"].values
        xrange = np.array([min(*rangea, *rangeb), max(*rangea, *rangeb)])
    else:
        xrange = limx[:2]
    if limy is None:
        yrange = summary.loc[["1%","99%"],f"{delta_prefix}{column}"].values
    else:
        yrange = limy[:2]
    yrange[1] *= 1.5
    mu, sigma = summary.loc[["mean","std"],f"{delta_prefix}{column}"].values

    g = sns.jointplot(data=comparison_table, x=f"{column} {labelx}", y=f"{delta_prefix}{column}", kind="hist", marginal_kws=dict(alpha=1.0, color=margins_color),
                      joint_kws=dict(color=filled_color), height=7, xlim=xrange, ylim=yrange)

    g.ax_joint.axhline(ls=":", lw=0.7, color=guide_color)
    g.ax_joint.axhline(mu, ls="-", lw=0.7, color=guide_color)
    g.ax_joint.axhline(mu-sigma, ls="--", lw=0.7, color=guide_color)
    g.ax_joint.axhline(mu+sigma, ls="--", lw=0.7, color=guide_color)

    g.ax_marg_y.axhline(ls=":", lw=0.7, color=guide_color)
    g.ax_marg_y.axhline(mu, ls="-", lw=0.7, color=guide_color)
    g.ax_marg_y.axhline(mu-sigma, ls="--", lw=0.7, color=guide_color)
    g.ax_marg_y.axhline(mu+sigma, ls="--", lw=0.7, color=guide_color)

    column_name = column.replace(f"~({unit})", "") if unit else column
    g.ax_joint.text(0.05, 0.05, f"{delta_prefix}{column_name}$\,={mu:.2f}\pm~{sigma:.2f}\,${unit}", ha="left", size="small", transform=g.ax_joint.transAxes)

    return g


def create_RGB(cube, wavelength, R_filt, G_filt, B_filt):
    """
    return RGB image object from a given cube and filters

    Parameters:
    -----------
    cube: 3-array_like
        cube to compute RGB image from
    {R,G,B}_filt: list_like
        a two-element object containing the range in wavelength of each filter

    Returns:
    --------
    RGB_image: image.Image object
        the RGB composite image    
    """

    Rdata = np.mean(cube[(R_filt[0] <= wavelength) & (
        wavelength <= R_filt[1])], axis=0) * 1.2
    Gdata = np.mean(cube[(G_filt[0] <= wavelength) & (
        wavelength <= G_filt[1])], axis=0) * 0.9
    Bdata = np.mean(cube[(B_filt[0] <= wavelength) & (
        wavelength <= B_filt[1])], axis=0) * 1.1

    scale_flux = 0.7  # 2.5
    RGBdata = np.zeros((Bdata.shape[0], Bdata.shape[1], 3), dtype=float)
    RGBdata[:, :, 0] = img_scale.sqrt(
        Rdata*scale_flux, scale_min=0.01, scale_max=2)
    RGBdata[:, :, 1] = img_scale.sqrt(
        Gdata*scale_flux, scale_min=0.01, scale_max=2)
    RGBdata[:, :, 2] = img_scale.sqrt(
        Bdata*scale_flux, scale_min=0.01, scale_max=2)

    RGBdata = RGBdata * 255
    RGBdata_int = RGBdata.astype('uint8')

    RGB_image = Image.fromarray(RGBdata_int)

    bright = ImageEnhance.Brightness(RGB_image)
    RGB_image = bright.enhance(1.2)
    contrast = ImageEnhance.Contrast(RGB_image)
    RGB_image = contrast.enhance(1.5)
    sharpness = ImageEnhance.Sharpness(RGB_image)
    RGB_image = sharpness.enhance(2.0)

    return RGB_image
