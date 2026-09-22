"""Smoothing and monotone-decay statistics for distance-binned expression profiles.

Shared by the ``*_empirical_distance_expression.py`` scripts (Xenium, Atera breast,
MERFISH cortex, CosMx human tonsil). Those scripts bin receiver cells by surface-to-
surface distance to their nearest sender and plot the per-bin mean expression of the
AMICI-implicated genes. With 1-2 um bins the per-bin means are dominated by sampling
noise, so this module provides two things:

1. A weighted local-linear (degree-1 Gaussian-kernel) smoother over the bin means, so
   the figures can show a smooth trend line with an analytic confidence band while the
   raw bin means stay visible in the background.

2. A per-gene test of whether the profile is a genuine monotone decay rather than
   noise. The central statistic is the weighted R^2 of an isotonic (non-increasing)
   fit to the bin means, calibrated against a parametric null in which the bin means
   are pure sampling noise around a flat profile. Isotonic regression fits noise well
   when there are many bins, so the raw R^2 is meaningless on its own; the null is
   what makes it interpretable.

Why not just the cell-level Spearman correlation: with 10k-40k receiver cells per
interaction, a correlation of -0.013 is "significant" at p = 0.009 while describing a
profile that is visually indistinguishable from flat. The statistics here are computed
on the binned profile and are effect-size driven.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm, spearmanr
from sklearn.isotonic import IsotonicRegression

# Bins holding very few cells have enormous SEMs and destabilise both the smoother and
# the isotonic fit; the calling scripts already drop bins below MIN_CELLS_PER_BIN.
MIN_BINS_FOR_STATS = 8
DEFAULT_N_NULL = 2000
DEFAULT_SEED = 0


def _weighted_mean(values, weights):
    return float(np.sum(weights * values) / np.sum(weights))


def local_linear_smooth(bin_centers, values, sems, weights, bandwidth, grid=None, n_grid=200):
    """Weighted local-linear smoother with a Gaussian kernel.

    Local *linear* rather than local constant (Nadaraya-Watson) because the quantity of
    interest lives at the left boundary of the domain: a local-constant smoother is
    biased towards the interior at distance 0 and flattens exactly the near-contact
    enrichment these plots are meant to show.

    Parameters
    ----------
    bin_centers, values, sems, weights
        Per-bin distance, mean expression, standard error of that mean, and a weight
        (the scripts pass the cell count per bin).
    bandwidth
        Gaussian kernel standard deviation, in the same units as ``bin_centers`` (um).
    grid
        Distances to evaluate at; defaults to ``n_grid`` points spanning the bin centres.

    Returns
    -------
    grid, fit, se
        ``se`` is the standard error of the fitted value, propagated analytically from
        the per-bin SEMs. Local-linear weights sum to one, so
        ``Var(fit(g)) = sum_i l_i(g)^2 * sem_i^2`` under independence across bins.
    """
    bin_centers = np.asarray(bin_centers, dtype=float)
    values = np.asarray(values, dtype=float)
    variances = np.asarray(sems, dtype=float) ** 2
    weights = np.asarray(weights, dtype=float)
    if grid is None:
        grid = np.linspace(bin_centers.min(), bin_centers.max(), n_grid)
    grid = np.asarray(grid, dtype=float)

    fit = np.empty(grid.shape)
    se = np.empty(grid.shape)
    for idx, point in enumerate(grid):
        offset = bin_centers - point
        kernel = np.exp(-0.5 * (offset / bandwidth) ** 2) * weights
        s0 = kernel.sum()
        if s0 <= 0:
            fit[idx], se[idx] = np.nan, np.nan
            continue
        s1 = np.sum(kernel * offset)
        s2 = np.sum(kernel * offset**2)
        denominator = s0 * s2 - s1**2
        # Degenerates to a local average when the local design is rank-deficient (too
        # few bins inside the kernel to identify a slope).
        equivalent_kernel = kernel / s0 if denominator <= 0 else kernel * (s2 - s1 * offset) / denominator
        fit[idx] = float(equivalent_kernel @ values)
        se[idx] = float(np.sqrt(np.sum(equivalent_kernel**2 * variances)))
    return grid, fit, se


def _isotonic_r2(bin_centers, values, weights, grand_mean=None):
    """Weighted R^2 of the best non-increasing step function through the bin means."""
    fitted = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(
        bin_centers, values, sample_weight=weights
    )
    center = _weighted_mean(values, weights) if grand_mean is None else grand_mean
    rss = float(np.sum(weights * (values - fitted) ** 2))
    tss = float(np.sum(weights * (values - center) ** 2))
    if tss <= 0:
        return 0.0, fitted
    return 1.0 - rss / tss, fitted


def _weighted_linear_trend(bin_centers, values, sems):
    """Inverse-variance weighted least-squares slope of the binned profile.

    The standard error is inflated by sqrt(chi2/df) whenever the residual scatter
    exceeds the per-bin SEMs, so that a profile with real but non-linear structure is
    not credited with a spuriously precise slope.
    """
    weights = 1.0 / np.asarray(sems, dtype=float) ** 2
    x_bar = _weighted_mean(bin_centers, weights)
    y_bar = _weighted_mean(values, weights)
    dx = bin_centers - x_bar
    sxx = float(np.sum(weights * dx**2))
    if sxx <= 0:
        return np.nan, np.nan, np.nan
    slope = float(np.sum(weights * dx * (values - y_bar)) / sxx)
    residuals = values - (y_bar + slope * dx)
    degrees_of_freedom = max(len(values) - 2, 1)
    overdispersion = float(np.sum(weights * residuals**2) / degrees_of_freedom)
    slope_se = float(np.sqrt(max(overdispersion, 1.0) / sxx))
    return slope, slope_se, overdispersion


def _exponential_decay_length(bin_centers, values, sems, max_distance):
    """Length scale of ``c + a * exp(-d / lambda)`` fitted to the binned profile.

    Returned for comparison with the length scales AMICI learns. It is only identifiable
    when the profile actually curves inside the fitted window: a non-positive amplitude,
    a length scale that rails against the upper bound (the profile is effectively linear
    over the window) or a standard error larger than the estimate all yield NaN rather
    than a number that would be over-read.
    """

    def model(distance, baseline, amplitude, length_scale):
        return baseline + amplitude * np.exp(-distance / length_scale)

    span = float(values.max() - values.min())
    try:
        popt, pcov = curve_fit(
            model,
            bin_centers,
            values,
            p0=[float(values.min()), max(span, 1e-6), max(max_distance / 5.0, 1.0)],
            sigma=sems,
            absolute_sigma=False,
            bounds=([-np.inf, 0.0, 0.5], [np.inf, np.inf, max_distance * 2.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return np.nan, np.nan
    length_scale = float(popt[2])
    length_scale_se = float(np.sqrt(np.diag(pcov))[2])
    upper_bound = max_distance * 2.0
    if popt[1] <= 1e-8 or length_scale >= 0.95 * upper_bound or not np.isfinite(length_scale_se):
        return np.nan, np.nan
    if length_scale_se > length_scale:
        return np.nan, np.nan
    return length_scale, length_scale_se


def monotone_decay_stats(
    bin_centers,
    values,
    sems,
    n_cells,
    max_distance,
    bandwidth,
    n_null=DEFAULT_N_NULL,
    seed=DEFAULT_SEED,
):
    """Quantify how much of a binned profile is monotone decay rather than noise.

    ``values`` and ``sems`` are expected in z-scored units so that amplitudes and slopes
    are comparable across genes with different expression levels.

    Returns a dict with, among others:

    ``mono_score``
        Isotonic R^2 rescaled so that 0 is the value expected from pure sampling noise
        and 1 is a perfectly clean monotone decay. This is the headline "is it monotone
        or is it noise" number.
    ``mono_pvalue``
        Parametric-bootstrap p-value for the isotonic R^2 against a flat profile with
        the observed per-bin sampling noise.
    ``slope_z_per_10um``
        Inverse-variance weighted linear slope, in z-score units per 10 um, with a
        standard error inflated for residual overdispersion.
    ``amplitude_z``
        Total drop of the smoothed curve from the first to the last distance bin, in z
        units, with ``amplitude_se``. This is the effect size to quote alongside the
        p-values, and it is read off the same smoother the figure draws so that the
        number and the picture agree. ``isotonic_amplitude_z`` is the equivalent from
        the isotonic fit, which is noisier because it is anchored on the two endpoint
        bins rather than on a local average.
    ``snr``
        Standard deviation of the distance-dependent signal divided by the standard
        deviation of the per-bin sampling noise, after subtracting the noise variance
        from the observed variance of the bin means.
    """
    bin_centers = np.asarray(bin_centers, dtype=float)
    values = np.asarray(values, dtype=float)
    sems = np.asarray(sems, dtype=float)
    n_cells = np.asarray(n_cells, dtype=float)

    order = np.argsort(bin_centers)
    bin_centers, values, sems, n_cells = bin_centers[order], values[order], sems[order], n_cells[order]

    # A bin in which every cell carries the identical value has a SEM of exactly zero,
    # which happens for sparse genes and would give that bin infinite inverse-variance
    # weight. Such bins carry no information about the noise level, so they are dropped
    # rather than allowed to dominate the isotonic fit and its null.
    n_bins_total = int(len(bin_centers))
    usable = np.isfinite(sems) & (sems > 0) & np.isfinite(values)
    bin_centers, values, sems, n_cells = bin_centers[usable], values[usable], sems[usable], n_cells[usable]

    empty = {
        "n_bins": n_bins_total,
        "n_bins_used": int(len(bin_centers)),
        "mono_score": np.nan,
        "mono_pvalue": np.nan,
        "isotonic_r2": np.nan,
        "isotonic_r2_null_median": np.nan,
        "slope_z_per_10um": np.nan,
        "slope_se": np.nan,
        "slope_pvalue": np.nan,
        "slope_overdispersion": np.nan,
        "amplitude_z": np.nan,
        "amplitude_se": np.nan,
        "isotonic_amplitude_z": np.nan,
        "spearman_bins_rho": np.nan,
        "spearman_bins_pvalue": np.nan,
        "signal_sd_z": np.nan,
        "noise_sd_z": np.nan,
        "snr": np.nan,
        "decay_length_um": np.nan,
        "decay_length_se_um": np.nan,
    }
    if len(bin_centers) < MIN_BINS_FOR_STATS:
        return empty

    # Isotonic weights are inverse-variance so that the fit, and the null it is compared
    # against, are driven by the same noise model.
    iso_weights = 1.0 / sems**2
    grand_mean = _weighted_mean(values, iso_weights)
    isotonic_r2, isotonic_fit = _isotonic_r2(bin_centers, values, iso_weights, grand_mean=grand_mean)

    # Null: the profile is flat and the bin means scatter only by their own sampling
    # error. This holds the number of bins and the per-bin precision fixed, which is
    # what makes the isotonic R^2 comparable across genes and interactions.
    rng = np.random.default_rng(seed)
    null_draws = grand_mean + rng.normal(size=(n_null, len(values))) * sems
    null_r2 = np.empty(n_null)
    for idx in range(n_null):
        null_r2[idx] = _isotonic_r2(bin_centers, null_draws[idx], iso_weights)[0]
    null_median = float(np.median(null_r2))
    mono_pvalue = float((1.0 + np.sum(null_r2 >= isotonic_r2)) / (n_null + 1.0))
    mono_score = float(np.clip((isotonic_r2 - null_median) / max(1.0 - null_median, 1e-9), 0.0, 1.0))

    slope, slope_se, overdispersion = _weighted_linear_trend(bin_centers, values, sems)
    slope_pvalue = float(2.0 * norm.sf(abs(slope / slope_se))) if np.isfinite(slope_se) and slope_se > 0 else np.nan

    rho, rho_pvalue = spearmanr(bin_centers, values)

    # Noise-corrected decomposition of the bin-to-bin variation, weighted by cell count.
    observed_var = float(np.sum(n_cells * (values - _weighted_mean(values, n_cells)) ** 2) / np.sum(n_cells))
    noise_var = float(np.sum(n_cells * sems**2) / np.sum(n_cells))
    signal_sd = float(np.sqrt(max(observed_var - noise_var, 0.0)))
    noise_sd = float(np.sqrt(noise_var))

    # The quoted effect size is the drop of the same smoother the figure draws, rather
    # than the isotonic fit's endpoint bins, which are single noisy bin means.
    _, endpoint_fit, endpoint_se = local_linear_smooth(
        bin_centers,
        values,
        sems,
        n_cells,
        bandwidth=bandwidth,
        grid=np.array([bin_centers[0], bin_centers[-1]]),
    )
    amplitude = float(endpoint_fit[0] - endpoint_fit[-1])
    amplitude_se = float(np.sqrt(endpoint_se[0] ** 2 + endpoint_se[-1] ** 2))

    decay_length, decay_length_se = _exponential_decay_length(bin_centers, values, sems, max_distance)

    return {
        "n_bins": n_bins_total,
        "n_bins_used": int(len(bin_centers)),
        "mono_score": mono_score,
        "mono_pvalue": mono_pvalue,
        "isotonic_r2": float(isotonic_r2),
        "isotonic_r2_null_median": null_median,
        "slope_z_per_10um": slope * 10.0,
        "slope_se": slope_se * 10.0,
        "slope_pvalue": slope_pvalue,
        "slope_overdispersion": overdispersion,
        "amplitude_z": amplitude,
        "amplitude_se": amplitude_se,
        "isotonic_amplitude_z": float(isotonic_fit[0] - isotonic_fit[-1]),
        "spearman_bins_rho": float(rho),
        "spearman_bins_pvalue": float(rho_pvalue),
        "signal_sd_z": signal_sd,
        "noise_sd_z": noise_sd,
        "snr": float(signal_sd / noise_sd) if noise_sd > 0 else np.nan,
        "decay_length_um": decay_length,
        "decay_length_se_um": decay_length_se,
    }


def benjamini_hochberg(pvalues):
    """BH-adjusted p-values, NaN-safe."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return adjusted
    values = pvalues[finite]
    n = len(values)
    order = np.argsort(values)
    ranked = values[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    adjusted[finite] = out
    return adjusted


def add_monotone_calls(stats, alpha=0.05, min_amplitude_z=0.05):
    """Add BH-adjusted q-values and the monotone-decay call to a per-gene stats frame.

    A gene passes the monotonic decay test when the Spearman correlation between a
    receiver cell's distance to its nearest sender and its expression of that gene is
    significantly negative after BH correction across every gene x interaction in the
    dataset.

    The correlation is computed per cell, so it needs no binning and every gene x pair is
    testable regardless of how few receivers fall inside the distance window. The binned
    profile is still used for the figures, and the bin-level correlation, isotonic R^2 and
    its bootstrap calibration are recorded alongside for reference.
    """
    stats = stats.copy()
    # A gene with too few usable distance bins cannot be tested at all; that is a
    # different outcome from being tested and found flat, and the two are kept apart so
    # the headline fraction is not diluted by data-starved panels.
    stats["testable"] = np.isfinite(stats["spearman_pvalue"].to_numpy())
    stats["spearman_qvalue"] = benjamini_hochberg(stats["spearman_pvalue"].to_numpy())
    stats["spearman_bins_qvalue"] = benjamini_hochberg(stats["spearman_bins_pvalue"].to_numpy())
    stats["mono_qvalue"] = benjamini_hochberg(stats["mono_pvalue"].to_numpy())
    stats["slope_qvalue"] = benjamini_hochberg(stats["slope_pvalue"].to_numpy())
    stats["monotone_decay"] = (stats["spearman_qvalue"] < alpha) & (stats["spearman_rho"] < 0)
    # Genes that pass but whose total drop is small are worth being able to identify, even
    # though the effect size no longer gates the call.
    stats["shallow"] = stats["monotone_decay"] & (stats["amplitude_z"] < min_amplitude_z)
    return stats


def smooth_summary(summary, value_col, sem_col, bandwidth, max_distance, n_grid=200):
    """Smoothed curve per gene for a binned-profile summary frame.

    Returns a long frame with columns ``gene``, ``distance``, ``fit``, ``se``.
    """
    records = []
    grid = np.linspace(0, max_distance, n_grid)
    for gene, gene_df in summary.groupby("gene", observed=True, sort=False):
        gene_df = gene_df.sort_values("bin_center")
        # Below this many bins the smoother has nothing to average over and would draw a
        # confident-looking curve through three points; the raw means are shown instead.
        if len(gene_df) < MIN_BINS_FOR_STATS:
            continue
        inside = (grid >= gene_df["bin_center"].min()) & (grid <= gene_df["bin_center"].max())
        _, fit, se = local_linear_smooth(
            gene_df["bin_center"].to_numpy(),
            gene_df[value_col].to_numpy(),
            gene_df[sem_col].to_numpy(),
            gene_df["n_cells"].to_numpy(),
            bandwidth=bandwidth,
            grid=grid[inside],
        )
        records.append(pd.DataFrame({"gene": gene, "distance": grid[inside], "fit": fit, "se": se}))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["gene", "distance", "fit", "se"])


def _gene_colors(genes):
    cmap = plt.get_cmap("tab10" if len(genes) <= 10 else "tab20")
    return dict(zip(genes, cmap(np.linspace(0, 1, len(genes), endpoint=False)), strict=False))


def plot_profiles(
    summaries,
    output_dir,
    output_prefix,
    suptitle,
    bandwidth,
    max_distance,
    value_col="mean_expression",
    sem_col="sem_expression",
    ylabel="Mean log1p-normalized expression",
    suffix="",
    zero_line=False,
    stats=None,
    xlabel="Surface-to-surface distance to nearest sender (µm)",
):
    """One panel per interaction: faint raw bin means behind a smoothed trend per gene.

    The raw per-bin means are kept in the background at low opacity so the figure still
    shows the data it is summarising; the bold line is the local-linear smoother and the
    band is its 95% confidence interval.
    """
    n_panels = len(summaries)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 4.6 * n_rows), squeeze=False)

    for panel_idx, (label, summary) in enumerate(summaries):
        ax = axes[panel_idx // n_cols][panel_idx % n_cols]
        genes = list(summary["gene"].drop_duplicates())
        colors = _gene_colors(genes)
        smoothed = smooth_summary(summary, value_col, sem_col, bandwidth=bandwidth, max_distance=max_distance)

        for gene in genes:
            color = colors[gene]
            gene_df = summary[summary["gene"] == gene].sort_values("bin_center")
            curve = smoothed[smoothed["gene"] == gene]
            if curve.empty:
                # Not enough bins to smooth: show the raw means at full weight, with
                # their own error band, rather than implying a trend that was not fitted.
                ax.plot(
                    gene_df["bin_center"],
                    gene_df[value_col],
                    marker="o",
                    ms=3.5,
                    lw=1.6,
                    color=color,
                    label=gene,
                    zorder=3,
                )
                ax.fill_between(
                    gene_df["bin_center"],
                    gene_df[value_col] - 1.96 * gene_df[sem_col],
                    gene_df[value_col] + 1.96 * gene_df[sem_col],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=2,
                )
                continue
            ax.plot(
                gene_df["bin_center"],
                gene_df[value_col],
                marker="o",
                ms=1.8,
                mew=0,
                lw=0.7,
                color=color,
                alpha=0.28,
                zorder=1,
            )
            ax.fill_between(
                curve["distance"],
                curve["fit"] - 1.96 * curve["se"],
                curve["fit"] + 1.96 * curve["se"],
                color=color,
                alpha=0.20,
                linewidth=0,
                zorder=2,
            )
            ax.plot(curve["distance"], curve["fit"], lw=2.2, color=color, label=gene, zorder=3, solid_capstyle="round")

        if zero_line:
            ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5, zorder=0)

        title = label
        if stats is not None:
            panel_stats = stats[stats["interaction"] == label]
            if len(panel_stats):
                testable = panel_stats[panel_stats["testable"]]
                if len(testable) == 0:
                    title = f"{label}\ntoo few cells per distance bin to test monotonicity"
                else:
                    n_monotone = int(testable["monotone_decay"].sum())
                    title = f"{label}\n{n_monotone}/{len(testable)} genes pass the monotonic decay test (BH q < 0.05)"
                    if len(testable) < len(panel_stats):
                        title += f"; {len(panel_stats) - len(testable)} not testable"
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max_distance)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.grid(alpha=0.25)

    for empty_idx in range(n_panels, n_rows * n_cols):
        axes[empty_idx // n_cols][empty_idx % n_cols].axis("off")

    fig.suptitle(suptitle, y=1.0, fontsize=13)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(output_dir, f"{output_prefix}{suffix}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def significance_stars(qvalue):
    """Conventional significance marking from a BH-adjusted p-value."""
    if not np.isfinite(qvalue):
        return ""
    if qvalue < 0.001:
        return "***"
    if qvalue < 0.01:
        return "**"
    if qvalue < 0.05:
        return "*"
    return "n.s."


def _draw_monotonicity_axis(ax, stats, palette, alpha=0.05, label_fontsize=7):
    """Horizontal bars of the bin-level Spearman correlation, one row per gene.

    The bar is the Spearman correlation between bin centre and bin mean expression, so a
    monotone decay sits at the negative end. Stars give the BH-adjusted significance, so
    the figure shows the correlation and the evidence for it side by side.
    """
    usable = stats[stats["testable"]].sort_values(["interaction", "spearman_rho"]).reset_index(drop=True)
    if usable.empty:
        ax.axis("off")
        return usable
    positions = np.arange(len(usable))[::-1]
    for pos, (_, row) in zip(positions, usable.iterrows(), strict=False):
        color = palette[row["interaction"]]
        called = bool(row["monotone_decay"])
        rho = row["spearman_rho"]
        ax.barh(pos, rho, height=0.72, color=color, alpha=0.92 if called else 0.30, edgecolor="none")
        ax.text(
            rho - 0.03 if rho < 0 else rho + 0.03,
            pos,
            significance_stars(row["spearman_qvalue"]),
            va="center",
            ha="right" if rho < 0 else "left",
            fontsize=label_fontsize - 0.5,
            color="black" if called else "0.45",
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(usable["gene"], fontsize=label_fontsize)
    ax.set_ylim(-0.8, len(usable) - 0.2)
    span = float(usable["spearman_rho"].abs().max())
    ax.set_xlim(-1.18 * span, max(0.35 * span, float(usable["spearman_rho"].max()) + 0.12 * span))
    ax.axvline(0, color="black", lw=0.9, alpha=0.6)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    return usable


def plot_monotonicity_summary(stats, output_dir, output_prefix, suptitle, alpha=0.05):
    """Per-gene monotone R^2 with bootstrap significance, for one dataset."""
    interactions = list(dict.fromkeys(stats[stats["testable"]]["interaction"]))
    if not interactions:
        return
    palette = _gene_colors(interactions)
    n_rows = int(stats["testable"].sum())
    fig, ax = plt.subplots(figsize=(7.6, max(3.4, 0.26 * n_rows + 1.5)))
    usable = _draw_monotonicity_axis(ax, stats, palette, alpha=alpha)
    ax.set_xlabel("Spearman correlation of expression with distance")

    handles = [plt.Line2D([], [], color=palette[key], lw=6, label=key) for key in interactions]
    handles.append(plt.Line2D([], [], color="none", label="*** q<0.001  ** q<0.01  * q<0.05"))
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    n_called = int(usable["monotone_decay"].sum())
    subtitle = f"{n_called}/{len(usable)} testable gene-interaction pairs pass the monotonic decay test"
    n_untestable = len(stats) - len(usable)
    if n_untestable:
        subtitle += f" ({n_untestable} excluded: fewer than {MIN_BINS_FOR_STATS} usable distance bins)"
    fig.suptitle(f"{suptitle}\n{subtitle}", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(output_dir, f"{output_prefix}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_monotonicity(datasets, output_dir, output_prefix, suptitle, alpha=0.05):
    """One figure over several datasets: per-gene monotone R^2 with significance.

    ``datasets`` is a sequence of ``(dataset_name, stats_frame)`` pairs, each frame as
    returned by :func:`add_monotone_calls`. Panels are sized in proportion to the number
    of testable genes so that the rows are the same height across datasets.
    """
    datasets = [(name, frame) for name, frame in datasets if frame["testable"].any()]
    if not datasets:
        return
    n_panels = len(datasets)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    row_counts = [int(frame["testable"].sum()) for _, frame in datasets]
    panel_height = max(row_counts[i * n_cols : (i + 1) * n_cols] + [1] for i in range(n_rows))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15.5, sum(max(3.0, 0.24 * h + 2.0) for h in panel_height)),
        squeeze=False,
        gridspec_kw={"height_ratios": [max(3.0, 0.24 * h + 2.0) for h in panel_height]},
    )

    for panel_idx, (name, frame) in enumerate(datasets):
        ax = axes[panel_idx // n_cols][panel_idx % n_cols]
        interactions = list(dict.fromkeys(frame[frame["testable"]]["interaction"]))
        palette = _gene_colors(interactions)
        usable = _draw_monotonicity_axis(ax, frame, palette, alpha=alpha, label_fontsize=6.5)
        n_called = int(usable["monotone_decay"].sum())
        n_untestable = len(frame) - len(usable)
        title = f"{name}\n{n_called}/{len(usable)} pass the monotonic decay test"
        if n_untestable:
            title += f" ({n_untestable} not testable)"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Spearman $\\rho$", fontsize=9)
        handles = [plt.Line2D([], [], color=palette[key], lw=5, label=key) for key in interactions]
        ax.legend(handles=handles, fontsize=6.0, frameon=False, loc="lower right")

    for empty_idx in range(n_panels, n_rows * n_cols):
        axes[empty_idx // n_cols][empty_idx % n_cols].axis("off")

    fig.suptitle(
        f"{suptitle}\nBars are the per-cell Spearman correlation between distance and expression; "
        f"*** q<0.001, ** q<0.01, * q<0.05",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{output_prefix}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
