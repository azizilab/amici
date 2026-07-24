# %% Import libraries
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SENSITIVITY_DIR = os.path.dirname(SCRIPT_DIR)
BENCHMARK_DIR = os.path.abspath(os.path.join(SENSITIVITY_DIR, ".."))
sys.path.insert(0, SENSITIVITY_DIR)

from current_amici_sweep import (  # noqa: E402
    REALISTIC_CONFIG,
    SYNTHETIC_CONFIG,
    generate_or_load_realistic_dataset,
    generate_or_load_synthetic_dataset,
)

from amici import AMICI  # noqa: E402
from amici._alt_module import AMICIUnconstrainedAttentionModule  # noqa: E402
from amici._unimodal_module import AMICIUnimodalAttentionModule  # noqa: E402


class UnconstrainedAttentionAMICI(AMICI):
    """AMICI wrapper for cached unconstrained positional-attention models."""

    _module_cls = AMICIUnconstrainedAttentionModule


class UnimodalAttentionAMICI(AMICI):
    """AMICI wrapper for cached unimodal distance-bias models."""

    _module_cls = AMICIUnimodalAttentionModule


# %% Paths
DATA_DIR = os.path.join(SENSITIVITY_DIR, "data")
FIGURES_DIR = os.path.join(SENSITIVITY_DIR, "figures", "amici_variants")
RESULTS_DIR = os.path.join(SENSITIVITY_DIR, "results", "amici_variants")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

VARIANT_SCORE_PATHS = {
    "Current AMICI": os.path.join(RESULTS_DIR, "current_amici", "current_amici_task_scores.csv"),
    "Unconstrained": os.path.join(
        SENSITIVITY_DIR,
        "results",
        "unconstrained_attention_sweep",
        "unconstrained_attention_task_scores.csv",
    ),
    "Unimodal": os.path.join(
        SENSITIVITY_DIR,
        "results",
        "unimodal_attention_sweep",
        "unimodal_attention_task_scores.csv",
    ),
}

LEGACY_VARIANT_SCORE_PATHS = {
    "Unconstrained": os.path.join(
        SENSITIVITY_DIR,
        "results",
        "unconstrained_attention_analysis",
        "unconstrained_attention_task_scores.csv",
    ),
    "Unimodal": os.path.join(
        SENSITIVITY_DIR,
        "results",
        "unimodal_attention_analysis",
        "unimodal_attention_task_scores.csv",
    ),
}


# %% AUPRC variant comparison
def load_variant_scores():
    """Load cached task scores for available AMICI variants."""
    frames = []
    for variant, path in VARIANT_SCORE_PATHS.items():
        if not os.path.exists(path) and variant in LEGACY_VARIANT_SCORE_PATHS:
            path = LEGACY_VARIANT_SCORE_PATHS[variant]
        if not os.path.exists(path):
            print(f"Missing {variant} task scores: {path}")
            continue
        df = pd.read_csv(path)
        df["variant"] = variant
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_task_boxplots(plot_df):
    """Plot AUPRC boxplots for current, unconstrained, and unimodal AMICI."""
    if plot_df.empty:
        print("No task scores available; skipping AUPRC plot.")
        return

    task_order = ["Gene", "Neighbor interaction", "Receiver subtype"]
    dataset_types = ["synthetic", "realistic_semisynthetic"]
    dataset_titles = {"synthetic": "Synthetic", "realistic_semisynthetic": "Realistic semi-synthetic"}
    variant_order = ["Current AMICI", "Unconstrained", "Unimodal"]
    variants = [variant for variant in variant_order if variant in set(plot_df["variant"])]
    colors = {"Current AMICI": "#009E73", "Unconstrained": "#6B7280", "Unimodal": "#2E5BBA"}
    if not variants:
        print("No variants with task scores available; skipping AUPRC plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    rng = np.random.default_rng(0)
    width = min(0.24, 0.75 / max(len(variants), 1))
    center = (len(variants) - 1) / 2
    offsets = {variant: (idx - center) * width for idx, variant in enumerate(variants)}

    for ax, dataset_type in zip(axes, dataset_types, strict=False):
        subset = plot_df[plot_df["dataset_type"] == dataset_type]
        for variant in variants:
            positions = np.arange(len(task_order)) + 1 + offsets[variant]
            data = [
                subset.loc[(subset["variant"] == variant) & (subset["task"] == task), "auprc"].dropna().to_numpy()
                for task in task_order
            ]
            bp = ax.boxplot(data, positions=positions, widths=width * 0.85, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[variant])
                patch.set_alpha(0.55)
            for part in ("medians", "whiskers", "caps"):
                for line in bp[part]:
                    line.set_color(colors[variant])
            for x_pos, values in zip(positions, data, strict=False):
                jitter = rng.uniform(-0.035, 0.035, size=len(values))
                ax.scatter(x_pos + jitter, values, color="black", s=13, alpha=0.5, zorder=3)

        handles = [plt.Line2D([0], [0], color=colors[v], lw=8, alpha=0.55, label=v) for v in variants]
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")
        ax.set_title(dataset_titles[dataset_type])
        ax.set_xticks(np.arange(len(task_order)) + 1)
        ax.set_xticklabels(task_order, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUPRC")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("AMICI variant task performance across matched dataset seeds", fontsize=13)
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGURES_DIR, f"amici_variant_task_auprc_boxplots.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# %% Unimodal peak-distance extraction
UNIMODAL_ATTENTION_THRESHOLD = 0.1
UNIMODAL_EFFECTIVE_LOGIT_DROP = -np.log(UNIMODAL_ATTENTION_THRESHOLD)


def label_mapping(model):
    """Return scvi label-category mapping for a loaded model."""
    return list(model.adata_manager.registry["field_registries"]["labels"]["state_registry"]["categorical_mapping"])


def inverse_huber_loss(loss_value, delta):
    """Return |x| where Huber(x, 0, delta) equals loss_value."""
    loss_value = np.asarray(loss_value, dtype=float)
    delta = float(delta)
    quadratic_cutoff = 0.5 * delta**2
    return np.where(
        loss_value <= quadratic_cutoff,
        np.sqrt(2.0 * loss_value),
        (loss_value / delta) + (0.5 * delta),
    )


def get_unimodal_peak_records(dataset_config):
    """Extract preferred-radius and effective upper-range parameters."""
    dataset_type = dataset_config["dataset_type"]
    model_root = os.path.join(SENSITIVITY_DIR, "saved_models", "unimodal_attention_sweep", dataset_type)
    if not os.path.isdir(model_root):
        model_root = os.path.join(SENSITIVITY_DIR, "saved_models", "unimodal_attention_analysis", dataset_type)
    records = []

    for dataset_seed in dataset_config["seeds"]:
        if dataset_type == "synthetic":
            adata, _ = generate_or_load_synthetic_dataset(dataset_config, DATA_DIR, dataset_seed)
        else:
            adata, _ = generate_or_load_realistic_dataset(dataset_config, DATA_DIR, dataset_seed)

        model_dir = os.path.join(model_root, f"dataset_seed_{dataset_seed}")
        best_model_path = os.path.join(model_dir, "best_model")
        best_params_path = os.path.join(model_dir, "best_model_params.json")
        if not os.path.exists(os.path.join(best_model_path, "model.pt")) or not os.path.exists(best_params_path):
            print(f"Missing unimodal best model for {dataset_type} seed {dataset_seed}")
            continue

        with open(best_params_path) as f:
            best_params = json.load(f)
        UnimodalAttentionAMICI.setup_anndata(
            adata,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(best_params["n_neighbors"]),
        )
        model = UnimodalAttentionAMICI.load(best_model_path, adata=adata)
        labels = label_mapping(model)
        preferred_radius = F.softplus(model.module.preferred_radius_raw).detach().cpu().numpy()
        radius_alpha = F.softplus(model.module.radius_alpha_raw).detach().cpu().numpy()
        unit_scale = float(model.module.distance_kernel_unit_scale)
        preferred_radius = preferred_radius * unit_scale
        effective_half_width = (
            inverse_huber_loss(
                UNIMODAL_EFFECTIVE_LOGIT_DROP / np.maximum(radius_alpha, 1e-12),
                float(model.module.unimodal_huber_delta),
            )
            * unit_scale
        )

        for interaction_name, interaction_config in dataset_config["gt_interactions"].items():
            receiver = interaction_config["receiver"]
            if receiver not in labels:
                print(f"Receiver {receiver} missing from label mapping for {dataset_type} seed {dataset_seed}")
                continue
            label_idx = labels.index(receiver)
            for head_idx, peak_distance in enumerate(preferred_radius[label_idx]):
                half_width = float(effective_half_width[label_idx, head_idx])
                records.append(
                    {
                        "variant": "Unimodal",
                        "dataset_type": dataset_type,
                        "dataset_seed": dataset_seed,
                        "interaction": interaction_name,
                        "interaction_label": f"{interaction_config['sender']} -> {receiver}",
                        "receiver_type": receiver,
                        "head_idx": head_idx,
                        "peak_interaction_distance": float(peak_distance),
                        "effective_lower_distance": max(0.0, float(peak_distance) - half_width),
                        "effective_upper_distance": float(peak_distance) + half_width,
                        "effective_width": 2.0 * half_width,
                        "radius_alpha": float(radius_alpha[label_idx, head_idx]),
                        "effective_logit_drop": UNIMODAL_EFFECTIVE_LOGIT_DROP,
                        "gt_length_scale": float(interaction_config["length_scale"]),
                        "best_train_seed": best_params.get("train_seed"),
                        "best_test_loss": best_params.get("test_loss"),
                    }
                )
    return records


def plot_unimodal_distance_summary(peak_df, value_col, x_label, title, file_prefix, color):
    """Plot a learned unimodal distance summary against ground truth."""
    if peak_df.empty:
        print(f"No unimodal distance summaries available; skipping {file_prefix}.")
        return

    dataset_types = ["synthetic", "realistic_semisynthetic"]
    dataset_titles = {"synthetic": "Synthetic", "realistic_semisynthetic": "Realistic semi-synthetic"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)

    for ax, dataset_type in zip(axes[0], dataset_types, strict=False):
        subset = peak_df[peak_df["dataset_type"] == dataset_type]
        interactions = subset[["interaction", "interaction_label", "gt_length_scale"]].drop_duplicates()
        interactions = interactions.sort_values("gt_length_scale")
        positions = np.arange(1, len(interactions) + 1)
        data = [
            subset.loc[subset["interaction"] == interaction, value_col].dropna().to_numpy()
            for interaction in interactions["interaction"]
        ]
        if not data:
            ax.set_title(dataset_titles[dataset_type])
            continue
        violins = ax.violinplot(data, positions=positions, vert=False, widths=0.8, showmeans=True, showextrema=True)
        for body in violins["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)
        for part in ("cmeans", "cmins", "cmaxes", "cbars"):
            violins[part].set_color(color)
            violins[part].set_linewidth(1.3)

        ax.scatter(
            interactions["gt_length_scale"].astype(float),
            positions,
            marker="x",
            color="black",
            s=130,
            linewidths=2.5,
            zorder=4,
            label="Ground truth",
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(interactions["interaction_label"])
        ax.set_xlabel(x_label)
        ax.set_title(dataset_titles[dataset_type])
        ax.grid(axis="x", alpha=0.25)
        ax.legend(frameon=False, loc="lower right")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{file_prefix}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_unimodal_peak_distances(peak_df):
    """Plot learned unimodal peak interaction distances against ground truth."""
    plot_unimodal_distance_summary(
        peak_df,
        value_col="peak_interaction_distance",
        x_label="Learned peak interaction distance",
        title="Unimodal AMICI learned peak distances across dataset seeds and heads",
        file_prefix="unimodal_peak_interaction_distance_distributions",
        color="#2E5BBA",
    )


def plot_unimodal_upper_ranges(peak_df):
    """Plot learned unimodal effective upper interaction ranges against ground truth."""
    plot_unimodal_distance_summary(
        peak_df,
        value_col="effective_upper_distance",
        x_label=f"Effective upper interaction range (relative weight = {UNIMODAL_ATTENTION_THRESHOLD:g})",
        title="Unimodal AMICI learned upper interaction ranges across dataset seeds and heads",
        file_prefix="unimodal_upper_interaction_range_distributions",
        color="#D97706",
    )


# %% Run plots
task_df = load_variant_scores()
if not task_df.empty:
    task_df.to_csv(os.path.join(RESULTS_DIR, "amici_variant_task_scores.csv"), index=False)
plot_task_boxplots(task_df)

peak_records = []
for config in (SYNTHETIC_CONFIG, REALISTIC_CONFIG):
    peak_records.extend(get_unimodal_peak_records(config))
peak_df = pd.DataFrame(peak_records)
peak_df.to_csv(os.path.join(RESULTS_DIR, "unimodal_peak_interaction_distances.csv"), index=False)
plot_unimodal_peak_distances(peak_df)
plot_unimodal_upper_ranges(peak_df)
