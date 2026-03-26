# %% Imports and constants
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import torch
from scipy import stats

from amici import AMICI
from amici.interpretation import AMICIAblationModule

seed = 18
pl.seed_everything(seed)

N_MC_ITERATIONS = 50
MC_CELL_TYPE = "LAMP3+_DCs"  # Small cell type (466 cells) for tractable runtime
MIN_GENE_VARIANCE = 0.1  # Exclude near-zero-variance genes from z-value collection

labels_key = "celltype_train_grouped"
data_date = "2025-05-01"
model_date = "2025-05-02"

fig_dir = "figures/null_z_distribution"
cache_path = f"{fig_dir}/null_z_values.npz"
log_path = f"{fig_dir}/mc_progress.log"
os.makedirs(fig_dir, exist_ok=True)


def log(msg):
    """Write to log file and stdout with immediate flush."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


# %% Monte Carlo simulation (skip if cache exists)
if os.path.exists(cache_path):
    log(f"Loading cached z-values from {cache_path}")
    cached = np.load(cache_path)
    all_z = cached["all_z"]
    N_MC_ITERATIONS = int(cached["n_mc"])
    n_ct_cells = int(cached["n_ct_cells"]) if "n_ct_cells" in cached else 466
    log(f"Loaded {len(all_z):,} z-values from {N_MC_ITERATIONS} iterations")
else:
    # Clear log
    open(log_path, "w").close()

    # Load data and model
    log("Loading adata...")
    adata = sc.read_h5ad(f"./data/xenium_sample1/xenium_sample1_filtered_{data_date}.h5ad")
    AMICI.setup_anndata(
        adata,
        labels_key=labels_key,
        coord_obsm_key="spatial",
        n_neighbors=50,
    )
    log("adata loaded and set up")

    saved_models_dir = f"saved_models/xenium_sample1_proseg_sweep_{data_date}_model_{model_date}"
    wandb_run_id = "te7pkv3z"
    wandb_sweep_id = "g3mucw4s"
    model_path = os.path.join(
        saved_models_dir,
        f"xenium_{seed}_sweep_{wandb_sweep_id}_{wandb_run_id}_params_{model_date}",
    )

    log("Loading model...")
    model = AMICI.load(model_path, adata=adata)
    model.module.eval()
    log("Model loaded")

    n_ct_cells = (adata.obs[labels_key] == MC_CELL_TYPE).sum()
    n_ct_types = len(adata.obs[labels_key].unique())
    batches_per_fwd = (n_ct_cells + 127) // 128
    log(f"Cell type: {MC_CELL_TYPE} ({n_ct_cells} cells, {batches_per_fwd} batches/fwd)")
    log(f"Forward passes per iter: 1 base + {n_ct_types} ablated = {n_ct_types + 1}")
    log(f"Gene variance threshold: {MIN_GENE_VARIANCE}")
    log(f"Starting {N_MC_ITERATIONS} MC iterations...")

    # Weight reinitialization
    def reinitialize_model_weights(module):
        for m in module.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    # Run MC simulation
    original_state = {k: v.clone() for k, v in model.module.state_dict().items()}

    all_z_values = []
    t_start = time.time()
    for i in range(N_MC_ITERATIONS):
        t_iter = time.time()
        torch.manual_seed(i)
        reinitialize_model_weights(model.module)
        model.module.eval()

        ablation = AMICIAblationModule.compute(
            model,
            cell_type=MC_CELL_TYPE,
            head_idx=None,
            adata=adata,
            compute_z_value=True,
        )

        df = ablation._ablation_scores_df
        # Filter out low-variance genes that produce degenerate correlations
        df_filtered = df[df["gene_variance"] > MIN_GENE_VARIANCE]
        z_cols = [c for c in df_filtered.columns if c.endswith("_z_value")]
        z_vals = df_filtered[z_cols].values.flatten()
        z_vals = z_vals[np.isfinite(z_vals) & (z_vals != 0)]
        all_z_values.append(z_vals)

        elapsed = time.time() - t_iter
        total_elapsed = time.time() - t_start
        avg = total_elapsed / (i + 1)
        eta = avg * (N_MC_ITERATIONS - i - 1)
        log(f"Iter {i+1}/{N_MC_ITERATIONS}: {len(z_vals)} z-vals, {elapsed:.1f}s (avg {avg:.1f}s, ETA {eta/60:.1f}m)")

    model.module.load_state_dict(original_state)

    all_z = np.concatenate(all_z_values)
    np.savez(cache_path, all_z=all_z, n_mc=N_MC_ITERATIONS, n_ct_cells=n_ct_cells)
    log(f"Done. Collected {len(all_z):,} z-values in {(time.time()-t_start)/60:.1f}m")
    log(f"Cached to {cache_path}")

# %% Histogram with overlaid normal distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: histogram
ax = axes[0]
ax.hist(
    all_z,
    bins=100,
    density=True,
    alpha=0.7,
    color="#56B4E9",
    edgecolor="white",
    linewidth=0.3,
    label="Empirical (random init)",
)

x = np.linspace(-5, 5, 300)
ax.plot(x, stats.norm.pdf(x, 0, 1), "k-", lw=2, label=r"$\mathcal{N}(0, 1)$")

mu_fit, std_fit = stats.norm.fit(all_z)
ax.plot(x, stats.norm.pdf(x, mu_fit, std_fit), "r--", lw=1.5,
        label=rf"Fitted $\mathcal{{N}}$({mu_fit:.2f}, {std_fit:.2f}$^2$)")

ks_stat, ks_pval = stats.kstest(all_z, "norm", args=(mu_fit, std_fit))

ax.set_xlabel("z-value", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title(
    f"Ablation z-values under random initialization\n"
    f"({N_MC_ITERATIONS} models, {MC_CELL_TYPE}, {n_ct_cells} cells)",
    fontsize=12,
)
ax.legend(fontsize=10)
ax.set_xlim(-5, 5)

# Right panel: QQ plot
ax2 = axes[1]
theoretical_q = np.linspace(0.001, 0.999, 500)
empirical_q = np.quantile(all_z, theoretical_q)
normal_q = stats.norm.ppf(theoretical_q, loc=mu_fit, scale=std_fit)
ax2.scatter(normal_q, empirical_q, s=3, alpha=0.5, color="#56B4E9")
qq_lim = max(abs(normal_q.min()), abs(normal_q.max()), abs(empirical_q.min()), abs(empirical_q.max()))
qq_lim = min(qq_lim, 5)
ax2.plot([-qq_lim, qq_lim], [-qq_lim, qq_lim], "k-", lw=1)
ax2.set_xlabel("Theoretical quantiles", fontsize=12)
ax2.set_ylabel("Empirical quantiles", fontsize=12)
ax2.set_title("Q-Q Plot", fontsize=12)
ax2.set_xlim(-qq_lim, qq_lim)
ax2.set_ylim(-qq_lim, qq_lim)
ax2.set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{fig_dir}/null_z_histogram.png", dpi=300)
plt.savefig(f"{fig_dir}/null_z_histogram.svg")
plt.show()

# %% Summary statistics
log(f"\n{'='*50}")
log("Null Z-Value Distribution Summary")
log(f"{'='*50}")
log(f"MC iterations:  {N_MC_ITERATIONS}")
log(f"Cell type:      {MC_CELL_TYPE}")
log(f"Gene var filter: > {MIN_GENE_VARIANCE}")
log(f"Total z-values: {len(all_z):,}")
log(f"Mean:           {all_z.mean():.4f}  (expected: 0)")
log(f"Std:            {all_z.std():.4f}  (expected: 1)")
log(f"Skewness:       {stats.skew(all_z):.4f}  (expected: 0)")
log(f"Kurtosis:       {stats.kurtosis(all_z):.4f}  (expected: 0)")
log(f"Fitted mu:      {mu_fit:.4f}")
log(f"Fitted sigma:   {std_fit:.4f}")
log(f"KS statistic:   {ks_stat:.4f}  (vs fitted normal)")
log(f"KS p-value:     {ks_pval:.2e}")
log(f"{'='*50}")
