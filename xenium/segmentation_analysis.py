# %% Import necessary libraries
import scanpy as sc
import numpy as np
import seaborn as sns
from scipy import sparse
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist

from scipy.stats import fisher_exact
from scipy.stats import chi2

# %% Load the anndata object
data_date = "2025-05-01"
adata = sc.read_h5ad(f"./data/xenium_sample1/xenium_sample1_filtered_{data_date}.h5ad")
adata_train = sc.read_h5ad(
    f"./data/xenium_sample1/xenium_sample1_filtered_train_{data_date}.h5ad"
)
adata_test = sc.read_h5ad(
    f"./data/xenium_sample1/xenium_sample1_filtered_test_{data_date}.h5ad"
)
labels_key = "celltype_train_grouped"

# %% Define the cell type pairs and genes of interest 
cell_type_pairs = [
    {
        "sender_type": "CD4+_T_Cells",
        "receiver_type": "CD8+_T_Cells", 
        "genes": ["GPR183", "SELL", "TCF7", "CCR7", "LTB", "IL7R"],
        "label": "CD4+ → CD8+ T cells"
    },
    {
        "sender_type": "Invasive_Tumor",
        "receiver_type": "Macrophages_1",
        "genes": ["APOC1", "FCGR3A", "SCD", "KRT7", "PPARG", "FOXA1", "FASN"],
        "label": "Tumor → Macrophages"
    },
    {
        "sender_type": "CD8+_T_Cells",
        "receiver_type": "Macrophages_1",
        "genes": ["WARS", "SLAMF7", "C15orf48"],
        "label": "CD8+ → Macrophages"
    },
    {
        "sender_type": "CD8+_T_Cells",
        "receiver_type": "Invasive_Tumor",
        "genes": ["AGR3", "ESR1", "SERPINA3"],
        "label": "CD8+ → Tumor"
    },
]
near_threshold = 20.0
far_threshold = 100.0

# %% Analyze each cell type pair
# Initialize lists to collect data for all pairs
all_gene_stats_data = []

for pair_idx, pair_config in enumerate(cell_type_pairs):
    sender_type = pair_config["sender_type"]
    receiver_type = pair_config["receiver_type"]
    genes = pair_config["genes"]
    pair_label = pair_config["label"]
    
    print(f"\n{'='*60}")
    print(f"Analyzing pair {pair_idx + 1}: {pair_label}")
    print(f"{'='*60}")
    
    # Identify cell subsets for this pair
    sender_mask = adata.obs[labels_key] == sender_type
    receiver_mask = adata.obs[labels_key] == receiver_type

    sender_ids = adata.obs_names[sender_mask]
    receiver_ids = adata.obs_names[receiver_mask]

    if len(sender_ids) == 0:
        print(f"WARNING: No cells found for sender type: {sender_type}")
        continue
    if len(receiver_ids) == 0:
        print(f"WARNING: No cells found for receiver type: {receiver_type}")
        continue

    print(f"Found {len(sender_ids)} {sender_type} cells and {len(receiver_ids)} {receiver_type} cells")

    sender_coords = adata.obsm["spatial"][sender_mask]
    receiver_coords = adata.obsm["spatial"][receiver_mask]

    # Calculate distances between all senders and receivers
    distances = cdist(receiver_coords, sender_coords)  # receivers x senders
    min_distances_to_senders = distances.min(axis=1)  # min distance for each receiver to any sender

    # Find receivers that are near senders
    near_receiver_mask = min_distances_to_senders <= near_threshold
    near_receiver_ids = receiver_ids[near_receiver_mask]
    near_distances = min_distances_to_senders[near_receiver_mask]

    # Find senders that are far from all receivers
    distances_senders_to_receivers = cdist(sender_coords, receiver_coords)  # senders x receivers
    min_distances_to_receivers = distances_senders_to_receivers.min(axis=1)  # min distance for each sender
    far_sender_mask = min_distances_to_receivers >= far_threshold
    far_sender_ids = sender_ids[far_sender_mask]
    far_sender_distances = min_distances_to_receivers[far_sender_mask]

    print(f"Near receivers (≤{near_threshold} units from any sender): {len(near_receiver_ids)}")
    print(f"Far senders (≥{far_threshold} units from all receivers): {len(far_sender_ids)}")

    if len(far_sender_ids) > 0:
        print(f"Distance statistics for far senders:")
        print(f"  Minimum distance to any receiver: {far_sender_distances.min():.1f}")
        print(f"  Mean distance to nearest receiver: {far_sender_distances.mean():.1f}")
        print(f"  Maximum distance to nearest receiver: {far_sender_distances.max():.1f}")
    
    if len(near_receiver_ids) == 0 or len(far_sender_ids) == 0:
        print(f"WARNING: Insufficient cells for analysis in pair: {pair_label}")
        continue

    # Set random seed for reproducibility (using pair index for reproducibility across pairs)
    np.random.seed(42 + pair_idx)

    # Find the minimum sample size among the three groups
    sample_sizes = [len(near_receiver_ids), len(far_sender_ids)]
    min_sample_size = min([s for s in sample_sizes if s > 0])

    print(f"\nOriginal sample sizes:")
    print(f"  Near receivers: {len(near_receiver_ids)}")
    print(f"  Far senders: {len(far_sender_ids)}")
    print(f"  Minimum sample size: {min_sample_size}")

    # Subsample each group to the minimum size
    if len(near_receiver_ids) > min_sample_size and min_sample_size > 0:
        near_receiver_sample_idx = np.random.choice(len(near_receiver_ids), min_sample_size, replace=False)
        near_receiver_ids_sampled = near_receiver_ids[near_receiver_sample_idx]
        near_distances_sampled = near_distances[near_receiver_sample_idx]
    else:
        near_receiver_ids_sampled = near_receiver_ids
        near_distances_sampled = near_distances

    if len(far_sender_ids) > min_sample_size and min_sample_size > 0:
        far_sender_sample_idx = np.random.choice(len(far_sender_ids), min_sample_size, replace=False)
        far_sender_ids_sampled = far_sender_ids[far_sender_sample_idx]
        far_sender_distances_sampled = far_sender_distances[far_sender_sample_idx]
    else:
        far_sender_ids_sampled = far_sender_ids
        far_sender_distances_sampled = far_sender_distances

    print(f"\nSubsampled sizes for statistical comparison:")
    print(f"  Near receivers: {len(near_receiver_ids_sampled)}")
    print(f"  Far senders: {len(far_sender_ids_sampled)}")

    distance_data = {
        'near_receivers': near_receiver_ids_sampled,  # Use subsampled data
        'far_senders': far_sender_ids_sampled,       # Use subsampled data
        'all_sender_cells': sender_ids,               # Keep original for spatial plots
        'all_receiver_cells': receiver_ids,           # Keep original for spatial plots
        'near_distances': near_distances_sampled,
        'far_sender_distances': far_sender_distances_sampled,
        'min_distances_to_receivers': min_distances_to_receivers,
        'min_distances_to_senders': min_distances_to_senders,
        # Store original unsampled data for reference
        'near_receivers_original': near_receiver_ids,
        'far_senders_original': far_sender_ids,
    }

    if isinstance(genes, str):
        genes = [genes]

    # Check if genes exist
    missing_genes = [g for g in genes if g not in adata.var_names]
    if missing_genes:
        print(f"Warning: Genes not found in data: {missing_genes}")
        genes = [g for g in genes if g in adata.var_names]

    if len(genes) == 0:
        print(f"WARNING: No valid genes found in the data for pair: {pair_label}")
        continue

    # Get expression matrix (dense for easier indexing)
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X

    # Get indices for cells and genes
    near_indices = adata.obs_names.get_indexer(distance_data['near_receivers'])
    far_sender_indices = adata.obs_names.get_indexer(distance_data['far_senders'])

    gene_indices = [adata.var_names.get_loc(g) for g in genes]

    # Extract expression data
    near_expr = (
        X[near_indices][:, gene_indices]
        if len(near_indices) > 0
        else np.array([]).reshape(0, len(genes))
    )
    far_sender_expr = (
        X[far_sender_indices][:, gene_indices]
        if len(far_sender_indices) > 0
        else np.array([]).reshape(0, len(genes))
    )

    expr_data = {
        'near_expression': near_expr,
        'far_sender_expression': far_sender_expr,
        'genes': genes,
        'gene_indices': gene_indices
    }

    genes_current = expr_data['genes']
    near_expr = expr_data['near_expression']
    far_sender_expr = expr_data['far_sender_expression']

    # Initialize lists to collect data for this pair
    pair_gene_stats_data = []

    for i, gene in enumerate(genes_current):
        plt.figure(figsize=(20, 5))
        ax = plt.gca()
        
        # Prepare data for plotting
        near_values = near_expr[:, i] if len(near_expr) > 0 else []
        far_sender_values = far_sender_expr[:, i] if len(far_sender_expr) > 0 else []
        
        # Create KDE plots
        has_data = False
        
        if len(near_values) > 1:  # Need at least 2 points for KDE
            sns.kdeplot(near_values, ax=ax, color='blue', fill=True, alpha=0.3, 
                       label=f'Near receivers (≤{near_threshold})', linewidth=2, common_norm=False)
            near_mean = np.mean(near_values)
            ax.axvline(near_mean, color='blue', linestyle='--', alpha=0.8, 
                      label=f'Near mean: {near_mean:.2f}')
            has_data = True
        elif len(near_values) == 1:
            ax.axvline(near_values[0], color='blue', linestyle='-', alpha=0.8, 
                      label=f'Near (n=1): {near_values[0]:.2f}')
            has_data = True
        
        if len(far_sender_values) > 1:  # Need at least 2 points for KDE
            sns.kdeplot(far_sender_values, ax=ax, color='orange', fill=True, alpha=0.3, 
                       label=f'Far senders (≥{far_threshold})', linewidth=2, common_norm=False)
            far_mean = np.mean(far_sender_values)
            ax.axvline(far_mean, color='orange', linestyle='--', alpha=0.8, 
                      label=f'Far sender mean: {far_mean:.2f}')
            has_data = True
        elif len(far_sender_values) == 1:
            ax.axvline(far_sender_values[0], color='orange', linestyle='-', alpha=0.8, 
                      label=f'Far (n=1): {far_sender_values[0]:.2f}')
            has_data = True


        if has_data:
            ax.set_xlabel('Expression')
            ax.set_ylabel('Density')
            ax.set_title(f'{gene} Expression Distribution ({pair_label})\n{receiver_type} near {sender_type} vs Far {sender_type}')
            ax.legend()
            
            # Add statistical analysis if we have enough data
            if len(near_values) > 0 and len(far_sender_values) > 0:
                # Basic statistics
                near_median = np.median(near_values)
                far_sender_median = np.median(far_sender_values)
                near_mean = np.mean(near_values)
                far_sender_mean = np.mean(far_sender_values)
                

                # Mann-Whitney U test: directional hypothesis test
                # H₀: near and far distributions are equal vs H₁: near values tend to be greater than far values
                if len(near_values) > 1 and len(far_sender_values) > 1:
                    # Calculate Mann-Whitney U statistic and p-value
                    u_statistic, mannwhitney_pval = stats.mannwhitneyu(
                        near_values, far_sender_values, alternative='greater'
                    )
                else:
                    u_statistic = np.nan
                    mannwhitney_pval = np.nan
                
                # Collect data for this pair
                pair_gene_stats_data.append({
                    'gene': gene,
                    'pair_label': pair_label,
                    'sender_type': sender_type,
                    'receiver_type': receiver_type,
                    'mannwhitney_pval': mannwhitney_pval,
                    'mannwhitney_statistic': u_statistic,
                    'near_mean': np.mean(near_values) if len(near_values) > 0 else np.nan,
                    'far_sender_mean': np.mean(far_sender_values) if len(far_sender_values) > 0 else np.nan,
                })
        else:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{gene} - No data ({pair_label})')

        plt.tight_layout()
        plt.show()

    # Add this pair's data to the overall collection
    all_gene_stats_data.extend(pair_gene_stats_data)
    print(f"Completed analysis for {len(pair_gene_stats_data)} genes in {pair_label}")

print(f"\n{'='*60}")
print(f"Analysis complete. Total genes analyzed: {len(all_gene_stats_data)}")
print(f"{'='*60}")

# %% Create segmentation test plot using Wald test results
def plot_segmentation_test_results(
    gene_stats_data,
    pval_threshold=0.05,
    label_genes=True,
    show=True,
    save_png=False,
    save_svg=False,
    save_dir="./figures",
):
    """
    Plot segmentation test results showing Mann-Whitney U test p-values for directional comparisons.
    
    Args:
        gene_stats_data (list): List of dictionaries with gene statistics including 'pair_label'
        pval_threshold (float): P-value significance threshold (default 0.05)
        label_genes (bool): Whether to show gene names (always True for this plot type)
        show (bool): Whether to display the plot
        save_png (bool): Whether to save as PNG
        save_svg (bool): Whether to save as SVG
        save_dir (str): Directory to save plots
    """
    import pandas as pd
    import matplotlib.colors as mcolors
    
    # Convert to DataFrame
    df = pd.DataFrame(gene_stats_data)
    
    # Filter out genes with NaN values
    df = df.dropna(subset=['mannwhitney_pval'])
    
    if len(df) == 0:
        print("No genes with valid Mann-Whitney U test p-value data")
        return
    
    # Group by pair and sort within each group by p-value (ascending = most significant first)
    df_grouped = []
    
    for pair_label in df['pair_label'].unique():
        pair_df = df[df['pair_label'] == pair_label].copy()
        pair_df = pair_df.sort_values('mannwhitney_pval', ascending=True)
        df_grouped.append(pair_df)
    
    # Concatenate all pairs (maintaining pair grouping)
    df_sorted = pd.concat(df_grouped, ignore_index=True)
    
    # Create the plot with increased height for multiple pairs
    fig, ax = plt.subplots(figsize=(12, max(8, len(df_sorted) * 0.5)))
    
    # Create scatter plot
    y_positions = np.arange(len(df_sorted))
    
    # Color points based on significance
    colors = ['red' if p < pval_threshold else 'lightblue' for p in df_sorted['mannwhitney_pval']]
    
    # Plot all genes with circle markers
    scatter = ax.scatter(
        df_sorted['mannwhitney_pval'], 
        y_positions,
        c=colors,
        s=120,  # Fixed size for all dots
        marker='o',  # All circles
        edgecolors='black',
        linewidth=0.8,
        alpha=0.8
    )
    
    # Add gene names with pair labels on y-axis
    gene_labels = [f"{row['gene']} ({row['pair_label']})" for _, row in df_sorted.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(gene_labels, fontsize=9)
    
    # Add significance threshold line
    ax.axvline(x=pval_threshold, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Significance threshold (p = {pval_threshold})')
    
    # Customize the plot
    ax.set_xlabel('Mann-Whitney U Test P-value', fontsize=12)
    ax.set_ylabel('Genes by Cell Type Pair', fontsize=12)
    ax.set_title('Segmentation Test Results\nMann-Whitney U Test: Near Receivers > Far Senders', 
                 fontsize=14, pad=20)
    
    # Set x-axis limits (0 to 1 for p-values)
    ax.set_xlim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend for colors and threshold line
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, label=f'Significant (p < {pval_threshold})'),
        Patch(facecolor='lightblue', alpha=0.8, label=f'Non-significant (p ≥ {pval_threshold})'),
        plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.7, 
                   label=f'Significance threshold (p = {pval_threshold})')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Add text box with summary statistics
    n_total = len(df_sorted)
    significant_genes = df_sorted[df_sorted['mannwhitney_pval'] < pval_threshold]
    n_significant = len(significant_genes)
    mean_pval = df_sorted['mannwhitney_pval'].mean()
    
    # Stats per pair
    pair_stats = []
    for pair_label in df_sorted['pair_label'].unique():
        pair_data = df_sorted[df_sorted['pair_label'] == pair_label]
        n_pair = len(pair_data)
        n_sig_pair = len(pair_data[pair_data['mannwhitney_pval'] < pval_threshold])
        pair_stats.append(f"{pair_label}: {n_sig_pair}/{n_pair}")
    
    stats_text = f"""Total genes: {n_total}
Significant: {n_significant} ({n_significant/n_total*100:.1f}%)
Mean p-value: {mean_pval:.3f}

Per pair:
{chr(10).join(pair_stats)}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_png:
        plt.savefig(
            f"{save_dir}/segmentation_test_results.png",
            dpi=300,
            bbox_inches="tight",
        )
    if save_svg:
        plt.savefig(
            f"{save_dir}/segmentation_test_results.svg",
            bbox_inches="tight",
        )
    
    if show:
        plt.show()
    
    return fig

# Generate the segmentation test results plot
if len(all_gene_stats_data) > 0:
    # Ensure figures directory exists
    import os
    os.makedirs("./figures/segmentation_analysis", exist_ok=True)
    
    plot_segmentation_test_results(
        gene_stats_data=all_gene_stats_data,
        pval_threshold=0.05,
        label_genes=True,
        show=True,
        save_png=True,
        save_svg=True,
        save_dir="./figures/segmentation_analysis"
    )
else:
    print("No gene statistics data collected for segmentation test plot")

# %%
