#!/usr/bin/env python
"""Plot AMICI runtime benchmark results."""

import argparse
import json
import os

import matplotlib.pyplot as plt


def load_results(filepath):
    """Load benchmark results from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    return data["metadata"], data["results"]


def print_peak_memory(results, accelerator):
    """Print peak GPU memory for reporting."""
    peak_memory_gb = [r["peak_memory_gb"] for r in results]
    max_mem = max(peak_memory_gb)
    print(f"\n{accelerator.upper()} Peak Memory Summary:")
    for r in results:
        print(f"  n_cells={r['n_cells']:>7d}: {r['peak_memory_gb']:.3f} GB")
    print(f"  Max across all sizes: {max_mem:.3f} GB")


def plot_benchmark(results, metadata, accelerator, output_dir):
    """Create a single runtime plot for this accelerator."""
    n_cells = [r["n_cells"] for r in results]
    runtime_setup = [r["runtime_setup_s"] for r in results]
    runtime_train = [r["runtime_train_s"] for r in results]
    runtime_total = [r["runtime_total_s"] for r in results]

    accel_label = accelerator.upper()
    max_epochs = metadata["train_params"]["max_epochs"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(n_cells, runtime_total, "o-", color="tab:blue", label="Total", linewidth=2)
    ax.plot(
        n_cells, runtime_train, "s--", color="tab:orange", label="Training", linewidth=1.5
    )
    ax.plot(
        n_cells, runtime_setup, "^--", color="tab:green", label="Setup (NN)", linewidth=1.5
    )
    ax.set_xlabel("Number of Cells", fontsize=12)
    ax.set_ylabel("Wall-Clock Time (seconds)", fontsize=12)
    ax.set_title(f"Runtime vs. Dataset Size ({accel_label}, {max_epochs} epochs)", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "svg"):
        save_path = os.path.join(output_dir, f"benchmark_{accelerator}.{ext}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
    plt.close(fig)


def plot_combined(all_results, all_metadata, output_dir):
    """Create a joint CPU vs GPU runtime plot with seconds/epoch on right axis."""
    fig, ax_left = plt.subplots(figsize=(8, 5))

    colors = {"cpu": "tab:blue", "gpu": "tab:orange"}
    markers = {"cpu": "o", "gpu": "s"}

    # Use only subset sizes common to both accelerators for a fair comparison
    common_sizes = None
    for results in all_results.values():
        sizes = {r["n_cells"] for r in results}
        common_sizes = sizes if common_sizes is None else common_sizes & sizes

    max_epochs = None
    for accelerator, results in all_results.items():
        filtered = [r for r in results if r["n_cells"] in common_sizes]
        n_cells = [r["n_cells"] for r in filtered]
        runtime_total = [r["runtime_total_s"] for r in filtered]
        label = accelerator.upper()
        ax_left.plot(
            n_cells, runtime_total, f"{markers[accelerator]}-",
            color=colors[accelerator], label=label, linewidth=2, markersize=6,
        )
        max_epochs = all_metadata[accelerator]["train_params"]["max_epochs"]

    ax_left.set_xlabel("Number of Cells", fontsize=12)
    ax_left.set_ylabel("Wall-Clock Time (seconds)", fontsize=12)
    ax_left.set_title(f"Runtime vs. Dataset Size ({max_epochs} epochs)", fontsize=13)
    ax_left.set_xscale("log")
    ax_left.legend(fontsize=11)
    ax_left.grid(True, alpha=0.3)
    ax_left.tick_params(labelsize=10)

    # Set y-axis minimum to zero
    _, y_max = ax_left.get_ylim()
    ax_left.set_ylim(0, y_max)

    # Right y-axis: seconds per epoch
    ax_right = ax_left.twinx()
    ax_right.set_ylabel("Seconds per Epoch", fontsize=12)
    ax_right.set_ylim(0, y_max / max_epochs)
    ax_right.tick_params(labelsize=10)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "svg"):
        save_path = os.path.join(output_dir, f"benchmark_combined.{ext}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
    plt.close(fig)


def write_csv(all_results, all_metadata, output_dir):
    """Write a CSV table with all benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")

    with open(csv_path, "w") as f:
        f.write("accelerator,n_cells,runtime_setup_s,runtime_train_s,runtime_total_s,peak_memory_gb\n")
        for accelerator in ("cpu", "gpu"):
            if accelerator not in all_results:
                continue
            for r in all_results[accelerator]:
                f.write(
                    f"{accelerator},{r['n_cells']},{r['runtime_setup_s']:.2f},"
                    f"{r['runtime_train_s']:.2f},{r['runtime_total_s']:.2f},"
                    f"{r['peak_memory_gb']:.3f}\n"
                )
    print(f"Saved {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot AMICI benchmark results")
    parser.add_argument(
        "--results-dir", type=str, default=".", help="Directory containing benchmark JSON files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures", help="Directory to save figures"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    all_results = {}
    all_metadata = {}

    for accelerator in ("cpu", "gpu"):
        filepath = os.path.join(args.results_dir, f"benchmark_results_{accelerator}.json")
        if not os.path.exists(filepath):
            print(f"No results found for {accelerator} at {filepath}, skipping.")
            continue
        metadata, results = load_results(filepath)
        all_results[accelerator] = results
        all_metadata[accelerator] = metadata
        print_peak_memory(results, accelerator)
        plot_benchmark(results, metadata, accelerator, args.output_dir)

    if len(all_results) >= 2:
        plot_combined(all_results, all_metadata, args.output_dir)

    if all_results:
        write_csv(all_results, all_metadata, args.output_dir)


if __name__ == "__main__":
    main()
