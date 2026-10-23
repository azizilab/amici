# Benchmark configs

Snakemake configs that reproduce each benchmarking figure, with the code they run on and the environments they
ran in. Run them from `benchmarks/` with
`snakemake --use-conda --cores <N> --resources gpu=1 --configfile <config>`; each config sets `datasets_to_run`, so
only its dataset runs.

| Config | Figures | Code | Published outputs |
| --- | --- | --- | --- |
| `fig2_pbmc_benchmark_config.yaml` | Fig. 2a-e, Fig. S2, Fig. S3 | any commit (PBMC settings unchanged since e768bc5) | khushi-gpu-migrated:`/home/khushidesai/amici_results`; Fig. 2 subset in `gs://justin-amici/semisyn/fig2_amici_results/` |
| `fig2_realistic_benchmark_config.yaml` | Fig. 2f-j | `3a3cc16` (2026-04-02); HEAD selects models the same way | not kept; values in an April-2026 message. Re-run's selected model, its scores and the sweep logs: `gs://justin-amici/semisyn/fig2_realistic_april_sweep/` |
| `figS35a_spatial_cv_benchmark_config.yaml` | Fig. S35a | `26a28ec` or later | `benchmarks/results_cv/breast_cancer_42/` |

The realistic configs both use `data/breast_cancer_42.h5ad` (md5 1e185c0bddb7795d30e8729d289ad8d3). The Fig. 2f-j
sweep is the committed April config with more attention heads, as recorded in an April-2026 message; the Fig. S35a
sweep is the wider 144-setting grid that later became the default in `benchmarks/benchmark_config.yaml`.
NicheDE jobs need about 30 GB of memory each, so cap them with
`--resources mem_mb=60000 --set-resources run_nichede_gene_task_model:mem_mb=40000`.

## Environments

The env files in `benchmarks/envs/` do not pin versions, so rebuilding them today resolves different packages; the
amici env currently pulls torch built for CUDA 13, which cannot use this machine's driver (CUDA 12.4). `envs/`
holds the exact package lists of the environments the runs used. Build each env from its yaml and then pin it:

```bash
pip install --no-deps -r ../reproducibility/benchmarks/envs/amici_pip_lock.txt
```

| Lock file | Built | Notes |
| --- | --- | --- |
| `envs/amici_pip_lock.txt` | 2026-06-04 | torch 2.6.0+cu124, transformer-lens 1.14.0; the env of the CV run and of the realistic re-run |
| `envs/gitiii_pip_lock.txt` | 2026-03-09 | |
| `envs/cgcom_pip_lock.txt` | 2026-03-28 | torch 2.11.0+cu130 cannot use the CUDA 12.4 driver, so CGCom trains on the CPU in it |
| `envs/basic_pip_lock.txt` | 2026-03-09 | ground-truth scoring |
| `envs/ncem_pip_lock.txt` | 2026-03-09 | |
| `envs/nichede_R_lock.txt` | 2026-03-09 | R packages; `generate_nichede_scores.R` installs nicheDE (87e0e89) and sceasy (d1e2b87) from GitHub at run time |

The PBMC run predates these environments and its own were not kept.

## Reproduction status

- **Fig. 2a-e**: the published per-seed AUPRCs reproduce every p-value in the Fig. 2 caption. The run's input
  datasets were not kept; regenerating them gives different datasets, and seeds 58 and 123 give the same one.
- **Fig. 2f-j**: reproduced to within 0.02 AUPRC. The published values (AUPRC; gene / sender / receiver) survive in an
  April-2026 message: AMICI 0.99 / 0.52 / 0.93, GITIII 0.93 / 0.39 / 0.95. Re-running
  `fig2_realistic_benchmark_config.yaml` (30 AMICI runs; outputs in
  `benchmarks/results_realistic_rerun_april_family/`) selects the 14-head, seed-42 model, which gives AMICI
  0.975 / 0.513 / 0.944; GITIII re-run at commit 3a3cc16 gives 0.933 / 0.410 / 0.949. NicheDE (sigma = 20) and NCEM
  reproduce the published bars; NicheDE sigma = 200 / 500 give 0.803 / 0.752 against 0.732 / 0.673.

  `fig2_realistic_april_sweep_scores.csv` lists all 30 AMICI models with their test loss and AUPRCs. They lie
  within 0.3 of each other in test loss, but their AUPRCs span gene 0.54-0.98, sender 0.03-0.59 and receiver
  0.10-0.94 (medians 0.83 / 0.35 / 0.87), and GPU training is not bitwise deterministic, so a re-run can select
  a different model and land elsewhere in these ranges. The committed April config alone (10 heads, 5 seeds)
  selected models scoring 0.81-0.85 on the gene task, and the 144-setting grid of Fig. S35a selected one scoring
  0.987 / 0.319 / 0.939.
- **Fig. S35a**: the outputs in `benchmarks/results_cv/` were produced by this config.
