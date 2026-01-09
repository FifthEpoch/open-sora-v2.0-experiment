## δ-TTA Experiments (SLOT-inspired) for Open-Sora v2.0

This directory contains experiments that implement **sample-specific test-time adaptation** using a very small set of learnable parameters **δ** (delta), inspired by the “single-vector” philosophy in SLOT ([arXiv:2505.12392](https://arxiv.org/pdf/2505.12392)).

The goal is to improve **video continuation** quality using **only the 33 conditioning frames** available at inference time (no access to future frames).

### Methods

- **Option A (global δ)**: a single δ vector added to the denoiser’s **global conditioning embedding** (`vec`) used for per-block modulation.
- **Option B (per-layer δ)**: a small set of δ vectors (grouped per depth) added to the modulation pathway per block.
- **Option C (output δ)**: a δ vector added as a residual correction to the denoiser output (predicted velocity).

### Running (cluster)

These scripts are intended to run on the Torch cluster via SLURM. Submit with your account:

```bash
sbatch --account=torch_pr_36_mren delta_experiment/sbatch/run_delta_a.sbatch
sbatch --account=torch_pr_36_mren delta_experiment/sbatch/run_delta_b.sbatch
sbatch --account=torch_pr_36_mren delta_experiment/sbatch/run_delta_c.sbatch

sbatch --account=torch_pr_36_mren delta_experiment/sbatch/evaluate_compare.sbatch
```

### Results layout

Each method writes:
- `results.json`: per-video metadata (paths, timings, final loss, success)
- `videos/`: generated mp4 files
- `config.json`: experiment configuration snapshot

### Evaluation

Evaluation is performed on frames **33–64** only (generated portion) and reports:
- PSNR
- SSIM
- LPIPS

Baseline and best-LoRA directories are taken from `lora_experiment/` to enable direct comparison.


