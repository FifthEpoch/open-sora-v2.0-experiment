#!/bin/bash
set -euo pipefail

# Convenience launcher (run on login node):
#   sbatch --account=torch_pr_36_mren delta_experiment/benchmarks/run_lora_best_100.sbatch
#   sbatch --account=torch_pr_36_mren delta_experiment/benchmarks/run_delta_a_100.sbatch
#   sbatch --account=torch_pr_36_mren delta_experiment/benchmarks/run_delta_b_lpips_winner_100.sbatch
#   sbatch --account=torch_pr_36_mren delta_experiment/benchmarks/evaluate_all.sbatch

echo "This is a convenience launcher. Submit the benchmark jobs with:"
echo "  sbatch --account=torch_pr_36_mren delta_experiment/benchmarks/run_lora_best_100.sbatch"
echo "  sbatch --account=torch_pr_36_mren delta_experiment/benchmarks/run_delta_a_100.sbatch"
echo "  sbatch --account=torch_pr_36_mren delta_experiment/benchmarks/run_delta_b_lpips_winner_100.sbatch"
echo "Then evaluate with:"
echo "  sbatch --account=torch_pr_36_mren delta_experiment/benchmarks/evaluate_all.sbatch"

