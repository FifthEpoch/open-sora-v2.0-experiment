## δ-Based Test-Time Adaptation for Open-Sora v2.0 (SLOT-inspired)

### 1. Motivation and Problem Setting

The current LoRA-based test-time adaptation (TTA) pipeline for Open-Sora v2.0 yields modest average gains over a strong baseline when evaluated on 100 stratified UCF-101 videos (33 conditioning frames, 32 generated frames). The best LoRA setting observed so far (rank=8, lr=2e-4, 20 steps) improves PSNR/SSIM/LPIPS by a small margin on average, but also exhibits meaningful per-video variance and occasional degradation, suggesting that even parameter-efficient fine-tuning can overfit or drift for some samples.

This motivates exploring *even lighter* per-sample adaptation mechanisms that:\n
1) introduce substantially fewer trainable parameters than LoRA,\n
2) can be optimized for a small number of steps at test time,\n
3) are reset/discarded after each sample,\n
4) use only the conditioning frames available at inference time (fairness constraint).

The SLOT framework (“Sample-specific Language Model Optimization at Test-time”) proposes sample-specific optimization of a lightweight additive parameter vector δ for LLMs, optimized for only a few steps and discarded afterwards, while avoiding heavy per-sample overhead through feature caching ([SLOT: arXiv:2505.12392](https://arxiv.org/pdf/2505.12392)). Although Open-Sora is a diffusion/flow-matching model rather than an autoregressive LLM, the conceptual idea of a *single small adaptive state* is potentially transferable.

This report proposes and implements three δ-based TTA variants for Open-Sora v2.0 video continuation, designed to be conceptually aligned with SLOT while respecting diffusion-model structure.

---

### 2. Background: Where Adaptation “Lives” in Open-Sora v2.0

Open-Sora v2.0 uses an MMDiT denoiser. At a high level, the denoiser receives:

- a packed latent token sequence corresponding to the noisy sample \(z_t\) at time \(t\),\n
- text context tokens from a large text encoder,\n
- a global conditioning vector `vec` derived from timestep embedding and CLIP embedding,\n
- an optional conditional embedding derived from the reference video (`cond` in v2v_head mode).

Crucially, the global vector `vec` is used repeatedly:

- In every DoubleStreamBlock, `vec` produces shift/scale/gate parameters (via a learned Modulation MLP) that modulate both the image stream and the text stream.\n
- In every SingleStreamBlock, `vec` controls the block’s modulation/gating.\n
- In the LastLayer, `vec` drives AdaLN modulation before producing the denoiser output (predicted velocity/noise in latent token space).

Therefore, small perturbations to `vec` can have global impact across the denoiser depth, while remaining low-dimensional compared to LoRA.

---

### 3. Fairness Constraint (No Access to Future Frames)

The experimental protocol must ensure that the TTA objective uses **only the conditioning frames** available at inference time.

- Each clip contains 65 frames total.\n
- Inference uses the first 33 frames as conditioning (v2v_head).\n
- The generation target is frames 33–64.\n
- During δ-optimization, **only the first 33 frames are used** (as latents) and the supervision signal is the flow-matching objective constructed from these conditioning latents and random noise.\n
- No ground-truth frames from 33–64 are used during δ-optimization.

Evaluation compares generated frames 33–64 against ground truth 33–64 using PSNR/SSIM/LPIPS.

---

### 4. Learning Signal: Flow-Matching MSE on Conditioning Latents

All three δ methods optimize δ using the same conditioning-only objective currently used for LoRA TTA:

1) Encode the first 33 frames with the VAE to obtain conditioning latents \(x\).\n
2) Sample random \(t \\sim U(0,1)\), and random noise \(\\epsilon \\sim \\mathcal{N}(0, I)\).\n
3) Construct \(z_t\) as a mixture of \(x\) and \(\\epsilon\) following Open-Sora’s rectified flow convention.\n
4) Define a target velocity \(v_{target}\) from \(x\) and \(\\epsilon\).\n
5) Run the denoiser to predict \(v_{pred}\).\n
6) Optimize δ to minimize \(\\|v_{pred} - v_{target}\\|_2^2\) (optionally with δ regularization).

This objective is computed entirely from conditioning frames and does not require future frames.

---

### 5. Three δ-Based Adaptation Methods

#### 5.1 Option A (Most SLOT-like): Global δ added to the conditioning embedding

**Idea.** Learn a single vector \(\\delta \\in \\mathbb{R}^{d}\) (where \(d = \\) hidden size of `vec`) and add it to the denoiser’s global conditioning vector:

\\[
\\text{vec}' = \\text{vec} + \\delta
\\]

**Where it adapts the diffusion model.** Because `vec` controls modulation in every block, δ acts as a sample-specific bias on the denoiser’s global conditioning pathway. This resembles SLOT’s additive vector before the output head, but in diffusion it affects *modulation* across many layers and also the final AdaLN head.

**Implementation.** The code patches the model’s `prepare_block_inputs` to add δ to `vec` before the forward pass proceeds through blocks. Only δ is trainable; all denoiser weights remain frozen. During sampling, the same δ patch is enabled so that every denoising step uses the adapted conditioning.

**Expected behavior.** Strong leverage (global impact) but risk of over-adaptation if δ grows too large; mitigated by few steps and optional L2 regularization.

---

#### 5.2 Option B: Grouped per-layer δ offsets

**Idea.** Learn a small set of δ vectors instead of one, to increase expressivity while remaining tiny. For example, with 4 groups for double blocks and 4 groups for single blocks:

- \(\\{\\delta^{double}_g\\}_{g=1..G_d}\)\n
- \(\\{\\delta^{single}_g\\}_{g=1..G_s}\)\n
- \(\\delta^{final}\)

Each block uses:

\\[
\\text{vec}_i' = \\text{vec} + \\delta_{group(i)}
\\]

**Where it adapts the diffusion model.** This modifies the modulation input differently at different depths, which can capture depth-specific corrections (early layers often affect global structure; late layers affect details). Compared to LoRA, this remains dramatically smaller but provides more control than a single δ.

**Implementation.** A custom forward function mirrors the model’s block loop and injects the grouped δ into `vec` at each block call. During sampling, the model forward is temporarily replaced so the denoiser uses the same per-layer offsets at every denoising step.

**Expected behavior.** More flexible than Option A and may reduce cases where a single δ is “too global”. Slightly increased risk of instability and overfit compared to A due to higher capacity, but still far smaller than LoRA.

---

#### 5.3 Option C: δ as an output correction

**Idea.** Keep the denoiser unchanged, and learn a small output-space vector \(\\delta_{out}\\) that shifts the denoiser prediction:

\\[
v'_{pred} = v_{pred} + \\delta_{out}
\\]

Here \(\\delta_{out} \\in \\mathbb{R}^{c}\) where \(c\) is the denoiser output dimension (for Open-Sora v2.0 typically 64 channels in packed latent token space).

**Where it adapts the diffusion model.** This directly adjusts the denoiser’s predicted velocity/noise field, without changing internal representations. Conceptually it resembles “logit bias” in language models, but applied to the velocity prediction.

**Implementation.** During δ-optimization, the loss is computed using the corrected output. During sampling, model.forward is temporarily wrapped so every denoising step uses the corrected prediction.

**Expected behavior.** Extremely lightweight and stable parameterization, but potentially limited expressivity: a constant per-token shift may not represent complex sample-specific corrections.

---

### 6. Experimental Protocol

**Video set.** Same 100 stratified videos (seed=42) for direct comparability with LoRA experiments.\n
**Conditioning.** 33 frames.\n
**Generation.** 65 total frames.\n
**Evaluation.** PSNR/SSIM/LPIPS computed on frames 33–64 only.\n
**Baselines.**\n
- baseline continuation results\n
- best LoRA setting (rank=8, lr=2e-4, 20 steps)\n
**Compute.** Designed for 1×H200 per job.

---

### 7. Practical Considerations and Failure Modes

1) **Denoising timestep dependence.** A single δ is applied across all timesteps during sampling. If performance is sensitive to timestep, future work can extend δ to a low-dimensional \(\\delta(t)\) schedule.\n
2) **Over-adaptation risk.** Even with few steps, δ can drift. Use small step counts, small LR, and optional L2 penalty on δ.\n
3) **Per-video variance.** As observed with LoRA, not all videos benefit. A follow-up is to add gating (apply δ only if a proxy score improves).\n
4) **No caching benefit.** Unlike SLOT, diffusion requires repeated forward/backward through the denoiser for δ updates. This is mitigated by optimizing δ using a single-step flow-matching objective rather than full sampling.\n

---

### 8. Files and How to Run

Run each method (Torch cluster):\n
\n
```bash
sbatch --account=torch_pr_36_mren delta_experiment/sbatch/run_delta_a.sbatch\n+sbatch --account=torch_pr_36_mren delta_experiment/sbatch/run_delta_b.sbatch\n+sbatch --account=torch_pr_36_mren delta_experiment/sbatch/run_delta_c.sbatch\n+```\n
\n
Then evaluate and compare:\n
\n
```bash
sbatch --account=torch_pr_36_mren delta_experiment/sbatch/evaluate_compare.sbatch\n+```\n
\n
Outputs are written under `delta_experiment/results/`.\n
\n
---
\n
### 9. Relationship to SLOT\n
\n
SLOT proposes optimizing a lightweight δ added right before the output head of a language model, minimizing loss on the input prompt and discarding δ after each sample ([arXiv:2505.12392](https://arxiv.org/pdf/2505.12392)). In this project, δ is optimized per video using conditioning-only flow-matching loss, and injected into the diffusion denoiser at three possible locations (global conditioning, per-layer modulation, output correction). These represent three diffusion-specific interpretations of “sample-specific parameter vector optimization at test time.”\n
\n

