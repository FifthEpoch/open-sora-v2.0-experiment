# Camera-ready diagram prompts (δ-TTA on Open-Sora v2.0) — mid-detail (nano banana)

These are **three standalone prompts** you can paste directly into **nano banana** to generate **publication-ready vector-style figures**.

**Design goal:** hit a middle ground:
- show enough Open-Sora v2.0 structure to orient the reader, but
- keep the main message dominant: **where δ is injected** and **δ is the only trainable component** (backbone frozen).

All prompts assume **Open-Sora v2.0 video continuation**:
- Conditioning frames: **0–32** (33 frames) as input.
- Output: frames **0–64** (65 frames total), where **33–64** are generated.
- Backbone: **Video VAE (encode/decode)** + **diffusion denoiser (MMDiT)** conditioned on **text + timestep** via a global vector `vec` used for **AdaLN-style modulation**.

**Nano banana style guidance (important, but not overly restrictive):**
- Make it a **clean schematic** suitable for a paper or slide: flat/2D (not photorealistic), crisp lines, high readability.
- **Allow tasteful styling**: rounded rectangles/cards, subtle drop shadows, light gradients, and small icons are OK.
- Keep colors **restrained and consistent**:
  - Use **blue** as the main highlight for **TRAINABLE** δ paths.
  - Use neutral gray for **FROZEN** backbone blocks.
  - Optional: a secondary neutral (light gray) for container boxes (sampling loop).
- Typography: prefer a clean serif or professional UI serif; avoid playful fonts.

---

## Prompt A — Option A (single global δ added to conditioning `vec`) — mid-detail

Create a **publication-ready schematic diagram** with a modern, clean look. You may use **rounded rectangles/cards**, **subtle shadows**, and **light gradients** for depth, as long as it stays clearly a diagram (not photorealistic). Keep the layout aligned on a grid with consistent spacing. The figure should be mid-detail: show the main modules, the sampling loop, and a simplified denoiser interior with *two stages* and *one modulation cue*.

### Baseline Open-Sora v2.0 architecture (mid-detail; must be included)
Draw a left-to-right pipeline:

1) **Inputs**
- Filmstrip labeled `x_cond` with subtitle “conditioning frames 0–32”.
- Text box labeled `prompt`.

2) **Frozen encoders**
- Box: “Text Encoder (T5/CLIP-like) [FROZEN]” producing `c_text` and `p_text`.
- Box: “Video VAE Encoder [FROZEN]” producing `z_cond`.

3) **Sampling loop**
- A large container labeled “Sampling loop (t = T…0) [FROZEN]”.
- Inside it: a small “Scheduler” box and a flow `z_t → denoiser → v_pred → latent update → z_{t-1}` (use simple arrows).

4) **Denoiser (mid-detail)**
Inside the sampling loop, draw a big box “Denoiser (MMDiT) [FROZEN]”.
Inside this denoiser box, include only:
- A small node: `vec = f(timestep, p_text)` (show timestep arrow + p_text arrow).
- Two stacked stage boxes:
  - “Stage 1: Double-stream blocks (video+text)”
  - “Stage 2: Single-stream blocks (unified)”
- One line under both stage boxes: `vec → AdaLN modulation (scale/shift/gate)`.
No per-layer internals beyond this.

5) **Frozen decoder**
- Box: “Video VAE Decoder [FROZEN]” producing filmstrip `x_out` labeled “frames 0–64 (33–64 generated)” with the generated part highlighted.

### Option A injection (main highlight)
- Add a **TRAINABLE** vector labeled `δ` (per-video) in accent blue.
- Add a `+` node at the denoiser conditioning path: **`vec' = vec + δ`**.
- Route `vec'` (not `vec`) to the “AdaLN modulation” line feeding both stage boxes.

### TTA note (small side panel)
Add a small side panel titled “TTA (per video)”:
- “Backbone frozen; optimize δ only”
- “K gradient steps”
- “Loss uses only frames 0–32”
Use a small lock icon on frozen boxes and a small gradient arrow pointing to δ.

---

## Prompt B — Option B (grouped per-layer δ offsets into modulation) — mid-detail

Create a **publication-ready schematic diagram** with a modern, clean look. You may use **rounded rectangles/cards**, **subtle shadows**, and **light gradients** for depth, as long as it stays clearly a diagram (not photorealistic). Use a clean serif/professional font and a grid layout. Mid-detail: show sampling loop + denoiser stages, but avoid full layer-level detail.

### Baseline Open-Sora v2.0 (same as Prompt A, mid-detail)
Include the same modules as Prompt A:
- Inputs (`x_cond`, `prompt`)
- Frozen Text Encoder → `c_text`, `p_text`
- Frozen VAE Encoder → `z_cond`
- Frozen sampling loop with Scheduler and latent update arrows
- Frozen denoiser box with `vec = f(timestep, p_text)` and “vec → AdaLN modulation”
- Frozen VAE Decoder → `x_out` (33–64 generated highlighted)

### Option B injection (main highlight)
Show **multiple trainable vectors** rather than a single δ:
- Add a **TRAINABLE** set `{δ_g}` with 4 vectors (`δ_1..δ_4`) in accent blue.
- Inside the denoiser, subdivide the two stage boxes into **early/late** halves (four regions total):
  - Stage1-early, Stage1-late, Stage2-early, Stage2-late.
- Draw arrows:
  - `δ_1` → Stage1-early modulation
  - `δ_2` → Stage1-late modulation
  - `δ_3` → Stage2-early modulation
  - `δ_4` → Stage2-late modulation
- Label each injection with: `vec_i' = vec + δ_group(i)` and show it feeding “AdaLN modulation”.

### TTA note (small side panel)
“Backbone frozen; optimize {δ_g} only; K steps; loss uses only frames 0–32.”
Use lock icons for frozen boxes, and show δ vectors as the only highlighted trainable elements.

---

## Prompt C — Option C (δ as denoiser output correction) — mid-detail

Create a **publication-ready schematic diagram** with a modern, clean look. You may use **rounded rectangles/cards**, **subtle shadows**, and **light gradients** for depth, as long as it stays clearly a diagram (not photorealistic). Use clean spacing. Mid-detail: show denoiser stages, but the δ injection must be clearly **after the denoiser output**.

### Baseline Open-Sora v2.0 (same as Prompt A, mid-detail)
Include the same pipeline blocks as Prompt A, including:
- Sampling loop container with Scheduler and latent update arrows.
- Denoiser box with `vec = f(timestep, p_text)`, two stage boxes, and “vec → AdaLN modulation”.
- Denoiser output labeled `v_pred`.

### Option C injection (main highlight)
- Add a **TRAINABLE** vector `δ_out` in accent blue.
- Add a tiny projection block `P(·)` (accent outline) producing `Δv`.
- Place a `+` node **after** the denoiser output: `v_pred' = v_pred + Δv`.
- Route `v_pred'` into the latent update path in the sampling loop.

### TTA note (small side panel)
“Backbone frozen; optimize δ_out (and tiny P) only; K steps; loss uses only frames 0–32; regularize ||δ_out||.”

