from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle


@dataclass(frozen=True)
class Style:
    # Global
    font_family: str = "DejaVu Serif"
    font_size: int = 9
    title_size: int = 10
    line_width: float = 1.0

    # Colors (paper-friendly)
    black: str = "#111111"
    gray: str = "#666666"
    light_gray: str = "#BBBBBB"
    accent: str = "#1f5cff"  # used for δ paths only

    # Fill (keep very subtle)
    box_face: str = "#FFFFFF"

    # Wrapping
    title_wrap_chars: int = 52


def setup_matplotlib(style: Style) -> None:
    mpl.rcParams.update(
        {
            "font.family": style.font_family,
            "font.size": style.font_size,
            "axes.linewidth": 0.0,
            "pdf.fonttype": 42,  # embed TrueType
            "ps.fonttype": 42,
        }
    )


def wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(s, width=width))


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str = ""

    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def right_mid(self) -> Tuple[float, float]:
        return (self.x + self.w, self.y + self.h / 2)

    def left_mid(self) -> Tuple[float, float]:
        return (self.x, self.y + self.h / 2)

    def top_mid(self) -> Tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h)

    def bottom_mid(self) -> Tuple[float, float]:
        return (self.x + self.w / 2, self.y)


def draw_box(
    ax,
    box: Box,
    style: Style,
    edge: Optional[str] = None,
    face: Optional[str] = None,
    label: Optional[str] = None,
    fontsize: Optional[int] = None,
    align: str = "center",
) -> None:
    edge = edge or style.black
    face = face or style.box_face
    ax.add_patch(
        Rectangle(
            (box.x, box.y),
            box.w,
            box.h,
            linewidth=style.line_width,
            edgecolor=edge,
            facecolor=face,
        )
    )
    text = label if label is not None else box.label
    if text:
        cx, cy = box.center()
        ha = "center" if align == "center" else "left"
        ax.text(cx, cy, text, ha=ha, va="center", fontsize=fontsize or style.font_size)


def draw_stack(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    n: int,
    style: Style,
    label: str,
    edge: Optional[str] = None,
    gap: float = 0.9,
) -> Box:
    edge = edge or style.black
    # draw n small boxes stacked vertically
    total_gap = gap * (n - 1)
    each_h = (h - total_gap) / n
    for i in range(n):
        yy = y + i * (each_h + gap)
        ax.add_patch(
            Rectangle((x, yy), w, each_h, linewidth=style.line_width, edgecolor=edge, facecolor=style.box_face)
        )
    ax.text(x + w / 2, y + h + 1.2, label, ha="center", va="bottom", fontsize=style.font_size)
    return Box(x=x, y=y, w=w, h=h, label=label)


def draw_arrow(
    ax,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    style: Style,
    color: Optional[str] = None,
    label: Optional[str] = None,
    label_offset: Tuple[float, float] = (0.0, 0.0),
    connectionstyle: str = "arc3",
) -> None:
    color = color or style.black
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=style.line_width,
            color=color,
            connectionstyle=connectionstyle,
        )
    )
    if label:
        mx = (p0[0] + p1[0]) / 2 + label_offset[0]
        my = (p0[1] + p1[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=style.font_size, color=color)


def draw_plus_node(ax, xy: Tuple[float, float], r: float, style: Style, color: Optional[str] = None) -> None:
    color = color or style.black
    ax.add_patch(Circle(xy, r, edgecolor=color, facecolor=style.box_face, linewidth=style.line_width))
    ax.text(xy[0], xy[1], "+", ha="center", va="center", fontsize=style.font_size, color=color)


@dataclass
class BaselineAnchors:
    # External anchors
    vec_point: Tuple[float, float]
    vpred_point: Tuple[float, float]
    mod_inset_point: Tuple[float, float]
    latent_update_in: Tuple[float, float]
    latent_update_out: Tuple[float, float]

    # Boxes for reference
    denoiser_box: Box


def build_baseline_detailed(ax, style: Style, title: str) -> BaselineAnchors:
    """
    Build a detailed Open-Sora v2.0 (conceptual) pipeline:
    VAE encode -> latent init + mask -> sampling loop with scheduler + denoiser -> latent update -> decode
    Denoiser shown as MMDiT with double/single stacks and AdaLN modulation inset.
    """
    # Canvas coords
    W, H = 120.0, 68.0
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_title(wrap(title, style.title_wrap_chars), fontsize=style.title_size, pad=6)

    # Inputs
    x_cond = Box(4, 48, 16, 10, "x_cond\nframes 0–32")
    prompt = Box(4, 34, 16, 10, "prompt")
    draw_box(ax, x_cond, style)
    draw_box(ax, prompt, style)

    # Encoders
    vae_enc = Box(24, 48, 16, 10, "Video VAE\nEncoder")
    text_enc = Box(24, 34, 16, 10, "Text Encoder\n(T5/CLIP)")
    draw_box(ax, vae_enc, style)
    draw_box(ax, text_enc, style)
    draw_arrow(ax, x_cond.right_mid(), vae_enc.left_mid(), style)
    draw_arrow(ax, prompt.right_mid(), text_enc.left_mid(), style)

    # Encoder outputs
    z_cond = Box(42, 50, 10, 6, "z_cond")
    c_text = Box(42, 38, 10, 6, "c_text")
    p_text = Box(42, 31, 10, 6, "p_text")
    draw_box(ax, z_cond, style)
    draw_box(ax, c_text, style)
    draw_box(ax, p_text, style)
    draw_arrow(ax, vae_enc.right_mid(), z_cond.left_mid(), style)
    draw_arrow(ax, text_enc.right_mid(), c_text.left_mid(), style)
    draw_arrow(ax, text_enc.right_mid(), p_text.left_mid(), style, connectionstyle="arc3,rad=-0.25")

    # Latent init + mask
    init = Box(56, 46, 18, 14, "Latent Init\n+ Mask\nz_T")
    draw_box(ax, init, style)
    draw_arrow(ax, z_cond.right_mid(), init.left_mid(), style, label="condition", label_offset=(0, 3))

    # Sampling loop region
    loop = Box(54, 14, 56, 30, "")
    ax.add_patch(
        Rectangle((loop.x, loop.y), loop.w, loop.h, linewidth=style.line_width, edgecolor=style.light_gray, facecolor="none")
    )
    ax.text(loop.x + 1.0, loop.y + loop.h + 1.5, "Sampling loop over timesteps  t = T … 0", ha="left", va="bottom", fontsize=style.font_size)

    # Scheduler
    scheduler = Box(58, 38, 18, 6, "Scheduler")
    draw_box(ax, scheduler, style, edge=style.gray)
    draw_arrow(ax, init.bottom_mid(), (init.bottom_mid()[0], loop.y + loop.h), style, color=style.gray)

    # Denoiser (MMDiT)
    denoiser = Box(78, 16, 30, 26, "Denoiser (MMDiT)")
    draw_box(ax, denoiser, style, fontsize=style.title_size)

    # Denoiser top conditioning (inside denoiser)
    e_t = Box(80, 36, 10, 4, "e_t")
    combine = Box(92, 36, 14, 4, "combine\nvec=f(e_t,p_text)")
    draw_box(ax, e_t, style, edge=style.gray)
    draw_box(ax, combine, style, edge=style.gray, fontsize=style.font_size)
    # p_text to combine
    draw_arrow(ax, p_text.right_mid(), (denoiser.x, p_text.right_mid()[1]), style, color=style.gray)
    draw_arrow(ax, (denoiser.x, p_text.right_mid()[1]), (combine.x, combine.y + combine.h / 2), style, color=style.gray, connectionstyle="arc3,rad=0.15")
    draw_arrow(ax, e_t.right_mid(), combine.left_mid(), style, color=style.gray)

    vec_point = (combine.x + combine.w, combine.y + combine.h / 2)

    # Double/single stacks (conceptual)
    double_stack = draw_stack(
        ax,
        x=81,
        y=24,
        w=12,
        h=10,
        n=3,
        style=style,
        label="N_double\nDouble-stream",
        edge=style.black,
        gap=0.7,
    )
    single_stack = draw_stack(
        ax,
        x=95,
        y=24,
        w=12,
        h=10,
        n=3,
        style=style,
        label="N_single\nSingle-stream",
        edge=style.black,
        gap=0.7,
    )
    # vec to stacks
    draw_arrow(ax, vec_point, (double_stack.x + double_stack.w / 2, double_stack.y + double_stack.h + 1.0), style, color=style.gray)
    draw_arrow(ax, vec_point, (single_stack.x + single_stack.w / 2, single_stack.y + single_stack.h + 1.0), style, color=style.gray, connectionstyle="arc3,rad=-0.15")

    # Modulation inset
    inset = Box(80, 17, 27, 6, "")
    ax.add_patch(
        Rectangle((inset.x, inset.y), inset.w, inset.h, linewidth=style.line_width, edgecolor=style.light_gray, facecolor="none")
    )
    ax.text(inset.x + 0.8, inset.y + inset.h - 0.6, "AdaLN modulation (per block)", ha="left", va="top", fontsize=style.font_size)
    ax.text(inset.x + 1.0, inset.y + 2.0, "vec → Mod MLP → (scale,shift,gate)", ha="left", va="center", fontsize=style.font_size, color=style.gray)
    mod_inset_point = (inset.x + 1.5, inset.y + 2.0)

    # Denoiser head output
    v_pred = Box(108, 26, 10, 6, "v_pred")
    draw_box(ax, v_pred, style)
    vpred_point = v_pred.left_mid()
    draw_arrow(ax, (denoiser.x + denoiser.w, denoiser.y + denoiser.h / 2), v_pred.left_mid(), style)

    # Latent update
    update = Box(96, 6, 18, 8, "Latent update\n(z_t→z_{t-1})")
    draw_box(ax, update, style, edge=style.gray)
    latent_update_in = update.left_mid()
    latent_update_out = update.right_mid()
    draw_arrow(ax, v_pred.bottom_mid(), update.top_mid(), style, color=style.gray)
    draw_arrow(ax, (denoiser.x + denoiser.w / 2, denoiser.y), update.top_mid(), style, color=style.gray, connectionstyle="arc3,rad=-0.35", label="z_t", label_offset=(0, -3))

    # Decode
    vae_dec = Box(116 - 14, 24, 14, 10, "Video VAE\nDecoder")
    draw_box(ax, vae_dec, style)
    x_out = Box(116 - 14, 10, 14, 10, "x_out\nframes 0–64\n(33–64 gen)")
    draw_box(ax, x_out, style)
    draw_arrow(ax, latent_update_out, vae_dec.left_mid(), style, color=style.gray)
    draw_arrow(ax, vae_dec.bottom_mid(), x_out.top_mid(), style)

    return BaselineAnchors(
        vec_point=vec_point,
        vpred_point=vpred_point,
        mod_inset_point=mod_inset_point,
        latent_update_in=latent_update_in,
        latent_update_out=latent_update_out,
        denoiser_box=denoiser,
    )


def new_figure(style: Style, figsize: Tuple[float, float] = (7.2, 3.9)):
    setup_matplotlib(style)
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

