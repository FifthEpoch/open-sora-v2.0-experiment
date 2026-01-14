from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

from .diagram_lib import (
    Style,
    Box,
    BaselineAnchors,
    draw_box,
    draw_arrow,
    draw_plus_node,
    wrap,
)


@dataclass(frozen=True)
class OverlayConfig:
    accent_label: str = "δ path"


def _tta_panel(
    ax,
    style: Style,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: Tuple[str, ...],
    accent: Optional[str] = None,
) -> None:
    accent = accent or style.accent
    panel = Box(x, y, w, h, "")
    draw_box(ax, panel, style, edge=style.black, face=style.box_face)
    ax.text(x + 1.0, y + h - 1.2, title, ha="left", va="top", fontsize=style.title_size)
    yy = y + h - 3.2
    for line in lines:
        ax.text(x + 1.2, yy, "• " + line, ha="left", va="top", fontsize=style.font_size)
        yy -= 2.2
    # small accent indicator line
    ax.plot([x + 0.6, x + 0.6], [y + 0.8, y + h - 0.8], color=accent, linewidth=2.0)


def overlay_delta_a(ax, style: Style, anchors: BaselineAnchors, cfg: OverlayConfig = OverlayConfig()) -> None:
    """
    Option A: single global δ added to global conditioning vector vec.
    """
    accent = style.accent

    # δ node near top-right of denoiser
    delta = Box(64, 60, 10, 5, "δ\n(per-video)")
    draw_box(ax, delta, style, edge=accent)

    # plus node to create vec' = vec + δ
    plus_xy = (anchors.vec_point[0] + 4.5, anchors.vec_point[1])
    draw_plus_node(ax, plus_xy, r=1.5, style=style, color=accent)

    # arrows: vec -> plus, δ -> plus, plus -> modulation inset
    draw_arrow(ax, anchors.vec_point, (plus_xy[0] - 1.5, plus_xy[1]), style, color=accent, label="vec")
    draw_arrow(ax, delta.bottom_mid(), (plus_xy[0], plus_xy[1] + 1.5), style, color=accent, label="δ")
    # vec' to modulation inset and into blocks (conceptual)
    draw_arrow(ax, (plus_xy[0] + 1.5, plus_xy[1]), (anchors.mod_inset_point[0], anchors.mod_inset_point[1]), style, color=accent, label="vec'")

    # small annotation near modulation inset
    ax.text(
        anchors.mod_inset_point[0] + 30,
        anchors.mod_inset_point[1],
        "use vec' in all\nmodulation MLPs",
        ha="right",
        va="center",
        fontsize=style.font_size,
        color=accent,
    )

    _tta_panel(
        ax,
        style,
        x=6,
        y=6,
        w=42,
        h=12,
        title="Test-Time Adaptation (Option A)",
        lines=(
            "freeze backbone weights",
            "optimize δ only (K steps per video)",
            "loss on conditioning frames 0–32 only",
        ),
        accent=accent,
    )


def overlay_delta_b(ax, style: Style, anchors: BaselineAnchors, cfg: OverlayConfig = OverlayConfig()) -> None:
    """
    Option B: grouped per-layer δ offsets added to modulation input per block group.
    """
    accent = style.accent

    # δ group stack
    stack = Box(62, 57, 14, 10, "")
    draw_box(ax, stack, style, edge=accent, label="{δ_g}\n(4 groups)")
    # individual labels
    ax.text(stack.x + stack.w + 1.2, stack.y + 8.4, "δ_1", color=accent, fontsize=style.font_size, va="center")
    ax.text(stack.x + stack.w + 1.2, stack.y + 6.1, "δ_2", color=accent, fontsize=style.font_size, va="center")
    ax.text(stack.x + stack.w + 1.2, stack.y + 3.8, "δ_3", color=accent, fontsize=style.font_size, va="center")
    ax.text(stack.x + stack.w + 1.2, stack.y + 1.5, "δ_4", color=accent, fontsize=style.font_size, va="center")

    # concept: vec_i' = vec + δ_group(i)
    plus_xy = (anchors.vec_point[0] + 4.5, anchors.vec_point[1])
    draw_plus_node(ax, plus_xy, r=1.5, style=style, color=accent)
    draw_arrow(ax, anchors.vec_point, (plus_xy[0] - 1.5, plus_xy[1]), style, color=accent, label="vec")
    draw_arrow(ax, (stack.x + stack.w / 2, stack.y), (plus_xy[0], plus_xy[1] + 1.5), style, color=accent, label="δ_group(i)")
    draw_arrow(ax, (plus_xy[0] + 1.5, plus_xy[1]), anchors.mod_inset_point, style, color=accent, label="vec_i'")

    # group assignment legend
    legend = Box(6, 20, 42, 18, "")
    draw_box(ax, legend, style, edge=style.black)
    ax.text(legend.x + 1.0, legend.y + legend.h - 1.2, "Grouped depth offsets", ha="left", va="top", fontsize=style.title_size)
    lines = (
        "early double-stream blocks → δ_1",
        "mid double-stream blocks → δ_2",
        "late double-stream blocks → δ_3",
        "single-stream blocks → δ_4",
    )
    yy = legend.y + legend.h - 3.4
    for line in lines:
        ax.text(legend.x + 1.2, yy, "• " + line, ha="left", va="top", fontsize=style.font_size)
        yy -= 2.1
    ax.text(legend.x + 1.2, legend.y + 1.6, "Apply: vec_i' = vec + δ_group(i)", ha="left", va="center", fontsize=style.font_size, color=accent)
    ax.plot([legend.x + 0.6, legend.x + 0.6], [legend.y + 0.8, legend.y + legend.h - 0.8], color=accent, linewidth=2.0)

    _tta_panel(
        ax,
        style,
        x=6,
        y=6,
        w=42,
        h=12,
        title="Test-Time Adaptation (Option B)",
        lines=(
            "freeze backbone weights",
            "optimize {δ_g} only (K steps per video)",
            "loss on conditioning frames 0–32 only",
        ),
        accent=accent,
    )


def overlay_delta_c(ax, style: Style, anchors: BaselineAnchors, cfg: OverlayConfig = OverlayConfig()) -> None:
    """
    Option C: δ_out projected to Δv and added to denoiser output v_pred.
    """
    accent = style.accent

    # δ_out and projection P(·)
    delta = Box(64, 6, 12, 6, "δ_out\n(per-video)")
    proj = Box(80, 6, 10, 6, "P(·)")
    draw_box(ax, delta, style, edge=accent)
    draw_box(ax, proj, style, edge=accent)
    draw_arrow(ax, delta.right_mid(), proj.left_mid(), style, color=accent)

    # plus node at the v_pred -> latent update path
    plus_xy = (anchors.vpred_point[0] + 6.0, anchors.vpred_point[1])
    draw_plus_node(ax, plus_xy, r=1.5, style=style, color=accent)
    # v_pred into plus
    draw_arrow(ax, anchors.vpred_point, (plus_xy[0] - 1.5, plus_xy[1]), style, color=accent, label="v_pred")
    # Δv into plus
    draw_arrow(ax, proj.top_mid(), (plus_xy[0], plus_xy[1] + 1.5), style, color=accent, label="Δv")
    # corrected output to latent update input
    draw_arrow(ax, (plus_xy[0] + 1.5, plus_xy[1]), anchors.latent_update_in, style, color=accent, label="v_pred'")

    ax.text(
        plus_xy[0],
        plus_xy[1] - 4.0,
        "output correction\nv_pred' = v_pred + Δv",
        ha="center",
        va="top",
        fontsize=style.font_size,
        color=accent,
    )

    _tta_panel(
        ax,
        style,
        x=6,
        y=6,
        w=42,
        h=12,
        title="Test-Time Adaptation (Option C)",
        lines=(
            "freeze backbone weights",
            "optimize δ_out (and tiny P) only",
            "loss on conditioning frames 0–32 only",
            "regularize ||δ_out|| for stability",
        ),
        accent=accent,
    )

