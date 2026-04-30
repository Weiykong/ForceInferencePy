#!/usr/bin/env python3
"""
Generate a VISUAL-HEAVY landscape portfolio poster for ForceInferencePy.
Optimized for Malt/Upwork with a prominent Case Study row showing Raw -> Labels -> Tensions.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "force_inference_portfolio_poster_landscape.svg"

WIDTH = 2400
HEIGHT = 1600

COLORS = {
    "bg": "#090D1A",
    "panel": "#121A2E",
    "panel_alt": "#0E1527",
    "stroke": "#26334F",
    "grid": "#1A2340",
    "text": "#ECF2FF",
    "muted": "#9EB1D5",
    "gold": "#FFB84D",
    "cyan": "#57D7FF",
    "mint": "#7EF7C7",
    "pink": "#FF7AC8",
    "red": "#FF6B6B",
    "blue": "#6CA7FF",
    "violet": "#A98BFF",
    "ink": "#060912",
}

# Detailed cell mosaic coordinates for 24 cells
# Formatted as (color_index, points_str)
CELL_DATA = [
    (0, "40,10 80,5 110,30 95,70 50,75 20,45"),
    (1, "110,30 150,20 180,50 160,90 120,100 95,70"),
    (2, "180,50 220,40 250,70 230,110 190,120 160,90"),
    (3, "250,70 290,60 320,90 300,130 260,140 230,110"),
    (4, "20,45 50,75 40,115 0,125 -20,95 -10,60"),
    (5, "50,75 95,70 85,110 40,115"),
    (0, "95,70 120,100 110,140 70,150 85,110"),
    (1, "120,100 160,90 190,120 180,160 140,170 110,140"),
    (2, "190,120 230,110 260,140 250,180 210,190 180,160"),
    (3, "260,140 300,130 330,160 310,200 270,210 250,180"),
    (4, "0,125 40,115 30,155 -10,165 -30,135"),
    (5, "40,115 85,110 70,150 30,155"),
    (0, "70,150 110,140 100,180 60,190 30,155"),
    (1, "110,140 140,170 130,210 90,220 100,180"),
    (2, "140,170 180,160 210,190 200,230 160,240 130,210"),
    (3, "210,190 250,180 280,210 270,250 230,260 200,230"),
    (4, "-10,165 30,155 20,195 -20,205 -40,175"),
    (5, "30,155 60,190 50,230 10,235 20,195"),
    (0, "60,190 100,180 90,220 50,230"),
    (1, "100,180 130,210 120,250 80,260 90,220"),
    (2, "130,210 160,240 150,280 110,290 120,250"),
    (3, "160,240 200,230 230,260 220,300 180,310 150,280"),
    (4, "10,235 50,230 40,270 0,275 -10,245"),
    (5, "50,230 90,220 80,260 40,270"),
]

def esc(text: str) -> str:
    return html.escape(text, quote=True)

def fmt_attrs(**attrs: object) -> str:
    bits = []
    for key, value in attrs.items():
        if value is None:
            continue
        name = key.rstrip("_").replace("__", ":").replace("_", "-")
        bits.append(f'{name}="{esc(str(value))}"')
    return " ".join(bits)

def wrap_lines(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    count = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and count + extra > max_chars:
            lines.append(" ".join(current))
            current = [word]
            count = len(word)
        else:
            current.append(word)
            count += extra
    if current:
        lines.append(" ".join(current))
    return lines

class SVG:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def text(self, x, y, text, size=28, fill=COLORS["text"], weight=400, family="'IBM Plex Sans', sans-serif", anchor="start", opacity=None):
        attrs = fmt_attrs(x=f"{x:.1f}", y=f"{y:.1f}", fill=fill, font_size=size, font_weight=weight, font_family=family, text_anchor=anchor, opacity=opacity)
        self.add(f'<text {attrs}>{esc(text)}</text>')

    def multiline(self, x, y, lines, size=24, fill=COLORS["text"], weight=400, family="'IBM Plex Sans', sans-serif", line_gap=1.32, anchor="start"):
        dy = size * line_gap
        tspans = [f'<tspan x="{x:.1f}" dy="{0 if idx == 0 else dy:.1f}" text-anchor="{anchor}">{esc(line)}</tspan>' for idx, line in enumerate(lines)]
        attrs = fmt_attrs(x=f"{x:.1f}", y=f"{y:.1f}", fill=fill, font_size=size, font_weight=weight, font_family=family)
        self.add(f'<text {attrs}>' + "".join(tspans) + "</text>")

    def save(self, path: Path) -> None:
        defs = f"""
<defs>
  <linearGradient id="bgGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#060912"/><stop offset="100%" stop-color="#0A1020"/></linearGradient>
  <linearGradient id="heroGrad" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#57D7FF"/><stop offset="55%" stop-color="#7EF7C7"/><stop offset="100%" stop-color="#FFB84D"/></linearGradient>
  <linearGradient id="panelGlow" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#141E3A"/><stop offset="100%" stop-color="#0A1022"/></linearGradient>
  <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="12" stdDeviation="15" flood-color="#000" flood-opacity="0.5"/></filter>
  <pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#202A42"/></pattern>
</defs>
"""
        root = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" fill="none" role="img">{defs}{"".join(self.parts)}</svg>'
        path.write_text(root, encoding="utf-8")

def rounded_card(svg: SVG, x, y, w, h, title, accent):
    svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="url(#panelGlow)" stroke="{COLORS["stroke"]}" stroke-width="1.5" filter="url(#softShadow)"/>')
    svg.add(f'<rect x="{x+24}" y="{y+24}" width="60" height="6" rx="3" fill="{accent}"/>')
    svg.text(x + 24, y + 55, title, size=22, weight=700)

def visual_workflow(svg: SVG, x: float, y: float, w: float):
    panel_w = (w - 80) / 3
    panel_h = 320
    
    # 1. Raw Membrane
    rounded_card(svg, x, y, panel_w, panel_h, "1. Input: Raw Membranes", COLORS["cyan"])
    svg.add(f'<g transform="translate({x + 40},{y + 80}) scale(0.95)">')
    for _, pts in CELL_DATA:
        # Simulate noisy membrane
        svg.add(f'<polygon points="{pts}" fill="#0A1020" stroke="#23314D" stroke-width="6" filter="blur(2px)" opacity="0.6"/>')
        svg.add(f'<polygon points="{pts}" fill="none" stroke="#26334F" stroke-width="1.5" opacity="0.4"/>')
    svg.add('</g>')
    
    # 2. Label driven topology
    rounded_card(svg, x + panel_w + 40, y, panel_w, panel_h, "2. Segmented Labels", COLORS["gold"])
    svg.add(f'<g transform="translate({x + panel_w + 80},{y + 80}) scale(0.95)">')
    palette = [COLORS["blue"], COLORS["violet"], COLORS["mint"], COLORS["pink"], COLORS["cyan"], COLORS["gold"]]
    for ci, pts in CELL_DATA:
        svg.add(f'<polygon points="{pts}" fill="{palette[ci % 6]}" stroke="#0B1020" stroke-width="2" opacity="0.3"/>')
    for _, pts in CELL_DATA: # Redraw edges for clarity
        svg.add(f'<polygon points="{pts}" fill="none" stroke="#ECF2FF" stroke-width="1.5" opacity="0.8"/>')
    svg.add('</g>')

    # 3. Tension Map
    rounded_card(svg, x + 2*panel_w + 80, y, panel_w, panel_h, "3. Inferred Tensions", COLORS["red"])
    svg.add(f'<g transform="translate({x + 2*panel_w + 120},{y + 80}) scale(0.95)">')
    # Use different stroke widths/colors for edges to show "tensions"
    tension_colors = [COLORS["cyan"], COLORS["gold"], COLORS["red"], COLORS["pink"]]
    for ci, pts in CELL_DATA:
        svg.add(f'<polygon points="{pts}" fill="#0B1020" stroke="{tension_colors[ci % 4]}" stroke-width="{3 + (ci%4)*2}" stroke-linejoin="round" opacity="0.9"/>')
    svg.add('</g>')
    
    # Arrows
    for ax in [x + panel_w + 10, x + 2*panel_w + 50]:
        svg.add(f'<path d="M {ax} {y + panel_h/2} L {ax + 20} {y + panel_h/2} M {ax + 12} {y + panel_h/2 - 8} L {ax + 20} {y + panel_h/2} L {ax + 12} {y + panel_h/2 + 8}" stroke="{COLORS["muted"]}" stroke-width="3" stroke-linecap="round"/>')

def equation_block(svg: SVG, x, y, w, title, eq, note, accent):
    svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="160" rx="20" fill="{COLORS["panel_alt"]}" stroke="{COLORS["stroke"]}" stroke-width="1.2"/>')
    svg.text(x + 18, y + 40, title, size=18, fill=accent, weight=700)
    svg.text(x + 18, y + 80, eq, size=24, weight=700, family="'IBM Plex Mono', monospace")
    svg.multiline(x + 18, y + 115, wrap_lines(note, 45), size=15, fill=COLORS["muted"])

def bullet_list(svg, x, y, items, size=16, max_chars=45):
    cursor = y
    for item in items:
        lines = wrap_lines(item, max_chars)
        svg.add(f'<circle cx="{x+7}" cy="{cursor-6}" r="3.5" fill="{COLORS["gold"]}"/>')
        svg.multiline(x + 20, cursor, lines, size=size, weight=450)
        cursor += max(32, size * 1.35 * len(lines) + 10)

def draw_pipeline(svg, x, y, w):
    node_h = 135
    gap = 20
    steps = [
        ("1", "Segmentation", "Cellpose-first path with grayscale fallback.", COLORS["cyan"]),
        ("2", "Topology", "Label-driven extraction avoids skeleton artifacts.", COLORS["gold"]),
        ("3", "Splitting", "High-degree junction replacement for stability.", COLORS["pink"]),
        ("4", "Geometry", "Circle fitting and analytical tangent calculation.", COLORS["mint"]),
        ("5", "Solvers", "Bayesian and Young-Laplace inverse mechanics.", COLORS["red"]),
        ("6", "Post-Process", "Per-cell stress tensors and 2.5D vertex maps.", COLORS["violet"]),
    ]
    for idx, (step, title, desc, accent) in enumerate(steps):
        py = y + idx * (node_h + gap)
        svg.add(f'<rect x="{x}" y="{py}" width="{w}" height="{node_h}" rx="22" fill="{COLORS["panel_alt"]}" stroke="{COLORS["stroke"]}" stroke-width="1"/>')
        svg.add(f'<circle cx="{x+30}" cy="{py+30}" r="15" fill="{accent}" opacity="0.2"/>')
        svg.text(x+30, py+36, step, size=16, fill=accent, weight=800, anchor="middle")
        svg.text(x+55, py+37, title, size=20, weight=700)
        svg.multiline(x+55, py+68, wrap_lines(desc, 36), size=14, fill=COLORS["muted"])
        if idx < len(steps) - 1:
            svg.add(f'<line x1="{x + w/2}" y1="{py + node_h}" x2="{x + w/2}" y2="{py + node_h + gap}" stroke="{COLORS["stroke"]}" stroke-width="2"/>')

def build_poster(out_path: Path):
    svg = SVG(WIDTH, HEIGHT)
    svg.add('<rect width="2400" height="1600" fill="url(#bgGrad)"/>')
    svg.add('<rect width="2400" height="1600" fill="url(#dotGrid)"/>')
    
    margin = 60
    
    # Header
    svg.add(f'<rect x="{margin}" y="{margin}" width="{WIDTH-2*margin}" height="220" rx="32" fill="#0D152A" filter="url(#softShadow)"/>')
    svg.add(f'<rect x="{margin+30}" y="{margin+30}" width="{WIDTH-2*margin-60}" height="12" rx="6" fill="url(#heroGrad)"/>')
    svg.text(margin + 40, margin + 95, "ForceInferencePy", size=72, weight=850)
    svg.text(margin + 40, margin + 140, "Robust epithelial mechanics from membrane imagery", size=28, fill=COLORS["muted"])
    
    # Showcase row
    visual_workflow(svg, margin, margin + 250, WIDTH - 2*margin)
    
    # Content Columns
    col_w = (WIDTH - 2*margin - 60) / 3
    top_y = margin + 600
    
    # Col 1: Features & Stats
    rounded_card(svg, margin, top_y, col_w, 420, "Core Capabilities", COLORS["gold"])
    bullet_list(svg, margin + 25, top_y + 100, [
        "Hybrid segmentation: Neural (Cellpose) + Grayscale fallback.",
        "Label-driven topology: Preserves junctions lost in skeletonization.",
        "Multi-modal solvers: Bayesian inference and Young-Laplace balance.",
        "Detailed Stress Analysis: Per-cell Batchelor stress tensors.",
        "2.5D Vertex Mapping: Support for height-aware tissue graphs."
    ])
    
    rounded_card(svg, margin, top_y + 450, col_w, 400, "Repository Metrics", COLORS["cyan"])
    stat_w = (col_w - 60) / 2
    for idx, (val, label, col) in enumerate([("12+", "Modules", "cyan"), ("8", "Examples", "mint"), ("30+", "Test cases", "pink"), ("100%", "Python", "gold")]):
        sx = margin + 20 + (idx % 2) * (stat_w + 20)
        sy = top_y + 520 + (idx // 2) * 150
        svg.add(f'<rect x="{sx}" y="{sy}" width="{stat_w}" height="120" rx="16" fill="{COLORS["panel_alt"]}" stroke="{COLORS["stroke"]}"/>')
        svg.text(sx + 20, sy + 55, val, size=38, fill=COLORS[col], weight=800)
        svg.text(sx + 20, sy + 85, label, size=16, fill=COLORS["muted"])

    # Col 2: Pipeline
    rounded_card(svg, margin + col_w + 30, top_y, col_w, 850, "End-to-end Pipeline", COLORS["cyan"])
    draw_pipeline(svg, margin + col_w + 48, top_y + 85, col_w - 36)

    # Col 3: Equations
    rounded_card(svg, margin + 2*col_w + 60, top_y, col_w, 850, "Mathematical Implementation", COLORS["red"])
    eq_y = top_y + 85
    equation_block(svg, margin + 2*col_w + 75, eq_y, col_w - 30, "Force Balance", "Σ T_e·u_e + Σ P_c·n_vc = 0", "Equilibrium at junctions couples edges and cells.", COLORS["cyan"])
    equation_block(svg, margin + 2*col_w + 75, eq_y + 180, col_w - 30, "Bayesian Inverse", "x* = argmin ||Ax||² + μ||Bx-g||²", "Regularized solver for indeterminate systems.", COLORS["gold"])
    equation_block(svg, margin + 2*col_w + 75, eq_y + 360, col_w - 30, "Young-Laplace Law", "ΔP = T · κ", "Interface curvature fits inferred tension field.", COLORS["pink"])
    equation_block(svg, margin + 2*col_w + 75, eq_y + 540, col_w - 30, "Batchelor Stress", "σ = -P·I + (1/A)Σ T·L (u⊗u)", "Summarizes cellular stress from edge forces.", COLORS["mint"])

    # Footer Bar
    bar_y = HEIGHT - 120
    svg.add(f'<rect x="{margin}" y="{bar_y}" width="{WIDTH-2*margin}" height="80" rx="20" fill="#0D152A" stroke="{COLORS["stroke"]}"/>')
    svg.text(margin + 30, bar_y + 48, "Visual Case Study: 1024x1024 test image processed in < 2s", size=20, fill=COLORS["text"], weight=700)
    svg.text(WIDTH - margin - 30, bar_y + 48, "GitHub: github.com/weiyuankong/ForceInferencePy", size=20, fill=COLORS["cyan"], weight=700, anchor="end")

    svg.save(out_path)
    print(f"Portfolio poster generated: {out_path}")

if __name__ == "__main__":
    build_poster(DEFAULT_OUT)
