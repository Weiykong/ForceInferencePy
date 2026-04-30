#!/usr/bin/env python3
"""
Generate a VISUAL-HEAVY portrait portfolio poster for ForceInferencePy.
Optimized for Malt/Upwork with a prominent Case Study row showing Raw -> Labels -> Tensions.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "force_inference_portfolio_poster.svg"

WIDTH = 1600
HEIGHT = 2400

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

def visual_workflow_portrait(svg: SVG, x: float, y: float, w: float):
    # Stack panels vertically or horizontal row? In portrait, maybe a horizontal row with arrows
    # Width is 1600. Let's do horizontal row
    panel_w = (w - 80) / 3
    panel_h = 320
    
    # Panels
    for idx, (title, accent, content_type) in enumerate([
        ("Raw Membranes", COLORS["cyan"], "membrane"),
        ("Segmented Labels", COLORS["gold"], "labels"),
        ("Inferred Tensions", COLORS["red"], "tension")
    ]):
        px = x + idx * (panel_w + 40)
        rounded_card(svg, px, y, panel_w, panel_h, title, accent)
        svg.add(f'<g transform="translate({px + 40},{y + 80}) scale(0.95)">')
        
        if content_type == "membrane":
            for _, pts in CELL_DATA:
                svg.add(f'<polygon points="{pts}" fill="#0A1020" stroke="#23314D" stroke-width="6" filter="blur(2px)" opacity="0.6"/>')
        elif content_type == "labels":
            palette = [COLORS["blue"], COLORS["violet"], COLORS["mint"], COLORS["pink"], COLORS["cyan"], COLORS["gold"]]
            for ci, pts in CELL_DATA:
                svg.add(f'<polygon points="{pts}" fill="{palette[ci % 6]}" stroke="#0B1020" stroke-width="1.5" opacity="0.4"/>')
        elif content_type == "tension":
            tension_colors = [COLORS["cyan"], COLORS["gold"], COLORS["red"], COLORS["pink"]]
            for ci, pts in CELL_DATA:
                svg.add(f'<polygon points="{pts}" fill="#0B1020" stroke="{tension_colors[ci % 4]}" stroke-width="{3 + (ci%4)*2}" opacity="0.9"/>')
        
        svg.add('</g>')

def build_poster(out_path: Path):
    svg = SVG(WIDTH, HEIGHT)
    svg.add('<rect width="1600" height="2400" fill="url(#bgGrad)"/>')
    svg.add('<rect width="1600" height="2400" fill="url(#dotGrid)"/>')
    
    margin = 58
    
    # Header
    svg.add(f'<rect x="{margin}" y="{margin}" width="{WIDTH-2*margin}" height="240" rx="36" fill="#0D152A" filter="url(#softShadow)"/>')
    svg.add(f'<rect x="{margin+30}" y="{margin+30}" width="{WIDTH-2*margin-60}" height="14" rx="7" fill="url(#heroGrad)"/>')
    svg.text(margin + 40, margin + 105, "ForceInferencePy", size=76, weight=850)
    svg.text(margin + 40, margin + 155, "Epithelial mechanics from membrane imagery", size=30, fill=COLORS["muted"])
    
    # Visual Showcase
    visual_workflow_portrait(svg, margin, margin + 280, WIDTH - 2*margin)
    
    # Layout Content
    left_x = margin
    left_w = WIDTH - 2*margin
    top_y = margin + 640
    
    # Pipeline
    rounded_card(svg, left_x, top_y, left_w, 400, "Inference Pipeline", COLORS["cyan"])
    # Simplified pipeline for portrait
    for idx, step in enumerate(["1. Segmentation", "2. Label Topology", "3. Curvature", "4. Mechanics", "5. Stress"]):
        sx = left_x + 30 + idx * 300
        svg.text(sx, top_y + 110, step, size=18, weight=700)
        svg.add(f'<rect x="{sx}" y="{top_y+130}" width="280" height="220" rx="16" fill="{COLORS["panel_alt"]}" stroke="{COLORS["stroke"]}"/>')
        # Tiny visual in each box
        svg.add(f'<circle cx="{sx+140}" cy="{top_y+240}" r="40" fill="{COLORS["cyan"]}" opacity="0.1"/>')

    # Features and Equations row
    content_y = top_y + 440
    content_w = (WIDTH - 2*margin - 40) / 2
    
    rounded_card(svg, left_x, content_y, content_w, 600, "Core Features", COLORS["gold"])
    bullet_list_portrait(svg, left_x + 30, content_y + 110, [
        "Hybrid segmentation: Neural (Cellpose) + Grayscale.",
        "Label-driven topology: Preserves junctions.",
        "Multi-modal solvers: Bayesian + Young-Laplace.",
        "Stress Analysis: Per-cell stress tensors.",
        "2.5D support for Z-coordinate vertex mapping."
    ], size=18)

    rounded_card(svg, left_x + content_w + 40, content_y, content_w, 600, "Key Equations", COLORS["red"])
    # Equation snippets
    eqs = [
        ("Force Balance", "Σ T_e·u_e + Σ P_c·n_vc = 0"),
        ("Bayesian Inverse", "x* = argmin ||Ax||² + μ||Bx-g||²"),
        ("Young-Laplace", "ΔP = T · κ"),
        ("Batchelor Stress", "σ = -P·I + (1/A)Σ T·L (u⊗u)")
    ]
    for idx, (title, eq) in enumerate(eqs):
        ey = content_y + 110 + idx * 125
        svg.text(left_x + content_w + 70, ey, title, size=18, fill=COLORS["cyan"], weight=700)
        svg.text(left_x + content_w + 70, ey + 40, eq, size=20, weight=700, family="'IBM Plex Mono', monospace")

    # Repo tree and bottom bar
    tree_y = content_y + 640
    rounded_card(svg, left_x, tree_y, left_w, 400, "Repository Structure", COLORS["violet"])
    # Simple tree
    for idx, entry in enumerate(["force_inference/", "  segmentation.py", "  topology_label.py", "  solvers.py", "examples/", "tests/"]):
        svg.text(left_x + 40, tree_y + 110 + idx * 45, entry, size=20, family="'IBM Plex Mono', monospace")

    # Footer
    svg.add(f'<rect x="{margin}" y="{HEIGHT-140}" width="{WIDTH-2*margin}" height="80" rx="20" fill="#0D152A"/>')
    svg.text(margin + 30, HEIGHT-92, "Visual Case Study: 1024x1024 input processed in < 2.0s", size=22, weight=700)
    svg.text(WIDTH-margin-30, HEIGHT-92, "github.com/weiyuankong/ForceInferencePy", size=22, fill=COLORS["cyan"], anchor="end")

    svg.save(out_path)
    print(f"Portrait poster generated: {out_path}")

def bullet_list_portrait(svg, x, y, items, size=16):
    cursor = y
    for item in items:
        lines = wrap_lines(item, 38)
        svg.add(f'<circle cx="{x+7}" cy="{cursor-6}" r="4" fill="{COLORS["gold"]}"/>')
        svg.multiline(x + 25, cursor, lines, size=size, weight=450)
        cursor += max(36, size * 1.35 * len(lines) + 12)

if __name__ == "__main__":
    build_poster(DEFAULT_OUT)
