"""
Script: generate_figure1.py
Description: Generates a publication-quality 3D schematic diagram (Figure 1)
calibrated for exactly 15 cm (6.0 inches) printed width.
All typography is strictly Times New Roman (matching manuscript text).
True 3D camera projection (Azimuth 32 deg, Elevation 22 deg) ensures
realistic, unambiguous 3D perspective and physical volume.

Subpanels:
  (a) Cross-section geometry with zero overlap on b(x) >= h(x) constraint
  (b) Investigated boundary conditions (C-F and C-P)
  (c) AFG material gradation bar (Al to SiC)

Output: 05_Drafts/figures/figure1_afg_microbeam_schematic.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
import matplotlib.colors as mcolors

os.makedirs("05_Drafts/figures", exist_ok=True)
out_png = os.path.abspath("05_Drafts/figures/figure1_afg_microbeam_schematic.png")

# Publication styling: Times New Roman, 15 cm width calibration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# 15.0 cm width = 6.0 in, height = 4.4 in
fig = plt.figure(figsize=(6.0, 4.4), dpi=300, facecolor="white")

# ==============================================================================
# MAIN 3D SCHEMATIC (Upper 58% of canvas)
# ==============================================================================
ax_main = fig.add_axes([0.02, 0.40, 0.96, 0.58])
ax_main.set_aspect("equal")
ax_main.axis("off")

# 3D Beam Geometry
L = 6.2
b0, h0 = 1.35, 0.85   # Root width and height
bL, hL = 0.70, 0.42   # Tip width and height

def b_fun(x):
    xi = x / L
    return b0 - (b0 - bL) * (1.15 * xi - 0.15 * xi**2)

def h_fun(x):
    xi = x / L
    return h0 - (h0 - hL) * (1.10 * xi - 0.10 * xi**2)

# Camera Basis Matrix (Azimuth = 32 deg, Elevation = 22 deg)
az_deg, el_deg = 32.0, 22.0
az = np.radians(az_deg)
el = np.radians(el_deg)
dist = 20.0
target = np.array([L * 0.48, 0.0, 0.0])

cam_pos = target + dist * np.array([np.sin(az)*np.cos(el), -np.cos(az)*np.cos(el), np.sin(el)])

forward = target - cam_pos
forward = forward / np.linalg.norm(forward)

up_world = np.array([0.0, 0.0, 1.0])
right = np.cross(forward, up_world)
right = right / np.linalg.norm(right)

true_up = np.cross(right, forward)
true_up = true_up / np.linalg.norm(true_up)

def proj(p3d):
    """Orthographic projection onto camera plane."""
    d = p3d - cam_pos
    u = np.dot(d, right)
    v = np.dot(d, true_up)
    return np.array([u, v])

# Colormap: Al (Light metallic slate/silver) at x=0 -> SiC (Deep ceramic navy) at x=L
colors_list = ["#E2E8F0", "#CBD5E1", "#94A3B8", "#64748B", "#334155", "#0F172A"]
cmap = mcolors.LinearSegmentedColormap.from_list("afg_metal_ceramic", colors_list, N=256)

# 1. Clamped Wall Support at x = 0
wy_min, wy_max = -b0 * 1.05, b0 * 1.05
wz_min, wz_max = -h0 * 1.35, h0 * 1.55

w_p1 = proj(np.array([0.0, wy_min, wz_min]))
w_p2 = proj(np.array([0.0, wy_max, wz_min]))
w_p3 = proj(np.array([0.0, wy_max, wz_max]))
w_p4 = proj(np.array([0.0, wy_min, wz_max]))

wall_poly = Polygon([w_p1, w_p2, w_p3, w_p4], closed=True,
                    facecolor="#F8FAFC", edgecolor="#475569", lw=1.3, zorder=1)
ax_main.add_patch(wall_poly)

# Support hatching behind/outside the wall (on left side)
n_hatches = 10
for i in range(n_hatches):
    frac = i / (n_hatches - 1)
    zy = wz_min + frac * (wz_max - wz_min)
    p_wall_edge = proj(np.array([0.0, wy_max, zy]))
    p_hatch_end = p_wall_edge + np.array([0.22, 0.22])
    ax_main.plot([p_wall_edge[0], p_hatch_end[0]], [p_wall_edge[1], p_hatch_end[1]],
                 color="#94A3B8", lw=0.9, zorder=0)

# Discretization along length
N_x = 60
x_vals = np.linspace(0, L, N_x + 1)

# 2. Top Surface (Z = +h/2)
for i in range(N_x):
    x1, x2 = x_vals[i], x_vals[i+1]
    b1, b2 = b_fun(x1), b_fun(x2)
    h1, h2 = h_fun(x1), h_fun(x2)
    
    p1 = proj(np.array([x1, -b1/2, h1/2]))
    p2 = proj(np.array([x2, -b2/2, h2/2]))
    p3 = proj(np.array([x2,  b2/2, h2/2]))
    p4 = proj(np.array([x1,  b1/2, h1/2]))
    
    frac = (x1 + x2) / (2.0 * L)
    c_base = np.array(cmap(frac)[:3])
    # Top face gets bright ambient light
    c_top = np.clip(c_base * 1.15, 0, 1)
    
    ax_main.add_patch(Polygon([p1, p2, p3, p4], closed=True,
                              facecolor=c_top, edgecolor="none", zorder=3))

# 3. Front Face (Y = -b/2 facing camera)
for i in range(N_x):
    x1, x2 = x_vals[i], x_vals[i+1]
    b1, b2 = b_fun(x1), b_fun(x2)
    h1, h2 = h_fun(x1), h_fun(x2)
    
    p1 = proj(np.array([x1, -b1/2, -h1/2]))
    p2 = proj(np.array([x2, -b2/2, -h2/2]))
    p3 = proj(np.array([x2, -b2/2,  h2/2]))
    p4 = proj(np.array([x1, -b1/2,  h1/2]))
    
    frac = (x1 + x2) / (2.0 * L)
    c_base = np.array(cmap(frac)[:3])
    # Front face gets directional shading
    c_front = c_base * 0.78
    
    ax_main.add_patch(Polygon([p1, p2, p3, p4], closed=True,
                              facecolor=c_front, edgecolor="none", zorder=4))

# 4. Tip Face at x = L
pt1 = proj(np.array([L, -bL/2, -hL/2]))
pt2 = proj(np.array([L,  bL/2, -hL/2]))
pt3 = proj(np.array([L,  bL/2,  hL/2]))
pt4 = proj(np.array([L, -bL/2,  hL/2]))

c_tip = np.array(cmap(1.0)[:3]) * 0.62
ax_main.add_patch(Polygon([pt1, pt2, pt3, pt4], closed=True,
                          facecolor=c_tip, edgecolor="#0F172A", lw=1.2, zorder=6))

# Outlines of visible edges
top_back = [proj(np.array([x, b_fun(x)/2, h_fun(x)/2])) for x in x_vals]
top_front = [proj(np.array([x, -b_fun(x)/2, h_fun(x)/2])) for x in x_vals]
bot_front = [proj(np.array([x, -b_fun(x)/2, -h_fun(x)/2])) for x in x_vals]

ax_main.plot([p[0] for p in top_back], [p[1] for p in top_back], color="#475569", lw=1.0, zorder=5)
ax_main.plot([p[0] for p in top_front], [p[1] for p in top_front], color="#0F172A", lw=1.3, zorder=5)
ax_main.plot([p[0] for p in bot_front], [p[1] for p in bot_front], color="#0F172A", lw=1.3, zorder=5)

# Tip centroid and axial load vector P
p_tip_c = proj(np.array([L, 0.0, 0.0]))
p_arrow_start = proj(np.array([L + 1.55, 0.0, 0.0]))

load_arrow = FancyArrowPatch(posA=p_arrow_start, posB=p_tip_c,
                             arrowstyle="simple,head_length=10,head_width=7,tail_width=2.8",
                             color="#DC2626", zorder=12)
ax_main.add_patch(load_arrow)

ax_main.text(p_arrow_start[0] + 0.12, p_arrow_start[1] + 0.04,
             r"$\mathbf{P}$",
             fontsize=12, fontweight="bold", color="#B91C1C", zorder=13, va="center")

# Neutral axis along centerline (dashed red line)
na_pts = [proj(np.array([x, 0.0, 0.0])) for x in np.linspace(0, L, 40)]
ax_main.plot([p[0] for p in na_pts], [p[1] for p in na_pts], color="#B91C1C", ls="-.", lw=1.0, zorder=5)

# Clamped support callout at x = 0
ax_main.annotate("Clamped root\n($x = 0$)",
                 xy=w_p4,
                 xytext=(w_p4[0] - 1.35, w_p4[1] + 0.40),
                 arrowprops=dict(arrowstyle="->", color="#334155", lw=1.0, connectionstyle="arc3,rad=-0.1"),
                 fontsize=8.5, fontweight="bold", color="#1E293B",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8),
                 zorder=12)

# Coordinate System Triad (Clear floating triad at lower left)
triad_origin = w_p1 + np.array([-0.65, -0.60])
axis_len = 0.80

v_x = proj(np.array([1.0, 0.0, 0.0])) - proj(np.array([0.0, 0.0, 0.0]))
v_x = v_x / np.linalg.norm(v_x) * axis_len

v_y = proj(np.array([0.0, 1.0, 0.0])) - proj(np.array([0.0, 0.0, 0.0]))
v_y = v_y / np.linalg.norm(v_y) * (axis_len * 0.70)

v_z = proj(np.array([0.0, 0.0, 1.0])) - proj(np.array([0.0, 0.0, 0.0]))
v_z = v_z / np.linalg.norm(v_z) * axis_len

# Add invisible points to guarantee autoscale includes triad completely
ax_main.plot([triad_origin[0] - 0.2, triad_origin[0] + v_x[0] + 0.2],
             [triad_origin[1] - 0.2, triad_origin[1] + v_z[1] + 0.3], alpha=0)

# X axis
ax_main.add_patch(FancyArrowPatch(posA=triad_origin, posB=triad_origin + v_x,
                                  arrowstyle="-|>,head_length=5,head_width=3.2",
                                  color="#0284C7", lw=1.6, zorder=10))
ax_main.text(triad_origin[0] + v_x[0] + 0.08, triad_origin[1] + v_x[1] - 0.06,
             r"$\mathbf{x}$", fontsize=9.5, fontweight="bold", color="#0369A1", zorder=10)

# Y axis
ax_main.add_patch(FancyArrowPatch(posA=triad_origin, posB=triad_origin + v_y,
                                  arrowstyle="-|>,head_length=5,head_width=3.2",
                                  color="#059669", lw=1.6, zorder=10))
ax_main.text(triad_origin[0] + v_y[0] + 0.08, triad_origin[1] + v_y[1] + 0.06,
             r"$\mathbf{y}$", fontsize=9.5, fontweight="bold", color="#047857", zorder=10)

# Z axis
ax_main.add_patch(FancyArrowPatch(posA=triad_origin, posB=triad_origin + v_z,
                                  arrowstyle="-|>,head_length=5,head_width=3.2",
                                  color="#7C3AED", lw=1.6, zorder=10))
ax_main.text(triad_origin[0] + v_z[0] - 0.04, triad_origin[1] + v_z[1] + 0.12,
             r"$\mathbf{z}$", fontsize=9.5, fontweight="bold", color="#6D28D9", zorder=10)

# Intermediate section at x_int: width b(x) and height h(x)
x_int = 2.7
b_int = b_fun(x_int)
h_int = h_fun(x_int)

p_int_t1 = proj(np.array([x_int,  b_int/2, h_int/2]))
p_int_t2 = proj(np.array([x_int, -b_int/2, h_int/2]))
p_int_f1 = proj(np.array([x_int, -b_int/2, -h_int/2]))

ax_main.plot([p_int_t1[0], p_int_t2[0], p_int_f1[0]],
             [p_int_t1[1], p_int_t2[1], p_int_f1[1]], color="#D97706", lw=1.8, ls="-", zorder=7)

# Dimension callout for b(x)
p_b_mid = 0.5 * (p_int_t1 + p_int_t2)
ax_main.annotate(r"$\mathbf{b(x)}$ (width)",
                 xy=p_b_mid,
                 xytext=(p_b_mid[0] - 0.35, p_b_mid[1] + 0.85),
                 arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.1),
                 fontsize=8.5, fontweight="bold", color="#B45309",
                 bbox=dict(boxstyle="round,pad=0.20", facecolor="#FEF3C7", edgecolor="#FCD34D", lw=0.8),
                 zorder=10)

# Dimension callout for h(x)
p_h_mid = 0.5 * (p_int_t2 + p_int_f1)
ax_main.annotate(r"$\mathbf{h(x)}$ (height)",
                 xy=p_h_mid,
                 xytext=(p_h_mid[0] + 0.45, p_h_mid[1] - 0.60),
                 arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.1),
                 fontsize=8.5, fontweight="bold", color="#B45309",
                 bbox=dict(boxstyle="round,pad=0.20", facecolor="#FEF3C7", edgecolor="#FCD34D", lw=0.8),
                 zorder=10)

# Dimension L (Span) below beam
p_L0 = proj(np.array([0.0, -b0/2, -h0/2])) + np.array([0.0, -0.75])
p_LL = proj(np.array([L,   -bL/2, -hL/2])) + np.array([0.0, -0.75])

ax_main.plot([proj(np.array([0.0, -b0/2, -h0/2]))[0], p_L0[0]],
             [proj(np.array([0.0, -b0/2, -h0/2]))[1], p_L0[1]], color="#64748B", lw=0.6, ls=":")
ax_main.plot([proj(np.array([L,   -bL/2, -hL/2]))[0], p_LL[0]],
             [proj(np.array([L,   -bL/2, -hL/2]))[1], p_LL[1]], color="#64748B", lw=0.6, ls=":")

dim_L = FancyArrowPatch(posA=p_L0, posB=p_LL,
                        arrowstyle="<|-|>,head_length=5.0,head_width=3.2",
                        color="#334155", lw=1.0, zorder=8)
ax_main.add_patch(dim_L)

p_L_mid = 0.5 * (p_L0 + p_LL)
ax_main.text(p_L_mid[0], p_L_mid[1] - 0.24, r"$\mathbf{L}$ (Span)",
             fontsize=9.0, fontweight="bold", color="#1E293B", ha="center")

# Tip annotation at x = L
ax_main.annotate("Tip ($x = L$)",
                 xy=pt3,
                 xytext=(pt3[0] + 0.55, pt3[1] + 0.55),
                 arrowprops=dict(arrowstyle="->", color="#334155", lw=1.0),
                 fontsize=8.5, color="#1E293B", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.20", facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.8),
                 zorder=10)


# ==============================================================================
# SUB-PANEL (a): Cross-Section Geometry (COMPLETELY OVERLAP-FREE)
# ==============================================================================
ax_sub1 = fig.add_axes([0.04, 0.04, 0.28, 0.32])
ax_sub1.set_xlim(-1.8, 1.8)
ax_sub1.set_ylim(-1.6, 1.6)
ax_sub1.set_aspect("equal")
ax_sub1.axis("off")

# Title
ax_sub1.text(0, 1.48, "(a) Cross-Section",
             fontsize=9.2, ha="center", color="#0F172A", fontweight="bold")

# Rectangle positioned slightly above center to leave ample space below
rect_b, rect_h = 2.0, 1.05
y_center = 0.28
rect = Rectangle((-rect_b/2, y_center - rect_h/2), rect_b, rect_h,
                 facecolor="#F1F5F9", edgecolor="#1E293B", lw=1.3, zorder=2)
ax_sub1.add_patch(rect)

# Axes in cross section (clipped neatly at the rectangle borders)
ax_sub1.plot([-rect_b/2, rect_b/2 + 0.35], [y_center, y_center], color="#991B1B", ls="-.", lw=1.0, zorder=3)
ax_sub1.plot([0, 0], [y_center - rect_h/2, y_center + rect_h/2 + 0.38], color="#7C3AED", ls="-.", lw=1.0, zorder=3)

ax_sub1.text(rect_b/2 + 0.42, y_center, r"$y$", fontsize=9.0, va="center", ha="left", color="#991B1B", fontweight="bold")
ax_sub1.text(0, y_center + rect_h/2 + 0.44, r"$z$", fontsize=9.0, ha="center", va="bottom", color="#7C3AED", fontweight="bold")

# Width b(x) dimension at TOP (centered with white halo)
y_dim_b = y_center + rect_h/2 + 0.16
ax_sub1.plot([-rect_b/2, -rect_b/2], [y_center + rect_h/2, y_dim_b + 0.08], color="#64748B", lw=0.6, ls=":")
ax_sub1.plot([rect_b/2, rect_b/2], [y_center + rect_h/2, y_dim_b + 0.08], color="#64748B", lw=0.6, ls=":")
ax_sub1.annotate("", xy=(-rect_b/2, y_dim_b), xytext=(rect_b/2, y_dim_b),
                 arrowprops=dict(arrowstyle="<->", color="#334155", lw=0.9), zorder=4)
ax_sub1.text(0.0, y_dim_b, r"$b(x)$", fontsize=9.0, ha="center", va="center", fontweight="bold", color="#1E293B",
             bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"), zorder=5)

# Height h(x) dimension on LEFT
x_dim_h = -rect_b/2 - 0.20
ax_sub1.plot([-rect_b/2, x_dim_h - 0.06], [y_center - rect_h/2, y_center - rect_h/2], color="#64748B", lw=0.6, ls=":")
ax_sub1.plot([-rect_b/2, x_dim_h - 0.06], [y_center + rect_h/2, y_center + rect_h/2], color="#64748B", lw=0.6, ls=":")
ax_sub1.annotate("", xy=(x_dim_h, y_center - rect_h/2), xytext=(x_dim_h, y_center + rect_h/2),
                 arrowprops=dict(arrowstyle="<->", color="#334155", lw=0.9))
ax_sub1.text(x_dim_h - 0.08, y_center, r"$h(x)$", fontsize=9.0, va="center", ha="right", fontweight="bold", color="#1E293B")

# In-plane buckling constraint box positioned cleanly at bottom with ZERO overlap
ax_sub1.text(0, -1.18,
             r"$b(x) \geq h(x)$" + "\n" + r"(In-plane buckling)",
             fontsize=8.5, ha="center", color="#1E293B", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8))


# ==============================================================================
# SUB-PANEL (b): Investigated Boundary Conditions
# ==============================================================================
ax_sub2 = fig.add_axes([0.36, 0.04, 0.31, 0.32])
ax_sub2.set_xlim(-0.6, 4.8)
ax_sub2.set_ylim(-1.6, 1.6)
ax_sub2.axis("off")

ax_sub2.text(2.1, 1.48, "(b) Boundary Conditions",
             fontsize=9.2, ha="center", color="#0F172A", fontweight="bold")

# 1. Clamped-Free (C-F)
y_cf = 0.40
ax_sub2.text(1.7, y_cf + 0.36, "Clamped–Free (C–F)", fontsize=8.5, ha="center", color="#1E293B", fontweight="bold")
ax_sub2.plot([0, 0], [y_cf - 0.24, y_cf + 0.24], color="#1E293B", lw=2.4)
for k in range(4):
    yw = y_cf - 0.18 + k * 0.12
    ax_sub2.plot([-0.20, 0], [yw - 0.08, yw], color="#64748B", lw=0.9)
ax_sub2.plot([0, 3.4], [y_cf, y_cf], color="#334155", lw=2.6)
ax_sub2.annotate("", xy=(3.4, y_cf), xytext=(4.2, y_cf),
                 arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=1.8, mutation_scale=11))
ax_sub2.text(4.35, y_cf, r"$P$", fontsize=10.5, fontweight="bold", color="#DC2626", va="center")

# 2. Clamped-Pinned (C-P)
y_cp = -0.62
ax_sub2.text(1.7, y_cp + 0.36, "Clamped–Pinned (C–P)", fontsize=8.5, ha="center", color="#1E293B", fontweight="bold")
ax_sub2.plot([0, 0], [y_cp - 0.24, y_cp + 0.24], color="#1E293B", lw=2.4)
for k in range(4):
    yw = y_cp - 0.18 + k * 0.12
    ax_sub2.plot([-0.20, 0], [yw - 0.08, yw], color="#64748B", lw=0.9)
ax_sub2.plot([0, 3.4], [y_cp, y_cp], color="#334155", lw=2.6)

# Pinned support at x=L
tri_x = [3.4, 3.25, 3.55]
tri_y = [y_cp, y_cp - 0.20, y_cp - 0.20]
ax_sub2.add_patch(Polygon(list(zip(tri_x, tri_y)), closed=True, facecolor="#64748B", edgecolor="#1E293B", lw=0.9))
ax_sub2.plot([3.15, 3.65], [y_cp - 0.23, y_cp - 0.23], color="#1E293B", lw=1.3)

ax_sub2.annotate("", xy=(3.4, y_cp), xytext=(4.2, y_cp),
                 arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=1.8, mutation_scale=11))
ax_sub2.text(4.35, y_cp, r"$P$", fontsize=10.5, fontweight="bold", color="#DC2626", va="center")


# ==============================================================================
# SUB-PANEL (c): Axially Functionally Graded (AFG) Material
# ==============================================================================
ax_sub3 = fig.add_axes([0.68, 0.04, 0.30, 0.32])
ax_sub3.set_xlim(-0.2, 4.4)
ax_sub3.set_ylim(-1.6, 1.6)
ax_sub3.axis("off")

ax_sub3.text(2.1, 1.48, "(c) Material Gradation",
             fontsize=9.2, ha="center", color="#0F172A", fontweight="bold")

# Arrow above gradient bar showing direction of increasing ceramic
ax_sub3.annotate("", xy=(3.9, 0.82), xytext=(0.3, 0.82),
                 arrowprops=dict(arrowstyle="->", color="#0F172A", lw=1.2))
ax_sub3.text(2.1, 0.94, r"Ceramic $V_r(x)$ increases $\rightarrow$",
             fontsize=8.2, ha="center", va="bottom", color="#0F172A", fontweight="bold")

# Gradient bar
bar_x0, bar_y0, bar_w, bar_h = 0.2, 0.20, 3.8, 0.38
N_bar = 100
for i in range(N_bar):
    fx = i / N_bar
    col = cmap(fx)
    ax_sub3.add_patch(Rectangle((bar_x0 + fx * bar_w, bar_y0), bar_w / N_bar + 0.01, bar_h,
                                facecolor=col, edgecolor="none"))
ax_sub3.add_patch(Rectangle((bar_x0, bar_y0), bar_w, bar_h, fill=False, edgecolor="#1E293B", lw=1.3))

# Labels below bar: clean, concise, with ample whitespace
ax_sub3.text(bar_x0, bar_y0 - 0.16, r"$\mathbf{Al}$" + "\n" + r"$x = 0$" + "\n" + r"$V_r = 0$",
             fontsize=8.5, color="#334155", va="top", ha="left")
ax_sub3.text(bar_x0 + bar_w, bar_y0 - 0.16, r"$\mathbf{SiC}$" + "\n" + r"$x = L$" + "\n" + r"$V_r \leq 1$",
             fontsize=8.5, ha="right", color="#0F172A", va="top")

# Save image
plt.savefig(out_png, dpi=300, facecolor="white")
plt.close()
print("Figure 1 successfully generated.")
