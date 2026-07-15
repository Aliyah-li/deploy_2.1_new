#!/usr/bin/env python3
"""
Generate dense stone field and extreme terrain geoms for M20.xml.
Outputs MuJoCo <geom .../> XML lines to stdout.
Usage: python3 generate_terrains.py
"""

import numpy as np

rng = np.random.default_rng(42)


def random_quat_euler(rng, max_roll=0.5, max_pitch=0.5, max_yaw=0.8):
    """Generate a random quaternion as 'w x y z' string from bounded Euler angles."""
    roll = rng.uniform(-max_roll, max_roll)
    pitch = rng.uniform(-max_pitch, max_pitch)
    yaw = rng.uniform(-max_yaw, max_yaw)
    # Euler → quaternion (ZYX intrinsic = XYZ extrinsic)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"


def geom_xml(name, pos, size, quat=None, euler=None, material="terrain_stone_dense",
             friction="1.6 0.01 0.01"):
    """Build a <geom .../> XML string."""
    parts = [
        f'<geom name="{name}" type="box"',
        f'pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}"',
        f'size="{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}"',
    ]
    if quat:
        parts.append(f'quat="{quat}"')
    elif euler is not None:
        if isinstance(euler, str):
            parts.append(f'euler="{euler}"')
        else:
            parts.append(f'euler="{euler[0]:.3f} {euler[1]:.3f} {euler[2]:.3f}"')
    parts.append(f'material="{material}"')
    parts.append(f'friction="{friction}"')
    parts.append('condim="3"')
    return " ".join(parts) + "/>"


# ============================================================
# SECTION 1: Dense Stone Field (y ≈ -3.2 to -5.3, x ≈ -0.5 to 11.5)
# ============================================================
def generate_dense_stone_field():
    """Generate a dense, irregular, interconnected stone field."""
    lines = []
    lines.append('')
    lines.append('        <!-- ====== Dense Stone Field: irregular, interconnected stones ====== -->')

    # Use a staggered (hexagonal-like) grid for natural packing
    dx = 0.22   # x spacing
    dy = 0.19   # y spacing (staggered)
    y_start = -3.2
    y_end = -5.3
    x_start = -0.5
    x_end = 11.5

    # Generate grid points
    row = 0
    y = y_start
    counter = 0

    while y > y_end:
        x = x_start
        col = 0
        # Stagger every other row
        x_offset = (dx / 2) if row % 2 == 1 else 0.0
        while x < x_end:
            # Random jitter
            jx = rng.uniform(-0.06, 0.06)
            jy = rng.uniform(-0.06, 0.06)
            px = x + x_offset + jx
            py = y + jy

            # Skip some points at edges for natural look
            if rng.random() < 0.08:
                x += dx
                col += 1
                continue

            # Random stone size
            sx = rng.uniform(0.06, 0.17)
            sy = rng.uniform(0.06, 0.16)
            sh = rng.uniform(0.02, 0.10)

            # Random 3D rotation
            quat = random_quat_euler(rng, max_roll=0.55, max_pitch=0.55, max_yaw=0.9)

            # Varied friction
            fric_val = round(rng.uniform(1.5, 1.8), 2)
            friction = f"{fric_val} 0.01 0.01"

            # Mix materials for visual variety
            mat_r = rng.random()
            if mat_r < 0.65:
                mat = "terrain_stone_dense"
            elif mat_r < 0.85:
                mat = "terrain_rough"
            else:
                mat = "terrain_uneven"

            lines.append(
                "        " + geom_xml(
                    f"stone_{counter:04d}",
                    (px, py, sh),
                    (sx, sy, sh),
                    quat=quat,
                    material=mat,
                    friction=friction,
                )
            )
            counter += 1
            x += dx
            col += 1
        y -= dy
        row += 1

    lines.append(f"        <!-- Total stones: {counter} -->")
    return lines


# ============================================================
# SECTION 2: Extreme Terrain (y ≈ -5.5 to -7.5, x ≈ 1 to 12)
# ============================================================
def generate_extreme_terrain():
    """Generate high-difficulty diverse extreme terrain zones."""
    lines = []
    lines.append('')
    lines.append('        <!-- ====== Extreme Terrain ====== -->')
    lines.append('        <!-- Zone A: Tall Blocks (x: 1-4) -->')

    # ---- Zone A: Tall Blocks ----
    zone_a_blocks = [
        # (x, y, height, sx, sy, euler_roll, euler_pitch, euler_yaw)
        (1.20, -5.80, 0.140, 0.22, 0.18, 0.10, 0.15, -0.30),
        (1.55, -6.10, 0.170, 0.20, 0.16, -0.08, -0.12, 0.40),
        (1.40, -6.60, 0.150, 0.24, 0.14, 0.15, 0.20, 0.15),
        (1.80, -5.90, 0.190, 0.18, 0.20, -0.12, 0.08, -0.50),
        (1.70, -6.40, 0.160, 0.22, 0.17, 0.05, -0.18, 0.35),
        (2.10, -6.00, 0.200, 0.26, 0.15, -0.18, 0.12, -0.20),
        (2.05, -6.70, 0.180, 0.20, 0.19, 0.12, -0.15, 0.60),
        (2.40, -5.75, 0.210, 0.19, 0.22, -0.06, 0.22, -0.45),
        (2.35, -6.35, 0.165, 0.23, 0.16, 0.20, -0.10, 0.25),
        (2.70, -6.10, 0.195, 0.21, 0.18, -0.15, 0.05, -0.35),
        (2.65, -6.80, 0.175, 0.25, 0.14, 0.08, 0.25, 0.55),
        (3.00, -5.85, 0.220, 0.17, 0.20, -0.20, -0.20, 0.10),
        (3.10, -6.50, 0.155, 0.22, 0.16, 0.16, 0.10, -0.40),
        (3.40, -6.00, 0.185, 0.24, 0.17, -0.10, -0.08, 0.50),
        (3.35, -6.60, 0.205, 0.20, 0.19, 0.22, 0.15, -0.55),
        (3.70, -6.25, 0.170, 0.23, 0.15, -0.14, 0.18, 0.30),
        (3.55, -6.90, 0.145, 0.21, 0.20, 0.06, -0.22, -0.15),
        (3.90, -5.70, 0.215, 0.18, 0.22, -0.18, -0.14, 0.65),
    ]
    for i, (px, py, sh, sx, sy, er, ep, ey) in enumerate(zone_a_blocks):
        euler = f"{er:.3f} {ep:.3f} {ey:.3f}"
        lines.append(
            "        " + geom_xml(
                f"extreme_a_{i:02d}", (px, py, sh), (sx, sy, sh),
                euler=euler, material="terrain_extreme",
                friction=f"{rng.uniform(1.4, 1.7):.2f} 0.01 0.01"
            )
        )

    # ---- Zone B: Narrow Gap Corridor (x: 4-7) ----
    lines.append('')
    lines.append('        <!-- Zone B: Narrow Gap Corridor (x: 4-7) -->')
    # Two wavy walls with ~0.35m gap between them
    # Upper wall (y ≈ -5.65), Lower wall (y ≈ -6.75), gap center ≈ y=-6.2
    wall_stones = []
    for i in range(25):
        x_pos = 4.0 + i * 0.14
        wave_upper = -5.65 + 0.08 * np.sin(i * 0.55) + 0.05 * np.cos(i * 1.1)
        wave_lower = -6.75 + 0.06 * np.sin(i * 0.65 + 1.5) + 0.04 * np.cos(i * 0.9)

        # Upper wall stone
        sh_u = rng.uniform(0.07, 0.15)
        sx_u = rng.uniform(0.08, 0.14)
        sy_u = rng.uniform(0.06, 0.12)
        quat_u = random_quat_euler(rng, 0.3, 0.3, 0.6)
        wall_stones.append((f"extreme_b_upper_{i:02d}", x_pos, wave_upper, sh_u, sx_u, sy_u, quat_u))

        # Lower wall stone
        sh_l = rng.uniform(0.07, 0.15)
        sx_l = rng.uniform(0.08, 0.14)
        sy_l = rng.uniform(0.06, 0.12)
        quat_l = random_quat_euler(rng, 0.3, 0.3, 0.6)
        wall_stones.append((f"extreme_b_lower_{i:02d}", x_pos, wave_lower, sh_l, sx_l, sy_l, quat_l))

    # Extra gap-filling stones to make walls dense
    for i in range(15):
        x_pos = 4.3 + i * 0.28
        # scatter a few extra in gap
        gap_y = -6.2 + rng.uniform(-0.08, 0.08)
        sh = rng.uniform(0.03, 0.06)
        quat = random_quat_euler(rng, 0.4, 0.4, 0.8)
        # Using slightly smaller name prefix to not conflict
        pass  # keep gap mostly clear

    for name, px, py, sh, sx, sy, quat in wall_stones:
        lines.append(
            "        " + geom_xml(
                name, (px, py, sh), (sx, sy, sh),
                quat=quat, material="terrain_extreme",
                friction=f"{rng.uniform(1.5, 1.8):.2f} 0.01 0.01"
            )
        )

    # ---- Zone C: Extreme Mixed (x: 7-10) ----
    lines.append('')
    lines.append('        <!-- Zone C: Extreme Mixed — steep slabs, tall blocks, irregular steps (x: 7-10) -->')

    # Steep tilted slabs
    slabs = [
        (7.20, -6.00, 0.065, 0.40, 0.70, 0.0, 0.60, 0.0),   # 35° pitch
        (7.80, -6.35, 0.075, 0.45, 0.65, 0.0, -0.70, 0.0),   # 40° pitch reverse
        (8.40, -5.90, 0.070, 0.50, 0.75, 0.05, 0.55, 0.15),  # mixed tilt
        (9.00, -6.50, 0.080, 0.42, 0.68, -0.04, -0.65, -0.10),
    ]
    for i, (px, py, sh, sx, sy, er, ep, ey) in enumerate(slabs):
        lines.append(
            "        " + geom_xml(
                f"extreme_c_slab_{i:02d}", (px, py, sh), (sx, sy, sh),
                euler=f"{er:.3f} {ep:.3f} {ey:.3f}",
                material="terrain_extreme",
                friction="1.6 0.01 0.01"
            )
        )

    # Tall isolated blocks
    tall_blocks = [
        (7.50, -6.80, 0.220, 0.16, 0.16, 0.15, 0.10, -0.40),
        (7.90, -5.70, 0.240, 0.14, 0.18, -0.20, -0.15, 0.35),
        (8.30, -6.60, 0.200, 0.17, 0.15, 0.08, 0.25, 0.50),
        (8.70, -5.85, 0.250, 0.15, 0.17, -0.12, -0.20, -0.55),
        (9.10, -6.40, 0.230, 0.18, 0.14, 0.18, 0.05, -0.30),
        (9.50, -6.90, 0.210, 0.16, 0.19, -0.22, 0.15, 0.60),
        (9.80, -5.75, 0.245, 0.15, 0.16, 0.10, -0.10, -0.45),
    ]
    for i, (px, py, sh, sx, sy, er, ep, ey) in enumerate(tall_blocks):
        lines.append(
            "        " + geom_xml(
                f"extreme_c_tall_{i:02d}", (px, py, sh), (sx, sy, sh),
                euler=f"{er:.3f} {ep:.3f} {ey:.3f}",
                material="terrain_extreme",
                friction="1.7 0.01 0.01"
            )
        )

    # Irregular step sequence
    steps = [
        (7.10, -6.15, 0.090, 0.30, 0.50, 0.0, 0.0, 0.0),
        (7.55, -6.15, 0.130, 0.28, 0.50, 0.02, 0.02, 0.0),
        (8.00, -6.15, 0.175, 0.32, 0.50, -0.03, -0.01, 0.0),
        (8.50, -6.15, 0.110, 0.26, 0.50, 0.0, 0.04, 0.0),
        (8.95, -6.15, 0.195, 0.34, 0.50, 0.01, -0.02, 0.0),
        (9.45, -6.15, 0.150, 0.30, 0.50, -0.02, 0.03, 0.0),
    ]
    for i, (px, py, sh, sx, sy, er, ep, ey) in enumerate(steps):
        lines.append(
            "        " + geom_xml(
                f"extreme_c_step_{i:02d}", (px, py, sh), (sx, sy, sh),
                euler=f"{er:.3f} {ep:.3f} {ey:.3f}",
                material="terrain_extreme",
                friction="1.5 0.01 0.01"
            )
        )

    # ---- Zone D: Rubble Descent (x: 10-12) ----
    lines.append('')
    lines.append('        <!-- Zone D: Rubble Descent (x: 10-12) -->')
    for i in range(60):
        x_pos = 10.0 + rng.uniform(0.0, 2.2)
        # Base y centers around -6.2, spread wider
        y_pos = -6.2 + rng.uniform(-0.9, 0.9)
        # Height decreases as x increases (descent)
        progress = (x_pos - 10.0) / 2.2
        base_height = 0.12 * (1.0 - progress) + 0.03
        sh = rng.uniform(base_height * 0.6, base_height * 1.4)
        sx = rng.uniform(0.06, 0.15)
        sy = rng.uniform(0.06, 0.14)
        quat = random_quat_euler(rng, 0.5, 0.5, 0.9)

        lines.append(
            "        " + geom_xml(
                f"extreme_d_rubble_{i:02d}", (x_pos, y_pos, sh), (sx, sy, sh),
                quat=quat, material="terrain_extreme",
                friction=f"{rng.uniform(1.5, 1.9):.2f} 0.01 0.01"
            )
        )

    return lines


# ============================================================
# SECTION 3: Connected Rubble Mountains (y ≈ -8 to -12)
# ============================================================
def generate_connected_rubble_mountains():
    """
    Generate a connected rubble mountain ridge with 3 peaks:
    Peak 1: 1.5m, Peak 2: 2.0m, Peak 3: 2.5m.
    Mountains overlap to form a continuous ridge.
    Uses larger stones for bigger scale.
    """
    lines = []
    lines.append('')
    lines.append('        <!-- ====== Connected Rubble Mountain Ridge (y ≈ -8 to -12) ====== -->')

    # Three peaks along a ridge at y ≈ -9.5
    peaks = [
        {"x": 2.5, "y": -9.5, "h": 1.5, "radius": 2.5},
        {"x": 5.5, "y": -9.5, "h": 2.0, "radius": 3.0},
        {"x": 9.0, "y": -9.5, "h": 2.5, "radius": 3.5},
    ]

    n_layers = 8

    for peak_idx, pk in enumerate(peaks):
        cx, cy, peak_h, base_r = pk["x"], pk["y"], pk["h"], pk["radius"]
        label = f"rmtn_p{peak_idx+1}"

        for layer in range(n_layers):
            t = layer / (n_layers - 1)
            layer_r = base_r * (1.0 - t * 0.82)
            base_z = peak_h * (t ** 0.85)
            n_stones = int(40 * (1.0 - t * 0.65))

            for si in range(n_stones):
                # Elliptical footprint (wider in y)
                px = cx + rng.uniform(-layer_r * 0.8, layer_r * 0.8)
                py = cy + rng.uniform(-layer_r, layer_r)
                dist = np.sqrt(((px - cx) / (layer_r * 0.8)) ** 2 +
                               ((py - cy) / layer_r) ** 2)
                if dist > 1.08:
                    continue

                max_size = 0.25 * (1.0 - t * 0.5) + 0.06
                sx = rng.uniform(0.06, max_size)
                sy = rng.uniform(0.06, max_size)
                sh = rng.uniform(0.04, max_size * 0.8)
                pz = base_z + sh

                quat = random_quat_euler(rng, 0.7, 0.7, 1.0)
                fric = f"{rng.uniform(1.5, 1.9):.2f} 0.01 0.01"
                mat = "terrain_stone_dense" if rng.random() < 0.6 else "terrain_rough"

                lines.append(
                    "        " + geom_xml(
                        f"{label}_l{layer:02d}_{si:03d}",
                        (px, py, pz), (sx, sy, sh),
                        quat=quat, material=mat, friction=fric
                    )
                )

    # Fill saddle zones between peaks
    for gap_idx in range(len(peaks) - 1):
        p1, p2 = peaks[gap_idx], peaks[gap_idx + 1]
        mid_x = (p1["x"] + p2["x"]) / 2
        mid_y = (p1["y"] + p2["y"]) / 2
        saddle_h = min(p1["h"], p2["h"]) * 0.45
        saddle_r = (p1["radius"] + p2["radius"]) / 2 * 0.8

        for layer in range(4):
            t = layer / 3
            lr = saddle_r * (1.0 - t * 0.7)
            bz = saddle_h * (t ** 0.85)
            n_stones = int(20 * (1.0 - t * 0.6))
            for si in range(n_stones):
                px = mid_x + rng.uniform(-lr * 0.7, lr * 0.7)
                py = mid_y + rng.uniform(-lr * 0.8, lr * 0.8)
                max_size = 0.18 * (1.0 - t * 0.5) + 0.05
                sx = rng.uniform(0.05, max_size)
                sy = rng.uniform(0.05, max_size)
                sh = rng.uniform(0.03, max_size * 0.7)
                pz = bz + sh
                quat = random_quat_euler(rng, 0.5, 0.5, 1.0)
                fric = f"{rng.uniform(1.5, 1.8):.2f} 0.01 0.01"
                mat = "terrain_stone_dense" if rng.random() < 0.6 else "terrain_rough"
                lines.append(
                    "        " + geom_xml(
                        f"rmtn_saddle_{gap_idx}_l{layer:02d}_{si:03d}",
                        (px, py, pz), (sx, sy, sh),
                        quat=quat, material=mat, friction=fric
                    )
                )

    return lines


# ============================================================
# SECTION 4: Smooth Hills (y ≈ -13 to -18)
# ============================================================
def generate_smooth_hills():
    """
    - Hill A: convex dome, 1.5m high
    - Hill B: concave bowl connected to A's edge, 0.25m deep
    - Hill C: slide (ramp up then down, sinusoidal arc)
    """
    lines = []
    lines.append('')
    lines.append('        <!-- ====== Smooth Hills (y ≈ -13 to -18) ====== -->')

    # ---- Hill A: Tall convex dome, 1.5m (x: 1-4.5, y: -14.5) ----
    lines.append('        <!-- Hill A: convex dome 1.5m -->')
    cx_a, cy_a = 2.7, -14.5
    radius_a = 3.0
    peak_a = 1.5
    n_rings = 15
    for ring in range(n_rings):
        r = radius_a * (ring + 0.5) / n_rings
        h = peak_a * np.cos((r / radius_a) * np.pi / 2)
        n_stones = max(8, int(2 * np.pi * r / 0.18))
        for i in range(n_stones):
            angle = 2 * np.pi * i / n_stones + rng.uniform(-0.06, 0.06)
            px = cx_a + r * np.cos(angle) + rng.uniform(-0.04, 0.04)
            py = cy_a + r * np.sin(angle) + rng.uniform(-0.04, 0.04)

            sz = max(0.03, h * 0.5 + 0.01)
            stone_w = 0.10 + rng.uniform(0.04, 0.10)
            sx = stone_w
            sy = stone_w * rng.uniform(0.7, 1.3)

            tilt = (r / radius_a) * 0.25
            euler = f"{rng.uniform(-0.06, 0.06):.3f} {tilt:.3f} {rng.uniform(-0.2, 0.2):.3f}"

            lines.append(
                "        " + geom_xml(
                    f"hill_a_{ring:02d}_{i:02d}",
                    (px, py, sz), (sx, sy, sz),
                    euler=euler, material="terrain_uneven",
                    friction="1.3 0.01 0.01"
                )
            )

    # ---- Hill B: Concave bowl connected to A's south edge ----
    # A center at y=-14.5, radius=3.0 → south edge at y≈-17.5
    # B center at y≈-19.0, so B's north rim touches A's south edge
    lines.append('        <!-- Hill B: concave bowl connected to Hill A -->')
    cx_b, cy_b = 2.7, -19.0
    radius_b = 2.2
    depth_b = 0.25
    n_rings_b = 10
    for ring in range(n_rings_b):
        r = radius_b * (ring + 0.5) / n_rings_b
        # h=0 at center, rises toward rim to depth_b
        h = depth_b * (r / radius_b) ** 1.6
        n_stones = max(5, int(2 * np.pi * r / 0.20))
        for i in range(n_stones):
            angle = 2 * np.pi * i / n_stones + rng.uniform(-0.08, 0.08)
            px = cx_b + r * np.cos(angle) + rng.uniform(-0.04, 0.04)
            py = cy_b + r * np.sin(angle) + rng.uniform(-0.04, 0.04)

            sz = max(0.02, h * 0.55 + 0.01)
            stone_w = 0.08 + rng.uniform(0.03, 0.08)
            sx, sy = stone_w, stone_w * rng.uniform(0.7, 1.3)

            tilt = -(r / radius_b) * 0.10
            euler = f"{rng.uniform(-0.04, 0.04):.3f} {tilt:.3f} {rng.uniform(-0.2, 0.2):.3f}"

            lines.append(
                "        " + geom_xml(
                    f"hill_b_{ring:02d}_{i:02d}",
                    (px, py, sz), (sx, sy, sz),
                    euler=euler, material="terrain_uneven",
                    friction="1.2 0.01 0.01"
                )
            )

    # Fill the connection zone between A and B
    lines.append('        <!-- Hill A-B connection fill -->')
    for si in range(40):
        px = cx_a + rng.uniform(-2.0, 2.0)
        py = -17.0 + rng.uniform(-0.5, 0.5)
        # Height: blend from A's slope to B's rim
        dist_a = np.sqrt((px - cx_a) ** 2 + (py - cy_a) ** 2)
        ta = np.clip(dist_a / radius_a, 0, 1)
        h_a = peak_a * np.cos(ta * np.pi / 2)
        dist_b = np.sqrt((px - cx_b) ** 2 + (py - cy_b) ** 2)
        tb = np.clip(dist_b / radius_b, 0, 1)
        h_b = depth_b * (tb ** 1.6)
        h = max(h_a, h_b) + rng.uniform(-0.02, 0.02)
        sz = max(0.02, h * 0.45 + 0.01)
        sx = rng.uniform(0.06, 0.14)
        sy = rng.uniform(0.06, 0.14)
        quat = random_quat_euler(rng, 0.3, 0.3, 0.6)
        lines.append(
            "        " + geom_xml(
                f"hill_ab_fill_{si:03d}",
                (px, py, sz), (sx, sy, sz),
                quat=quat, material="terrain_uneven",
                friction="1.3 0.01 0.01"
            )
        )

    # ---- Hill C: Slide — sinusoidal ramp up then down (x: 7-12, y: -14.5) ----
    lines.append('        <!-- Hill C: slide — sinusoidal up-down ramp -->')
    slide_x0, slide_x1 = 7.0, 11.5
    slide_y0, slide_y1 = -16.0, -13.0
    slide_h_max = 0.9
    slide_length = slide_x1 - slide_x0
    n_slabs_x = 24
    n_slabs_y = 10
    for ix in range(n_slabs_x):
        tx = (ix + 0.5) / n_slabs_x
        px = slide_x0 + tx * slide_length
        # Sinusoidal: up then down (one full bump)
        h = slide_h_max * np.sin(tx * np.pi)  # 0 → max → 0
        # Smooth the bottom transitions
        if tx < 0.08:
            h *= tx / 0.08
        elif tx > 0.92:
            h *= (1.0 - tx) / 0.08

        for iy in range(n_slabs_y):
            ty = (iy + 0.5) / n_slabs_y
            py = slide_y0 + ty * (slide_y1 - slide_y0)
            sx = slide_length / n_slabs_x * 0.55
            sy = (slide_y1 - slide_y0) / n_slabs_y * 0.55
            sz = max(0.02, h * 0.48 + 0.015)

            # Pitch follows the slope of sin: derivative = cos
            slope = slide_h_max * np.pi / slide_length * np.cos(tx * np.pi)
            pitch = np.arctan(slope)  # actual slope angle
            euler = f"{rng.uniform(-0.02, 0.02):.3f} {pitch:.3f} {rng.uniform(-0.03, 0.03):.3f}"

            lines.append(
                "        " + geom_xml(
                    f"hill_c_slide_{ix:02d}_{iy:02d}",
                    (px, py, sz), (sx, sy, sz),
                    euler=euler, material="terrain_uneven",
                    friction="1.2 0.01 0.01"
                )
            )

    return lines


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    stone_lines = generate_dense_stone_field()
    extreme_lines = generate_extreme_terrain()
    rubble_mtn_lines = generate_connected_rubble_mountains()
    smooth_hill_lines = generate_smooth_hills()

    for line in stone_lines:
        print(line)
    for line in extreme_lines:
        print(line)
    for line in rubble_mtn_lines:
        print(line)
    for line in smooth_hill_lines:
        print(line)
