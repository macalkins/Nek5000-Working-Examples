'''
    Geometric grading version of write_box_file_annulus.py for Nek5000.

    Replaces piecewise-uniform BL zones with smooth geometric expansion
    from each wall toward the domain center. Eliminates element size jumps
    at zone boundaries for better conditioning of the pressure solver.

    Nelx and Nelz are computed automatically from BL requirements.

    Usage:
        python write_box_file_geometric.py <model> <Ek> <Nely> [growth_rate]

    Example:
        python write_box_file_geometric.py annulus 9e-8 30 1.15
'''

import sys
import re
import math
import numpy as np

pi = np.pi

# ============================================================
# Parse arguments
# ============================================================
if len(sys.argv) < 4:
    print("Usage: python write_box_file_geometric.py <model> <Ek> <Nely> [growth_rate]")
    sys.exit(1)

model = str(sys.argv[1])
Ek = float(sys.argv[2])
Nely = int(sys.argv[3])
growth_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 1.30

# Number of elements within each BL zone
# With p=7: 2 elements give 15 GLL points (>= 10 target)
N_BL = 2

filename = model + '.box'

# Boundary conditions
Velocity_BCs = "W ,W ,P ,P ,W ,W ,"
Temperature_BCs = "t ,t ,P ,P ,I ,I ,"

# ============================================================
# Read domain parameters from .par file
# ============================================================
def read_par(parfile):
    params = {}
    with open(parfile) as f:
        for line in f:
            for key in ['userParam02', 'userParam03', 'userParam04', 'userParam06']:
                if key in line:
                    parts = re.split(r'[=#]', line.strip())
                    params[key] = float(parts[1].strip())
    return params

params = read_par(model + '.par')
k_c = params['userParam02']
num_box = params['userParam03']
aspect = params['userParam06']

lamb_c = 2 * pi / k_c
L_y = int(num_box) * lamb_c

# Domain extents
x_in, x_out = -0.5, 0.5
z_bot, z_top = -0.5 * aspect, 0.5 * aspect
x_half = (x_out - x_in) / 2.0
z_half = (z_top - z_bot) / 2.0
x_center = (x_in + x_out) / 2.0
z_center = (z_bot + z_top) / 2.0

# BL thicknesses
delta_E = 10.0 * np.sqrt(Ek)
delta_S1 = 1.5 * Ek**(1.0 / 3.0)

# ============================================================
# Geometric grading functions
# ============================================================
def compute_symmetric_grading(L_half, delta, g, n_bl):
    """Compute element boundaries with geometric grading from both walls.

    Places n_bl elements within delta from each wall, then continues
    geometric growth until meeting at the domain center. Returns
    coordinates for one half (wall to center).
    """
    if delta >= L_half:
        raise ValueError(f"BL thickness {delta:.4e} >= half-domain {L_half:.4e}")

    # Wall element size so that n_bl elements span delta
    h_1 = delta * (g - 1) / (g**n_bl - 1)

    # Total elements from wall to center
    ratio = 1 + L_half * (g - 1) / h_1
    N_half = math.ceil(math.log(ratio) / math.log(g))

    # Recompute h_1 for exact fit to L_half
    h_1 = L_half * (g - 1) / (g**N_half - 1)

    # Build coordinates (0 = wall, L_half = center)
    coords = np.zeros(N_half + 1)
    for i in range(N_half):
        coords[i + 1] = coords[i] + h_1 * g**i
    coords[-1] = L_half  # ensure exact

    # Count elements actually within delta
    n_bl_actual = 0
    cumul = 0.0
    for i in range(N_half):
        cumul += h_1 * g**i
        if cumul <= delta * 1.001:
            n_bl_actual += 1

    return coords, N_half, h_1, n_bl_actual


def build_full_grid(x_min, x_max, half_coords, N_half):
    """Build the full grid from wall-to-center half-grid by mirroring."""
    center = (x_min + x_max) / 2.0

    # Left half: wall at x_min, grading toward center
    left = x_min + half_coords

    # Right half: wall at x_max, grading toward center (mirror)
    right = x_max - half_coords[::-1]

    # Combine (left ends at center, right starts at center — share that point)
    full = np.concatenate([left[:-1], right])
    return full


def count_gll_in_bl(h_1, g, n_bl_actual, delta, p=7):
    """Count GLL points within delta from the wall.

    GLL reference points for p=7 on [0, 1]:
    0, 0.0642, 0.2042, 0.3954, 0.6046, 0.7958, 0.9358, 1.0
    """
    gll_ref = np.array([
        0.0, 0.06428243, 0.20414990, 0.39535040,
        0.60464960, 0.79585010, 0.93571757, 1.0
    ])

    n_gll = 0
    cumul = 0.0
    for i in range(n_bl_actual + 2):  # check a couple extra elements
        h_i = h_1 * g**i
        for j in range(len(gll_ref)):
            pos = cumul + gll_ref[j] * h_i
            if pos <= delta * 1.001:
                if i == 0 or j > 0:  # avoid double-counting shared nodes
                    n_gll += 1
            else:
                return n_gll
        cumul += h_i
    return n_gll


# ============================================================
# Compute graded element distributions
# ============================================================
g = growth_rate

# X direction (radial — Stewartson E^1/3 layers)
x_half_coords, Nx_half, hx_wall, nx_bl = compute_symmetric_grading(
    x_half, delta_S1, g, N_BL
)
x = build_full_grid(x_in, x_out, x_half_coords, Nx_half)
Nelx = len(x) - 1
n_gll_stew = count_gll_in_bl(hx_wall, g, nx_bl, delta_S1)

# Z direction (vertical — Ekman layers)
z_half_coords, Nz_half, hz_wall, nz_bl = compute_symmetric_grading(
    z_half, delta_E, g, N_BL
)
z = build_full_grid(z_bot, z_top, z_half_coords, Nz_half)
Nelz = len(z) - 1
n_gll_ekman = count_gll_in_bl(hz_wall, g, nz_bl, delta_E)

# Y direction (azimuthal — uniform)
y = [0.0, L_y, 1.0]
y_edges = np.linspace(0, L_y, Nely + 1)

Nels = [Nelx, -Nely, Nelz]

# ============================================================
# Save element edges for postprocessing
# ============================================================
np.savetxt('element_edges_x.txt', x, header=f'Nelx={Nelx}', fmt='%.12f')
np.savetxt('element_edges_y.txt', y_edges, header=f'Nely={Nely}', fmt='%.12f')
np.savetxt('element_edges_z.txt', z, header=f'Nelz={Nelz}', fmt='%.12f')

# ============================================================
# Print diagnostics
# ============================================================
hx_max = hx_wall * g**(Nx_half - 1)
hz_max = hz_wall * g**(Nz_half - 1)

print("=" * 60)
print("  Geometric grading mesh for Nek5000")
print("=" * 60)
print(f"  Ek = {Ek:.2e}, growth rate = {g}")
print()
print(f"  X (radial, Stewartson E^1/3):")
print(f"    delta_S1 = {delta_S1:.6f}")
print(f"    Nelx = {Nelx} ({Nx_half} per half)")
print(f"    h_wall = {hx_wall:.6f}, h_center = {hx_max:.6f}, ratio = {hx_max/hx_wall:.1f}")
print(f"    {nx_bl} elements within delta_S1, {n_gll_stew} GLL points in BL")
print()
print(f"  Z (vertical, Ekman):")
print(f"    delta_E = {delta_E:.6f}")
print(f"    Nelz = {Nelz} ({Nz_half} per half)")
print(f"    h_wall = {hz_wall:.6f}, h_center = {hz_max:.6f}, ratio = {hz_max/hz_wall:.1f}")
print(f"    {nz_bl} elements within delta_E, {n_gll_ekman} GLL points in BL")
print()
print(f"  Y (azimuthal, uniform): Nely = {Nely}, L_y = {L_y:.4f}")
print()
print(f"  Total elements: {Nelx} x {Nely} x {Nelz} = {Nelx * Nely * Nelz}")
print("=" * 60)

# ============================================================
# Write .box file
# ============================================================
def split_list_by_char_count(input_list, char_limit):
    """Split coordinate list into lines respecting character limit."""
    result = []
    current_chunk = []
    current_char_count = 0
    for item in input_list:
        item_length = len(str(item))
        if current_char_count + item_length <= char_limit:
            current_chunk.append(item)
            current_char_count += item_length
        else:
            if current_chunk:
                result.append(current_chunk)
            current_chunk = [item]
            current_char_count = item_length
    if current_chunk:
        result.append(current_chunk)
    return result

limit = 120
x_split = split_list_by_char_count(x, limit)
z_split = split_list_by_char_count(z, limit)

with open(filename, 'w', newline='') as f:
    f.write('-3\n')
    f.write('2\n')
    f.write('Box\n')
    f.write(' '.join(str(v) for v in Nels) + '\n')
    for row in x_split:
        f.write(' '.join(f'{v:.12f}' if isinstance(v, float) else str(v) for v in row) + '\n')
    f.write(' '.join(str(v) for v in y) + '\n')
    for row in z_split:
        f.write(' '.join(f'{v:.12f}' if isinstance(v, float) else str(v) for v in row) + '\n')
    f.write(Velocity_BCs + '\n')
    f.write(Temperature_BCs + '\n')

print(f"\nCreated {filename}")
print("You may now run genbox")
