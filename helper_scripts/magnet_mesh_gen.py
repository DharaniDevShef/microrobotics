"""
Magnet Mesh Gen - generates a small cylindrical magnet STL mesh (2mm
diameter x 2mm height by default) via trimesh and writes it to
../meshes/Magnet.stl, for use in the module CAD/MJCF assemblies.
"""

import os
import trimesh

# ==========================
# Dimensions (mm)
# ==========================

diameter = 2.0   # mm
height = 2.0     # mm

# Convert to meters
radius = (diameter / 2) / 1000
height = height / 1000

# ==========================
# Create cylinder
# ==========================

mesh = trimesh.creation.cylinder(
    radius=radius,
    height=height,
    sections=128  # Increase for smoother cylinder
)

# ==========================
# Check mesh
# ==========================

print("Watertight:", mesh.is_watertight)
print("Volume:", mesh.volume)

# ==========================
# Export
# ==========================

output_path = "../meshes/Magnet.stl"

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Export (overwrites existing file)
mesh.export(output_path)

print(f"STL created successfully: {output_path}")