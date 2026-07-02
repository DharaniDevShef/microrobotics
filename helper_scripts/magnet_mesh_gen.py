import numpy as np
import trimesh


# ==========================
# Dimensions (mm)
# ==========================

side_length = 1.27
height = 1.8

# mm -> meter
s = side_length / 1000
h = height / 1000


# ==========================
# Vertices
# ==========================

vertices = []

# bottom + top hexagon
for z in [0, h]:

    for i in range(6):

        angle = np.deg2rad(60*i)

        x = s * np.cos(angle)
        y = s * np.sin(angle)

        vertices.append([x, y, z])


vertices = np.array(vertices)


# ==========================
# Triangular faces
# ==========================

faces = []


# Bottom face triangles
faces.append([0,1,2])
faces.append([0,2,3])
faces.append([0,3,4])
faces.append([0,4,5])


# Top face triangles
faces.append([6,8,7])
faces.append([6,9,8])
faces.append([6,10,9])
faces.append([6,11,10])


# Side walls
for i in range(6):

    j = (i+1) % 6

    faces.append([i,j,i+6])
    faces.append([j,j+6,i+6])


faces = np.array(faces)


# ==========================
# Create STL mesh
# ==========================

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces
)


mesh.fix_normals()


# Check solid
print("Watertight:", mesh.is_watertight)
print("Volume:", mesh.volume)


# Export
mesh.export("Magnet.stl")

print("STL created successfully")