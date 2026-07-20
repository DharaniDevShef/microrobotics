import json
import numpy as np
import networkx as nx
from xml.dom import minidom
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------
# 1. Coordinate Math Helpers
# ---------------------------------------------------------------------
def quaternion_to_matrix(q):
    """Converts a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x**2 - 2*y**2]
    ])

def matrix_to_quaternion(R):
    """Converts a 3x3 rotation matrix to a quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def get_rotation_z(angle_rad):
    """Returns a 3x3 rotation matrix around the Z axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

# ---------------------------------------------------------------------
# 2. Graph and Setup
# ---------------------------------------------------------------------
with open("graph6.json", "r") as f:
    data = json.load(f)

G = nx.DiGraph()

for node in data["nodes"]:
    G.add_node(
        node["id"],
        module_type=node["module_type"],
        hinge_angle=node["hinge_angle"],
        pos=node["_pos"]
    )

for edge in data["edges"]:
    G.add_edge(
        edge["source"],
        edge["target"],
        connector1=edge["connector1"],
        connector2=edge["connector2"]
    )

# ---------------------------------------------------------------------
# 3. Structural Constants & Assignments
# ---------------------------------------------------------------------
connector_assignments = {}
for node_id in G.nodes:
    connector_assignments[node_id] = {1: "connectorC", 2: "connectorC", 3: "connectorC"}

for u, v, d in G.edges(data=True):
    c1 = d["connector1"]
    c2 = d["connector2"]
    connector_assignments[u][c1] = "connectorA"
    connector_assignments[v][c2] = "connectorB"

# Extracted exact structural coordinates from working XML
CONN_POSITIONS = {
    3: "-0.002017680 -0.001164957 -0.002500001",
    2: "0.002017941 -0.001165050 -0.002500001",
    1: "0.000000439 0.002330720 -0.002500000"
}

CONN_QUATS = {
    3: "0.683012546 0.183013001 -0.683012546 0.183013001",
    2: "0.683013104 0.183012443 0.683013104 -0.183012443",
    1: "0.500000000 -0.500000000 0.500000000 0.500000000"
}

# ---------------------------------------------------------------------
# Input Geometric Variables
# ---------------------------------------------------------------------
CONN1_ROT = 180
CONN2_ROT = 120
CONN3_ROT = -120
PARENT_CHILD_DIS = 0.0089  # mm

# ---------------------------------------------------------------------
# Dynamic Vector Projection Math
# ---------------------------------------------------------------------
# We calculate the (x, y) coordinates using sine and cosine. 
# For Connector 1 (180°), it points straight down along the inverted Y-axis.
PARENT_CONNECTOR_REL_OFFSETS = {
    1: np.array([
        PARENT_CHILD_DIS * np.sin(np.radians(180 - CONN1_ROT)), 
        PARENT_CHILD_DIS * np.cos(np.radians(180 - CONN1_ROT)), 
        0.0
    ]), 
    2: np.array([
        PARENT_CHILD_DIS * np.sin(np.radians(CONN2_ROT)), 
        PARENT_CHILD_DIS * np.cos(np.radians(CONN2_ROT)), 
        0.0
    ]), 
    3: np.array([
        PARENT_CHILD_DIS * np.sin(np.radians(CONN3_ROT)), 
        PARENT_CHILD_DIS * np.cos(np.radians(CONN3_ROT)), 
        0.0
    ])  
}

# Local orientation steps mapping connector outward-facing headings (120 degree separations around Z)
PARENT_CONNECTOR_ROTATIONS = {
    1: get_rotation_z(np.radians(CONN1_ROT)),       # Top Slot
    2: get_rotation_z(np.radians(CONN2_ROT)),      # Bottom-Left Slot
    3: get_rotation_z(np.radians(CONN3_ROT + 180))        # Bottom-Right Slot
}

# ---------------------------------------------------------------------
# 4. Corrected Kinematic Solve (BFS Loop Closure Fix)
# ---------------------------------------------------------------------
positions = {}
quaternions = {}

# Set origin module anchor
positions["module_1"] = np.array([0.0, 0.0, 0.0])
quaternions["module_1"] = np.array([1.0, 0.0, 0.0, 0.0])

visited = {"module_1"}
queue = ["module_1"]

# Extract the Z-offset dynamically from Connector 1 (ignoring the string vs numeric type)
# Strings like "-0.002017680 -0.001164957 -0.002500001" split to get the 3rd index
if isinstance(CONN_POSITIONS[1], str):
    CONNECTOR_Z_OFFSET = abs(float(CONN_POSITIONS[1].split()[2]))
else:
    CONNECTOR_Z_OFFSET = abs(CONN_POSITIONS[1][2])

# This calculates the total thickness/lift discrepancy automatically!
VALLEY_FLIP_Z_ADJUSTMENT = 2 * CONNECTOR_Z_OFFSET

while queue:
    current = queue.pop(0)
    curr_pos = positions[current]
    curr_quat = quaternions[current]
    R_curr = quaternion_to_matrix(curr_quat)
    
    # Process both child links and parent links cleanly
    for nbr in list(G.successors(current)) + list(G.predecessors(current)):
        if nbr not in visited:
            nbr_type = G.nodes[nbr].get("module_type", "non-foldable")
            
            if G.has_edge(current, nbr):
                # Standard Outbound step (Parent -> Child)
                parent_slot = G[current][nbr]["connector1"]
                
                # Dynamic Z compensation for Valley fold flips
                base_offset = PARENT_CONNECTOR_REL_OFFSETS[parent_slot].copy()
                if nbr_type == "valley fold":
                    base_offset[2] = -VALLEY_FLIP_Z_ADJUSTMENT
                elif G.nodes[current].get("module_type") == "valley fold":
                    base_offset[2] = VALLEY_FLIP_Z_ADJUSTMENT
                
                # Rotate local translation step cleanly into current global heading orientation
                positions[nbr] = curr_pos + R_curr @ base_offset
                R_nbr = R_curr @ PARENT_CONNECTOR_ROTATIONS[parent_slot]
                quaternions[nbr] = matrix_to_quaternion(R_nbr)
            else:
                # Inbound step (Child <- Parent) - REQUIRES TRANSPOSED MATRICES
                child_slot = G[nbr][current]["connector2"]
                
                base_offset = PARENT_CONNECTOR_REL_OFFSETS[child_slot].copy()
                if G.nodes[nbr].get("module_type") == "valley fold":
                    base_offset[2] = -VALLEY_FLIP_Z_ADJUSTMENT
                elif nbr_type == "valley fold":
                    base_offset[2] = VALLEY_FLIP_Z_ADJUSTMENT
                
                # Reverse orientation lookup safely 
                R_nbr = R_curr @ PARENT_CONNECTOR_ROTATIONS[child_slot].T
                quaternions[nbr] = matrix_to_quaternion(R_nbr)
                positions[nbr] = curr_pos - R_nbr @ base_offset
            
            visited.add(nbr)
            queue.append(nbr)

# ---------------------------------------------------------------------
# 5. XML Templates
# ---------------------------------------------------------------------
fold_body_tpl = """
        <body name="module_{module_id}" pos="{module_pos}" quat="{module_macro_quat}">
            <freejoint name="free_module_{module_id}"/>
            <inertial pos="0 0 0" mass="1e-05" diaginertia="1e-08 1e-08 1e-08"/>
            <body name="bodyBase_{module_id}" pos="0.000000000 0.000000000 0.000000000" quat="{module_quat}">
                <inertial pos="-0.000000 -0.002067 -0.002280" mass="0.000089" diaginertia="1.651610e-09 1.384291e-09 1.198679e-09"/>
                <geom name="geom_bodyBase_{module_id}" type="mesh" mesh="bodyBase" rgba="0.2 0.2 0.8 1" fluidshape="ellipsoid" density="1200" fluidcoef="0.6 0.25 1.5 1.0 1.0"/>
                <body name="connector1_{module_id}" pos="{connector1_pos}" quat="{connector1_quat}">
                    <inertial pos="-0.000000 -0.000000 0.001027" mass="0.000035" diaginertia="1.315291e-10 1.315224e-10 1.494471e-10"/>
                    <geom name="geom_connector1_{module_id}" type="mesh" mesh="{connector1_mesh}" rgba="1 1 1 1"/>
                    <geom name="magnet_connector1_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009"/>
                </body>
                <body name="connector2_{module_id}" pos="{connector2_pos}" quat="{connector2_quat}">
                    <inertial pos="0.000000 -0.000000 0.001102" mass="0.000037" diaginertia="1.661707e-10 1.661707e-10 1.944327e-10"/>
                    <geom name="geom_connector2_{module_id}" type="mesh" mesh="{connector2_mesh}" rgba="0 0 0 1"/>
                    <geom name="magnet_connector2_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009"/>
                </body>
                <body name="bodyLink_{module_id}" pos="0 0 0" quat="1 0 0 0">
                    <inertial pos="-0.000000 0.002372 -0.002389" mass="0.000078" diaginertia="1.262893e-09 1.530211e-09 1.260111e-09"/>
                    <joint name="joint_{module_id}" type="hinge" axis="1 0 0" pos="0 0.001 -0.0055" range="-45 0" limited="true" armature="0.001" damping="0"/>
                    <geom name="joint_marker_bodyLink_{module_id}" type="cylinder" size="0.0002 0.008" pos="0 0.001 -0.0055" quat="0.7071 0 0.7071 0" rgba="0 1 0 1" mass="0"/>
                    <geom name="geom_bodyLink_{module_id}" type="mesh" mesh="bodyLink" rgba="0.2 0.2 0.8 1" fluidshape="ellipsoid" density="1200" fluidcoef="0.6 0.25 1.5 1.0 1.0"/>
                    <body name="connector3_{module_id}" pos="{connector3_pos}" quat="{connector3_quat}">
                        <inertial pos="0.000000 -0.000000 0.001102" mass="0.000037" diaginertia="1.661707e-10 1.661707e-10 1.944327e-10"/>
                        <geom name="geom_connector3_{module_id}" type="mesh" mesh="{connector3_mesh}" rgba="0.2 0.2 0.8 1"/>
                        <geom name="magnet_connector3_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009"/>
                    </body>
                </body>
            </body>
        </body>
"""

rigid_body_tpl = """
        <body name="module_{module_id}" pos="{module_pos}" quat="{module_macro_quat}">
            <freejoint name="free_module_{module_id}"/>
            <inertial pos="0 0 0" mass="1e-05" diaginertia="1e-08 1e-08 1e-08"/>
            <body name="bodyRigid_{module_id}" pos="0.000000000 0.000000000 0.000000000" quat="{module_quat}">
                <inertial pos="-0.000000 0.000000 -0.002331" mass="0.000167" diaginertia="2.914502e-09 2.914502e-09 2.458790e-09"/>
                <geom name="geom_bodyRigid_{module_id}" type="mesh" mesh="bodyRigid" rgba="0.2 0.2 0.8 1" fluidshape="ellipsoid" density="1200" fluidcoef="0.6 0.25 1.5 1.0 1.0"/>
                <body name="connector1_{module_id}" pos="{connector1_pos}" quat="{connector1_quat}">
                    <inertial pos="-0.000000 -0.000000 0.001027" mass="0.000035" diaginertia="1.315291e-10 1.315224e-10 1.494471e-10"/>
                    <geom name="geom_connector1_{module_id}" type="mesh" mesh="{connector1_mesh}" rgba="1 1 1 1"/>
                    <geom name="magnet_connector1_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009"/>
                </body>
                <body name="connector2_{module_id}" pos="{connector2_pos}" quat="{connector2_quat}">
                    <inertial pos="0.000000 0.000000 0.001471" mass="0.000057" diaginertia="3.133186e-10 3.133186e-10 2.768016e-10"/>
                    <geom name="geom_connector2_{module_id}" type="mesh" mesh="{connector2_mesh}" rgba="0 0 0 1"/>
                </body>
                <body name="connector3_{module_id}" pos="{connector3_pos}" quat="{connector3_quat}">
                    <inertial pos="0.000000 0.000000 0.001471" mass="0.000057" diaginertia="3.133186e-10 3.133186e-10 2.768016e-10"/>
                    <geom name="geom_connector3_{module_id}" type="mesh" mesh="{connector3_mesh}" rgba="0.2 0.2 0.8 1"/>
                </body>
            </body>
        </body>
"""

base_xml_skeleton = """<?xml version="1.0" ?>
<mujoco model="FusionExportAssembly">
    <compiler meshdir="../meshes" autolimits="false"/>
    <size nconmax="500" njmax="1500" nstack="100000"/>
    <option timestep="0.01" viscosity="0.0009" integrator="implicitfast">
        <flag contact="enable"/>
    </option>
    <asset>
        <mesh name="bodyBase" file="BodyFoldedSide1.stl" scale="0.001 0.001 0.001"/>
        <mesh name="bodyLink" file="BodyFoldedSide2.stl" scale="0.001 0.001 0.001"/>
        <mesh name="bodyRigid" file="Body.stl" scale="0.001 0.001 0.001"/>
        <mesh name="connectorA" file="SGA.stl" scale="0.001 0.001 0.001"/>
        <mesh name="connectorB" file="SGB.stl" scale="0.001 0.001 0.001"/>
        <mesh name="connectorC" file="SGX.stl" scale="0.001 0.001 0.001"/>
        <material name="silver" specular="1" shininess="0.8" rgba="0.85 0.85 0.9 1"/>
        <mesh name="Magnet" file="Magnet.stl" scale="1 1 1" inertia="shell"/>
        <material name="submerged_glass" rgba="0.6 0.8 0.9 0.4" shininess="0.9" specular="1"/>
    </asset>
    <worldbody>
        <light directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" pos="0 0 1" dir="0 0 -1"/>
        <geom name="wet_glass_floor" 
              type="plane" 
              size="1 1 0.1" 
              material="submerged_glass" 
              friction="0.25 0.005 0.0001" 
              solimp="0.9 0.95 0.001 0.5 2" 
              solref="0.01 1" 
              condim="3"/>
    </worldbody>
</mujoco>
"""

# ---------------------------------------------------------------------
# 6. Build the XML Structure
# ---------------------------------------------------------------------
root = ET.fromstring(base_xml_skeleton)
worldbody = root.find("worldbody")

fold_joints = []
modules_info = {}

for node_id in sorted(G.nodes, key=lambda x: int(x.split('_')[1])):
    num_id = node_id.replace("module_", "")
    m_type = G.nodes[node_id]["module_type"]
    
    modules_info[node_id] = m_type
    
    pos_str = " ".join(f"{x:.9f}" for x in positions[node_id])
    macro_quat_str = " ".join(f"{x:.9f}" for x in quaternions[node_id])
    
    c1_idx = 1
    c2_idx = 2
    c3_idx = 3

    if m_type == "valley fold":
        # swap 2 and 3 connectors positions
        c2_idx = 3
        c3_idx = 2

    # Structural constants mapping
    c1_pos, c1_quat = CONN_POSITIONS[c1_idx], CONN_QUATS[c1_idx]
    c2_pos, c2_quat = CONN_POSITIONS[c2_idx], CONN_QUATS[c2_idx]
    c3_pos, c3_quat = CONN_POSITIONS[c3_idx], CONN_QUATS[c3_idx]
    
    c1_mesh = connector_assignments[node_id][1]
    c2_mesh = connector_assignments[node_id][2]
    c3_mesh = connector_assignments[node_id][3]
    
    # Local module hinge flipping logic
    if m_type == "Mountain fold":
        m_quat = "1 0 0 0"
        fold_joints.append(num_id)
    elif m_type == "valley fold":
        m_quat = "0 1 0 0"  # 180 flip along X axis
        fold_joints.append(num_id)
    else:
        m_quat = "1 0.000000 0.000000 0.000000"
        
    if m_type == "non-foldable":
        body_xml = rigid_body_tpl.format(
            module_id=num_id, module_pos=pos_str, module_macro_quat=macro_quat_str, module_quat=m_quat,
            connector1_pos=c1_pos, connector1_quat=c1_quat, connector1_mesh=c1_mesh,
            connector2_pos=c2_pos, connector2_quat=c2_quat, connector2_mesh=c2_mesh,
            connector3_pos=c3_pos, connector3_quat=c3_quat, connector3_mesh=c3_mesh
        )
    else:
        body_xml = fold_body_tpl.format(
            module_id=num_id, module_pos=pos_str, module_macro_quat=macro_quat_str, module_quat=m_quat,
            connector1_pos=c1_pos, connector1_quat=c1_quat, connector1_mesh=c1_mesh,
            connector2_pos=c2_pos, connector2_quat=c2_quat, connector2_mesh=c2_mesh,
            connector3_pos=c3_pos, connector3_quat=c3_quat, connector3_mesh=c3_mesh
        )
    
    body_element = ET.fromstring(body_xml)
    worldbody.append(body_element)

# ---------------------------------------------------------------------
# 7. Generate Contact, Actuators, and Welds
# ---------------------------------------------------------------------
contact_elem = ET.SubElement(root, "contact")
equality_elem = ET.SubElement(root, "equality")
actuator_elem = ET.SubElement(root, "actuator")

for node_id, m_type in modules_info.items():
    num_id = node_id.replace("module_", "")
    if m_type != "non-foldable":
        ET.SubElement(contact_elem, "exclude", body1=f"bodyBase_{num_id}", body2=f"bodyLink_{num_id}")
        ET.SubElement(contact_elem, "exclude", body1=f"bodyBase_{num_id}", body2=f"connector1_{num_id}")
        ET.SubElement(contact_elem, "exclude", body1=f"bodyBase_{num_id}", body2=f"connector2_{num_id}")
        ET.SubElement(contact_elem, "exclude", body1=f"bodyLink_{num_id}", body2=f"connector3_{num_id}")
    else:
        ET.SubElement(contact_elem, "exclude", body1=f"bodyRigid_{num_id}", body2=f"connector1_{num_id}")
        ET.SubElement(contact_elem, "exclude", body1=f"bodyRigid_{num_id}", body2=f"connector2_{num_id}")
        ET.SubElement(contact_elem, "exclude", body1=f"bodyRigid_{num_id}", body2=f"connector3_{num_id}")

# Cross-module edge assembly welds & filters
for u, v, d in G.edges(data=True):
    u_num = u.replace("module_", "")
    v_num = v.replace("module_", "")
    c1 = d["connector1"]
    c2 = d["connector2"]
    
    # Base connector names
    body1 = f"connector{c1}_{u_num}"
    body2 = f"connector{c2}_{v_num}"
    
    # DYNAMIC FIX: If a slot belongs to a moving link, we route the weld 
    # constraint to the base body so the hinge physics do not invert.
    if G.nodes[u].get("module_type") in ["Mountain fold", "valley fold"] and c1 == 3:
        body1 = f"bodyBase_{u_num}"
    if G.nodes[v].get("module_type") in ["Mountain fold", "valley fold"] and c2 == 3:
        body2 = f"bodyBase_{v_num}"
        
    ET.SubElement(contact_elem, "exclude", body1=f"connector{c1}_{u_num}", body2=f"connector{c2}_{v_num}")
    ET.SubElement(equality_elem, "weld", body1=body1, body2=body2, solref="0.02 1", solimp="0.95 0.99 0.001")

module_ids = list(modules_info.keys())
for i in range(len(module_ids)):
    for j in range(i + 1, len(module_ids)):
        ET.SubElement(contact_elem, "exclude", body1=module_ids[i], body2=module_ids[j])

for idx, num_id in enumerate(fold_joints, start=1):
    ET.SubElement(actuator_elem, "position", 
                  name=f"ctrl_joint{idx}", 
                  joint=f"joint_{num_id}", 
                  kp="1", ctrlrange="-45 0", ctrllimited="true")

# ---------------------------------------------------------------------
# 8. Clean Formatting and Save
# ---------------------------------------------------------------------
xml_str = ET.tostring(root, encoding="utf-8")
parsed_xml = minidom.parseString(xml_str)
pretty_xml = parsed_xml.toprettyxml(indent="    ")

clean_xml = "\n".join([line for line in pretty_xml.splitlines() if line.strip()])

with open("../models/graph6_assembly.xml", "w") as f:
    f.write(clean_xml)

print("Successfully generated 'graph6_assembly.xml' with fixed topological macro-rotations.")