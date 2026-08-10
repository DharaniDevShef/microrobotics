"""
Fixed-capacity ("padded") MJCF generator for GPU-batched parallel simulation.

mjcf_generator.build_assembly() compiles one XML per graph, sized exactly
to that graph's module count and edge set -- every candidate produces a
structurally different mujoco.MjModel. That's incompatible with MJWarp's
batched-world GPU parallelism (see src/notebook_creator.py), which compiles
ONE Model and only varies per-world Data (qpos, eq_active) across worlds.

This module builds that one fixed-shape template instead:
  - PAD_MODULES identical, structurally uniform module slots.
  - One weld equality per (module, connector-slot) pair appearing in the
    union of every candidate graph's edges ("edge superset"), inactive by
    default.
  - One hinge-lock equality per slot, inactive by default.
Then, for a given candidate graph, configure_candidate() computes the qpos
and eq_active values for just that graph -- everything a single MJWarp
world needs -- without touching the Model.

Structural simplifications versus mjcf_generator.py (deliberate, matching
the tradeoffs documented in src/notebook_creator.py -- refine later once
this scaffold is validated):
  - Every slot uses the foldable body shape (bodyBase + bodyLink + hinge).
    A "non-foldable" candidate module is approximated by *locking* the
    hinge via an equality constraint rather than swapping in the
    single-body rigid mesh, so its mass distribution is slightly off.
  - The hinge pivot is fixed at the valley-fold joint_z with a widened
    +-90 degree range so both valley- and mountain-fold graphs can share
    the slot; mountain-fold's real ~6mm different joint_z offset is not
    represented.
  - All three connector sites always use the unconnected "connectorC"
    mesh (real mass/inertia, but not the true A/B mesh a mated connector
    would have), since mesh choice is fixed at compile time and can't
    vary per candidate/world.

This script only builds/validates the padded XML on CPU MuJoCo -- batched
GPU execution via mujoco_warp is future work (not installed on this
machine yet).
"""
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import networkx as nx
import numpy as np
import mujoco

try:  # package import (e.g. `from helper_scripts import mjcf_generator_padded`)
    from . import mjcf_generator as mg
except ImportError:  # direct script execution fallback (`python mjcf_generator_padded.py`)
    import mjcf_generator as mg

# Fixed regardless of the current population's actual max module count, so
# every graph -- past, present, or not-yet-drawn -- compiles to the same
# structural shape (same nbody/njnt/nq/neq) and can share one MJWarp model.
PAD_MODULES = 40

CONN_POSITIONS = {
    3: "-0.002017680 -0.001164957 -0.002500001",
    2: "0.002017941 -0.001165050 -0.002500001",
    1: "0.000000439 0.002330720 -0.002500000",
}
CONN_QUATS = {
    3: "0.683012546 0.183013001 -0.683012546 0.183013001",
    2: "0.683013104 0.183012443 0.683013104 -0.183012443",
    1: "0.500000000 -0.500000000 0.500000000 0.500000000",
}
# Relative pose (connector-body-2-in-connector-body-1's-frame) for a mated
# pair of connector slots. A weld with no explicit `relpose` freezes to the
# bodies' *compile-time* qpos0 relative pose -- for padded slots that's the
# arbitrary parked pose, not the real mate, since actual candidate poses are
# only written into `qpos` at runtime. This relpose is purely a function of
# which two slots (1/2/3) mate -- independent of module position/orientation
# -- so it can be baked in statically. Values were derived by compiling a
# real (non-padded) two-module assembly per slot pair with
# mjcf_generator.build_assembly() and reading off the connector bodies'
# relative xpos/xquat after mj_forward.
CONNECTOR_RELPOSE = {
    (1, 1): "0 0.000000878 0.003838560   0 -1 0 0",
    (1, 2): "0 0.000000431 0.003839167   0  1 0 0",
    (1, 3): "0 0.000000397 0.003839439   0  0 -1 0",
    (2, 1): "0 0.000000428 0.003839167   0 -1 0 0",
    (2, 2): "0 -0.000000019 0.003839774  0 -1 0 0",
    (2, 3): "0 -0.000000054 0.003840046  0  0 -1 0",
    (3, 1): "0 -0.000000393 0.003839439  0  0 1 0",
    (3, 2): "0 0.000000054 0.003840046   0  0 1 0",
    (3, 3): "0 0.000000088 0.003840319   0  1 0 0",
}

CONN1_ROT, CONN2_ROT, CONN3_ROT = 180, 120, -120
PARENT_CHILD_DIS = 0.0085  # meters

PARENT_CONNECTOR_REL_OFFSETS = {
    1: np.array([PARENT_CHILD_DIS * np.sin(np.radians(180 - CONN1_ROT)),
                 PARENT_CHILD_DIS * np.cos(np.radians(180 - CONN1_ROT)), 0.0]),
    2: np.array([PARENT_CHILD_DIS * np.sin(np.radians(CONN2_ROT)),
                 PARENT_CHILD_DIS * np.cos(np.radians(CONN2_ROT)), 0.0]),
    3: np.array([PARENT_CHILD_DIS * np.sin(np.radians(CONN3_ROT)),
                 PARENT_CHILD_DIS * np.cos(np.radians(CONN3_ROT)), 0.0]),
}
PARENT_CONNECTOR_ROTATIONS = {
    1: mg.get_rotation_z(np.radians(0)),
    2: mg.get_rotation_z(np.radians(-CONN2_ROT)),
    3: mg.get_rotation_z(np.radians(-CONN3_ROT)),
}
CHILD_CONNECTOR_ROTATIONS = {
    1: mg.get_rotation_z(np.radians(180)),
    2: mg.get_rotation_z(np.radians(60)),
    3: mg.get_rotation_z(np.radians(-60)),
}

SLOT_TPL = """
<body name="module_{i}" pos="{park_pos}" quat="1 0 0 0">
    <freejoint name="free_module_{i}"/>
    <inertial pos="0 0 0" mass="1e-08" diaginertia="1e-08 1e-08 1e-08"/>
    <body name="bodyBase_{i}" pos="0 0 0" quat="1 0 0 0">
        {bodyBase_inertial}
        <geom name="geom_bodyBase_{i}" type="mesh" mesh="bodyBase" rgba="0.2 0.2 0.8 1" contype="0" conaffinity="0"/>
        {rf_sites}
        <body name="connector2_{i}" pos="{c2_pos}" quat="{c2_quat}">
            {connector2_inertial}
            <geom name="geom_connector2_{i}" type="mesh" mesh="connectorC" rgba="0 0 0 1" contype="0" conaffinity="0"/>
        </body>
        <body name="connector3_{i}" pos="{c3_pos}" quat="{c3_quat}">
            {connector3_inertial}
            <geom name="geom_connector3_{i}" type="mesh" mesh="connectorC" rgba="0.2 0.2 0.8 1" contype="0" conaffinity="0"/>
        </body>
        <body name="bodyLink_{i}" pos="0 0 0" quat="1 0 0 0">
            {bodyLink_inertial}
            <joint name="joint_{i}" type="hinge" axis="1 0 0" pos="0 0.001 0.0005" range="-90 90" limited="true" armature="1e-04" damping="0"/>
            <geom name="geom_bodyLink_{i}" type="mesh" mesh="bodyLink" rgba="0.2 0.2 0.8 1" contype="0" conaffinity="0"/>
            <body name="connector1_{i}" pos="{c1_pos}" quat="{c1_quat}">
                {connector1_inertial}
                <geom name="geom_connector1_{i}" type="mesh" mesh="connectorC" rgba="1 1 1 1" contype="0" conaffinity="0"/>
            </body>
        </body>
    </body>
</body>
"""

BASE_XML_SKELETON = """<?xml version="1.0" ?>
<mujoco model="roblet_padded_batch">
    <compiler meshdir="{meshdir}" autolimits="false"/>
    <size nconmax="4000" njmax="8000"/>
    <option timestep="0.01" integrator="implicitfast">
        <flag contact="enable"/>
    </option>
    <asset>
        <mesh name="bodyBase" file="BodyFoldedSide1.stl" scale="0.001 0.001 0.001"/>
        <mesh name="bodyLink" file="BodyFoldedSide2.stl" scale="0.001 0.001 0.001"/>
        <mesh name="connectorC" file="SGX.stl" scale="0.001 0.001 0.001"/>
        <material name="glass" rgba="0.6 0.8 0.9 0.4" shininess="0.9" specular="1"/>
    </asset>
    <worldbody>
        <light directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" pos="0 0 1" dir="0 0 -1"/>
        <geom name="glass_floor" type="plane" size="1 1 0.1" material="glass"
            friction="0.4 0.005 0.0001" solimp="0.9 0.95 0.001 0.5 2" solref="0.02 1" condim="3"/>
        <geom name="wall_north" type="box" pos="0 1.0 0.1" size="1.05 0.02 0.1" group="1"/>
        <geom name="wall_south" type="box" pos="0 -1.0 0.1" size="1.05 0.02 0.1" group="1"/>
        <geom name="wall_east"  type="box" pos="1.0 0 0.1" size="0.02 1.05 0.1" group="1"/>
        <geom name="wall_west"  type="box" pos="-1.0 0 0.1" size="0.02 1.05 0.1" group="1"/>
    </worldbody>
</mujoco>
"""


# module_1 doubles as the control/sensor module: mount the 4 wall-facing
# rangefinders on it, matching mjcf_generator.py. Rangefinders cast along
# their site's local +Z axis, so each site's quat rotates +Z to point at
# the wall it's named after.
RF_SITES = """<site name="rf_east" pos="0 0 0.001" quat="0.70710678 0 0.70710678 0" rgba="0 0 0 0"/>
        <site name="rf_west" pos="0 0 0.001" quat="0.70710678 0 -0.70710678 0" rgba="0 0 0 0"/>
        <site name="rf_north" pos="0 0 0.001" quat="0.70710678 -0.70710678 0 0" rgba="0 0 0 0"/>
        <site name="rf_south" pos="0 0 0.001" quat="0.70710678 0.70710678 0 0" rgba="0 0 0 0"/>"""


def _park_pos(i):
    return f"{(i // 8) * 0.08 - 1.5:.3f} {(i % 8) * 0.08 - 1.5:.3f} -1.0"


def load_candidate(json_path):
    """Parse a graph JSON into module count/type, edges (by 1-based module
    index + connector slot), and each module's global pose -- the exact
    per-candidate information a padded world needs to be configured."""
    with open(json_path) as f:
        data = json.load(f)
    G = nx.node_link_graph(data, edges="edges")
    if not G.is_directed():
        raise ValueError(f"{json_path} is undirected; needs directed edges (module_1 hierarchy)")

    idx_of = {n: int(n.split("_")[1]) for n in G.nodes}
    module_type = {idx_of[n]: G.nodes[n]["module_type"] for n in G.nodes}

    edge_lookup = {}
    for u, v, d in G.edges(data=True):
        edge_lookup[frozenset((u, v))] = {"u": u, "v": v, "s_u": d["connector1"], "s_v": d["connector2"]}

    def edge_local_transform(current, nbr):
        info = edge_lookup[frozenset((current, nbr))]
        parent_slot, child_slot = (
            (info["s_u"], info["s_v"]) if info["u"] == current else (info["s_v"], info["s_u"])
        )
        R_p = PARENT_CONNECTOR_ROTATIONS[parent_slot]
        pos_p = PARENT_CONNECTOR_REL_OFFSETS[parent_slot].copy()
        R_c = CHILD_CONNECTOR_ROTATIONS[child_slot]
        return pos_p, R_p @ R_c.T

    UG = G.to_undirected()
    global_pose = {}
    for comp_idx, comp in enumerate(nx.connected_components(UG)):
        comp_nodes = sorted(comp, key=lambda x: int(x.split("_")[1]))
        start = comp_nodes[0]
        global_pose[start] = (np.array([comp_idx * 0.5, 0.0, 0.0]), np.eye(3))
        queue, visited = [start], {start}
        while queue:
            cur = queue.pop(0)
            cur_pos, cur_R = global_pose[cur]
            for nbr in list(G.successors(cur)) + list(G.predecessors(cur)):
                if nbr in visited:
                    continue
                local_pos, local_R = edge_local_transform(cur, nbr)
                global_pose[nbr] = (cur_pos + cur_R @ local_pos, cur_R @ local_R)
                visited.add(nbr)
                queue.append(nbr)

    # Kept directed (u=parent/connector1-side, v=child/connector2-side): the
    # mate's relpose depends on which side is the "parent" slot, so this
    # order must survive into the weld lookup, not just a sorted pair.
    edges_idx = sorted({
        (idx_of[u], d["connector1"], idx_of[v], d["connector2"])
        for u, v, d in G.edges(data=True)
    })

    return dict(
        name=os.path.splitext(os.path.basename(json_path))[0],
        n_modules=len(G.nodes),
        module_type=module_type,
        edges_idx=edges_idx,
        poses={idx_of[n]: global_pose[n] for n in G.nodes},
    )


def edge_superset(candidates):
    s = set()
    for c in candidates:
        s.update(c["edges_idx"])
    return sorted(s)


def build_padded_template(candidates, pad_n=PAD_MODULES, meshdir=None,
                           weld_solref="0.01 1", weld_solimp="0.99 0.999 0.0001"):
    """Compile one fixed-shape MJCF string covering every candidate's
    module count and edge/connector-slot usage."""
    if meshdir is None:
        meshdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meshes")

    max_modules = max((c["n_modules"] for c in candidates), default=0)
    if max_modules > pad_n:
        raise ValueError(f"a candidate needs {max_modules} modules > PAD_MODULES={pad_n}")

    edges = edge_superset(candidates)
    for (i, _, j, _) in edges:
        if i > pad_n or j > pad_n:
            raise ValueError(f"edge touching module {max(i, j)} exceeds PAD_MODULES={pad_n}")

    xml_parts = [BASE_XML_SKELETON.format(meshdir=meshdir).replace("    </worldbody>\n</mujoco>\n", "")]
    for i in range(1, pad_n + 1):
        xml_parts.append(SLOT_TPL.format(
            i=i, park_pos=_park_pos(i),
            c1_pos=CONN_POSITIONS[1], c1_quat=CONN_QUATS[1],
            c2_pos=CONN_POSITIONS[2], c2_quat=CONN_QUATS[2],
            c3_pos=CONN_POSITIONS[3], c3_quat=CONN_QUATS[3],
            rf_sites=RF_SITES if i == 1 else "",
            bodyBase_inertial=mg.body_inertial("bodyBase"),
            bodyLink_inertial=mg.body_inertial("bodyLink"),
            connector1_inertial=mg.connector_inertial("connectorC", "site1"),
            connector2_inertial=mg.connector_inertial("connectorC", "site23_fold"),
            connector3_inertial=mg.connector_inertial("connectorC", "site23_fold"),
        ))
    xml_parts.append("\n  </worldbody>\n  <equality>")

    for i in range(1, pad_n + 1):
        xml_parts.append(
            f'\n    <joint name="lock_{i}" joint1="joint_{i}" polycoef="0 0 0 0 0" active="false"/>'
        )
    for (i, ci, j, cj) in edges:
        xml_parts.append(
            f'\n    <weld name="weld_{i}_{ci}_{j}_{cj}" body1="connector{ci}_{i}" body2="connector{cj}_{j}" '
            f'relpose="{CONNECTOR_RELPOSE[(ci, cj)]}" '
            f'active="false" solref="{weld_solref}" solimp="{weld_solimp}"/>'
        )
    xml_parts.append("\n  </equality>\n</mujoco>\n")
    return "".join(xml_parts), edges


def configure_candidate(model, cand, pad_n=PAD_MODULES, spawn_height=0.05):
    """qpos (nq,) and eq_active (neq,) for one candidate against a
    compiled padded model -- everything a single MJWarp world needs."""
    qpos = np.zeros(model.nq, dtype=np.float64)
    eq_active = np.zeros(model.neq, dtype=bool)

    active_slots = set(cand["module_type"].keys())
    for i in range(1, pad_n + 1):
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"free_module_{i}")]
        if i in active_slots:
            pos, R = cand["poses"][i]
            quat = mg.matrix_to_quaternion(R)
            qpos[adr:adr + 3] = pos + np.array([0.0, 0.0, spawn_height])
            qpos[adr + 3:adr + 7] = quat
        else:
            px, py, pz = _park_pos(i).split()
            qpos[adr:adr + 3] = [float(px), float(py), float(pz)]
            qpos[adr + 3] = 1.0

    for (i, ci, j, cj) in cand["edges_idx"]:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_{i}_{ci}_{j}_{cj}")
        if eq_id < 0:
            raise KeyError(f"{cand['name']}: edge {(i, ci)}-{(j, cj)} not in template's edge superset")
        eq_active[eq_id] = True
    for i, mtype in cand["module_type"].items():
        if mtype == "non-foldable":
            eq_active[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"lock_{i}")] = True

    return qpos, eq_active


def build_padded_assembly(graph_json_path, out_xml_path, pad_n=PAD_MODULES, meshdir=None,
                           weld_solref="0.01 1", weld_solimp="0.99 0.999 0.0001", spawn_height=0.05):
    """Single-graph convenience wrapper -- same call shape as
    mjcf_generator.build_assembly(graph_json_path, out_xml_path): reads one
    graph JSON, writes one ready-to-load XML file. Unlike build_assembly,
    the written model is padded to a fixed PAD_MODULES capacity (so every
    graph saved from the pattern editor, now or later, compiles to the same
    structural shape), and this candidate's pose/active welds/locks are
    baked in as static attributes so the file simulates correctly with no
    Python-side Data setup -- just `mujoco.MjModel.from_xml_path(...)`.
    """
    cand = load_candidate(graph_json_path)
    template_xml, _ = build_padded_template(
        [cand], pad_n=pad_n, meshdir=meshdir, weld_solref=weld_solref, weld_solimp=weld_solimp)
    root = ET.fromstring(template_xml)

    active_slots = set(cand["module_type"].keys())
    for i in range(1, pad_n + 1):
        if i not in active_slots:
            continue
        pos, R = cand["poses"][i]
        pos = pos + np.array([0.0, 0.0, spawn_height])
        quat = mg.matrix_to_quaternion(R)
        body = root.find(f".//body[@name='module_{i}']")
        body.set("pos", " ".join(f"{x:.9f}" for x in pos))
        body.set("quat", " ".join(f"{x:.9f}" for x in quat))

    for (i, ci, j, cj) in cand["edges_idx"]:
        root.find(f".//weld[@name='weld_{i}_{ci}_{j}_{cj}']").set("active", "true")
    for i, mtype in cand["module_type"].items():
        if mtype == "non-foldable":
            root.find(f".//joint[@name='lock_{i}']").set("active", "true")

    xml_str = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")
    clean_xml = "\n".join(line for line in pretty_xml.splitlines() if line.strip())
    with open(out_xml_path, "w", encoding="utf-8") as f:
        f.write(clean_xml)

    print(f"Padded MuJoCo assembly XML written (PAD_MODULES={pad_n}, "
          f"{cand['n_modules']} active modules, {len(cand['edges_idx'])} welds)!")

    return dict(
        name=cand["name"], n_modules=cand["n_modules"],
        n_welds=len(cand["edges_idx"]), pad_n=pad_n,
    )


def _test_against_graphs(graphs_dir):
    json_paths = sorted(glob.glob(os.path.join(graphs_dir, "*.json")))
    candidates = [load_candidate(p) for p in json_paths]

    template_xml, edges = build_padded_template(candidates)
    model = mujoco.MjModel.from_xml_string(template_xml)
    print(f"Compiled padded template: PAD_MODULES={PAD_MODULES}, nbody={model.nbody}, "
          f"njnt={model.njnt}, nq={model.nq}, neq={model.neq} "
          f"({len(edges)} welds + {PAD_MODULES} locks)")

    data = mujoco.MjData(model)
    for cand in candidates:
        qpos, eq_active = configure_candidate(model, cand)
        data.qpos[:] = qpos
        data.eq_active[:] = eq_active
        mujoco.mj_forward(model, data)

        # Sanity check independent of the solver: with correct qpos, each
        # mated connector pair's *actual* relative pose should already match
        # its baked-in CONNECTOR_RELPOSE (the mate isn't body-origin
        # coincidence -- connectors mate ~3.84mm apart, face-to-face at the
        # embedded magnet -- so compare against the expected offset, not zero).
        max_mate_err = 0.0
        for (i, ci, j, cj) in cand["edges_idx"]:
            b1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"connector{ci}_{i}")
            b2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"connector{cj}_{j}")
            rel_p = np.array([float(x) for x in CONNECTOR_RELPOSE[(ci, cj)].split()[:3]])
            R1 = data.xmat[b1].reshape(3, 3)
            expected_p2 = data.xpos[b1] + R1 @ rel_p
            max_mate_err = max(max_mate_err, float(np.linalg.norm(data.xpos[b2] - expected_p2)))

        for _ in range(50):
            mujoco.mj_step(model, data)
        n_bad = int(np.sum(~np.isfinite(data.qpos)))
        n_welds = len(cand["edges_idx"])
        n_locks = sum(1 for t in cand["module_type"].values() if t == "non-foldable")
        status = "OK" if n_bad == 0 else f"FAILED ({n_bad} non-finite qpos)"
        print(f"  {cand['name']:>16s}  modules={cand['n_modules']:2d}  "
              f"welds={n_welds:2d}  locks={n_locks:2d}  "
              f"max_mate_pose_err={max_mate_err * 1000:.4f}mm  {status}")

    print("\nbuild_padded_assembly() single-graph export check:")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    for json_path in json_paths:
        name = os.path.splitext(os.path.basename(json_path))[0]
        out_xml = os.path.join(out_dir, f"padded_{name}.xml")
        build_padded_assembly(json_path, out_xml)
        m = mujoco.MjModel.from_xml_path(out_xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for _ in range(50):
            mujoco.mj_step(m, d)
        n_bad = int(np.sum(~np.isfinite(d.qpos)))
        print(f"  {name:>16s}  reloaded from {out_xml}  {'OK' if n_bad == 0 else f'FAILED ({n_bad} non-finite)'}")


if __name__ == "__main__":
    graphs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "graphs")
    _test_against_graphs(graphs_dir)
