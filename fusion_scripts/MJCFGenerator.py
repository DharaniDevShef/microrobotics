"""
MJCF Generator - Fusion 360 script add-in: reads every component
occurrence's absolute transform in the active design and writes out an
MJCF XML assembly (~/Documents/assembly_model.xml) with each body
positioned/oriented to match its Fusion layout. Run from Fusion 360's
Scripts and Add-Ins panel.
"""

import adsk.core, adsk.fusion, adsk.cam
import os
import traceback
import math

def get_absolute_transform(occ):
    mat = occ.transform.copy()
    parent = occ.assemblyContext
    while parent:
        mat.transformBy(parent.transform)
        parent = parent.assemblyContext
    return mat

# Quaternion helper math functions
def conjugate_quat(q):
    return [q[0], -q[1], -q[2], -q[3]]

def multiply_quat(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]

def rotate_vector(v, q):
    q_conj = conjugate_quat(q)
    v_quat = [0, v[0], v[1], v[2]]
    return multiply_quat(multiply_quat(q, v_quat), q_conj)[1:]

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox('No active Fusion design found.')
            return

        home_dir = os.path.expanduser('~')
        output_file = os.path.join(home_dir, 'Documents', 'assembly_model.xml')

        root = design.rootComponent
        occurrences = root.allOccurrences
        
        # --- FIRST PASS: Gather all components and calculate raw world transforms ---
        valid_data = []
        unique_components = {}
        
        anchor_quat = [1.0, 0.0, 0.0, 0.0]
        anchor_pos = [0.0, 0.0, 0.0]
        found_anchor = False

        for occ in occurrences:
            comp = occ.component
            if comp.bRepBodies.count == 0:
                continue
                
            if comp.name not in unique_components:
                safe_name = "".join([c for c in comp.name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).rstrip()
                unique_components[comp.name] = safe_name
                
            instance_name = occ.name.replace(':', '_').replace(' ', '_')
            world_trans = get_absolute_transform(occ)
            
            # RAW POSITIONS (meters)
            pos_x = world_trans.translation.x * 0.01
            pos_y = world_trans.translation.y * 0.01
            pos_z = world_trans.translation.z * 0.01
            
            r_data = world_trans.asArray()
            tr = r_data[0] + r_data[5] + r_data[10]
            if tr > 0:
                S = math.sqrt(tr + 1.0) * 2
                qw = 0.25 * S
                qx = (r_data[9] - r_data[6]) / S
                qy = (r_data[2] - r_data[8]) / S
                qz = (r_data[4] - r_data[1]) / S
            else:
                if (r_data[0] > r_data[5]) and (r_data[0] > r_data[10]):
                    S = math.sqrt(1.0 + r_data[0] - r_data[5] - r_data[10]) * 2
                    qw = (r_data[9] - r_data[6]) / S
                    qx = 0.25 * S
                    qy = (r_data[1] + r_data[4]) / S
                    qz = (r_data[2] + r_data[8]) / S
                elif r_data[5] > r_data[10]:
                    S = math.sqrt(1.0 + r_data[5] - r_data[0] - r_data[10]) * 2
                    qw = (r_data[2] - r_data[8]) / S
                    qx = (r_data[1] + r_data[4]) / S
                    qy = 0.25 * S
                    qz = (r_data[6] + r_data[9]) / S
                else:
                    S = math.sqrt(1.0 + r_data[10] - r_data[0] - r_data[5]) * 2
                    qw = (r_data[4] - r_data[1]) / S
                    qx = (r_data[2] + r_data[8]) / S
                    qy = (r_data[6] + r_data[9]) / S
                    qz = 0.25 * S

            q_len = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            raw_q = [qw/q_len, qx/q_len, qy/q_len, qz/q_len]
            
            if "BodyFoldedSide1" in comp.name and "_1" in instance_name:
                anchor_quat = raw_q
                anchor_pos = [pos_x, pos_y, pos_z]
                found_anchor = True
                
            valid_data.append({
                'occ': occ,
                'comp': comp,
                'instance_name': instance_name,
                'raw_pos': [pos_x, pos_y, pos_z],
                'raw_quat': raw_q
            })

        # --- TILT CORRECTION CALCULATION ---
        z_prime = rotate_vector([0, 0, 1], anchor_quat)
        axis = [-z_prime[1], z_prime[0], 0] 
        norm = math.sqrt(axis[0]**2 + axis[1]**2)
        
        if norm > 0.0001:
            axis = [axis[0]/norm, axis[1]/norm, axis[2]/norm]
            angle = math.acos(max(-1.0, min(1.0, z_prime[2])))
            s = math.sin(-angle / 2)
            inv_tilt_q = [math.cos(-angle / 2), axis[0]*s, axis[1]*s, axis[2]*s]
        else:
            inv_tilt_q = [1.0, 0.0, 0.0, 0.0]

        # --- SECOND PASS: Apply tilt correction and evaluate bounding box floor ---
        min_z_floor = float('inf')
        
        for data in valid_data:
            rel_pos = [data['raw_pos'][0] - anchor_pos[0], data['raw_pos'][1] - anchor_pos[1], data['raw_pos'][2] - anchor_pos[2]]
            leveled_rel_pos = rotate_vector(rel_pos, inv_tilt_q)
            
            data['fixed_pos'] = [leveled_rel_pos[0] + anchor_pos[0], leveled_rel_pos[1] + anchor_pos[1], leveled_rel_pos[2] + anchor_pos[2]]
            data['fixed_quat'] = multiply_quat(inv_tilt_q, data['raw_quat'])
            
            # FIXED: Grab boundingBox directly from the Component, not physicalProperties
            bbox = data['comp'].boundingBox
            local_half_height = (bbox.maxPoint.z - bbox.minPoint.z) * 0.005 
            
            lowest_point = data['fixed_pos'][2] - local_half_height
            if lowest_point < min_z_floor:
                min_z_floor = lowest_point

        z_offset_correction = -min_z_floor

        # --- WRITE FINAL LEVEL MJCF ---
        xml_lines = []
        xml_lines.append('<mujoco model="FusionExportAssembly">')
        xml_lines.append('    <compiler meshdir="meshes" autolimits="true"/>')
        xml_lines.append('    <option timestep="0.002" gravity="0 0 -9.81"/>')
        xml_lines.append('    <asset>')
        
        for name, safe_name in unique_components.items():
            xml_lines.append(f'        <mesh name="{safe_name}" file="{safe_name}.stl" scale="0.001 0.001 0.001"/>')
        xml_lines.append('    </asset>')
        
        xml_lines.append('    <worldbody>')
        xml_lines.append('        <light directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" pos="0 0 1" dir="0 0 -1"/>')
        xml_lines.append('        <geom name="ground" type="plane" size="1 1 0.01" pos="0 0 0" rgba="0.3 0.3 0.3 1"/>')

        for data in valid_data:
            mesh_name = unique_components[data['comp'].name]
            
            final_x = data['fixed_pos'][0]
            final_y = data['fixed_pos'][1]
            final_z = data['fixed_pos'][2] + z_offset_correction
            
            fq = data['fixed_quat']

            physical_props = data['comp'].getPhysicalProperties(adsk.fusion.CalculationAccuracy.HighCalculationAccuracy)
            mass_kg = physical_props.mass
            com = physical_props.centerOfMass
            com_x = com.x * 0.01
            com_y = com.y * 0.01
            com_z = com.z * 0.01
            
            (_, ixx, iyy, izz, _, _, _) = physical_props.getXYZMomentsOfInertia()
            ixx_m2 = ixx * 0.0001
            iyy_m2 = iyy * 0.0001
            izz_m2 = izz * 0.0001

            if "BodyFoldedSide1" in data['comp'].name:
                rgba = "0.8 0.3 0.3 1"
            elif "BodyFoldedSide2" in data['comp'].name:
                rgba = "0.3 0.3 0.8 1"
            elif "SGA" in data['comp'].name:
                rgba = "0.2 0.7 0.2 1"
            elif "SGB" in data['comp'].name:
                rgba = "0.7 0.2 0.7 1"
            elif "SGX" in data['comp'].name:
                rgba = "0.2 0.7 0.7 1"
            else:
                rgba = "0.5 0.5 0.5 1"

            xml_lines.append(f'        <body name="{data["instance_name"]}" pos="{final_x:.6f} {final_y:.6f} {final_z:.6f}" quat="{fq[0]:.6f} {fq[1]:.6f} {fq[2]:.6f} {fq[3]:.6f}">')
            xml_lines.append(f'            <inertial pos="{com_x:.6f} {com_y:.6f} {com_z:.6f}" mass="{mass_kg:.6f}" diaginertia="{ixx_m2:.6e} {iyy_m2:.6e} {izz_m2:.6e}"/>')
            
            # if "BodyFoldedSide1" in data['comp'].name and "_1" in data['instance_name']:
            #     xml_lines.append('            <joint type="free"/>')
                
            if "BodyFoldedSide2" in data['comp'].name:
                xml_lines.append('            <joint name="rotation_joint_' + data['instance_name'] + '" type="hinge" axis="1 0 0" pos="0 0.001 -0.0055"/>')
                xml_lines.append('            <geom name="joint_marker_' + data['instance_name'] + '" type="cylinder" size="0.0002 0.008" pos="0 0.001 -0.0055" quat="0.7071 0 0.7071 0" rgba="0 1 0 1" mass="0"/>')
                
            xml_lines.append(f'            <geom name="geom_{data["instance_name"]}" type="mesh" mesh="{mesh_name}" rgba="{rgba}"/>')
            xml_lines.append('        </body>')

        xml_lines.append('    </worldbody>')
        xml_lines.append('</mujoco>')

        with open(output_file, 'w') as f:
            f.write('\n'.join(xml_lines))
            
        ui.messageBox(f'Perfect leveled floor assembly generated at:\n{output_file}')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))