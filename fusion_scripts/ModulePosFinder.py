import adsk.core, adsk.fusion, adsk.cam
import os
import traceback
import re

def get_absolute_transform(occ):
    mat = occ.transform.copy()
    parent = occ.assemblyContext
    while parent:
        mat.transformBy(parent.transform)
        parent = parent.assemblyContext
    return mat

def get_inverse_matrix(matrix):
    inv = matrix.copy()
    inv.invert()
    return inv

def matrix_to_quat(matrix):
    data = matrix.asArray()
    r11, r12, r13 = data[0], data[1], data[2]
    r21, r22, r23 = data[4], data[5], data[6]
    r31, r32, r33 = data[8], data[9], data[10]
    
    tr = r11 + r22 + r33
    if tr > 0:
        s = 0.5 / (tr + 1.0) ** 0.5
        w = 0.25 / s
        x = (r32 - r23) * s
        y = (r13 - r31) * s
        z = (r21 - r12) * s
    elif (r11 > r22) and (r11 > r33):
        s = 2.0 * (1.0 + r11 - r22 - r33) ** 0.5
        w = (r32 - r23) / s
        x = 0.25 * s
        y = (r12 + r21) / s
        z = (r13 + r31) / s
    elif r22 > r33:
        s = 2.0 * (1.0 + r22 - r11 - r33) ** 0.5
        w = (r13 - r31) / s
        x = (r12 + r21) / s
        y = 0.25 * s
        z = (r23 + r32) / s
    else:
        s = 2.0 * (1.0 + r33 - r11 - r22) ** 0.5
        w = (r21 - r12) / s
        x = (r13 + r31) / s
        y = (r23 + r32) / s
        z = 0.25 * s
        
    norm = (w*w + x*x + y*y + z*z) ** 0.5
    return f"{w/norm:.6f} {x/norm:.6f} {y/norm:.6f} {z/norm:.6f}"

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox('No active Fusion design found.')
            return

        xml_path = r"C:\Users\elp25ds\workspace\microrobotics\models\assembly_model.xml"
        if not os.path.exists(xml_path):
            ui.messageBox(f"Target XML file not found at: {xml_path}")
            return

        root_comp = design.rootComponent
        occurrences = root_comp.allOccurrences

        # Explicit mapping layout based directly on your true XML hierarchy
        module_roots = {
            "module_1": "body_1",
            "module_2": "bodyfoldedside1_1",
            "module_3": "body_2",
            "module_4": "bodyfoldedside1_2",
            "module_5": "bodyfoldedside1_3",
            "module_6": "bodyfoldedside1_4",
            "module_7": "bodyfoldedside1_5",
            "module_8": "body_3",
            "module_9": "body_4",
            "module_10": "body_5",
            "module_11": "body_6"
        }

        # 1. Build a clean dictionary of CAD components using flexible string normalization
        cad_transforms = {}
        for occ in occurrences:
            if occ.component.bRepBodies.count == 0:
                continue
            
            # Normalize names: convert "BodyFoldedSide1:2" or "BodyFoldedSide1 (1)" -> "bodyfoldedside1_2"
            raw_name = occ.name.lower().replace(' ', '').replace(':', '_').replace('(', '_').replace(')', '')
            cad_transforms[raw_name] = get_absolute_transform(occ)

        # Helper function to find a transform even if names are slightly mixed up in CAD
        def find_cad_transform(target_token):
            if target_token in cad_transforms:
                return cad_transforms[target_token]
            # Try a partial fallback match
            for cad_name, tx in cad_transforms.items():
                if target_token in cad_name or cad_name in target_token:
                    return tx
            return None

        # Find our core global reference tracking frame anchor (Body_1)
        base_matrix = find_cad_transform("body_1")
        if base_matrix is None:
            ui.messageBox("Error: Could not find 'Body_1' or any variant in your Fusion assembly to set origin layout.")
            return

        inv_base_matrix = get_inverse_matrix(base_matrix)

        # 2. Modify XML entries 
        with open(xml_path, 'r') as file:
            lines = file.readlines()

        new_lines = []
        modified_count = 0

        for line in lines:
            updated_line = line
            
            # Look for any module definition line
            if '<body name="module_' in line or "<body name='module_" in line:
                try:
                    m_name = line.split('name="')[1].split('"')[0] if 'name="' in line else line.split("name='")[1].split("'")[0]
                    m_name_lower = m_name.lower()

                    if m_name_lower in module_roots:
                        cad_token = module_roots[m_name_lower]
                        target_matrix = find_cad_transform(cad_token)

                        if m_name_lower == "module_1":
                            new_pos = "0.000000 0.000000 0.000000"
                            new_quat = "1.000000 0.000000 0.000000 0.000000"
                        elif target_matrix is not None:
                            # Run transform relativity math directly
                            rel_matrix = target_matrix.copy()
                            rel_matrix.transformBy(inv_base_matrix)
                            
                            mx = rel_matrix.translation.x * 0.01
                            my = rel_matrix.translation.y * 0.01
                            mz = rel_matrix.translation.z * 0.01
                            new_pos = f"{mx:.6f} {my:.6f} {mz:.6f}"
                            new_quat = matrix_to_quat(rel_matrix)
                        else:
                            # Skip if CAD variant is truly missing from open workspace viewport
                            new_lines.append(line)
                            continue

                        # Clean existing attributes cleanly using regex
                        line_stripped = line.split('>')[0]
                        line_stripped = re.sub(r'\s+pos="[^"]*"', '', line_stripped)
                        line_stripped = re.sub(r"\s+pos='[^']*'", '', line_stripped)
                        line_stripped = re.sub(r'\s+quat="[^"]*"', '', line_stripped)
                        line_stripped = re.sub(r"\s+quat='[^']*'", '', line_stripped)
                        
                        updated_line = f'{line_stripped} pos="{new_pos}" quat="{new_quat}">\n'
                        modified_count += 1
                except:
                    pass

            # Zero-out the immediate sub-bodies so they don't compound rotation math
            elif '<body name="' in line or "<body name='" in line:
                try:
                    b_name = line.split('name="')[1].split('"')[0] if 'name="' in line else line.split("name='")[1].split("'")[0]
                    b_name_lower = b_name.lower()
                    
                    if b_name_lower in module_roots.values():
                        line_stripped = line.split('>')[0]
                        line_stripped = re.sub(r'\s+pos="[^"]*"', '', line_stripped)
                        line_stripped = re.sub(r"\s+pos='[^']*'", '', line_stripped)
                        line_stripped = re.sub(r'\s+quat="[^"]*"', '', line_stripped)
                        line_stripped = re.sub(r"\s+quat='[^']*'", '', line_stripped)
                        
                        updated_line = f'{line_stripped} pos="0.000000 0.000000 0.000000" quat="1.000000 0.000000 0.000000 0.000000">\n'
                except:
                    pass

            new_lines.append(updated_line)

        with open(xml_path, 'w') as file:
            file.writelines(new_lines)

        ui.messageBox(f"Alignment Finalized Successfully!\nProcessed and completely snapped {modified_count} modules into place.")

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))