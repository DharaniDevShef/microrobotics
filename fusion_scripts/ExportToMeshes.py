"""
Export To Meshes - Fusion 360 script add-in: exports every uniquely-named
component in the active design's assembly tree as an STL file into
~/Documents/meshes, skipping components with no solid bodies or already
exported. Run from Fusion 360's Scripts and Add-Ins panel.
"""

import adsk.core, adsk.fusion, adsk.cam
import os
import traceback

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        
        # Get active design
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox('No active Fusion design found. Open your model first.')
            return

        # SAFE FALLBACK: Target the user's local Documents directory directly
        # This completely avoids cloud folder permission conflicts
        home_dir = os.path.expanduser('~')
        output_folder = os.path.join(home_dir, 'Documents', 'meshes')
            
        # Create the 'meshes' folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Gather all individual component references in the assembly tree
        occurrences = design.rootComponent.allOccurrences
        
        # Keep track of component names we have already exported to avoid duplicate files
        exported_components = set()
        export_mgr = design.exportManager
        
        count = 0
        for occ in occurrences:
            comp = occ.component
            
            # Skip if this component definition has already been written out
            if comp.name in exported_components:
                continue
                
            # Skip components that contain no physical solid bodies
            if comp.bRepBodies.count == 0:
                continue

            # Clean up the component name to make it a safe operating system file name
            safe_name = "".join([c for c in comp.name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).rstrip()
            file_path = os.path.join(output_folder, f"{safe_name}.stl")
            
            # Configure the STL export options
            stl_options = export_mgr.createSTLExportOptions(comp, file_path)
            stl_options.isBinaryFormat = True
            stl_options.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
            
            # Execute the export operation
            export_mgr.execute(stl_options)
            
            exported_components.add(comp.name)
            count += 1
            
        ui.messageBox(f'Successfully exported {count} unique components to:\n{output_folder}')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))