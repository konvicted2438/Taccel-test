import os
import sys
import gmsh
import numpy as np

def convert_geometry_to_msh(cad_file, mesh_size=1.0, order=1, output_file=None, center_at_origin=True):
    """
    Convert STL/STEP/IGES geometry to MSH using Gmsh.
    """
    if output_file is None:
        output_file = os.path.splitext(cad_file)[0] + ".msh"

    ext = os.path.splitext(cad_file)[1].lower()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(os.path.basename(output_file))

    print(f"Loading geometry: {cad_file}")
    if ext in {".step", ".stp", ".iges", ".igs", ".brep"}:
        gmsh.model.occ.importShapes(cad_file)
        gmsh.model.occ.synchronize()
        
        if center_at_origin:
            # Get all entities
            entities = gmsh.model.occ.getEntities()
            
            if entities:
                # Calculate overall bounding box from all entities
                min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
                max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
                
                for dim, tag in entities:
                    try:
                        bbox = gmsh.model.occ.getBoundingBox(dim, tag)
                        min_x = min(min_x, bbox[0])
                        min_y = min(min_y, bbox[1])
                        min_z = min(min_z, bbox[2])
                        max_x = max(max_x, bbox[3])
                        max_y = max(max_y, bbox[4])
                        max_z = max(max_z, bbox[5])
                    except:
                        continue
                
                center_x = (min_x + max_x) / 2.0
                center_y = (min_y + max_y) / 2.0
                center_z = (min_z + max_z) / 2.0
                
                print(f"Original bounding box: min=({min_x:.3f}, {min_y:.3f}, {min_z:.3f}), "
                      f"max=({max_x:.3f}, {max_y:.3f}, {max_z:.3f})")
                print(f"Original center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
                print(f"Moving geometry to origin...")
                
                # Translate all entities to center at origin
                gmsh.model.occ.translate(entities, -center_x, -center_y, -center_z)
                gmsh.model.occ.synchronize()
                
                # Verify new bounding box
                min_x_new, min_y_new, min_z_new = float('inf'), float('inf'), float('inf')
                max_x_new, max_y_new, max_z_new = float('-inf'), float('-inf'), float('-inf')
                
                entities_new = gmsh.model.occ.getEntities()
                for dim, tag in entities_new:
                    try:
                        bbox = gmsh.model.occ.getBoundingBox(dim, tag)
                        min_x_new = min(min_x_new, bbox[0])
                        min_y_new = min(min_y_new, bbox[1])
                        min_z_new = min(min_z_new, bbox[2])
                        max_x_new = max(max_x_new, bbox[3])
                        max_y_new = max(max_y_new, bbox[4])
                        max_z_new = max(max_z_new, bbox[5])
                    except:
                        continue
                
                print(f"New bounding box: min=({min_x_new:.3f}, {min_y_new:.3f}, {min_z_new:.3f}), "
                      f"max=({max_x_new:.3f}, {max_y_new:.3f}, {max_z_new:.3f})")
    else:
        # STL file handling
        gmsh.merge(cad_file)
        
        # Create volume from surface
        faces = gmsh.model.getEntities(2)
        if not faces:
            raise RuntimeError("No surfaces found in the imported geometry.")
        
        if center_at_origin:
            # Get bounding box of all surfaces
            min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
            max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
            
            for dim, tag in faces:
                bbox = gmsh.model.getBoundingBox(dim, tag)
                min_x = min(min_x, bbox[0])
                min_y = min(min_y, bbox[1])
                min_z = min(min_z, bbox[2])
                max_x = max(max_x, bbox[3])
                max_y = max(max_y, bbox[4])
                max_z = max(max_z, bbox[5])
            
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            center_z = (min_z + max_z) / 2.0
            
            print(f"Original center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
            print(f"Moving geometry to origin...")
            
            # Translate all surfaces
            for dim, tag in faces:
                gmsh.model.geo.translate([(dim, tag)], -center_x, -center_y, -center_z)
            gmsh.model.geo.synchronize()
        
        surf_loop = gmsh.model.geo.addSurfaceLoop([f[1] for f in faces])
        gmsh.model.geo.addVolume([surf_loop])
        gmsh.model.geo.synchronize()

    # Set mesh parameters
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", mesh_size)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    print("Generating volumetric mesh...")
    gmsh.model.mesh.generate(3)
    
    # Post-process: Center the mesh after generation
    if center_at_origin:
        print("Post-processing: Centering mesh nodes...")
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        
        if len(node_coords) > 0:
            # Reshape coordinates
            coords = node_coords.reshape(-1, 3)
            
            # Calculate current center
            current_center = coords.mean(axis=0)
            print(f"Current mesh center: {current_center}")
            
            # If center is significantly off, translate all nodes
            if np.linalg.norm(current_center) > 0.01:  # Tolerance of 0.01 units
                print(f"Adjusting mesh center by: {-current_center}")
                
                # Translate all nodes to true center
                coords_centered = coords - current_center
                
                # Set new node positions
                for i, tag in enumerate(node_tags):
                    gmsh.model.mesh.setNode(tag, 
                                           [coords_centered[i, 0], 
                                            coords_centered[i, 1], 
                                            coords_centered[i, 2]],
                                           [])  # Empty list for parametric coordinates
                
                # Verify final centering
                final_center = coords_centered.mean(axis=0)
                print(f"Final mesh center after adjustment: {final_center}")
                print(f"Final mesh bounds: min={coords_centered.min(axis=0)}, max={coords_centered.max(axis=0)}")
            else:
                print(f"Mesh is already well-centered (within tolerance)")
                print(f"Final mesh bounds: min={coords.min(axis=0)}, max={coords.max(axis=0)}")
    else:
        # Just print bounds without centering
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        if len(node_coords) > 0:
            coords = node_coords.reshape(-1, 3)
            print(f"Final mesh bounds: min={coords.min(axis=0)}, max={coords.max(axis=0)}")
            print(f"Final mesh center: {coords.mean(axis=0)}")

    print(f"Saving mesh to {output_file}")
    gmsh.write(output_file)
    gmsh.finalize()
    print("Mesh generation complete.")
    return output_file


if __name__ == "__main__":
    cad_file = "finray_test/w_shape.STEP"

    if not os.path.isfile(cad_file):
        print(f"Error: geometry file not found at {cad_file}")
        sys.exit(1)

    output_file = "finray_test/w_shape.msh"
    mesh_size = 1

    convert_geometry_to_msh(
        cad_file, 
        mesh_size=mesh_size, 
        output_file=output_file,
        center_at_origin=True  # Enable centering
    )