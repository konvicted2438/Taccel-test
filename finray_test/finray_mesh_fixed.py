import gmsh
import os
import sys
import numpy as np
import pyvista as pv

def mesh(path_finger, mesh_sizes, design_name, surf=False, finger_mesh=True):
    gmsh.initialize()
    gmsh.model.add("soft-finger")

    # Load the STEP file
    v = gmsh.model.occ.importShapes(path_finger)
    
    # Synchronize the CAD model with the Gmsh model
    gmsh.model.occ.synchronize()

    # IMPORTANT: Add physical groups for ALL entities
    volumes = gmsh.model.getEntities(3)
    if volumes:
        volume_tags = [vol[1] for vol in volumes]
        physical_tag = gmsh.model.addPhysicalGroup(3, volume_tags, 1)
        gmsh.model.setPhysicalName(3, physical_tag, "Volume")
        print(f"Added physical group 'Volume' with {len(volume_tags)} volumes")

    # Set mesh parameters for finer mesh
    sizefactor = mesh_sizes[0]
    minMeshSize = mesh_sizes[1]
    maxMeshSize = mesh_sizes[2]
    curvatureSetting = mesh_sizes[3]

    gmsh.option.setNumber("Mesh.MeshSizeFactor", sizefactor)
    gmsh.option.setNumber("Mesh.MeshSizeMin", minMeshSize)
    gmsh.option.setNumber("Mesh.MeshSizeMax", maxMeshSize)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvatureSetting)
    
    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    # Save in multiple formats for compatibility
    if finger_mesh:
        # Save as MSH
        msh_path = "./finray_test/finray_test.msh"
        gmsh.write(msh_path)
        print(f"Mesh saved to {msh_path}")
        
        # Also save as VTK for better compatibility
        vtk_path = "./finray_test/finray_test.vtk"
        gmsh.write(vtk_path)
        print(f"Mesh saved to {vtk_path}")

    # Extract mesh data for PyVista
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    vertices = node_coords.reshape(-1, 3)
    
    # Get tetrahedral elements
    tetra_type = gmsh.model.mesh.getElementType("tetrahedron", 1)
    element_tags, element_node_tags = gmsh.model.mesh.getElementsByType(tetra_type)
    # Convert to 0-based indexing for PyVista
    tetra_cells = element_node_tags.reshape(-1, 4).astype(np.int64) - 1
    
    # Create PyVista mesh
    cell_connectivity = np.hstack(
        (np.full((tetra_cells.shape[0], 1), 4, dtype=np.int64), tetra_cells)
    ).ravel()
    cell_types = np.full(tetra_cells.shape[0], pv.CellType.TETRA, dtype=np.uint8)
    pyvista_mesh = pv.UnstructuredGrid(cell_connectivity, cell_types, vertices)
    
    # Save PyVista mesh directly
    pyvista_mesh.save("./finray_test/finray_test_pv.vtk")

    print(f"Generated mesh with {len(vertices)} vertices and {len(tetra_cells)} tetrahedra")

    # Launch GUI if requested
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()
    
    return f'fem-{design_name}-{minMeshSize}-{maxMeshSize}-{curvatureSetting}.msh', pyvista_mesh


if __name__ == "__main__":
    step_file_path = "./finray_test/O_whole21.STEP"
    # Use finer mesh settings
    mesh_sizes = [3.0, 3.0, 1.0, 1]  # [sizefactor, minMeshSize, maxMeshSize, curvatureSetting]
    
    mesh_name, pv_mesh = mesh(step_file_path, mesh_sizes, "test", surf=False, finger_mesh=True)
    print(f"Mesh generation complete: {mesh_name}")