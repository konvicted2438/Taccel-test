import gmsh
import os
import sys
import numpy as np
import pyvista as pv
import meshio

def mesh(path_finger, mesh_sizes, design_name, surf=False, finger_mesh=True):
    gmsh.initialize()
    gmsh.model.add("soft-finger")

    # Load the STEP file
    v = gmsh.model.occ.importShapes(path_finger)
    
    # Synchronize the CAD model with the Gmsh model
    gmsh.model.occ.synchronize()

    # Add physical groups for proper mesh export
    volumes = gmsh.model.getEntities(3)
    if volumes:
        volume_tags = [vol[1] for vol in volumes]
        physical_tag = gmsh.model.addPhysicalGroup(3, volume_tags, 1)
        gmsh.model.setPhysicalName(3, physical_tag, "Volume")

    # Set mesh parameters
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

    # Save the mesh
    if finger_mesh:
        output_path = "./finray_test/finray_test.msh"
        gmsh.write(output_path)
        print(f"Mesh saved to {output_path}")

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

    print(f"Generated mesh with {len(vertices)} vertices and {len(tetra_cells)} tetrahedra")
    print(f"Physical groups added: Volume (tag 1)")

    # Launch GUI if requested
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()
    
    mesh_name = f'fem-{design_name}-{minMeshSize}-{maxMeshSize}-{curvatureSetting}.msh'
    
    return mesh_name, pyvista_mesh


if __name__ == "__main__":
    step_file_path = "./finray_test/O_whole21.STEP"
    mesh_sizes = [6, 6.0, 1.0, 1.0]  # [sizefactor, minMeshSize, maxMeshSize, curvatureSetting]
    
    mesh_name, pv_mesh = mesh(step_file_path, mesh_sizes, "test", surf=False, finger_mesh=True)
    print(f"Mesh generation complete: {mesh_name}")
    print(f"Mesh type: {type(pv_mesh)}")
    print(f"Cell types in mesh: {pv_mesh.celltypes}")