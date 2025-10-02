import os
import sys
import gmsh
import pyvista as pv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Configuration ---
RADIUS = 0.1
MESH_SIZE = 0.03
OUTPUT_FILENAME = "sphere.msh"

# --- Script ---
gmsh.initialize()
gmsh.model.add("sphere")

# Create geometry
gmsh.model.occ.addSphere(0, 0, 0, RADIUS)
gmsh.model.occ.synchronize()

# Set meshing options for a good quality tetrahedral mesh
gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal-Delaunay for 3D
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), MESH_SIZE)

# Generate 3D tetrahedral mesh
gmsh.model.mesh.generate(3)

# Save the mesh to .msh file
mesh_file_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILENAME)
gmsh.write(mesh_file_path)

gmsh.finalize()

print(f"Mesh successfully generated and saved to: {mesh_file_path}")

# --- Validation Step using PyVista ---
print("\n--- Validating Final Mesh ---")
try:
    # Load the generated mesh
    grid = pv.read(mesh_file_path)
    print(f"Successfully loaded mesh as: {type(grid)}")
    
    # Check for tetrahedral cells (cell type 10)
    if 10 in grid.celltypes:
        num_tetras = grid.extract_cells_by_type(10).n_cells
        print(f"✅ Mesh contains {num_tetras} tetrahedral cells.")
    else:
        print("❌ Warning: Mesh does not contain any tetrahedral cells.")
    print(f"Cell types found: {grid.celltypes}")

except Exception as e:
    print(f"An error occurred during PyVista validation: {e}")