# ------------------------------------------------------------------------------
#  Gmsh Python 
#  STEP import and manipulation, geometry partitioning
# ------------------------------------------------------------------------------

# The OpenCASCADE CAD kernel allows to import STEP files and to modify them. 

import gmsh
import math
import os
import sys
# from finray_design_auto import save_path
import meshio
from pymeshfix import _meshfix
import numpy as np
import pyvista as pv
#mesh/finray_test/O_whole2.step
# Define the path to the STEP or STL file
step_file_path = "./finray_test/O_whole21.STEP"  # Update this path to your actual file
# Save the mesh to a file
mesh_file_path = "./finray_test/O_whole21.msh"  # Output path for the numpy file
mesh_size = 0.25  # Mesh density for STEP files
num_divisions = 3  # Number of mesh divisions per 2π radians for STEP files
target_density = "3"  # Target mesh density for STL files
display_model = True  # Whether to display the model using Gmsh's GUI
# mesh(path_basic, mesh_sizes, design_name, surf=False, finger_mesh=False)

def mesh(path_finger, mesh_sizes, design_name, surf=False, finger_mesh=True):

    gmsh.initialize()
    gmsh.model.add("soft-finger")

    # Load a STEP file (using `importShapes' instead of `merge' allows to directly
    # retrieve the tags of the highest dimensional imported entities):

    path = os.path.dirname(os.path.abspath(__file__))

    
    #initial model to generate mesh
    # v = gmsh.model.occ.importShapes(os.path.join(path, 'data/finray_

    v = gmsh.model.occ.import_shapes(path_finger)

    # If we had specified
    #
    # gmsh.option.setString('Geometry.OCCTargetUnit', 'M')
    #
    # before merging the STEP file, OpenCASCADE would have converted the units to
    # meters (instead of the default, which is millimeters).

    # Get the bounding box of the volume:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(
        v[0][0], v[0][1])

    # We want to slice the model into N slices, and either keep the volume slices
    # or just the surfaces obtained by the cutting:

    N = 6  # Number of slices
    dir = 'X' # Direction: 'X', 'Y' or 'Z'
    # surf = False  # Keep only surfs

    dx = (xmax - xmin)
    dy = (ymax - ymin)
    dz = (zmax - zmin)
    L = dz if (dir == 'X') else dx
    H = dz if (dir == 'Y') else dy

    # Create the first cutting plane:
    s = []
    s.append((2, gmsh.model.occ.addRectangle(xmin, ymin, zmin, L, H)))
    if dir == 'X':
        gmsh.model.occ.rotate([s[0]], xmin, ymin, zmin, 0, 1, 0, -math.pi/2)
    elif dir == 'Y':
        gmsh.model.occ.rotate([s[0]], xmin, ymin, zmin, 1, 0, 0, math.pi/2)
    tx = dx / N if (dir == 'X') else 0
    ty = dy / N if (dir == 'Y') else 0
    tz = dz / N if (dir == 'Z') else 0
    gmsh.model.occ.translate([s[0]], tx, ty, tz)

    # Create the other cutting planes:
    for i in range(1, N-1):
        s.extend(gmsh.model.occ.copy([s[0]]))
        gmsh.model.occ.translate([s[-1]], i * tx, i * ty, i * tz)

    # Fragment (i.e. intersect) the volume with all the cutting planes:
    gmsh.model.occ.fragment(v, s)

    # Now remove all the surfaces (and their bounding entities) that are not on the
    # boundary of a volume, i.e. the parts of the cutting planes that "stick out" of
    # the volume:
    gmsh.model.occ.remove(gmsh.model.occ.getEntities(2), True)

    gmsh.model.occ.synchronize()

    if surf:
        # If we want to only keep the surfaces, retrieve the surfaces in bounding
        # boxes around the cutting planes...
        eps = 1e-3
        s = []
        for i in range(1, N):
            xx = xmin if (dir == 'X') else xmax
            yy = ymin if (dir == 'Y') else ymax
            zz = zmin if (dir == 'Z') else zmax
            s.extend(gmsh.model.getEntitiesInBoundingBox(
                xmin - eps + i * tx, ymin - eps + i * ty, zmin - eps + i * tz,
                xx + eps + i * tx, yy + eps + i * ty, zz + eps + i * tz, 2))
        # ...and remove all the other entities (here directly in the model, as we
        # won't modify any OpenCASCADE entities later on):
        dels = gmsh.model.getEntities(2)
        for e in s:
            dels.remove(e)
        gmsh.model.removeEntities(gmsh.model.getEntities(3))
        gmsh.model.removeEntities(dels)
        gmsh.model.removeEntities(gmsh.model.getEntities(1))
        gmsh.model.removeEntities(gmsh.model.getEntities(0))

    # Finally, let's specify a global mesh size and mesh the partitioned model:
    # minMeshSize = round(float(sys.argv[2]), 2)
    # maxMeshSize = round(float(sys.argv[3]), 2)
    # curvatureSetting = round(float(sys.argv[4]), 2)

    # mesh_sizes = [1, 2, 3, 2]

    # Manual Override
    sizefactor = mesh_sizes[0]
    minMeshSize = mesh_sizes[1]
    maxMeshSize = mesh_sizes[2]
    curvatureSetting = mesh_sizes[3]

    

    gmsh.option.setNumber("Mesh.MeshSizeFactor", sizefactor)
    gmsh.option.setNumber("Mesh.MeshSizeMin", minMeshSize)
    gmsh.option.setNumber("Mesh.MeshSizeMax", maxMeshSize)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvatureSetting)
    
    gmsh.model.mesh.generate(3)

    a = str(design_name)

   
    if finger_mesh:
        gmsh.write("./finray_test/finray_test.msh")
        
        
        mesh_name = f'/mesh/fem-{a}-{minMeshSize}-{maxMeshSize}-{curvatureSetting}.msh'
        


        # Launch the GUI to see the results:
    if '-nopopup' in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()

    mesh_result = meshio.read("./finray_test/finray_test.msh")
    vertices = mesh_result.points

    tetra_cells = mesh_result.get_cells_type("tetra")
    if tetra_cells.size == 0:
        raise ValueError("Generated mesh does not contain tetrahedral cells.")
    tetra_cells = tetra_cells.astype(np.int64)

    meshio.write(
        "./finray_test/finray_test.msh",
        meshio.Mesh(points=vertices, cells=[("tetra", tetra_cells)]),
        file_format="gmsh",
    )

    cell_connectivity = np.hstack(
        (np.full((tetra_cells.shape[0], 1), 4, dtype=np.int64), tetra_cells)
    ).ravel()
    cell_types = np.full(tetra_cells.shape[0], pv.CellType.TETRA, dtype=np.uint8)
    pyvista_mesh = pv.UnstructuredGrid(cell_connectivity, cell_types, vertices)

    faces = mesh_result.cells_dict.get("triangle")
    if faces is not None and faces.size:
        meshfix = _meshfix.PyTMesh()
        meshfix.load_array(vertices, faces)
        meshfix.clean(max_iters=10, inner_loops=3)
        meshfix.save_file(mesh_file_path)

    print("optimised done")
    print(type(pyvista_mesh))
    print(pyvista_mesh.celltypes)
    assert isinstance(pyvista_mesh, pv.UnstructuredGrid)

    return mesh_name, pyvista_mesh


if __name__ == "__main__":
    mesh(step_file_path, [5, 5.0, 1.0, 1.0], "ini test")






