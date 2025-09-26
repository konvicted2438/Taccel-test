import os
import sys

import math
import tetgen
import subprocess
import pyvista as pv
import matplotlib.pyplot as plt
import meshio
import numpy as np
import trimesh

def pv_viewmesh(mesh_filepath: str = None):
    """
    IMPORTANT: The following file extensions are supported by pv.read()

    Abaqus (.inp), ANSYS msh (.msh), AVS-UCD (.avs), CGNS (.cgns), DOLFIN XML (.xml), 
    Exodus (.e, .exo), FLAC3D (.f3grid), H5M (.h5m), Kratos/MDPA (.mdpa), Medit (.mesh, .meshb),
    MED/Salome (.med), Nastran (bulk data, .bdf, .fem, .nas), Netgen (.vol, .vol.gz),
    Neuroglancer precomputed format, Gmsh (format versions 2.2, 4.0, and 4.1, .msh),
    OBJ (.obj), OFF (.off), PERMAS (.post, .post.gz, .dato, .dato.gz), PLY (.ply),
    STL (.stl), Tecplot .dat, TetGen .node/.ele, SVG (2D output only) (.svg), 
    SU2 (.su2), UGRID (.ugrid), VTK (.vtk), VTU (.vtu), WKT (TIN) (.wkt), XDMF (.xdmf, .xmf).
    """

    mesh_pv = pv.read(mesh_filepath) #pv.read(os.path.join('mesh_files', filename))
    pl = pv.Plotter(off_screen = False) #optional headless rendering
    pl.add_axes()
    pl.add_mesh(mesh_pv, color='green', opacity = 0.2, show_edges=True, 
                            edge_color = 'black', line_width = 0.5)
    pl.show()

def create_rigid_box_stl_trimesh(x_length = 0.06, y_length = 0.01, z_length = 0.04, output_file_path = "assets/objects/soft_gripper/rigid_box_softgripper.stl"):
    """
    Box dimensions (full length along each axis): x_length (x-axis), y_length (y-axis), z_length (z-axis)
    """
    # Create a box centered at the origin
    box = trimesh.creation.box(extents=(x_length, y_length, z_length))

    # Export to STL
    box.export(output_file_path)

def kabsch(P, Q):
    """Compute the best-fit rotation matrix between two point sets P and Q,
        where P are points in the undeformed frame, and Q are points in a 
        deformed frame"""
    
    # Center the points
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    # Covariance matrix
    H = P_centered.T @ Q_centered

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    P_origin = np.array([0,0,0])
    P_estimated = (Q - Q_centroid) @ R + P_centroid
    # Correct placement of R.T: verify by swapping R <----> R.T for P_estimated and Q_estimated/see which returns lowest 'error'
    Q_estimated = (P_origin - P_centroid) @ R.T + Q_centroid 
    # # Check the difference (should be near zero)
    error = np.linalg.norm(P_estimated - P)
    # print(f"Alignment error: {error}")
    # print("Is q_estimated (0, 0, 1)? Ans: ", Q_estimated)

    #NOTE: R = R_G/R_B (rotates World frame stuff into Geom frame), Q_estimated functions as t_G
    return R, Q_estimated, error 

def voxel_3D_binary_array_to_stl(voxel_array, voxel_size=0.001, save_filepath ='D1_leftfinger.stl'):
    """
    Convert a 3D boolean voxel array to an STL by triangulating only surface faces.

    Args:
        voxel_array: 3D numpy array of dtype=bool, shape (nely, nelx, nelz)
        voxel_size: physical size of a voxel (defaults to 1.0)
        filename: output filename for STL
    """
    nely, nelx, nelz = voxel_array.shape
    triangles = []

    # Define cube face normals and relative vertex indices
    face_configs = [
        # Each face: normal direction, and the 4 corner offsets of that face
        ([0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),  # -y (bottom)
        ([0, 1, 0],  [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),  # +y (top)
        ([0, 0, -1], [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]]),  # -z (back)
        ([0, 0, 1],  [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]),  # +z (front)
        ([-1, 0, 0], [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]]),  # -x (left)
        ([1, 0, 0],  [[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]])   # +x (right)
    ]

    # Iterate through the voxel array
    for j in range(nely):
        for i in range(nelx):
            for k in range(nelz):
                if not voxel_array[j, i, k]:
                    continue  # skip empty voxels

                for normal, face in face_configs:
                    # Calculate neighbouring voxel
                    ni, nj, nk = i + normal[0], j + normal[1], k + normal[2]
                    # Is neighbouring voxel in our domain? If not, triangulate this face and check
                    if (0 <= nj < nely) and (0 <= ni < nelx) and (0 <= nk < nelz):
                        if voxel_array[nj, ni, nk]:
                            continue  # neighboring voxel is filled → face is internal

                    # This face is on the surface — construct two triangles
                    base = np.array([i, j, k], dtype=float) * voxel_size
                    vertices = [base + np.array(offset) * voxel_size for offset in face]
                    # print('base: ', base)
                    # print("normal: ", normal)
                    # print("vertices = ", vertices)
                    # quit()

                    # Add two triangles per face
                    triangles.append([vertices[0], vertices[1], vertices[2]])
                    triangles.append([vertices[0], vertices[2], vertices[3]])

    # Convert to Trimesh mesh and export
    mesh = trimesh.Trimesh(vertices=np.array(triangles).reshape(-1, 3),
                           faces=np.arange(len(triangles)*3).reshape(-1, 3),
                           process=True)
    
    # mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    #                    faces=[[0, 2, 1], [2,3,1], [0,3,2], [0,1,3]],
    #                    process=False)
    print(f"\nIs {save_filepath} watertight: ", mesh.is_watertight)
    # Apply 180-degree rotation about the x-axis to get mesh coordinate axes aligned as desired
    R = trimesh.transformations.rotation_matrix(
        angle=np.pi,              # 180 degrees
        direction=[1, 0, 0],      # x-axis
        point=[0, 0, 0]           # rotate about origin
    )
    mesh.apply_transform(R)
    mesh.export(save_filepath)
    print(f"STL file written to: {save_filepath}")

def create_D0_binary_array(nelx = 150, nely = 70, nelz = 30):

    # Initialise 2D D0 array
    D0 = np.ones(nely*nelx, dtype = float)
    passive = np.zeros((nely, nelx), dtype=int)
    for i in range(nelx):
        for j in range(nely):
            # Check if the element (i, j) is within the triangular void region
            if j >= -i + nelx:
                passive[j, i] = 1  # Set to 1 for void (non-design) region
            # Check if the element (i,j) is along the bottom boundary
            if j == nely - 1 and j < -i + nelx:
                passive[j, i] = 2  # Set to 2 for solid (non-design) region

    D0.reshape(nely, nelx, order = 'F')[passive == 1] = 0
    #D0.reshape(nely, nelx, order = 'F')[passive == 2] = 1
    return D0, passive

def create_D0_modified_binary_array(nelx = 150, nely = 70, nelz = 30):

    D0 = np.ones(nely*nelx, dtype = float)
    init_passive = np.zeros((nely, nelx), dtype=int)
    passive = np.zeros((nely, nelx), dtype=int)

    for i in range(nelx):
        for j in range(nely):
            # Check if the element (i, j) is within the triangular void region
            if nely - 1 - int(0.55*i) > j:
                init_passive[j, i] = 1
            if j >= -i + nelx:
                passive[j, i] = 1  # Set to 1 for void (non-design) region
            
    D0.reshape(nely, nelx, order = 'F')[init_passive == 1] = 0
    D0.reshape(nely, nelx, order = 'F')[passive == 1] = 0
    return D0, passive

def filter_points_on_line(points, line_start, line_end, epsilon=1e-4):
    """
    Returns indices of points that lie on the line segment from line_start to line_end.

    Args:
        points (np.ndarray): Array of shape (N, 3) with point coordinates.
        line_start (array-like): 3D coordinates of line start.
        line_end (array-like): 3D coordinates of line end.
        epsilon (float): Tolerance for floating-point comparison.

    Returns:
        np.array: Indices of points within the line, 
        indexed from id = 0,...,n_points - 1. GMSH is indexed from 1.
    """
    line_start = np.array(line_start, dtype=np.float64)
    line_end = np.array(line_end, dtype=np.float64)
    line_vec = line_end - line_start
    line_len_sq = np.dot(line_vec, line_vec)

    result_ids = []

    for i, p in enumerate(points):
        vec = p - line_start
        cross = np.cross(vec, line_vec)
        if np.linalg.norm(cross) > epsilon:
            continue  # not on line (not parallel)

        t = np.dot(vec, line_vec) / line_len_sq
        # checking that we are on the line AND within the start/end point (not beyond it)
        if -epsilon <= t <= 1 + epsilon: 
            result_ids.append(i)

    # + 1 for 1-based indexing as in GMSH
    return np.array(result_ids)

def plot_1D_binary_or_density(x, nelx, nely, iteration):

    x = x.reshape((nely, nelx), order = 'F')
    plt.figure(1, figsize = (8*2, 6*2))
    plt.imshow(-x, cmap='gray', interpolation='none')
    title = r'$\rho(\mathbf{{x}})$ at iteration {}'.format(iteration)
    plt.title(title)
    plt.show()

def plot_2D_binary_or_density(x, iteration):

    plt.figure(1, figsize = (8*2, 6*2))
    plt.imshow(-x, cmap='gray', interpolation='none')
    title = r'$\rho(\mathbf{{x}})$ at iteration {}'.format(iteration)
    plt.title(title)
    plt.show()

def plot_3D_binary_or_density(x, iteration):

    # Prepare the figure and 3D axis once for plotting below
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap("Greys")
    norm = plt.Normalize(0,1)

    ax.voxels(x, facecolors=cmap(norm(x)), edgecolors='black', alpha=0.8)
    ax.set_aspect('equal')
    ax.view_init(elev=-159, azim=-48, roll=0)
    ax.set_title(r'$\rho(\mathbf{{x}})$ at iteration {}'.format(iteration))
    ax.set_axis_off()
    plt.show()
    
def plot_CPs2maxforces(cps2maxforces: dict, force_mag: float, title: str, 
                       mesh_filepath: str, save_filepath: str = None):
    
    CPs = np.array(list(cps2maxforces.keys()))
    forces = np.array(list(cps2maxforces.values()))
    pv_mesh = pv.read(mesh_filepath)

    pl = pv.Plotter(off_screen=True)  # Off-screen rendering needed for saving PNG
    pl.add_axes()
    pl.add_title(title, font_size=20)
    pl.add_mesh(pv_mesh, color='green', opacity=0.2, show_edges=True, 
                edge_color='black', line_width=0.5)
    pl.add_points(CPs, color='red', point_size=10)
    pl.add_arrows(CPs, forces, mag=force_mag, show_scalar_bar=False, color='purple')


    if save_filepath:
        pl.screenshot(save_filepath)  # Save as PNG
    else:
        pl.show()
    return

def pv_tetrahedralize(mesh_filepath: str = 'D1_leftfinger.stl', save_filename: str = "D1_leftfinger", PLOT: bool = False):

    pv.set_plot_theme('document')
    mesh = pv.read(mesh_filepath)

    tet = tetgen.TetGen(mesh)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid
    grid.save(f"{save_filename}" + ".vtk")

    if PLOT:
        grid.plot(show_edges=True)

def run_fTetWild(input_mesh_filepath, output_mesh_filepath, lr = 0.08,
                 epsr = 1e-3, COARSEN = True, 
                 executable_path="../fTetWild/build/FloatTetwild_bin"):
    
    if COARSEN:
        command = [
            executable_path,
            "--input", input_mesh_filepath,
            "--lr", str(lr), 
            "--epsr", str(epsr), #  epsilon-envelope represents the maximal deviation from the input surface the user is willing to accept. 
            "--no-binary",
            "--no-color",
            "--coarsen",
            "--output",  output_mesh_filepath
        ]
    else:
        command = [
            executable_path,
            "--input", input_mesh_filepath,
            "--lr", str(lr), # 0.5
            "--epsr", str(epsr), #  epsilon-envelope represents the maximal deviation from the input surface the user is willing to accept. 
            "--no-binary",
            "--no-color",
            "--output",  output_mesh_filepath
        ]
    print(f"\nTetrahedralizing {input_mesh_filepath} via fTetWild...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("fTetWild STDOUT:\n", result.stdout)
        print("fTetWild STDERR:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("fTetWild failed with error:\n", e.stderr)

def translate_fTetWild_msh(mesh_filepath: str):

    # Load original mesh
    mesh = meshio.read(mesh_filepath)

    # Apply translation
    translation = np.array([0, 0.07, 0.03])
    translated_points = mesh.points + translation

    # Create new mesh object with translated points
    translated_mesh = meshio.Mesh(points=translated_points, cells=mesh.cells, point_data=mesh.point_data, cell_data=mesh.cell_data)

    # Save as Gmsh 2.2 ASCII
    meshio.write(mesh_filepath, translated_mesh, file_format="gmsh22", binary = False) # Gmsh 2.2

def main():


    # MESH_DIR = os.path.join(os.getcwd(),"assets", "objects", "soft_gripper", "grasp_sphere_E=5.1e5")
    # run_fTetWild(os.path.join(MESH_DIR, f"D1_leftfinger.stl"), os.path.join(MESH_DIR, f"D1_leftfinger.msh"), 
    #              executable_path="../fTetWild/build/FloatTetwild_bin")
    # create_rigid_box_stl_trimesh()

    MESH_DIR = os.path.join("assets", "objects", "paper1_eval", "test_grasp_stars_hourglass_pyramid")
    run_fTetWild(os.path.join(MESH_DIR, f"gear.stl"), os.path.join(MESH_DIR, f"test_gear_vol.msh"), 
                executable_path="../fTetWild/build/FloatTetwild_bin", COARSEN=False)
    # Offset .msh generated by fTetWild by [0, 0.07, 0.03] to ensure we have our desired Cartesian coordinate system for TO and simulation/kabsch.
    #translate_fTetWild_msh(os.path.join(MESH_DIR, f"test_gear_vol.msh"))
if __name__ == "__main__":
    main()
