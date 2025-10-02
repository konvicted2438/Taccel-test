import os
import sys
import gmsh

def convert_stl_to_msh(stl_file, mesh_size=1.0, order=1, output_file=None):
    """
    Convert STL file to MSH file using GMSH.
    
    Args:
        stl_file (str): Path to the STL file
        mesh_size (float): Global mesh size (default=1.0)
        order (int): Element order (1=linear, 2=quadratic)
        output_file (str): Output MSH file path (default: same as input with .msh extension)
    """
    if output_file is None:
        output_file = os.path.splitext(stl_file)[0] + ".msh"
    
    # Initialize GMSH
    gmsh.initialize()
    
    # Set verbosity level (0=silent, 1=normal, 2=verbose)
    gmsh.option.setNumber("General.Terminal", 1)
    
    # Import the STL file - will create a surface model
    print(f"Loading STL file: {stl_file}")
    gmsh.merge(stl_file)
    
    # Create a volume from the surface
    print("Creating volume from surface")
    s = gmsh.model.getEntities(2)  # Get all surfaces
    l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
    v = gmsh.model.geo.addVolume([l])
    gmsh.model.geo.synchronize()
    
    # Set mesh size
    print(f"Setting mesh size to {mesh_size}")
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", mesh_size)
    
    # Set element order
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    
    # 3D mesh algorithm (1=Delaunay, 4=Frontal, 7=MMG3D, 9=R-tree)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    
    # Optimize the mesh
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    
    # Generate 3D mesh
    print("Generating volumetric mesh...")
    gmsh.model.mesh.generate(3)
    
    # Save the mesh file
    print(f"Saving mesh to {output_file}")
    gmsh.write(output_file)
    
    # Clean up
    gmsh.finalize()
    print("Mesh generation complete")
    
    return output_file


if __name__ == "__main__":
    # Input STL file
    stl_file = "finray_test/sphere2.STL"
    
    # Check if file exists
    if not os.path.isfile(stl_file):
        print(f"Error: STL file not found at {stl_file}")
        sys.exit(1)
    
    # Output MSH file
    output_file = "finray_test/sphere2.msh"
    
    # Convert with smaller mesh size for finer details
    mesh_size = 5  # Adjust as needed for your simulation
    
    # Run conversion
    convert_stl_to_msh(stl_file, mesh_size=mesh_size, output_file=output_file)