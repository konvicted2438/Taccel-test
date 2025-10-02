import meshio
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def plot_largest_x_nodes(points, num_nodes=10, title="Nodes with Largest X Coordinates"):
    """
    Find nodes with the largest x coordinates and plot them with their index numbers
    
    Args:
        points (numpy.ndarray): Array of points with shape (n, 3) representing 3D points
        num_nodes (int): Number of largest x nodes to find and plot
        title (str): Title for the plot
        
    Returns:
        numpy.ndarray: Indices of the nodes with the largest x coordinates
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Get x coordinates
    x_coordinates = points[:, 0]
    
    # Find indices of largest x coordinates (using negative indices to get from the end of the sorted array)
    largest_x_indices = np.argsort(x_coordinates)[-num_nodes:]
    
    # Get the corresponding points
    largest_x_points = points[largest_x_indices]
    
    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(largest_x_points[:, 0], largest_x_points[:, 1], largest_x_points[:, 2], 
               color='blue', s=50)
    
    # Label points with their indices
    for i, idx in enumerate(largest_x_indices):
        ax.text(largest_x_points[i, 0], largest_x_points[i, 1], largest_x_points[i, 2], 
                f"{idx}", fontsize=12)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()
    
    return largest_x_indices


# Load your mesh
mesh_finray = meshio.read("./finray_test/finray_test_vol.msh")
mesh_object = meshio.read("./finray_test/sphere.msh")
# Define the 8 corner points of the box
# box_corners = np.array([
#     [10, -1.9, -35.9], [10, -7.9, -1.4], [-10, -1.9, -35.9], [-10, -7.9, -1.4],
#     [10, -0.1, -0.1], [-10, -0.1, -0.1], [10, 6.1, -34.5], [-10, 6.1, -34.5]
# ])
box_corners = np.array([
    [-8.4, -9.6, -29.2], [-8.4, -15.6, 5.4], [-8.4, -23.5, 4.0], [-8.4, -17.5, -30.6],
    [11.7, -9.6, -29.2], [11.7, -15.6, 5.4], [11.7, -23.5, 4.0], [11.7, -17.5, -30.6]
])
# Extract the points (vertices)
points_finray = mesh_finray.points  # This will be an array of [x, y, z] coordinates
points_obj = mesh_object.points
# Extract the tetrahedral elements (indices)
# `tetra` corresponds to tetrahedral cells (4 indices per element)
if 'tetra' in mesh_finray.cells_dict:
    tet_indices = mesh_finray.cells_dict['tetra']
else:
    raise ValueError("No tetrahedral elements found in the mesh.")


# Use Delaunay triangulation to create a convex hull
hull = Delaunay(box_corners)

inside_indices = []
for idx, point in enumerate(points_finray):
    if hull.find_simplex(point) >= 0:
        inside_indices.append(idx)

inside_indices = np.asarray(inside_indices, dtype=np.int32)
print("inside_indices", inside_indices)
np.save("./finray_test/in_box_indices.npy", inside_indices)


