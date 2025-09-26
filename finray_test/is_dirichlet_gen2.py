import meshio
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def plot_largest_x_nodes(points, num_nodes=10, title="Nodes with Largest X Coordinates"):
    """
    找到具有最大x坐标的节点，并绘制它们及其索引号
    
    Args:
        points (numpy.ndarray): 点的数组，形状为(n, 3)表示3D点
        num_nodes (int): 要查找和绘制的最大x节点数量
        title (str): 图的标题
        
    Returns:
        numpy.ndarray: 具有最大x坐标的节点的索引
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 获取x坐标
    x_coordinates = points[:, 0]
    
    # 找到最大x坐标的索引 (使用负索引从排序后的数组末尾获取)
    largest_x_indices = np.argsort(x_coordinates)[-num_nodes:]
    
    # 获取对应的点
    largest_x_points = points[largest_x_indices]
    
    # 创建3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点
    ax.scatter(largest_x_points[:, 0], largest_x_points[:, 1], largest_x_points[:, 2], 
               color='blue', s=50)
    
    # 用索引标记点
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
mesh_finray = meshio.read("./finray_test/finray_test.msh")
mesh_object = meshio.read("./finray_test/sphere.msh")
# 定义盒子的8个角点
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


# 使用Delaunay三角剖分创建凸包
hull = Delaunay(box_corners)

inside_indices = []
for idx, point in enumerate(points_finray):
    if hull.find_simplex(point) >= 0:
        inside_indices.append(idx)

inside_indices = np.asarray(inside_indices, dtype=np.int32)
print("inside_indices", inside_indices)
np.save("./finray_test/in_box_indices.npy", inside_indices)


