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
mesh_finray = meshio.read("./mesh/finray_test/finray_test.msh")
mesh_object = meshio.read("./mesh/finray_test/sphere.msh")
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
# 检查点是否在凸包内
is_in_box = np.zeros(len(points_finray), dtype=int)
for i, point in enumerate(points_finray):
    if hull.find_simplex(point) >= 0:  # 如果点在凸包内
        is_in_box[i] = 1

# 将列表转换为numpy数组
is_in_box_array = np.array(is_in_box)




# Assuming you have another array named `another_array`
empty_obj = np.zeros(len(mesh_object.points))  # replace this with your actual array
empty_finray = np.zeros(len(mesh_finray.points))
all_obj_ones = np.ones(len(points_obj), dtype=int)

# fix the object bottom layer:

y_coordinates = [point[1] for point in points_obj]

# Get the 10 smallest unique y-coordinates
min_y_obj = sorted(set(y_coordinates))[:10]

# Generate a list where 1 means the y-coordinate of the point is in the 10 smallest unique y-coordinates, 0 means not
is_in_min_y = [1 if point[1] in min_y_obj else 0 for point in points_obj]


# Get the 10 largest unique y-coordinates
max_y_obj = sorted(set(y_coordinates), reverse=True)[:10]

# Generate a list where 1 means the y-coordinate of the point is in the 10 largest unique y-coordinates, 0 means not
is_in_max_y = [1 if point[1] in max_y_obj else 0 for point in points_obj]

# 为finray找到x坐标最小的10个节点
smallest_x_finray = plot_largest_x_nodes(points_finray, num_nodes=150, 
                                          title="Finray: Nodes with Smallest X")

# 为物体找到x坐标最小的10个节点
#smallest_x_obj = plot_smallest_x_nodes(points_obj, num_nodes=10, 
#                                       title="Object: Nodes with Smallest X")

#print(f"Finray节点索引: {smallest_x_finray}")
#print(f"Object节点索引: {smallest_x_obj}")


# Convert the list to a numpy array
is_slide_obj = np.array(is_in_min_y)

# 根据需要连接数组
is_fixed_obj_dirichlet = np.concatenate((is_in_box_array, empty_obj))
# Concatenate `is_bottom_layer_array` and `another_array`
is_slided_finray_dirichlet = np.concatenate((empty_finray, all_obj_ones))
#is_fixed_obj_dirichlet = np.concatenate((empty_finray, is_fixed_obj))
#is_fixed_obj_dirichlet = np.concatenate((is_slided_finray, is_fixed_obj))

totalnum = len(points_finray) + len(points_obj)

if len(is_slided_finray_dirichlet) == totalnum and len(is_fixed_obj_dirichlet) == totalnum:

   np.save('./mesh/finray_test/is_fixed_obj_dirichlet.npy', is_fixed_obj_dirichlet)
   np.save('./mesh/finray_test/is_slided_finray_dirichlet.npy', is_slided_finray_dirichlet)

# Convert the list to a numpy array

# 创建要追踪的索引列表
tracked_indices = [212, 216, 220, 210, 1700, 227, 221, 217, 213]

# 为finray创建一个全零数组
is_tracked_finray = np.zeros(len(points_finray), dtype=int)

# 将需要追踪的节点设置为1
for idx in tracked_indices:
    is_tracked_finray[idx] = 1

# 为物体节点创建全零数组（因为我们只追踪finray上的点）
tracked_obj_zeros = np.zeros(len(points_obj), dtype=int)

# 连接两个数组
is_tracked_finray_dirichlet = np.concatenate((is_tracked_finray, tracked_obj_zeros))

# 保存结果
np.save('./mesh/finray_test/is_tracked_finray_dirichlet.npy', is_tracked_finray_dirichlet)

print(f"已创建追踪数组，标记了 {sum(is_tracked_finray)} 个finray节点，共 {len(is_tracked_finray_dirichlet)} 个元素")


# 找到z坐标在5.0到5.4之间的所有点
z_filtered_indices = []
for i, point in enumerate(points_finray):
    if 5.2 <= point[2] <= 5.4:  # 第三个坐标是z轴
        z_filtered_indices.append(i)

# 转换为numpy数组
z_filtered_indices = np.array(z_filtered_indices)

# 可视化这些点（可选）
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 获取对应的点
z_filtered_points = points_finray[z_filtered_indices]
print("Z坐标值:")
for point in z_filtered_points:
    print(f"z = {point[2]}")
# 创建3D图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制点
ax.scatter(z_filtered_points[:, 0], z_filtered_points[:, 1], z_filtered_points[:, 2], 
           color='green', s=50)

# 用索引标记点
for i, idx in enumerate(z_filtered_indices):
    ax.text(z_filtered_points[i, 0], z_filtered_points[i, 1], z_filtered_points[i, 2], 
            f"{idx}", fontsize=10)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(5.0, 5.5)  # 固定Z轴的显示范围
ax.set_title("Points with Z between 5.0 and 5.4")

plt.tight_layout()
plt.show()

# 创建一个Dirichlet数组，用于标记z坐标在5.0到5.4之间的点
is_z_filtered_finray = np.zeros(len(points_finray), dtype=int)

# 将z坐标在5.0到5.4之间的点设置为1
for idx in z_filtered_indices:
    is_z_filtered_finray[idx] = 1

# 为物体节点创建全零数组
z_filtered_obj_zeros = np.zeros(len(points_obj), dtype=int)

# 连接两个数组
is_z_filtered_dirichlet = np.concatenate((is_z_filtered_finray, z_filtered_obj_zeros))

# 保存结果
np.save('./mesh/finray_test/is_z_filtered_dirichlet.npy', is_z_filtered_dirichlet)

print(f"已创建z坐标过滤数组，标记了 {sum(is_z_filtered_finray)} 个z坐标在5.0到5.4之间的finray节点，共 {len(is_z_filtered_dirichlet)} 个元素")