import os
import sys
import argparse
import pathlib
from typing import List
import gmsh
from copy import copy

"""
Code sourced from: https://github.com/mohammad200h/GMSHConverter/tree/main
The code was created in response to this MuJoCo discussion thread I initiated: https://github.com/google-deepmind/mujoco/issues/1543#issuecomment-2025785431
"""

class BaseExtractor(object):
  def __init__(self, input_file_path, output_file_path):
    input_path = pathlib.Path(input_file_path)
    gmsh.open(input_path.as_posix())
    self.output_file_path = pathlib.Path(output_file_path)

    self.process()

  def process(self):
    self.write41_mesh_format_41()
    self.write41_entity_header()
    self.write41_node_header()
    self.write41_nodes_indices()
    self.write41_nodes()
    self.write41_nodes_header_end()
    self.write41_element_header()
    self.write41_elements()
    self.write41_element_header_end()

  def has_msh_extension(self,filename):
    filename_without_extension, extension = os.path.splitext(filename)
    return filename_without_extension, extension.lower() == '.msh'

  def split_path_and_filename(self,path):
    directory, filename = os.path.split(path)
    return directory + os.sep, filename

  def produce_path(self, path, is_volume = True,is_obj = False) -> str:
    path,filename = self.split_path_and_filename(path)
    filename_without_extension, there_is_an_extension = self.has_msh_extension(filename)
    new_file_name = ""
    if there_is_an_extension:
      new_file_name = filename_without_extension
    else:
      new_file_name = filename

    if is_volume:
      new_file_name +="_vol.msh"
    else:
      if is_obj:
        new_file_name +="_surf.obj"
      else:
        new_file_name +="_surf.msh"

    return  path + new_file_name

  def get_num_points(self):
    #getting nodes in write format
    node_indexes,points,_ = gmsh.model.mesh.getNodes()

    return len(node_indexes)

  def get_points(self):
    return gmsh.model.mesh.getNodes()[1].reshape(-1,3)

  def get_max_xyz(self):
    points = self.get_points()

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    max_x = max(x)
    max_y = max(y)
    max_z = max(z)

    min_x = min(x)
    min_y = min(y)
    min_z = min(z)

    return (min_x, min_y, min_z, max_x, max_y, max_z)

  def write41_mesh_format_41(self):
    with self.output_file_path.open('w') as f:
      f.write('$MeshFormat\n')
      f.write('4.1 0 8\n')
      f.write('$EndMeshFormat\n')

  def write41_nodes_header_end(self):
    with self.output_file_path.open('a') as f:
      f.write('$EndNodes\n')

  def write41_nodes_indices(self):
    num_nodes = self.get_num_points()
    with self.output_file_path.open('a') as f:
      for i in range(num_nodes):
        f.write(f"{i+1}\n")

  def write41_nodes(self):
    nodes = self.get_points()
    with self.output_file_path.open('a') as f:
      for node in nodes:
        f.write(f"{node[0]} {node[1]} {node[2]}\n")

  def write41_element_header_end(self):
    with self.output_file_path.open('a') as f:
      f.write('$EndElements\n')

  def write41_entity_header(self):
    raise NotImplementedError("Please Implement write41_entity_header")

  def write41_node_header(self):
    raise NotImplementedError("Please Implement write41_node_header")

  def write41_element_header(self):
    raise NotImplementedError("Please Implement write41_element_header")

  def write41_elements(self):
    raise NotImplementedError("Please Implement write41_elements")


class VolumeExtractor(BaseExtractor):
  def __init__(self, input_file_path, output_file_path):
    #create a custom file name for volume
    output_file_path = self.produce_path(output_file_path)
    super().__init__(input_file_path, output_file_path)

  def get_elements(self):

    tetrahedronElementType = gmsh.model.mesh.getElementType("tetrahedron", 1)
    element_index, node_indexes = gmsh.model.mesh.getElementsByType(tetrahedronElementType)
    node_indexes = node_indexes.astype(int).reshape(-1,4)

    return node_indexes

  def get_num_elements(self) -> int:
    elements = self.get_elements()
    num_elements = len(elements)
    if num_elements>0:
      return num_elements

    raise ValueError('There are no elements representing volume')


  def write41_entity_header(self):
    min_max = self.get_max_xyz()
    with self.output_file_path.open('a') as f:
      f.write('$Entities\n')
      f.write('0 0 0 1\n')
      f.write('0 ' + str(min_max[0]) + ' ' + str(min_max[1]) + ' ' +
              str(min_max[2]) + ' ' + str(min_max[3]) + ' ' + str(min_max[4]) +
              ' ' + str(min_max[5]) + ' 0 0 \n')
      f.write('$EndEntities\n')

  def write41_node_header(self):
    num_nodes = self.get_num_points()
    with self.output_file_path.open('a') as f:
      f.write('$Nodes\n')
      f.write(f'1 {num_nodes} 1 {num_nodes}\n')
      f.write(f'3 0 0 {num_nodes}\n')

  def write41_element_header(self):
    num_elems = self.get_num_elements()
    with self.output_file_path.open('a') as f:
      f.write('$Elements\n')
      f.write(f'1 {num_elems} 1 {num_elems}\n')
      f.write(f'3 0 4 {num_elems}\n')

  def write41_elements(self):
    elements = self.get_elements()
    with self.output_file_path.open('a') as f:
      index = 1
      for element in elements:
        f.write(f'{index} {element[0]} {element[1]} {element[2]} '
                f'{element[3]}\n')
        index += 1

if __name__ == '__main__':

  # Initialize Gmsh
  gmsh.initialize()

  input_filepath = os.path.join(os.getcwd(), "finray_test/sphere2.msh")
  output_filepath = copy(input_filepath)
  VolumeExtractor(input_filepath, output_filepath)

  # Finalize Gmsh
  gmsh.finalize()
