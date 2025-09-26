
import os
import gmsh
import pathlib
from typing import List
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

def create_testsphere(radius=0.03, mesh_size=0.005, save_filepath="testsphere.msh", view_mesh = True):

    gmsh.initialize()
    gmsh.model.occ.addSphere(0, 0, 0, radius, tag=1)
    gmsh.model.occ.synchronize()

    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    # Save to file
    gmsh.write(save_filepath)

    print(f"Mesh saved to {save_filepath}")
    
    # Optional: Launch GUI if desired
    if view_mesh:
        gmsh.fltk.run()

    gmsh.finalize()

def create_testbox(length_x = 0.06, length_y = 0.04, length_z = 0.06, mesh_size=0.009, save_filepath="testbox.msh", view_mesh = True):


    gmsh.initialize()
    gmsh.model.occ.addBox(0, 0, 0, length_x, length_y, length_z)
    gmsh.model.occ.synchronize()

    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    # Generate the 3D mesh
    gmsh.model.mesh.generate(3)

    # Save to file
    gmsh.write(save_filepath)

    print(f"Mesh saved to {save_filepath}")
    
    # Optional: Launch GUI if desired
    if view_mesh:
        gmsh.fltk.run()

    gmsh.finalize()

def create_leftfinger(save_filepath, view_msh = True, ASCII = True, mesh_size = 0.02, thickness = 0.02):

    # Initialize Gmsh - inputs used for a binary file
    if ASCII: 
        # Initialize Gmsh
        gmsh.initialize()

    # Output a binary file
    else:
        gmsh.initialize(argv=["","-bin"])

    mesh_size = mesh_size

    # rectangle width: in paper = 0.15m
    r_width = 0.15
    # rectangle height: in paper = 0.07m
    r_height = 0.07
    thickness = thickness # 0.03m

    # Points of 2D domain: assume origin at bottom left corner
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size) # bottom left corner
    p2 = gmsh.model.geo.addPoint(r_width - r_height, 0, 0, mesh_size) # bottom right corner
    p3 = gmsh.model.geo.addPoint(r_width, r_height, 0, mesh_size) # top right corner
    p4 = gmsh.model.geo.addPoint(0, r_height, 0, mesh_size) # top left corner

    # Lines 
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    cl = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4])
    surface = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.extrude([(2, surface)], 0, 0, thickness)

    # # Synchronize the CAD model again
    gmsh.model.geo.synchronize()

    # Generate a mesh for the model
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(save_filepath)

    if view_msh:
        gmsh.fltk.run()

    # Finalize the GMSH API
    gmsh.finalize()

def create_testsphere_vol(radius=0.03, mesh_size=0.005, save_filepath="testsphere.msh", view_mesh = True):
  
    create_testsphere(radius, mesh_size, save_filepath, view_mesh)
    gmsh.initialize()
    VolumeExtractor(save_filepath, save_filepath)
    gmsh.finalize()

def create_testbox_vol(length_x = 0.06, length_y = 0.04, length_z = 0.06, mesh_size=0.009, save_filepath="testbox.msh", view_mesh = True):
  
    create_testbox(length_x, length_y, length_z, mesh_size, save_filepath, view_mesh)
    gmsh.initialize()
    input_filepath = save_filepath
    output_filepath = copy(save_filepath)
    VolumeExtractor(input_filepath, output_filepath)
    gmsh.finalize()

def create_D0_leftfinger_vol(save_filepath, view_msh = True, ASCII = True, mesh_size = 0.009, thickness = 0.03):

    create_leftfinger(save_filepath, view_msh = view_msh, ASCII = ASCII, mesh_size = mesh_size, thickness = thickness)
    input_filepath = save_filepath
    output_filepath = copy(input_filepath)
    gmsh.initialize()
    VolumeExtractor(input_filepath, output_filepath)
    gmsh.finalize()

def init_mesh_dirs_SDTO_parameter_sweep(demo_name: str, output_folder: str, create_box_or_sphere = "box"):

    assert create_box_or_sphere in ["box", "sphere"], "You must specify if you want these directories to have a test sphere or box."
    MESH_DIR = os.path.join(os.getcwd(),"assets", "objects", demo_name, output_folder)
    os.makedirs(MESH_DIR, exist_ok=True)

    if create_box_or_sphere == "box":
        save_filepath = os.path.join(MESH_DIR, "testbox.msh")
        create_testbox_vol(length_x = 0.06, length_y = 0.04, length_z = 0.06, mesh_size=0.009, save_filepath= save_filepath, view_mesh = False)
    else:
        save_filepath = os.path.join(MESH_DIR, "testsphere.msh")
        create_testsphere_vol(radius=0.03, mesh_size=0.005, save_filepath= save_filepath, view_mesh = False)

def init_mesh_dirs_with_object_and_D0(demo_name: str, output_folder: str, create_box_or_sphere = "box"):

    assert create_box_or_sphere in ["box", "sphere"], "You must specify if you want these directories to have a test sphere or box."
    MESH_DIR = os.path.join(os.getcwd(),"assets", "objects", demo_name, output_folder)
    os.makedirs(MESH_DIR, exist_ok=True)

    if create_box_or_sphere == "box":
        save_filepath = os.path.join(MESH_DIR, "testbox.msh")
        create_testbox_vol(length_x = 0.06, length_y = 0.04, length_z = 0.06, mesh_size=0.009, save_filepath= save_filepath, view_mesh = False)
    else:
        save_filepath = os.path.join(MESH_DIR, "testsphere.msh")
        create_testsphere_vol(radius=0.03, mesh_size=0.005, save_filepath= save_filepath, view_mesh = False)

    save_filepath = os.path.join(MESH_DIR, "D0_leftfinger.msh")
    create_D0_leftfinger_vol(save_filepath, view_msh = False, ASCII = False, mesh_size = 0.007, thickness = 0.03)

if __name__ == "__main__":

  for density_multiplier in [0.25, 0.50, 1, 2, 4, 8]:
    for softfinger_E in [0.46, 0.50, 0.60, 0.90, 1.39, 11.51, 39.98]:
        for TO_volfrac in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:

          init_mesh_dirs_SDTO_parameter_sweep("delete_me", f"box_rho={density_multiplier:.2f}*1000_YM={softfinger_E:.2f}MPa_vf={TO_volfrac:.2f}", create_box_or_sphere = "sphere") 
