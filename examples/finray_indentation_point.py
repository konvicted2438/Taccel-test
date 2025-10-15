import numpy as np
import os
import warp as wp
import pyvista as pv
import trimesh as tm
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
from warp.sim.render import SimRendererOpenGL
import numpy as np
from examples.example_utils import init_robot_demo
from utils.math_utils import Rx, Ry, Rz
from warp_ipc.ipc_integrator import IPCIntegrator
from warp_ipc.utils.constants import VolMaterialType
from warp_ipc.sim_model import ASRModel
from warp_ipc.utils import log

from tqdm.rich import tqdm_rich as tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--num_envs", default=1, type=int)
parser.add_argument("--viz", action="store_true")

args = parser.parse_args()

OUT_DIR = init_robot_demo(args, "finray_indentation_pointy", "finray_indentation_pointy")

dt = 1 / 50
env_pos = (
    ASRModel.get_env_pos(
        args.num_envs,
        int(np.sqrt(args.num_envs)),
        int(args.num_envs / np.sqrt(args.num_envs)),
        0.5, #env_spacing
    )
    .cpu()
    .numpy()
)

if __name__ == "__main__":

    wp.init()
    model = ASRModel(
        num_envs=args.num_envs,
        viz_envs=list(range(args.num_envs)) if args.viz else [],
    )    
    model.set_kinematic_stiffness(1e5)
    model.dhat = 1e-4
    model.kappa = 0.9
    model.tol = 1e-3
    model.k_elasticity_damping = 0
    model.epsv = 1e-2
    model.gravity = wp.vec3d([0, -9.81, 0])
    model.add_plane(np.array([0, 1, 0], dtype=np.float64), np.zeros((3,), dtype=np.float64), 0.3) # normal, origin, mu (friction)

    #left_softfinger = pv.read("assets/objects/soft_gripper/output_voxels.stl")
    left_softfinger = pv.read("finray_test/finray_test_vol.msh")  # or .stl
    objects_pointy = pv.read("finray_test/pointy_vol.msh")

    scale = 0.01  # uniform scale factor

    left_softfinger.points *= scale
    objects_pointy.points *= 0.015


    slide_idx = np.load(os.path.join("finray_test", "in_box_indices.npy")).astype(np.int64)
    #print("slide idx", slide_idx)
    slide_mask = np.zeros(left_softfinger.n_points, dtype=np.int32)
    slide_mask[slide_idx] = 1

    move_idx = np.arange(objects_pointy.n_points, dtype=np.int64)
    move_mask = np.zeros(objects_pointy.n_points, dtype=np.int32)
    move_mask[move_idx] = 1


    left_softfinger_handles,  objects_pointy_handles = [],[] 
    for env_id in range(args.num_envs):
        mesh1 = left_softfinger.copy()
        # mesh2 = left_softfinger.copy()
        mesh3 = objects_pointy.copy()

        # Rx/y/z(theta) rotates theta degrees CCW when looking from origin toward +x/y/z
        # NOTE: From left to right, Rx acts first, then Ry - and the matrix multiplication is NOT commutative!
        mesh1.points = mesh1.points @ Rx(np.radians(-90)) @ Ry(np.radians(90)) @ Rz(np.radians(180)) + env_pos[env_id] + np.array([0.2, 0.4, 0.0]) #(106,3) @ (3,3) + (3,)
        left_softfinger_handle = model.add_soft_vol_body(mesh1, density=1, E=5.3e6, nu=0.4, mu=1.0, env_id=env_id) 
        left_softfinger_handles.append(left_softfinger_handle)

        # #  # Rx/y/z(theta) rotates theta degrees CCW when looking from origin toward +x/y/z
        # # # NOTE: From left to right, Rx acts first, then Ry - and the matrix multiplication is NOT commutative!
        # mesh2.points = mesh2.points @ Rx(np.radians(-90)) @ Ry(np.radians(-90)) @ Rz(np.radians(90)) + env_pos[env_id] + np.array([0.5, 0.4, 0.0]) #(106,3) @ (3,3) + (3,)
        # right_softfinger_handle = model.add_soft_vol_body(mesh2, density=1e3, E=1.3e6, nu=0.4, mu=1.0, env_id=env_id) 
        # right_softfinger_handles.append(right_softfinger_handle)


        # #  # Rx/y/z(theta) rotates theta degrees CCW when looking from origin toward +x/y/z
        # # # NOTE: From left to right, Rx acts first, then Ry - and the matrix multiplication is NOT commutative!
        mesh3.points = mesh3.points @ Rx(np.radians(-90)) @ Ry(np.radians(-90)) @ Rz(np.radians(90)) + env_pos[env_id] + np.array([0.4, 0.02, 0.0]) #(106,3) @ (3,3) + (3,)
        objects_pointy_handle = model.add_soft_vol_body(mesh3, density=1e3, E=1.3e6, nu=0.4, mu=1.0, env_id=env_id) 
        objects_pointy_handles.append(objects_pointy_handle)

    
    # Setup simulation
    model.init()


    # capture rest pose for later use
    rest_left_positions = model.get_body_x(left_softfinger_handles[0]).detach().clone()

    for env_id in range(args.num_envs):
        model.set_soft_kinematic_constraint(left_softfinger_handles[env_id], slide_mask)
        model.set_soft_kinematic_constraint(objects_pointy_handles[env_id], move_mask)

    model.kinematic_helper.set_initial_stiffness(1e7)


        # IMPORTANT: Finalize model before setting up renderer
    model.finalize()

        # Integrator setup
    integrator = IPCIntegrator()
    integrator.use_hard_kinematic_constraint = False
    integrator.use_cpu = False
    integrator.max_newton_iter = 30
    integrator.max_cg_iter = 300
    integrator.cg_rel_tol = 1e-3
    integrator.use_inversion_free_step_size_filter = True
    integrator.inversion_free_im_tol = 1e-6
    integrator.inversion_free_cubic_coef_tol = 1e-10
    integrator.soft_vol_material_type = VolMaterialType.NEO_HOOKEAN


    # Set up visualization AFTER model.finalize()
    if args.viz:
        stage_path = os.path.join(OUT_DIR, "finray_drop.usd")
        renderer = SimRendererOpenGL(
            model,
            stage_path,
            scaling=1,
            near_plane=0.001,
            far_plane=20.0,
            camera_fov=75.0,
            camera_pos=(0.5, 0.0, 2.0),
            camera_front=(0.0, 0.0, -2.0),
            camera_up=(0.0, 1.0, 0.0),
        )


    # parameter for the kinematic control
    SLIDE_DISTANCE = 0.1
    LIFT_DISTANCE = 0.05
    total_timesteps = 100

    # Start simulation
    sim_time = 0
    for t in tqdm(range(total_timesteps)):
        
        for env_id in range(args.num_envs):
            # SLIDING
            if t < int(total_timesteps/2):
                # Get current positions
                object_targets = model.get_body_x(objects_pointy_handles[env_id])
                # Move object down
                object_targets[move_idx, 1] -= (SLIDE_DISTANCE)/int(total_timesteps/2)
                # SET THE TARGET (this was missing!)
                model.set_soft_kinematic_target(objects_pointy_handles[env_id], object_targets)
                
            # LIFTING
            else:
                # Get current positions
                object_targets = model.get_body_x(objects_pointy_handles[env_id])
                # Move object up
                object_targets[move_idx, 1] += (SLIDE_DISTANCE)/int(total_timesteps/2)
                # SET THE TARGET (this was missing!)
                model.set_soft_kinematic_target(objects_pointy_handles[env_id], object_targets)
        
            # Handle finger (keep it fixed)
            left_targets = model.get_body_x(left_softfinger_handles[env_id])
            left_targets[slide_idx] = rest_left_positions[slide_idx]
            model.set_soft_kinematic_target(left_softfinger_handles[env_id], left_targets)
        
        integrator.simulate(model, dt=dt, control=None)
        
        if args.viz:
            renderer.begin_frame(model.elapsed_time)
            renderer.render(model.state())
            
            renderer.end_frame()
            
        
        sim_time += integrator.profile_helper.current_timestep_data["total_timestep"]        
        model.write_scene(os.path.join(OUT_DIR, f"frames/frame_{t}.ply"))






#  # --- Render spheres for marker points using current state ---
# marker_sphere_radius = 0.05  # Adjust size as needed
# marker_sphere_color = (0.1, 0.1, 0.1) # Green color for markers

# # Get current particle positions directly from the state being rendered
# current_positions_np = wp.array.numpy(self.state_0.particle_q)


# # Iterate through the marker indices
# for i, marker_idx in enumerate(self.marker_list):
#     # Get the position of the particle with this index
#     # Ensure the index is within bounds
#     if 0 <= marker_idx < len(current_positions_np):
#         pos = current_positions_np[marker_idx]
#         sphere_pos = wp.vec3(pos[0], pos[1], pos[2])
#         sphere_name = f"marker_sphere_{marker_idx}" # Use marker_idx for a more stable name

#         self.renderer.render_sphere(
#             name=sphere_name,
#             pos=sphere_pos,
#             rot=wp.quat_identity(), # No rotation needed
#             radius=marker_sphere_radius,
#             color=marker_sphere_color
#         )