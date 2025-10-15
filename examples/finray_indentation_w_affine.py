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
import torch
import json

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--num_envs", default=1, type=int)
parser.add_argument("--viz", action="store_true")

args = parser.parse_args()

OUT_DIR = init_robot_demo(args, "finray_indentation_w", "finray_indentation_w")
marker_sphere_radius = 0.015  # Adjust size as needed
marker_sphere_color = (0.1, 0.1, 0.1) # Green color for markers
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
    model.set_kinematic_stiffness(1e8)
    model.dhat = 1e-4
    #model.kappa = 
    model.tol = 1e-3
    model.k_elasticity_damping = 0
    model.epsv = 1e-2
    model.gravity = wp.vec3d([0, -9.81, 0])
    model.add_plane(np.array([0, 1, 0], dtype=np.float64), np.zeros((3,), dtype=np.float64), 0.3) # normal, origin, mu (friction)

    #left_softfinger = pv.read("assets/objects/soft_gripper/output_voxels.stl")
    left_softfinger = pv.read("finray_test/finray_test_vol.msh")  # or .stl
    objects_pointy = pv.read("finray_test/w_shape_vol.msh")
    tracked_indices = [212, 216, 220, 210, 1700, 227, 221, 217, 213]

    scale = 0.01  # uniform scale factor

    left_softfinger.points *= scale
    objects_pointy.points *= 0.01


    slide_idx = np.load(os.path.join("finray_test", "in_box_indices.npy")).astype(np.int64)
    #print("slide idx", slide_idx)
    slide_mask = np.zeros(left_softfinger.n_points, dtype=np.int32)
    slide_mask[slide_idx] = 1

    move_idx = np.arange(objects_pointy.n_points, dtype=np.int64)
    move_mask = np.zeros(objects_pointy.n_points, dtype=np.int32)
    move_mask[move_idx] = 1
    initial_pointy_pos = np.array([0.46, 0.66, 0.0])

    left_softfinger_handles, objects_pointy_handles = [], []
    for env_id in range(args.num_envs):
        mesh1 = left_softfinger.copy()
        mesh3 = objects_pointy.copy()

        mesh3.points = mesh3.points @ Rx(np.radians(-90)) @ Ry(np.radians(-90)) @ Rz(np.radians(90)) + env_pos[env_id]
        
        # Extract surface from volumetric mesh (if loading .msh)
        if mesh3.faces is None:
            mesh3_surface = mesh3.extract_surface()
            vertices = mesh3_surface.points
            faces = mesh3_surface.faces.reshape(-1, 4)[:, 1:]
        else:
            # Already a surface mesh (if loading .stl)
            vertices = mesh3.points
            faces = mesh3.faces.reshape(-1, 4)[:, 1:]
    
        # Add as affine body
        objects_pointy_handle = model.add_affine_body(
            x=vertices,
            face=faces.astype(np.int32),
            density=1e3,
            E=1.3e9,
            mu=1.0,
            mass_xi=None,  # For closed mesh
            env_id=env_id,
            nu=0.3
        )
        objects_pointy_handles.append(objects_pointy_handle)

        # Transform finger mesh
        mesh1.points = mesh1.points @ Rx(np.radians(-90)) @ Ry(np.radians(90)) @ Rz(np.radians(180)) + env_pos[env_id] + np.array([0.2, 0.4, 0.0])
        left_softfinger_handle = model.add_soft_vol_body(mesh1, density=1e1, E=1.3e6, nu=0.4, mu=1.0, env_id=env_id)
        left_softfinger_handles.append(left_softfinger_handle)


    # Setup simulation
    model.init()


    # capture rest pose for later use
    rest_left_positions = model.get_body_x(left_softfinger_handles[0]).detach().clone()

    # Set kinematic constraints (ONLY for soft bodies, NOT affine bodies)
    for env_id in range(args.num_envs):
        model.set_soft_kinematic_constraint(left_softfinger_handles[env_id], slide_mask)
        # REMOVE this line - affine bodies don't use soft kinematic constraints!
        # model.set_soft_kinematic_constraint(objects_pointy_handles[env_id], move_mask)
        
        # Enable kinematic control for affine body (like in the peg example)
        model.enable_affine_kinematic_constraint(objects_pointy_handles[env_id])
        start_pos = initial_pointy_pos + env_pos[env_id]
        model.set_affine_state(objects_pointy_handles[env_id], np.eye(3), start_pos)
        #model.set_affine_kinematic_target(objects_pointy_handles[env_id], np.eye(3), start_pos)

    model.kinematic_helper.set_initial_stiffness(1e7)
    model.finalize()

        # Integrator setup
    integrator = IPCIntegrator()
    integrator.use_hard_kinematic_constraint = True
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
            camera_pos=(0.5, 0.5, 0.7),
            camera_front=(0.0, 0.0, -2.0),
            camera_up=(0.0, 1.0, 0.0),
        )


    # parameter for the kinematic control
    SLIDE_DISTANCE = 0.18
    LIFT_DISTANCE = 0.05
    total_timesteps = 300

    # Store initial position for the pointy object


    # Start simulation
    sim_time = 0
    json_path = os.path.join(OUT_DIR, "simulation_data.json")
    if not os.path.exists(json_path):
        print(f"Creating new JSON file at: {json_path}")
        with open(json_path, "w") as f:
            f.write("[\n")
    else:
        print(f"JSON file already exists at: {json_path}")

    sim_time = 0
    for t in tqdm(range(total_timesteps)):
        for env_id in range(args.num_envs):
            # Control the affine body (pointy object)
            if t < int(total_timesteps/2):
                # SLIDING down - move pointy object up to indent finger
                current_y = initial_pointy_pos[1] - (SLIDE_DISTANCE * t / int(total_timesteps/2))
                current_translation = np.array([initial_pointy_pos[0], current_y, initial_pointy_pos[2]]) + env_pos[env_id]
            else:
                # LIFTING up - move pointy object back down
                t_lift = t - int(total_timesteps/2)
                current_y = initial_pointy_pos[1] - SLIDE_DISTANCE + (SLIDE_DISTANCE * t_lift / int(total_timesteps/2))
                current_translation = np.array([initial_pointy_pos[0], current_y, initial_pointy_pos[2]]) + env_pos[env_id]
            
            # Set affine kinematic target (like the hole in the peg example)
            model.set_affine_kinematic_target(
                objects_pointy_handles[env_id],
                np.eye(3),
                current_translation,
            )

            rot, trans = model.get_affine_body_state(objects_pointy_handles[env_id])
            print(f"[t={t:03d}] env {env_id} target={current_translation}, state={trans}, Δ={trans - current_translation}")
            
            # Handle finger (keep it fixed) - this stays the same
            left_targets = model.get_body_x(left_softfinger_handles[env_id])
            left_targets[slide_idx] = rest_left_positions[slide_idx]
            model.set_soft_kinematic_target(left_softfinger_handles[env_id], left_targets)
        
        integrator.simulate(model, dt=dt, control=None)
        
        if args.viz:
            renderer.begin_frame(model.elapsed_time)
            renderer.render(model.state())

            for i, marker_idx in enumerate(tracked_indices):
                # Get the position of the particle with this index
                # Ensure the index is within bounds
                if 0 <= marker_idx < len(left_targets):
                    pos = left_targets[marker_idx]
                    sphere_pos = wp.vec3(pos[0], pos[1], pos[2])
                    sphere_name = f"marker_sphere_{marker_idx}" # Use marker_idx for a more stable name

                    renderer.render_sphere(
                        name=sphere_name,
                        pos=sphere_pos,
                        rot=wp.quat_identity(), # No rotation needed
                        radius=marker_sphere_radius,
                        color=marker_sphere_color
                    )


            renderer.end_frame()
        
        sim_time += integrator.profile_helper.current_timestep_data["total_timestep"]
        model.write_scene(os.path.join(OUT_DIR, f"frames/frame_{t}.ply"))

        leftfinger_nodal_contactforces = model.get_body_nodal_contact_force(left_softfinger_handles[env_id], dt)
        leftfinger_nodal_frictionforces = model.get_body_nodal_friction_force(left_softfinger_handles[env_id], dt)
        contact_force_magnitudes = torch.norm(leftfinger_nodal_contactforces, dim=1)
        friction_force_magnitudes = torch.norm(leftfinger_nodal_frictionforces, dim=1)
        total_force_magnitudes = contact_force_magnitudes + friction_force_magnitudes
        contact_threshold = 1e-6
        nodes_in_contact = torch.where(total_force_magnitudes > contact_threshold)[0]

        contact_node_positions = []
        if len(nodes_in_contact) > 0:
            left_positions = model.get_body_x(left_softfinger_handles[env_id])
            contact_positions = left_positions[nodes_in_contact]
            for i, node_idx in enumerate(nodes_in_contact):
                contact_node_positions.append(
                    {
                        "node_id": node_idx.item(),
                        "position": contact_positions[i].cpu().numpy().tolist(),
                        "contact_force": leftfinger_nodal_contactforces[node_idx].cpu().numpy().tolist(),
                        "friction_force": leftfinger_nodal_frictionforces[node_idx].cpu().numpy().tolist(),
                        "force_magnitude": total_force_magnitudes[node_idx].item(),
                    }
                )
        net_Fz_left_contact = leftfinger_nodal_contactforces[:, 2].sum().item()
        net_Fz_left_friction = leftfinger_nodal_frictionforces[:, 2].sum().item()
        fingerforce = net_Fz_left_contact + net_Fz_left_friction

        record_dict = {
            "timestamp": t * dt,
            "timestep": t,
            "youngs_modulus": 1.3e6,
            "pointy_position": current_translation.tolist(),
            "slide_distance": SLIDE_DISTANCE,
            "phase": "compression" if t < int(total_timesteps / 2) else "release",
            "finger_force": fingerforce,
            "contact_info": {
                "num_nodes_in_contact": len(nodes_in_contact),
                "contact_node_indices": nodes_in_contact.cpu().numpy().tolist() if len(nodes_in_contact) > 0 else [],
                "max_contact_force": contact_force_magnitudes.max().item() if len(nodes_in_contact) > 0 else 0.0,
                "total_contact_force": contact_force_magnitudes.sum().item() if len(nodes_in_contact) > 0 else 0.0,
                "total_friction_force": friction_force_magnitudes.sum().item() if len(nodes_in_contact) > 0 else 0.0,
                "contact_points": contact_node_positions,
            },
            "net_forces": {
                "contact_z": net_Fz_left_contact,
                "friction_z": net_Fz_left_friction,
                "total_z": fingerforce,
            },
            "tracked_nodes": {
                "indices": tracked_indices,
                "positions": [
                    left_targets[idx].cpu().numpy().tolist() if idx < len(left_targets) else [0, 0, 0]
                    for idx in tracked_indices
                ],
                "deformations": [
                    (left_targets[idx] - rest_left_positions[idx]).cpu().numpy().tolist() if idx < len(left_targets) else [0, 0, 0]
                    for idx in tracked_indices
                ],
            },
        }

        with open(json_path, "a") as f:
            json_str = json.dumps(record_dict, separators=(",", ":"))
            json_str = json_str.replace('{"timestamp":', '{\n  "timestamp":')
            json_str = json_str.replace(',"timestep":', ',\n  "timestep":')
            json_str = json_str.replace(',"youngs_modulus":', ',\n  "youngs_modulus":')
            json_str = json_str.replace(',"pointy_position":', ',\n  "pointy_position":')
            json_str = json_str.replace(',"slide_distance":', ',\n  "slide_distance":')
            json_str = json_str.replace(',"phase":', ',\n  "phase":')
            json_str = json_str.replace(',"finger_force":', ',\n  "finger_force":')
            json_str = json_str.replace(',"contact_info":', ',\n  "contact_info":')
            json_str = json_str.replace(',"net_forces":', ',\n  "net_forces":')
            json_str = json_str.replace(',"tracked_nodes":', ',\n  "tracked_nodes":')
            json_str = json_str.replace("}", "\n}")
            f.write(json_str)
            if t < total_timesteps - 1:
                f.write(",\n")
            else:
                f.write("\n]")

    print(f"\nSimulation complete!")
    print(f"Output directory: {OUT_DIR}")
    print(f"JSON file saved to: {json_path}")
    print(f"PLY frames saved to: {os.path.join(OUT_DIR, 'frames/')}")