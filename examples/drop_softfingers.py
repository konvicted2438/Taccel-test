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

OUT_DIR = init_robot_demo(args, "drop_softfingers", "finray_testplay")

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

def plot_net_Fz_vs_simtime(net_Fz_list, dt, weight_force_object: float,
                           current_design_iter, total_sim_time):

    net_Fzs = np.array(net_Fz_list)
    time = np.linspace(dt, total_sim_time, num=len(net_Fzs), endpoint=True)

    plt.figure(figsize=(8, 4))  # Optional: control figure size
    plt.plot(time, net_Fzs, 'o', label='Net $F_z$ Gripper')

    label = '$m_{obj}g = $'
    label += f"{round(weight_force_object, 2)}N"
    plt.axhline(weight_force_object, color='red', linestyle='--', label=label)

    # Compute dynamic y-axis limits
    y_min = min(net_Fzs.min(), weight_force_object)
    y_max = max(net_Fzs.max(), weight_force_object)
    y_range = y_max - y_min
    padding = 0.05 * y_range if y_range > 0 else 1.0  # avoid zero padding
    plt.ylim(y_min - padding, y_max + padding)

    plt.xlabel('t (s)')
    plt.ylabel('Force (N)')
    plt.title(f"{current_design_iter} Force-Generation Capability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{current_design_iter}_net_Fz_vs_time.png"))
    plt.close() 


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
    objects_sphere = pv.read("finray_test/sphere_vol.msh")

    scale = 0.005  # uniform scale factor

    left_softfinger.points *= scale
    objects_sphere.points *= 1

    stick_idx = np.load(os.path.join("finray_test", "in_box_indices.npy")).astype(np.int64)
    stick_mask = np.zeros(left_softfinger.n_points, dtype=np.int32)
    stick_mask[stick_idx] = 1

    left_softfinger_handles, right_softfinger_handles, objects_sphere_handles = [],[],[]
    for env_id in range(args.num_envs):
        mesh1 = left_softfinger.copy()
        mesh2 = left_softfinger.copy()
        mesh3 = objects_sphere.copy()

        # Rx/y/z(theta) rotates theta degrees CCW when looking from origin toward +x/y/z
        # NOTE: From left to right, Rx acts first, then Ry - and the matrix multiplication is NOT commutative!
        mesh1.points = mesh1.points @ Rx(np.radians(-90)) @ Ry(np.radians(90)) @ Rz(np.radians(-90)) + env_pos[env_id] + np.array([0.2, 0.4, 0.0]) #(106,3) @ (3,3) + (3,)
        left_softfinger_handle = model.add_soft_vol_body(mesh1, density=1e3, E=1e5, nu=0.4, mu=1.0, env_id=env_id) 
        left_softfinger_handles.append(left_softfinger_handle)

        #  # Rx/y/z(theta) rotates theta degrees CCW when looking from origin toward +x/y/z
        # # NOTE: From left to right, Rx acts first, then Ry - and the matrix multiplication is NOT commutative!
        mesh2.points = mesh2.points @ Rx(np.radians(-90)) @ Ry(np.radians(-90)) @ Rz(np.radians(90)) + env_pos[env_id] + np.array([0.5, 0.4, 0.0]) #(106,3) @ (3,3) + (3,)
        right_softfinger_handle = model.add_soft_vol_body(mesh2, density=1e3, E=1e5, nu=0.4, mu=1.0, env_id=env_id) 
        right_softfinger_handles.append(right_softfinger_handle)


        mesh3.points = mesh3.points + env_pos[env_id] + np.array([0.35, 0.1, 0.0])
        objects_sphere_handle = model.add_soft_vol_body(mesh3, density=1000, E=1e5, nu=0.4, mu=1.0, env_id=env_id)
        objects_sphere_handles.append(objects_sphere_handle)
    
    
    
    # Setup simulation
    model.init()

    # Integrator setup
    integrator = IPCIntegrator()
    integrator.use_hard_kinematic_constraint = True
    integrator.use_cpu = False
    integrator.max_newton_iter = 30
    integrator.max_cg_iter = 300
    integrator.cg_rel_tol = 1e-5
    integrator.use_inversion_free_step_size_filter = True
    integrator.inversion_free_im_tol = 1e-6
    integrator.inversion_free_cubic_coef_tol = 1e-10
    integrator.soft_vol_material_type = VolMaterialType.NEO_HOOKEAN
    model.kinematic_helper.set_initial_stiffness(1e8)

    # IMPORTANT: Finalize model before setting up renderer
    model.finalize()

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
            camera_pos=(2.0, 1.0, 2.0),
            camera_front=(0.0, -1.0, -2.0),
            camera_up=(0.0, 1.0, 0.0),
        )

    # Start simulation
    sim_time = 0
    for t in tqdm(range(50)):
        integrator.simulate(model, dt=dt, control=None)
        
        if args.viz:
            renderer.begin_frame(model.elapsed_time)
            renderer.render(model.state())
            renderer.end_frame()
            
        model.write_scene(os.path.join(OUT_DIR, f"frames/frame_{t}.ply"))