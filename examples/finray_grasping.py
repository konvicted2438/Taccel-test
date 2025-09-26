import os
import sys
sys.path.append(".")

import numpy as np
import warp as wp
import pyvista as pv
import trimesh as tm
from scipy.spatial.transform import Rotation as R, Slerp
from warp.sim.render import SimRendererOpenGL

from examples.example_utils import init_robot_demo
from warp_ipc.ipc_integrator import IPCIntegrator
from warp_ipc.utils.constants import VolMaterialType
from warp_ipc.sim_model import ASRModel
from warp_ipc.utils import log

from tqdm.rich import tqdm_rich as tqdm

import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--num_envs", default=1, type=int)
parser.add_argument("--viz", action="store_true")
args = parser.parse_args()

# initialize output folder
DATETIME_TAG, OUT_DIR = init_robot_demo(args, "finray_test")

dt = 1 / 50
grid_x = max(1, math.isqrt(args.num_envs))
grid_y = math.ceil(args.num_envs / grid_x)
env_pos = (
    ASRModel.get_env_pos(
        args.num_envs,
        grid_x,
        grid_y,
        0.15,
    )
    .cpu()
    .numpy()
)


if __name__ == "__main__":
    # Clear cache and initialize warp
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
    
    # Create ground plane
    model.add_plane(np.array([0, 1, 0], dtype=np.float64), np.zeros((3,), dtype=np.float64), 0.3)
    
    # Load finray mesh
    finray = pv.read("finray_test/finray_test_vol.msh")
    scale = 0.005  # uniform scale factor
    finray_points = np.asarray(finray.points) * scale
    translated_points = finray_points + 1
    finray.points = translated_points
    
    # Add finray to simulation
    finray_handle = model.add_soft_vol_body(
        finray,
        density=1e3,
        E=1e5,
        nu=0.4,
        mu=1.0,
        env_id=0,
    )
    
    
    # Initialize model
    model.init()

    # Set up visualization
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

    # Configure integrator
    integrator = IPCIntegrator()
    integrator.use_cpu = False
    integrator.max_newton_iter = 30
    integrator.max_cg_iter = 300
    integrator.cg_rel_tol = 1e-3
    integrator.use_inversion_free_step_size_filter = True
    integrator.inversion_free_im_tol = integrator.inversion_free_im_tol = 1e-6
    integrator.inversion_free_cubic_coef_tol = integrator.inversion_free_cubic_coef_tol = 1e-10
    integrator.use_hard_kinematic_constraint = False
    integrator.soft_vol_material_type = VolMaterialType.NEO_HOOKEAN
    model.kinematic_helper.set_initial_stiffness(1e8)

    # Finalize model
    model.finalize()

    # Position finray above ground - MOVED AFTER finalize()
    init_offset = np.array([0.0, 0.5, 0.0])  # Adjusted for Y-up coordinate system
    model.set_soft_state(finray_handle, translated_points + init_offset)

    # Create output directory for frames
    os.makedirs(os.path.join(OUT_DIR, "frames"), exist_ok=True)
        # Attempting kinematic control
    COMPRESS_DISTANCE = 0.07
    LIFT_DISTANCE = 0.02
    total_timesteps = 200 #FPS = 1/DT = 50fps i.e. 4s sim



    # Run simulation
    for t in tqdm(range(1000)):
        integrator.simulate(model, dt=dt,)
        if args.viz:
            renderer.begin_frame(model.elapsed_time)
            renderer.render(model.state())
            renderer.end_frame()
        model.write_scene(os.path.join(OUT_DIR, f"frames/frame_{t}.ply"))
    
    # Keep visualization window open
    if args.viz:
        while renderer.is_open():
            renderer.begin_frame(model.elapsed_time)
            renderer.render(model.state())
            renderer.end_frame()
