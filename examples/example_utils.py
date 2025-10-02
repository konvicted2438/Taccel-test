import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import warp as wp

def set_seed(seed: int = None):
    seed = random.randint(0, 2**32 - 1) if seed is None else seed

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_robot_demo(args, demo_name: str, output_folder: str = "grasping_test", cache_dir: str = "./ptx"):
    #DATETIME_TAG = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #OUT_DIR = f"./output/{demo_name}/{DATETIME_TAG}"
    OUT_DIR = f"./output/{demo_name}/{output_folder}"
    os.makedirs(os.path.join(OUT_DIR, "frames"), exist_ok=True)
    json.dump(vars(args), open(os.path.join(OUT_DIR, "args.json"), "w"))
    set_seed(args.seed if hasattr(args, "seed") else 42)
    wp.config.kernel_cache_dir = cache_dir
    wp.config.cuda_output = "ptx"
    wp.config.ptx_target_arch = 86

    return OUT_DIR

def init_robot_demo_SDTO(args, demo_name: str, output_folder: str, current_design_iter: str, cache_dir: str = "./ptx"):
    #OUT_DIR = f"./output/{demo_name}/cd_{str(args.compress_distance)}_ld_{str(args.lift_distance)}"
    OUT_DIR = f"./output/{demo_name}/{output_folder}"
    MESH_DIR = os.path.join(os.getcwd(),"assets", "objects", demo_name, output_folder)
    os.makedirs(os.path.join(OUT_DIR, f"frames_{current_design_iter}"), exist_ok=True)
    os.makedirs(MESH_DIR, exist_ok=True)
    json.dump(vars(args), open(os.path.join(OUT_DIR,f"frames_{current_design_iter}", "args.json"), "w"))
    set_seed(args.seed if hasattr(args, "seed") else 42)
    wp.config.kernel_cache_dir = cache_dir
    wp.config.cuda_output = "ptx"
    wp.config.ptx_target_arch = 86

    return OUT_DIR, MESH_DIR

def init_robot_demo_SDTO_eval(args, demo_name: str, output_folder: str, softfinger_mesh_filename: str, cache_dir: str = "./ptx"):

    OUT_DIR = f"./output/{demo_name}/{output_folder}"
    os.makedirs(os.path.join(OUT_DIR, f"frames_{softfinger_mesh_filename}"), exist_ok=True)
    json.dump(vars(args), open(os.path.join(OUT_DIR,f"frames_{softfinger_mesh_filename}", "args.json"), "w"))
    set_seed(args.seed if hasattr(args, "seed") else 42)
    wp.config.kernel_cache_dir = cache_dir
    wp.config.cuda_output = "ptx"
    wp.config.ptx_target_arch = 86

    return OUT_DIR


def init_robot_demo_parallel(args, demo_name: str, env_output_folder: str, cache_dir: str = "./ptx"):
    #OUT_DIR = f"./output/{demo_name}/cd_{str(args.compress_distance)}_ld_{str(args.lift_distance)}"
    DEMO_DIR = f"./output/{demo_name}"
    OUT_DIR = os.path.join(DEMO_DIR,f"{env_output_folder}")
    MESH_DIR = os.path.join(os.getcwd(),"assets", "objects", demo_name, env_output_folder)
    os.makedirs(os.path.join(DEMO_DIR, f"frames"), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MESH_DIR, exist_ok=True)
    json.dump(vars(args), open(os.path.join(DEMO_DIR,f"frames", "args.json"), "w"))
    set_seed(args.seed if hasattr(args, "seed") else 42)
    wp.config.kernel_cache_dir = cache_dir
    wp.config.cuda_output = "ptx"
    wp.config.ptx_target_arch = 86
    return OUT_DIR, MESH_DIR

def get_interp_step(start, end, n_steps, step):
    return start + (end - start) * step / n_steps

def init_parallel_example(demo_name: str, cache_dir: str = "./ptx"):

    FRAME_DIR = f"./output/{demo_name}"
    os.makedirs(os.path.join(FRAME_DIR, f"frames"), exist_ok=True)

    config_filepath = os.path.join(os.getcwd(),"config", demo_name + ".json")
    with open(config_filepath, "r") as f:
        config = json.load(f)
        json.dump(config, open(os.path.join(FRAME_DIR, "args.json"), "w"))
    
    set_seed(42)
    wp.config.kernel_cache_dir = cache_dir
    wp.config.cuda_output = "ptx"
    wp.config.ptx_target_arch = 86
    return config, FRAME_DIR

def get_OUT_DIR_and_MESH_DIR(demo_name: str, env_output_folder: str):
    DEMO_DIR = f"./output/{demo_name}"
    OUT_DIR = os.path.join(DEMO_DIR,f"{env_output_folder}")
    MESH_DIR = os.path.join(os.getcwd(),"assets", "objects", demo_name, env_output_folder)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MESH_DIR, exist_ok=True)
    return OUT_DIR, MESH_DIR


def init_parallel_SDTO(demo_name: str, cache_dir: str = "./ptx"):

    FRAME_DIR = f"./output/{demo_name}"
    os.makedirs(os.path.join(FRAME_DIR, f"frames_D0"), exist_ok=True)

    config_filepath = os.path.join(os.getcwd(),"config", demo_name + ".json")
    with open(config_filepath, "r") as f:
        config = json.load(f)
        json.dump(config, open(os.path.join(FRAME_DIR, "args.json"), "w"))
    
    set_seed(42)
    wp.config.kernel_cache_dir = cache_dir
    wp.config.cuda_output = "ptx"
    wp.config.ptx_target_arch = 86
    return config, FRAME_DIR
