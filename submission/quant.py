import submitit
import os
import sys
import debugpy
import torch

# vscode changes the cwd to the file's directory, so we need to add the workspace to the path
# Set the working directory to the base of the workspace
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
sys.path.insert(0, workspace_dir)

from src import paths
from src.datasets import (
    extract_the_dataset_on_compute_node,
    move_data_to_compute_node,
    resolve_data_directories,
)
from src.utils import EXPERIMENT_PREFIX_SEP, DatasetSwitch, determine_device
from src import quant_measures_grads


def extract_the_grads_dataset_on_compute_node(COMPUTE_DATA_DIR, EXT, TARGET_DIR):
    extract_the_dataset_on_compute_node(COMPUTE_DATA_DIR, EXT, TARGET_DIR)

    # extract the sub directories
    os.system(
        f'ls {TARGET_DIR} | xargs -I {{}} -P 16 sh -c "tar -xf {TARGET_DIR}/{{}}/*.tar -C {TARGET_DIR}/{{}}"'
    )


def main(args):
    print(args)
    name = args["name"]
    args["dataset"] = DatasetSwitch.GRADS
    
    if args["port"] is not None and args["port"] > 0:
        job_env = submitit.JobEnvironment()
        print(f"Debugger is running on node {job_env.hostname} port {args['port']}")
        debugpy.listen((job_env.hostname, args["port"]))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    determine_device(args)

    (
        DATA_DIR,
        COMPUTE_DATA_DIR,
        EXT,
        COMPUTE_DATA_DIR_BASE_DIR,
        TARGET_DIR,
        COMPUTE_OUTPUT_DIR,
        LOCAL_OUTPUT_DIR,
    ) = resolve_data_directories(args)

    os.system("module load Fpart/1.5.1-gcc-8.5.0")

    move_data_to_compute_node(DATA_DIR, EXT == "tgz", COMPUTE_DATA_DIR)

    extract_the_grads_dataset_on_compute_node(COMPUTE_DATA_DIR, EXT, TARGET_DIR)

    print("Running main job...")
    print(f"Data is in {COMPUTE_DATA_DIR_BASE_DIR}")
    results = quant_measures_grads.main(
        root_path=TARGET_DIR,
        output_dir=paths.LOCAL_QUANTS_DIR,
        **args,
    )

    os.makedirs(paths.LOCAL_QUANTS_DIR, exist_ok=True)
    file_name = os.path.join(
        paths.LOCAL_QUANTS_DIR, f"{name}{EXPERIMENT_PREFIX_SEP}quants.pt"
    )
    torch.save(results, file_name)
