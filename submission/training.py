import submitit
import os
import sys
import debugpy

from torch.utils.tensorboard import SummaryWriter

# vscode changes the cwd to the file's directory, so we need to add the workspace to the path
# Set the working directory to the base of the workspace
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
sys.path.insert(0, workspace_dir)

from src import datasets
from src.datasets import (
    extract_the_dataset_on_compute_node,
    move_data_to_compute_node,
    resolve_data_directories,
)
from src.utils import determine_device
from src import training_and_val


def main(args):
    print(args)
    
    if args["port"] is not None and args["port"] > 0:
        job_env = submitit.JobEnvironment()
        print(f"Debugger is running on node {job_env.hostname} port {args['port']}")
        debugpy.listen((job_env.hostname, args["port"]))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    determine_device(args)
    init_tensorboard_writer(args)

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

    extract_the_dataset_on_compute_node(COMPUTE_DATA_DIR, EXT, TARGET_DIR)

    print("Running main job...")
    print(f"Data is in {COMPUTE_DATA_DIR_BASE_DIR}")
    training_and_val.main(
        root_path=COMPUTE_DATA_DIR,
        **args,
    )


def init_tensorboard_writer(args):
    if args["port"] != 0:
        tb_postfix = args["tb_postfix"]
        writer = SummaryWriter(
            log_dir=f"logs/runs/{tb_postfix}",
        )
        args["writer"] = writer
    else:
        args["writer"] = None


if __name__ == "__main__":
    args = training_and_val.get_inputs()
    assert "dataset" in args, "Please provide a dataset"
    assert "block_main" in args, "Please provide block_main"
    assert args["dataset"] in datasets.registered_datasets, "Dataset not found"
    assert "port" in args, "Please provide a port"

    print("submitting jobs")

    executor = submitit.AutoExecutor(folder="logs/%j")
    executor.update_parameters(
        timeout_min=args["timeout"],
        cpus_per_task=8,
        slurm_additional_parameters={
            "constraint": "thin",
            "reservation": "safe",
            "gpus": 1,
        },
    )
    if args["port"] == 0:
        print("Running in locally")
        main(args)
    else:
        job = executor.submit(main, args)
        print("Job submitted")
        # wait until the job has finished
        if args["block_main"]:
            job.wait()
