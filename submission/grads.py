import os
import sys
import debugpy
import submitit

# vscode changes the cwd to the file's directory, so we need to add the workspace to the path
# Set the working directory to the base of the workspace
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
sys.path.insert(0, workspace_dir)

from src import datasets
from src.utils import determine_device, get_experiment_prefix
from src.datasets import (
    extract_the_dataset_on_compute_node,
    move_data_to_compute_node,
    resolve_data_directories,
    move_output_compute_node,
)
from src import compute_grad


def main(args):
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

    extract_the_dataset_on_compute_node(COMPUTE_DATA_DIR, EXT, TARGET_DIR)

    print("Running main job...")
    print(f"Data is in {COMPUTE_DATA_DIR_BASE_DIR}")
    compute_grad.main(
        root_path=COMPUTE_DATA_DIR,
        output_dir=COMPUTE_OUTPUT_DIR,
        **args,
    )

    experiment_prefix = get_experiment_prefix(
        model_name=args["model_name"],
        activation=args["activation"],
        augmentation=args["augmentation"],
        bias=args["bias"],
        epoch=args["epoch"],
        add_inverse=args["add_inverse"],
        pre_act=args["pre_act"],
    )
    move_output_compute_node(
        COMPUTE_OUTPUT_DIR,
        LOCAL_OUTPUT_DIR,
        experiment_prefix,
    )


if __name__ == "__main__":
    args = compute_grad.get_inputs()
    assert "dataset" in args, "Please provide a dataset"
    assert "block_main" in args, "Please provide block_main"
    assert args["dataset"] in datasets.registered_datasets, "Dataset not found"
    assert "port" in args, "Please provide a port"
    assert "augmentation" in args, "Please provide an augmentation"
    assert isinstance(
        args["augmentation"], datasets.AugmentationSwitch
    ), "Invalid augmentation"

    if args["port"] == 0:
        print("Running in locally")
        main(args)
    else:
        executor = submitit.AutoExecutor(folder="logs/%j")
        executor.update_parameters(
            timeout_min=60,
            cpus_per_task=8,
            slurm_additional_parameters={
                "constraint": "thin",
                "reservation": "safe",
                "gpus": 1,
            },
        )
        job = executor.submit(main, args)
        print("Job submitted")
        # wait until the job has finished
        if args["block_main"]:
            print("Waiting for job to finish...")
            job.wait()
