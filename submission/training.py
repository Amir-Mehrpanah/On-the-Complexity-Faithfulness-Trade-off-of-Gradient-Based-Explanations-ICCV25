import subprocess
import submitit
import os
import sys
import debugpy

# vscode changes the cwd to the file's directory, so we need to add the workspace to the path
# Set the working directory to the base of the workspace
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
sys.path.insert(0, workspace_dir)

from src.paths import get_local_data_dir, get_remote_data_dir
from src import datasets
from src import training_and_val


def main(args):
    if args["port"] is not None and args["port"] > 0:
        job_env = submitit.JobEnvironment()
        print(f"Debugger is running on node {job_env.hostname} port {args['port']}")
        debugpy.listen((job_env.hostname, args["port"]))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # data preparation commands here
    DATA_DIR = datasets.registered_datasets[args["dataset"]].__root_path__
    if DATA_DIR.endswith(".tgz"):
        BASE_DIR = os.path.basename(DATA_DIR)[:-4]
    else:
        BASE_DIR = os.path.basename(DATA_DIR)

    if args["port"] == 0:
        # Running locally
        COMPUTE_DATA_DIR = get_local_data_dir(args["dataset"])
    else:
        COMPUTE_DATA_DIR = get_remote_data_dir(args["dataset"])

    COMPUTE_DATA_DIR_BASE_DIR = os.path.join(COMPUTE_DATA_DIR, BASE_DIR)
    os.makedirs(COMPUTE_DATA_DIR_BASE_DIR, exist_ok=True)

    os.system("module load Fpart/1.5.1-gcc-8.5.0")

    result = subprocess.run(
        [
            "time",
            "fpsync",
            "-n",
            "8",
            "-m",
            "tarify",
            "-s",
            "2000M",
            DATA_DIR,
            COMPUTE_DATA_DIR,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to sync data: {result.stderr}")

    result = subprocess.run(
        [
            "time",
            "ls",
            f"{COMPUTE_DATA_DIR}/*.tar",
            "|",
            "xargs",
            "-n",
            "1",
            "-P",
            "8",
            "-I",
            "@",
            "tar",
            "-xf",
            "@",
            "-C",
            f"{COMPUTE_DATA_DIR_BASE_DIR}",
        ],
        capture_output=True,
        text=True,
        shell=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to extract data")

    print("Running main job...")
    print(f"Data is in {COMPUTE_DATA_DIR_BASE_DIR}")
    # training_and_val.main(
    #     root_path=COMPUTE_DATA_DIR,
    #     **args,
    # )


if __name__ == "__main__":
    args = training_and_val.get_inputs()
    assert "dataset" in args, "Please provide a dataset"
    assert args["dataset"] in datasets.registered_datasets, "Dataset not found"
    assert "port" in args, "Please provide a port"

    print("submitting jobs")
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
    if args["port"] == 0:
        print("Running in locally")
        main(args)
    else:
        job = executor.submit(main, args)
        print("Job submitted")
        # wait until the job has finished
        if args["block_main"]:
            job.wait()
