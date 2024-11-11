import submitit
import os
import sys

# vscode changes the cwd to the file's directory, so we need to add the workspace to the path
# Set the working directory to the base of the workspace
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
sys.path.insert(0, workspace_dir)

from src import paths
from src import training_and_val


def main(args):
    # Load modules and set up the environment
    os.system("module load Fpart/1.5.1-gcc-8.5.0")
    os.environ["DATA_DIR"] = paths.registered_datasets[args["dataset"]].__default_path__
    os.environ["DATA_TAR_DIR"] = f"/scratch/local/tar_dir/{args['dataset']}"

    # Run your data preparation commands
    os.system("time fpsync -n 8 -m tarify -s 2000M $DATA_DIR $DATA_TAR_DIR")
    os.system(
        "time ls $DATA_TAR_DIR/*.tar | xargs -n 1 -P 8 tar -x -C /scratch/local/ -f"
    )

    print("Running main job...")
    # Your training script here
    training_and_val.main(args)


if __name__ == "__main__":
    args = training_and_val.get_inputs()
    assert "dataset" in args, "Please provide a dataset"

    print("submitting jobs")
    # executor = submitit.AutoExecutor(folder="logs")
    # executor.update_parameters(timeout_min=60, gpus_per_node=1, cpus_per_task=8)
    main()
    # job = executor.submit(main)
