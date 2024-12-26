from datetime import datetime
from itertools import product
import os
import submitit
import pandas as pd
import torch
from src import paths
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from submission import training, grads, quant
from src.utils import EXPERIMENT_PREFIX_SEP, get_experiment_prefix, get_save_path


def submit_training(
    *,
    block_main,
    port,
    timeout,
    warmup_epochs_ratio,
    # batch_size,
    # lr,
    **args,
):
    # now = datetime.now().strftime("%Y%m%d-%H")
    args = pd.DataFrame(
        list(
            product(
                *args.values(),
            )
        ),
        columns=args.keys(),
    )
    # args["lr"] = args["activation"].map(lr)
    # args["batch_size"] = args["activation"].map(batch_size)
    args["tb_postfix"] = args.apply(
        lambda x: get_experiment_prefix(**x),
        axis=1,
    )
    args["checkpoint_path"] = args.apply(
        lambda x: get_save_path(
            **x,
        ),
        axis=1,
    )
    base_path = args["checkpoint_path"][0].split("/")[1]
    os.makedirs(f"checkpoints/{base_path}", exist_ok=True)
    checkpoint_exists = args["checkpoint_path"].apply(lambda x: os.path.exists(x))
    print("Checkpoints skipped because they do already exist")
    args[checkpoint_exists]["checkpoint_path"].apply(print)

    valid_args = args[~checkpoint_exists]
    valid_args["port"] = port
    valid_args["block_main"] = block_main
    valid_args["timeout"] = timeout
    valid_args["warmup_epochs"] = (valid_args["epochs"] * warmup_epochs_ratio).astype(
        int
    )
    print("Valid args:")
    valid_args["checkpoint_path"].apply(print)
    return execute_job_submission(block_main, port, timeout, valid_args, training.main)


def submit_grads(
    *,
    block_main,
    port,
    timeout,
    # batch_size,
    # lr,
    **args,
):
    print(f"time: {datetime.now()}")
    args = pd.DataFrame(
        list(
            product(
                *args.values(),
            )
        ),
        columns=args.keys(),
    )

    args["port"] = port
    args["block_main"] = block_main
    args["timeout"] = timeout
    # args["lr"] = args["activation"].map(lr)
    # args["batch_size"] = args["activation"].map(batch_size)
    args["experiment_prefix"] = args.apply(
        lambda x: get_experiment_prefix(**x)
        + f"{EXPERIMENT_PREFIX_SEP}{x.gaussian_noise_var}",
        axis=1,
    )
    args["experiment_output_dir"] = args.apply(
        lambda x: os.path.join(
            paths.LOCAL_OUTPUT_DIR,
            x.experiment_prefix,
        ),
        axis=1,
    )
    args["checkpoint_path"] = args.apply(
        lambda x: get_save_path(
            **x,
        ),
        axis=1,
    )
    output_dir_exists = args["experiment_output_dir"].apply(lambda x: os.path.exists(x))
    checkpoint_exists = args["checkpoint_path"].apply(lambda x: os.path.exists(x))
    valid_ids = checkpoint_exists & ~output_dir_exists
    valid_args = args[valid_ids]

    print("Checkpoints not available:")
    args[~checkpoint_exists]["checkpoint_path"].apply(print)
    print("Output dirs skipped:")
    args[output_dir_exists]["experiment_output_dir"].apply(print)
    print("Valid args:")
    args[valid_ids]["experiment_output_dir"].apply(print)

    return execute_job_submission(block_main, port, timeout, valid_args, grads.main)


def submit_measurements(
    *,
    block_main,
    port,
    timeout,
    **args,
):
    print(f"time: {datetime.now()}")
    args = pd.DataFrame(
        list(
            product(
                *args.values(),
            )
        ),
        columns=args.keys(),
    )

    args["port"] = port
    args["block_main"] = block_main
    args["timeout"] = timeout

    return execute_job_submission(
        block_main,
        port,
        timeout,
        args,
        quant.main,
        num_gpus=0,
        cpus_per_task=16,
        mem_gb=64,
    )


def execute_job_submission(
    block_main,
    port,
    timeout,
    args,
    func,
    num_gpus=1,
    cpus_per_task=16,
    mem_gb=64,
):
    jobs_args = args.to_dict(orient="records")

    repr_args = args.copy()
    repr_args = repr_args.map(str)
    nunique = repr_args.nunique()
    print(nunique)
    print("total num of jobs", len(args))
    if len(args) == 0:
        print("No jobs to run exiting")
        return

    if port != None:
        print("Running only the first job because of the debug flag")
        jobs_args = [jobs_args[0]]

    print("Do you want to continue? [y/n]")
    if input() != "y":
        print("Aborted")
        return

    print("submitting jobs")
    executor = submitit.AutoExecutor(folder="logs/%j")
    executor.update_parameters(
        timeout_min=timeout,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        slurm_additional_parameters={
            "constraint": "thin",
            "reservation": "safe",
            "gpus": num_gpus,
        },
    )

    if port == 0:
        print("Running in locally")
        func(jobs_args)
    else:
        jobs = executor.map_array(func, jobs_args)
        print("Job submitted")
        # wait until the job has finished
        if block_main:
            print("Waiting for job to finish")
            results = [job.result() for job in jobs]
            print("All jobs finished")
            return results


def visualize_hooks(
    Dataset,
    hook_samples,
    keys,
):
    for j in hook_samples[0]:
        os.makedirs(f".tmp/visualizations/{j}", exist_ok=True)
        glob_path = f".tmp/quants/hooks/{Dataset}*/{j}.pt"
        paths = glob(glob_path)
        print(glob_path, len(paths))
        for path in paths:
            data = torch.load(path)
            corrects = data["correct"]
            batch_size = data["batch_size"]
            print(os.path.basename(glob_path), "corrects", corrects, batch_size)
            prefix = path.split("/")[-2]
            for key in keys:
                if key == "image":
                    if os.path.exists(f".tmp/visualizations/{j}/{key}.png"):
                        continue
                    temp = np.transpose(data[key], (1, 2, 0))
                    plt.imshow(temp)
                    plt.savefig(f".tmp/visualizations/{j}/{key}.png")
                else:
                    temp = data[key]
                    plt.imshow(temp)
                    plt.savefig(f".tmp/visualizations/{j}/{key}_{prefix}.png")
                plt.close()
