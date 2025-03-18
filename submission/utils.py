from datetime import datetime
from itertools import product
import os
import submitit
import pandas as pd
from src import paths
from submission import explainers, training, grads, quant
from src.utils import EXPERIMENT_PREFIX_SEP, get_experiment_prefix, get_save_path


def submit_training(
    *,
    block_main,
    port,
    timeout,
    warmup_epochs_ratio,
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
    dir_names = args["checkpoint_path"].apply(os.path.dirname)
    for checkpoint_path_dir_name in dir_names.unique():
        os.makedirs(checkpoint_path_dir_name, exist_ok=True)

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


def submit_explainers(
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
    args["experiment_prefix"] = args.apply(
        lambda x: get_experiment_prefix(
            **x,
        )
        + f"{EXPERIMENT_PREFIX_SEP}{x.explainer}",
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

    return execute_job_submission(
        block_main, port, timeout, valid_args, explainers.main
    )


def submit_grads(
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
    args["experiment_prefix"] = args.apply(
        lambda x: get_experiment_prefix(**x)
        + f"{EXPERIMENT_PREFIX_SEP}{x.e_gaussian_noise_var}{EXPERIMENT_PREFIX_SEP}{x.e_gaussian_blur_var}",
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

    args.gaussian_noise_var = args.e_gaussian_noise_var
    args.gaussian_blur_var = args.e_gaussian_blur_var

    output_dir_exists = args["experiment_output_dir"].apply(lambda x: os.path.exists(x))
    checkpoint_exists = args["checkpoint_path"].apply(lambda x: os.path.exists(x))
    valid_ids = checkpoint_exists & ~output_dir_exists
    valid_args = args[valid_ids]

    print("Checkpoints are available:")
    args[checkpoint_exists]["checkpoint_path"].apply(print)
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

    if "checkpoint_path" in nunique:
        if nunique["checkpoint_path"] != len(args):
            print("ATTENTION!!!\n Some checkpoint paths seem to be the same!")

    if len(args) == 0:
        print("No jobs to run exiting")
        return

    if port != None:
        print("Running only the first job because of the debug flag")
        jobs_args = [jobs_args[0]]

    print("Do you want to continue? [y/n]", flush=True)
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
