from datetime import datetime
from itertools import product
import submitit
import pandas as pd
from submission import training, grads


def submit_training(
    *,
    block_main,
    port,
    timeout,
    batch_size,
    warmup_epochs_ratio,
    **args,
):
    now = datetime.now().strftime("%Y%m%d-%H")
    args = pd.DataFrame(
        list(
            product(
                *args.values(),
            )
        ),
        columns=args.keys(),
    )
    args["tb_postfix"] = args.apply(
        lambda x: f"{now}_{x.activation}_{x.dataset}_{x.layers}_{x.model_name.name}",
        axis=1,
    )
    args["port"] = port
    args["block_main"] = block_main
    args["timeout"] = timeout
    args["batch_size"] = args["activation"].map(batch_size)
    args["warmup_epochs"] = (args["epochs"] * warmup_epochs_ratio).astype(int)

    return execute_job_submission(block_main, port, timeout, args, training.main)


def submit_grads(
    *,
    block_main,
    port,
    timeout,
    batch_size,
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
    args["batch_size"] = args["activation"].map(batch_size)

    return execute_job_submission(block_main, port, timeout, args, grads.main)


def execute_job_submission(block_main, port, timeout, args, func):
    jobs_args = args.to_dict(orient="records")

    repr_args = args.copy()
    repr_args = repr_args.map(str)
    nunique = repr_args.nunique()
    print(nunique)
    print("total num of jobs", len(args))

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
        cpus_per_task=8,
        slurm_additional_parameters={
            "constraint": "thin",
            "reservation": "safe",
            "gpus": 1,
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
