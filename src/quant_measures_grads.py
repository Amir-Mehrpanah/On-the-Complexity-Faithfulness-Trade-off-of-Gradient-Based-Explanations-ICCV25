import argparse

from src.datasets import get_grad_dataloader


def get_inputs():
    parser = argparse.ArgumentParser(description="Get inputs for the model.")
    parser.add_argument(
        "--block_main",
        action="store_true",
        help="Block the main thread",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen for debugger attach",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of workers for dataloaders",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="prefetch factor for dataloaders",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="timeout for the job",
    )

    args = parser.parse_args()
    args = vars(args)
    return args

def measure_grads(data):
    return {"sum_grads": data.grad.sum().item()}


def main(
    *,
    root_path,
    num_workers,
    prefetch_factor,
    **kwargs,
):
    print(f"num_workers: {num_workers}")
    print(f"prefetch_factor: {prefetch_factor}")
    print(f"kwargs: {kwargs}")

    dataloader = get_grad_dataloader(
        root_path,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    measurements = []
    for data in dataloader:
        measurements.append(measure_grads(data))

    return measurements
