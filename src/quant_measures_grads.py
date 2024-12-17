import os
from src.datasets import get_grad_dataloader


def measure_grads(data):
    return {"sum_grads": data["mean_rank"].sum().item()}


def main(
    *,
    root_path,
    num_workers,
    prefetch_factor,
    hook_samples,
    output_dir,
    **kwargs,
):
    print(f"num_workers: {num_workers}")
    print(f"prefetch_factor: {prefetch_factor}")
    print(f"kwargs: {kwargs}")
    hooks_dir = os.path.join(output_dir, "hooks")
    os.makedirs(hooks_dir, exist_ok=True)
    dataloader = get_grad_dataloader(
        root_path,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    measurements = []
    for data in dataloader:
        measurements.append(measure_grads(data))
        if data["index"] in hook_samples:
            address = data["address"][0]
            parent_dir = os.path.basename(os.path.dirname(address))
            os.system(f"rsync -a {address} {hooks_dir}/{parent_dir}/")  # faster
            # torch.save(data, f"{hooks_dir}/{data['address']}")

    return measurements
