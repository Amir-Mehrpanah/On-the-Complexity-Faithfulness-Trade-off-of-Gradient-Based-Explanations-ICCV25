DATASETS_COMMON = "/proj/azizpour-group/datasets"

MNIST_ROOT = f"{DATASETS_COMMON}/MNIST"
CIFAR10_ROOT = f"{DATASETS_COMMON}/cifar-10-batches-py"
CIFAR100_ROOT = f"{DATASETS_COMMON}/cifar-100-python"
IMAGENETTE_ROOT = f"{DATASETS_COMMON}/imagenette2.tgz"
# incomplete
IMAGENET_TRAIN_ROOT = f"{DATASETS_COMMON}/imagenet/train/"
IMAGENET_VAL_ROOT = f"{DATASETS_COMMON}/imagenet/val/"

COMPUTE_OUTPUT_DIR = "/scratch/local/outputs/"
LOCAL_OUTPUT_DIR = ".tmp/outputs/"


def get_local_data_dir(dataset):
    """
    This function is for visualization or debugging purposes. It returns the local path of the dataset.
    """
    if str(dataset) == "CIFAR10":
        return f"/home/x_amime/x_amime/projects/kernel-view-to-explainability/.tmp/CIFAR10/"
    elif str(dataset) == "IMAGENETTE":
        return f"/proj/azizpour-group/datasets/"
    else:
        raise ValueError(f"Unknown dataset {dataset}")


def get_remote_data_dir(dataset):
    return f"/scratch/local/{dataset}/"
