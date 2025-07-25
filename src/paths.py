WORKDIR = "/home/x_amime/x_amime/projects/kernel-view-to-explainability"
DATASETS_COMMON = "/proj/azizpour-group/datasets"

MNIST_ROOT = f"{DATASETS_COMMON}/MNIST"
FASHION_MNIST_ROOT = f"{DATASETS_COMMON}/FashionMNIST"
CIFAR10_ROOT = f"{DATASETS_COMMON}/cifar-10-batches-py"
CIFAR100_ROOT = f"{DATASETS_COMMON}/cifar-100-python"
IMAGENETTE_ROOT = f"{DATASETS_COMMON}/imagenette2.tgz"
IMAGENET_ROOT = f"{DATASETS_COMMON}/imagenet.tgz"
GRADS_ROOT = f"{WORKDIR}/.tmp/outputs/"

CHECKPOINTS_DIR = f"{WORKDIR}/checkpoints/"
COMPUTE_OUTPUT_DIR = "/scratch/local/outputs/"
LOCAL_OUTPUT_DIR = f"{WORKDIR}/.tmp/outputs/"
LOCAL_QUANTS_DIR = f"{WORKDIR}/.tmp/quants/"


def get_local_data_dir(dataset):
    """
    This function is for visualization or debugging purposes. It returns the local path of the dataset.
    """
    if str(dataset) == "CIFAR10":
        return f"{WORKDIR}/.tmp/CIFAR10/"
    elif str(dataset) == "IMAGENETTE":
        return DATASETS_COMMON
    elif str(dataset) == "MNIST":
        return f"{WORKDIR}/.tmp/MNIST/"
    elif str(dataset) == "FASHION_MNIST":
        return f"{WORKDIR}/.tmp/FASHION_MNIST/"
    elif str(dataset) == "GRADS":
        return GRADS_ROOT
    else:
        raise ValueError(f"Unknown dataset {dataset}")


def get_remote_data_dir(dataset):
    return f"/scratch/local/{dataset}/"
