DATASETS_COMMON = "/proj/azizpour-group/datasets/"

MNIST_ROOT = f"{DATASETS_COMMON}/MNIST/"
CIFAR10_ROOT = f"{DATASETS_COMMON}/cifar-10-batches-py"
CIFAR100_ROOT = f"{DATASETS_COMMON}/cifar-100-python/"
# incomplete
IMAGENET_TRAIN_ROOT = f"{DATASETS_COMMON}/imagenet/train/"
IMAGENET_VAL_ROOT = f"{DATASETS_COMMON}/imagenet/val/"


def get_local_data_dir(dataset):
    return (
        f"/home/x_amime/x_amime/projects/kernel-view-to-explainability/.tmp/{dataset}/"
    )


def get_remote_data_dir(dataset):
    return f"/scratch/local/{dataset}/"
