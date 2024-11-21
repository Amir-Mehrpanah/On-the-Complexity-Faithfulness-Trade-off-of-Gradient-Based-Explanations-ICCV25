import argparse
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision
from src.models import get_model
from src.datasets import get_training_and_test_data
from src.utils import (
    ActivationSwitch,
    LossSwitch,
    DatasetSwitch,
    ModelSwitch,
    convert_str_to_activation_fn,
    convert_str_to_loss_fn,
    save_pth,
    get_save_path,
)


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
        "--dataset",
        type=DatasetSwitch.convert,
        required=True,
        help="Dataset to use (e.g., cifar10, mnist)",
    )
    parser.add_argument(
        "--activation",
        type=ActivationSwitch.convert,
        required=True,
        help="Activation function (e.g., relu, softplus)",
    )
    parser.add_argument(
        "--loss",
        required=True,
        type=LossSwitch.convert,
        help="Loss function (e.g., ce, mse)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Image size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.set_defaults(bias=True)
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_false",
        help="if the network has no bias",
    )
    parser.add_argument(
        "--ckpt_mod",
        type=int,
        default=1,
        help="checkpoint if epoch % ckpt_mod == 0",
    )
    parser.add_argument(
        "--add_inverse",
        action="store_true",
        help="add the inverse of the input image to the input",
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
        "--tb_postfix",
        type=str,
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="postfix for tensorboard logs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="patience for early stopping",
    )
    parser.add_argument(
        "--model_name",
        type=ModelSwitch.convert,
        required=True,
        help="model name",
    )
    parser.add_argument(
        "--lr_decay_gamma",
        type=float,
        default=0.95,
        help="gamma for lr scheduler",
    )

    args = parser.parse_args()
    args = vars(args)
    return args


def train(dataloader, model, loss_fn, optimizer, epoch, device, writer):
    model.train()
    size = len(dataloader.dataset)
    total_loss, total_correct = 0, 0
    for step, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)

        loss = loss_fn(pred, y)
        total_loss += loss.item()
        loss = loss / x.size(0)
        loss.backward()
        optimizer.step()

        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        total_correct += correct
        if writer is not None:
            writer.add_scalar("Loss/train", loss.item(), step + size * epoch)
            writer.add_scalar(
                "Accuracy/train", correct / x.size(0), step + size * epoch
            )

    total_loss = total_loss / size
    total_correct = total_correct / size
    if writer is not None:
        writer.add_scalar("Loss/train_epoch", total_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", total_correct, epoch)
    print(f"train accuracy: {(100*total_correct):>0.1f}%, train loss: {total_loss:>8f}")
    return total_loss, total_correct


def test(dataloader, model, loss_fn, epoch, device, writer):
    size = len(dataloader.dataset)
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            total_loss += loss_fn(pred, y).item()
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            total_correct += correct

    total_loss /= size
    total_correct /= size
    if writer is not None:
        writer.add_scalar("Loss/test_epoch", total_loss, epoch)
        writer.add_scalar("Accuracy/test_epoch", total_correct, epoch)
    print(f"test accuracy: {(100*total_correct):>0.1f}%, test loss: {total_loss:>8f}")
    return total_loss, total_correct


def main(
    *,
    root_path,
    activation,
    model_name,
    loss,
    batch_size,
    img_size,
    epochs,
    lr,
    bias,
    ckpt_mod,
    add_inverse,
    dataset,
    port,
    num_workers,
    prefetch_factor,
    patience,
    lr_decay_gamma,
    writer,
    **kwargs,
):
    activation_fn = convert_str_to_activation_fn(activation)
    loss_fn = convert_str_to_loss_fn(loss)

    # DATASET AND DATALOADERS
    train_dataloader, test_dataloader, input_shape, num_classes = (
        get_training_and_test_data(
            dataset,
            root_path,
            batch_size,
            img_size=img_size,
            add_inverse=add_inverse,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    )

    # MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu" if port == 0 else device
    print(f"Using {device} device")

    torch.cuda.empty_cache()
    model = get_model(
        input_shape=input_shape,
        model_name=model_name,
        num_classes=num_classes,
        activation_fn=activation_fn,
        bias=bias,
        add_inverse=add_inverse,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_gamma)
    print(
        f"Experiment activation {activation} loss {loss} bias {bias} add_inverse {add_inverse} ({batch_size},{input_shape})"
    )
    old_test_loss = np.inf
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, train_acc = train(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            epoch,
            device,
            writer,
        )
        test_loss, test_acc = test(
            test_dataloader,
            model,
            loss_fn,
            epoch,
            device,
            writer,
        )
        if save_ckpt_criteria(ckpt_mod, epoch, test_loss, old_test_loss):
            save_pth(
                model,
                path=get_save_path(
                    activation,
                    bias,
                    epoch,
                    add_inverse,
                ),
            )

            write_an_example_image(
                model,
                train_dataloader,
                writer,
                epoch,
                device,
                add_inverse,
            )
        scheduler.step()

        # early stopping
        if test_loss > old_test_loss:
            patience_counter -= 1
            if patience_counter == 0:
                print("Early stopping")
                break
        else:
            patience_counter = patience
        old_test_loss = test_loss


def write_an_example_image(
    model,
    train_dataloader,
    writer: SummaryWriter,
    epoch,
    device,
    add_inverse,
):
    """
    this function writes an example image to tensorboard for visualization purposes
    """

    if writer is not None:
        model.eval()

        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            break
        x = x[[0], ...]
        x.requires_grad = True
        pred = model(x)  # get the first image
        pred = pred.max()
        pred.backward()
        grad = torch.norm(x.grad.squeeze(), 2, dim=0).cpu().numpy()
        grad = (grad - grad.min()) / (grad.max() - grad.min()) * 256
        writer.add_image(
            "Grad Image",
            grad,
            epoch,
            dataformats="HW",
        )

        if add_inverse:
            x = x[0, :3]
        else:
            x = x[0]

        x = (x - x.min()) / (x.max() - x.min()) * 256
        writer.add_image(
            "Input Image",
            x.detach().cpu(),
            epoch,
        )


def save_ckpt_criteria(ckpt_mod, epoch, test_loss, old_test_loss, warmup_epochs=30):
    return (
        (epoch % ckpt_mod == 0)
        and (epoch > warmup_epochs)
        and (test_loss < old_test_loss)
    )


if __name__ == "__main__":
    # INPUT VARS
    args = get_inputs()

    main(**args)
