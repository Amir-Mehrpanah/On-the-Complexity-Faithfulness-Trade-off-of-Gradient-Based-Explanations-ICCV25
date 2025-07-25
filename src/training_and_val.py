import argparse
import torch
from datetime import datetime

from src.models.utils import get_model
from src.datasets import get_training_and_test_dataloader
from src.utils import (
    ActivationSwitch,
    AugmentationSwitch,
    LossSwitch,
    DatasetSwitch,
    ModelSwitch,
    convert_str_to_activation_fn,
    convert_str_to_loss_fn,
    save_pth,
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
        "--augmentation",
        type=AugmentationSwitch.convert,
        required=True,
        help="use data augmentation",
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
    parser.add_argument(
        "--l2_reg",
        type=float,
        default=0.0,
        help="l2 regularization",
    )
    parser.add_argument(
        "--pre_act",
        action="store_true",
        help="use preact architecture",
    )
    parser.add_argument(
        "--gaussian_noise_var",
        type=float,
        default=1e-5,
        help="variance of gaussian noise",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="number of layers",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="number of warmup epochs",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="timeout for the job",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    args = parser.parse_args()
    args = vars(args)
    return args


def train(dataloader, model, loss_fn, optimizer, epoch, device, writer):
    model.train()
    size = len(dataloader.dataset)
    total_loss, total_correct, grad_norm = 0, 0, 0
    for step, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)

        loss = loss_fn(pred, y)
        total_loss += loss.item()
        loss = loss / x.size(0)
        loss.backward()
        # grad_norm += torch.norm(
        #     torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        # ).item()
        optimizer.step()

        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        total_correct += correct
        # if writer is not None:
        #     writer.add_scalar("Loss/train", loss.item(), step + size * epoch)
        #     writer.add_scalar(
        #         "Accuracy/train", correct / x.size(0), step + size * epoch
        #     )

    total_loss = total_loss / size
    total_correct = total_correct / size
    # grad_norm = grad_norm / size
    if writer is not None:
        writer.add_scalar("Loss/train_epoch", total_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", total_correct, epoch)
        # writer.add_scalar(
        #     "ParamsNorm",
        #     torch.norm(torch.cat([p.view(-1) for p in model.parameters()])),
        #     epoch,
        # )
        # writer.add_scalar(
        #     "GradNorm",
        #     grad_norm,
        #     epoch,
        # )
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
    num_workers,
    prefetch_factor,
    patience,
    lr_decay_gamma,
    l2_reg,
    writer,
    pre_act,
    gaussian_noise_var,
    gaussian_blur_var,
    layers,
    warmup_epochs,
    seed,
    device,
    min_test_acc,
    checkpoint_path,
    **kwargs,
):
    torch.manual_seed(seed)
    activation_fn = convert_str_to_activation_fn(activation)
    loss_fn = convert_str_to_loss_fn(loss)

    # DATASET AND DATALOADERS
    train_dataloader, test_dataloader, input_shape, num_classes = (
        get_training_and_test_dataloader(
            dataset,
            root_path,
            batch_size,
            img_size=img_size,
            augmentation=AugmentationSwitch.TRAIN,
            add_inverse=add_inverse,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            gaussian_noise_var=gaussian_noise_var,
            gaussian_blur_var=gaussian_blur_var,
        )
    )

    # MODEL
    torch.cuda.empty_cache()
    model = get_model(
        input_shape=input_shape,
        model_name=model_name,
        num_classes=num_classes,
        activation_fn=activation_fn,
        bias=bias,
        add_inverse=add_inverse,
        pre_act=pre_act,
        layers=layers,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_gamma)
    print(
        f"Experiment model_name {model_name} activation {activation} layers {layers}"
        f" loss {loss} bias {bias} add_inverse {add_inverse} l2_reg {l2_reg}"
        f" ({batch_size},{input_shape}) pre_act {pre_act}"
        f" gaussian_noise_var {gaussian_noise_var} gaussian_blur_var {gaussian_blur_var}"
    )
    old_test_acc = 0
    saved_any_checkpoint = False
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
        if save_ckpt_criteria(
            ckpt_mod,
            epoch,
            test_acc,
            old_test_acc,
            warmup_epochs,
            min_test_acc,
        ):
            saved_any_checkpoint = True
            save_pth(
                model,
                train_acc,
                test_acc,
                path=checkpoint_path,
            )

        scheduler.step()

        # early stopping
        if early_stopping_criteria(
            epoch,
            warmup_epochs,
            test_acc,
            old_test_acc,
            min_test_acc,
            tol=1e-3,
        ):
            patience_counter -= 1
            if patience_counter == 0:
                print(
                    "Early stopping met when test_acc is ",
                    test_acc,
                    " at epoch ",
                    epoch,
                    " if saved checkpoints: ",
                    saved_any_checkpoint,
                )
                break
        else:
            patience_counter = patience
        old_test_acc = test_acc

    if not saved_any_checkpoint:
        print(
            "No checkpoints saved for ",
            checkpoint_path,
        )

    return test_acc


def early_stopping_criteria(
    epoch,
    warmup_epochs,
    test_acc,
    old_test_acc,
    min_test_acc,
    tol,
):
    #
    # to ensure all models are trained equally
    return (
        (epoch > warmup_epochs)
        and (test_acc > min_test_acc)
        # and (test_acc - old_test_acc < tol)
    )


def save_ckpt_criteria(
    ckpt_mod,
    epoch,
    test_acc,
    old_test_acc,
    warmup_epochs,
    min_test_acc,
):
    return (
        (epoch % ckpt_mod == 0)
        and (epoch > warmup_epochs)
        and (test_acc > min_test_acc)
        # and (test_acc > old_test_acc)
    )


if __name__ == "__main__":
    # INPUT VARS
    args = get_inputs()
    main(**args)
