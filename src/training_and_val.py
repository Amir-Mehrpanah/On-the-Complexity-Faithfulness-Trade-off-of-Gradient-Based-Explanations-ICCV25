import argparse
import torch
import tqdm
from torch import nn

from src.models import SimpleConvNet
from src.datasets import get_training_and_test_data
from src.utils import ActivationSwitch, LossSwitch, DatasetSwitch


def get_inputs():
    parser = argparse.ArgumentParser(description="Get inputs for the model.")
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
        default=64,
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
        "--no-bias",
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

    args = parser.parse_args()
    return vars(args)


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    progress = tqdm.tqdm(dataloader)
    for X, y in progress:
        X, y = X.to(device), y.to(device)
        pred = model(X)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress.set_description(f"loss: {loss.item():>7f}")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            y = y.argmax(1)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, test loss: {test_loss:>8f} \n"
    )


def save_pth(model, path):
    """Saves the model to a .pth file.

    Args:
      model: The model to save.
      path: The path to save the model to.
    """
    torch.save(model.state_dict(), path)


def convert_str_to_activation_fn(activation):
    if ActivationSwitch.RELU == activation:
        return nn.ReLU()

    str_activation = str(activation)
    if "SOFTPLUS" in str_activation:
        beta = str_activation.replace("SOFTPLUS_B", "")
        beta = float(beta)

        return nn.Softplus(beta)

    raise NameError(str_activation)


def convert_str_to_loss_fn(loss):
    if LossSwitch.MSE == loss:
        return nn.MSELoss()

    if LossSwitch.CE == loss:
        return nn.CrossEntropyLoss()

    raise NameError(loss)


def main(
    *,
    activation,
    loss,
    batch_size,
    img_size,
    epochs,
    lr,
    bias,
    ckpt_mod,
    add_inverse,
    dataset,
):
    activation_fn = convert_str_to_activation_fn(activation)
    loss_fn = convert_str_to_loss_fn(loss)

    # DATASET AND DATALOADERS
    train_dataloader, test_dataloader = get_training_and_test_data(
        dataset,
        batch_size,
        img_size=img_size,
        add_inverse=add_inverse,
    )

    # MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    model = SimpleConvNet(
        activation=activation_fn,
        conv_bias=bias,
        fc_bias=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Experiment activation {activation} loss {loss} bias {bias}")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
        if epoch % ckpt_mod == 0:
            save_pth(model, f"checkpoints/{activation}_{bias}.pth")
            break


if __name__ == "__main__":
    # INPUT VARS
    args = get_inputs()

    main(**args)
