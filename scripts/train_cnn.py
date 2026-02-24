import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from lutneuro.config import LUTConfig
from lutneuro.models import lut_resnet20, resnet20
from lutneuro.ops.lut_conv2d import LUTConv2d

_PROGRESS_COLUMNS = (
    TextColumn("[bold cyan]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeRemainingColumn(),
    TimeElapsedColumn(),
    TextColumn("[purple]loss: {task.fields[loss]:.4f}"),
)


def main(args: argparse.Namespace):
    accelerator = Accelerator()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    criterion = nn.CrossEntropyLoss()
    model = lut_resnet20(config=LUTConfig.load_from_yaml(args.config)) if args.lut else resnet20()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    model, optimizer, trainloader, testloader = accelerator.prepare(model, optimizer, trainloader, testloader)

    with Progress(*_PROGRESS_COLUMNS, disable=not accelerator.is_main_process) as progress:
        epoch_bar = progress.add_task("Training", total=args.epochs, loss=0.0)

        for epoch in range(args.epochs):
            model.train()
            step_bar = progress.add_task(f"  Epoch {epoch + 1}/{args.epochs}", total=len(trainloader), loss=0.0)
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(trainloader):
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                if args.lut:
                    lut_loss = []
                    for _name, module in model.named_modules():
                        if isinstance(module, LUTConv2d):
                            lut_loss.append(module.linear.lut_loss)
                    lut_loss = torch.stack(lut_loss).mean()
                    loss += lut_loss
                accelerator.backward(loss)
                optimizer.step()

                running_loss += (loss.item() - running_loss) / (i + 1)
                progress.update(step_bar, advance=1, loss=running_loss)

            progress.update(epoch_bar, advance=1, loss=running_loss)
            progress.remove_task(step_bar)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            _, predicted = torch.max(model(images), 1)
            predicted, labels = accelerator.gather_for_metrics((predicted, labels))
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accelerator.print(f"Accuracy of the network on the 10000 test images: {100 * correct // total:.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="number of workers")
    parser.add_argument("--lut", action="store_true", default=False, help="use LUT")
    parser.add_argument("--config", type=str, default="configs/resnet.yaml", help="configuration file")
    args = parser.parse_args()
    main(args)
