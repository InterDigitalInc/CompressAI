# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import shutil
import sys

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import compressai.transforms as transforms

from compressai._utils.distributed import unwrap_model
from compressai._utils.meters import AverageMeter
from compressai.datasets import ModelNetDataset
from compressai.losses import ChamferPccRateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.registry import MODELS
from compressai.zoo import pointcloud_models


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    args,
):
    model.train()
    device = next(model.parameters()).device
    unwrapped_model = unwrap_model(model)

    for i, d in enumerate(train_dataloader):
        d = {k: v.to(device) for k, v in d.items()}
        batch_size = next(iter(d.values())).shape[0]

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        aux_loss = unwrapped_model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            metrics = {
                "samples": i * batch_size,
                "progress": 100.0 * i / len(train_dataloader),
                "loss": out_criterion["loss"].item(),
                "bpp_loss": out_criterion["bpp_loss"].item(),
                "rec_loss": out_criterion["rec_loss"].item(),
                "aux_loss": aux_loss.item(),
            }
            print(
                f"Train epoch {epoch}: ["
                f"{metrics['samples']}/{len(train_dataloader.dataset)} "
                f"({metrics['progress']:.0f}%)] "
                f"Loss: {metrics['loss']:.3f} | "
                f"Bpp loss: {metrics['bpp_loss']:.3f} | "
                f"Rec loss: {metrics['rec_loss']:.4f} | "
                # f"Aux loss: {aux_loss.item():.0f} | "
                "\n"
            )


def test_epoch(epoch, test_dataloader, model, criterion, args):
    model.eval()
    unwrapped_model = unwrap_model(model)
    unwrapped_model.update(force=True, update_quantiles=True)
    device = next(model.parameters()).device

    meter_keys = ["loss", "bpp_loss", "rec_loss", "aux_loss"]
    meters = {key: AverageMeter() for key in meter_keys}

    with torch.no_grad():
        for d in test_dataloader:
            d = {k: v.to(device) for k, v in d.items()}
            batch_size = next(iter(d.values())).shape[0]

            out_net = model(d)
            out_criterion = criterion(out_net, d)
            metrics = {
                "loss": out_criterion["loss"].item(),
                "bpp_loss": out_criterion["bpp_loss"].item(),
                "rec_loss": out_criterion["rec_loss"].item(),
                "aux_loss": unwrapped_model.aux_loss().item(),
            }
            for key, value in metrics.items():
                meters[key].update(value, batch_size)

    print(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {meters['loss'].avg:.3f} | "
        f"Bpp loss: {meters['bpp_loss'].avg:.3f} | "
        f"Rec loss: {meters['rec_loss'].avg:.4f} | "
        # f"Aux loss: {meters['aux_loss'].avg:.0f} | "
        "\n"
    )

    return meters["loss"].avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="sfu2023-pcc-rec-pointnet",
        choices=pointcloud_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=100,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    num_points = 1024

    train_dataset = ModelNetDataset(
        args.dataset,
        split="train",
        pre_transform=Compose(
            [
                transforms.ToDict(wrapper="torch_geometric.data.Data"),
                transforms.SamplePointsV2(
                    num=8192, remove_faces=True, include_normals=True, static_seed=1234
                ),
                transforms.NormalizeScaleV2(center=True, scale_method="l2"),
                transforms.ToDict(wrapper="dict"),
            ]
        ),
        transform=Compose(
            [
                transforms.ToDict(wrapper="torch_geometric.data.Data"),
                transforms.RandomSample(num=num_points, attrs=["pos", "normal"]),
                transforms.ToDict(wrapper="dict"),
            ]
        ),
    )

    test_dataset = ModelNetDataset(
        args.dataset,
        split="test",
        pre_transform=Compose(
            [
                transforms.ToDict(wrapper="torch_geometric.data.Data"),
                transforms.SamplePointsV2(
                    num=8192, remove_faces=True, include_normals=True, static_seed=1234
                ),
                transforms.NormalizeScaleV2(center=True, scale_method="l2"),
                transforms.ToDict(wrapper="dict"),
            ]
        ),
        transform=Compose(
            [
                transforms.ToDict(wrapper="torch_geometric.data.Data"),
                transforms.RandomSample(
                    num=num_points, attrs=["pos", "normal"], static_seed=1234
                ),
                transforms.ToDict(wrapper="dict"),
            ]
        ),
    )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = MODELS[args.model]()
    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = ChamferPccRateDistortionLoss(lmbda={"bpp": 1.0, "rec": args.lmbda})

    last_epoch = 0
    best_loss = float("inf")
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("loss", float("inf"))
        unwrap_model(net).load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, args)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": unwrap_model(net).state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
            )


if __name__ == "__main__":
    main(sys.argv[1:])


# NOTE: A more complete trainer with experiment tracking, visualizations, etc
# that uses CompressAI Trainer can be found at:
#
# https://github.com/multimedialabsfu/learned-point-cloud-compression-for-classification
