import network
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets.rzdg import RZDGSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn


from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--dataset", type=str, default='rzdg',
                        choices=['voc', 'cityscapes', 'rzdg'],
                        help='Dataset name')

    parser.add_argument("--data_root", type=str,
                        default='/home/rax/Documents/team_project/RZDG_real_seg',
                        help="Path to RZDG dataset")

    parser.add_argument("--num_classes", type=int, default=None)

    # Model Options
    available_models = sorted(
        name for name in network.modeling.__dict__
        if name.islower() and callable(network.modeling.__dict__[name])
    )

    parser.add_argument("--model", type=str,
                        default='deeplabv3plus_resnet50',
                        choices=available_models)

    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Training Options
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--loss_type", type=str, default='cross_entropy')
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1)

    return parser


def get_dataset(opts):
    """ Dataset and Augmentation """

    # Shared normalization
    normalize = et.ExtNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if opts.dataset == 'rzdg':
        train_transform = et.ExtCompose([
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            normalize,
        ])

        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            normalize,
        ])

        train_dst = RZDGSegmentation(
            root=opts.data_root,
            split='train',
            transform=train_transform
        )

        val_dst = RZDGSegmentation(
            root=opts.data_root,
            split='val',
            transform=val_transform
        )

    elif opts.dataset == 'voc':
        raise NotImplementedError("VOC not used in this project")

    elif opts.dataset == 'cityscapes':
        raise NotImplementedError("Cityscapes not used in this project")

    return train_dst, val_dst


def main():
    opts = get_argparser().parse_args()

    # 🔑 RZDG has 3 classes
    if opts.dataset == 'rzdg':
        opts.num_classes = 3

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)

    # FIX: Set num_workers=0 to avoid multiprocessing data corruption
    train_loader = data.DataLoader(
        train_dst,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=0,  # Changed from 2 to 0
        drop_last=True
    )

    val_loader = data.DataLoader(
        val_dst,
        batch_size=opts.val_batch_size,
        shuffle=False,
        num_workers=0  # Changed from 2 to 0
    )

    print(f"Dataset: RZDG | Train: {len(train_dst)} | Val: {len(val_dst)}")

    # Model
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes,
        output_stride=opts.output_stride
    )

    model = nn.DataParallel(model)
    model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opts.lr,
        weight_decay=opts.weight_decay
    )

    metrics = StreamSegMetrics(opts.num_classes)

    # ========= TRAIN LOOP =========
    for epoch in range(opts.epochs):
        print(f"\nEpoch {epoch + 1}/{opts.epochs}")
        model.train()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Debug: Check label values (remove after confirming it works)
            if batch_idx == 0 and epoch == 0:
                print(f"DEBUG: Label min={labels.min()}, max={labels.max()}, unique={torch.unique(labels)}")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # ===== Validation =====
        model.eval()
        metrics.reset()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                metrics.update(
                    labels.cpu().numpy(),
                    preds.cpu().numpy()
                )

        scores = metrics.get_results()
        print(metrics.to_str(scores))

    print("Training finished.")


if __name__ == '__main__':
    main()