#!/usr/bin/env python3
"""
Training script for PointPillars on RZDG dataset
"""

import argparse
import os
import torch
from tqdm import tqdm
import pdb

from pointpillars.utils import setup_seed
from pointpillars.dataset import RZDG, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.loss import Loss
from torch.utils.tensorboard import SummaryWriter


def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)


def main(args):
    setup_seed()
    
    # Load datasets
    train_dataset = RZDG(data_root=args.data_root, split='train')
    val_dataset = RZDG(data_root=args.data_root, split='val')
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False)

    # Initialize model with 2 classes (barrier, road_beacon)
    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=2).cuda()
    else:
        pointpillars = PointPillars(nclasses=2)
    
    loss_func = Loss()

    # Setup optimizer and scheduler
    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=pointpillars.parameters(),
                                  lr=init_lr,
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=init_lr * 10,
                                                    total_steps=max_iters)

    # Setup tensorboard
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)

    # Setup checkpoint directory
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.max_epoch):
        # Training phase
        pointpillars.train()
        train_loss_dict = {'cls_loss': 0, 'reg_loss': 0, 'total_loss': 0}
        train_step = 0

        pbar = tqdm(train_dataloader, desc=f'Train Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            pts = batch['pts']
            gt_bboxes_3d = batch['gt_bboxes_3d']
            gt_labels = batch['gt_labels']

            if not args.no_cuda:
                pts = pts.cuda()
                gt_bboxes_3d = gt_bboxes_3d.cuda()
                gt_labels = gt_labels.cuda()

            # Forward pass
            output = pointpillars(pts)

            # Calculate loss
            loss_dict = loss_func(output, gt_bboxes_3d, gt_labels)

            # Backward pass
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), 10.0)
            optimizer.step()
            scheduler.step()

            # Accumulate losses
            train_loss_dict['cls_loss'] += loss_dict['cls_loss'].item()
            train_loss_dict['reg_loss'] += loss_dict['reg_loss'].item()
            train_loss_dict['total_loss'] += loss_dict['loss'].item()
            train_step += 1

            # Update progress bar
            pbar.set_postfix({
                'cls_loss': loss_dict['cls_loss'].item(),
                'reg_loss': loss_dict['reg_loss'].item(),
                'lr': optimizer.param_groups[0]['lr']
            })

            global_step += 1

        # Average training losses
        for k in train_loss_dict:
            train_loss_dict[k] /= train_step
        
        print(f"Epoch {epoch} - Train Loss: {train_loss_dict['total_loss']:.4f}")

        # Log training metrics
        save_summary(writer, train_loss_dict, epoch, 'train',
                    lr=optimizer.param_groups[0]['lr'])

        # Validation phase
        if epoch % args.val_interval == 0:
            pointpillars.eval()
            val_loss_dict = {'cls_loss': 0, 'reg_loss': 0, 'total_loss': 0}
            val_step = 0

            with torch.no_grad():
                pbar = tqdm(val_dataloader, desc=f'Val Epoch {epoch}')
                for batch in pbar:
                    pts = batch['pts']
                    gt_bboxes_3d = batch['gt_bboxes_3d']
                    gt_labels = batch['gt_labels']

                    if not args.no_cuda:
                        pts = pts.cuda()
                        gt_bboxes_3d = gt_bboxes_3d.cuda()
                        gt_labels = gt_labels.cuda()

                    output = pointpillars(pts)
                    loss_dict = loss_func(output, gt_bboxes_3d, gt_labels)

                    val_loss_dict['cls_loss'] += loss_dict['cls_loss'].item()
                    val_loss_dict['reg_loss'] += loss_dict['reg_loss'].item()
                    val_loss_dict['total_loss'] += loss_dict['loss'].item()
                    val_step += 1

                    pbar.set_postfix({
                        'cls_loss': loss_dict['cls_loss'].item(),
                        'reg_loss': loss_dict['reg_loss'].item()
                    })

            # Average validation losses
            for k in val_loss_dict:
                val_loss_dict[k] /= val_step

            print(f"Epoch {epoch} - Val Loss: {val_loss_dict['total_loss']:.4f}")

            # Log validation metrics
            save_summary(writer, val_loss_dict, epoch, 'val')

            # Save best model
            if val_loss_dict['total_loss'] < best_val_loss:
                best_val_loss = val_loss_dict['total_loss']
                ckpt_path = os.path.join(args.ckpt_dir, 'best_model.pth')
                torch.save(pointpillars.state_dict(), ckpt_path)
                print(f"Saved best model: {ckpt_path}")

        # Save checkpoint every epoch
        if epoch % args.ckpt_interval == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f'epoch_{epoch}.pth')
            torch.save(pointpillars.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    writer.close()
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointPillars training on RZDG dataset')
    parser.add_argument('--data_root', type=str, default='/home/rax/Documents/team_project/RZDG_real_object',
                        help='Root directory of RZDG dataset')
    parser.add_argument('--log_dir', type=str, default='./logs_rzdg',
                        help='Directory to save tensorboard logs')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints_rzdg',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--max_epoch', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--init_lr', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validation interval in epochs')
    parser.add_argument('--ckpt_interval', type=int, default=5,
                        help='Checkpoint save interval in epochs')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')

    args = parser.parse_args()
    main(args)
