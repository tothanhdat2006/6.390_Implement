import os
from pathlib import Path
import logging
import tqdm
import wandb
import matplotlib.pyplot as plt
import argparse

import numpy as np
import torch
import torch.nn as nn
from unet.unet import UNET
from utils.load_data import ImageCustomDataset, MaskCustomDataset
from torch.utils.data import random_split, DataLoader

dir_images = Path('./data/imgs')
dir_masks = Path('./data/masks')
dir_checkpoint = Path('./checkpoints/')

def train_model(
        model, 
        device, 
        n_epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.99,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    dataset = MaskCustomDataset(dir_images, dir_masks, img_scale)

    # tmp_img = dataset[0]['image'].permute(1, 2, 0)
    # plt.imshow(np.asarray(tmp_img))
    # plt.show()

    # 2. Split into train / validation partitions
    n_val = int(val_percent * len(dataset))
    n_train = len(dataset) - n_val
    train_set, test_set = random_split(dataset, [n_train, n_val], generator = torch.Generator().manual_seed(86))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu.count(), pin_memory=True)
    train_dataloader = DataLoader(train_set, shuffle=True, **loader_args)
    test_dataloader = DataLoader(test_set, shuffle=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    
    # 5. Begin training
    return  

def get_args():
    parser = argparse.ArgumetParser(description='Train UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == "__main__":
    print("Training...")
    args = get_args()
    
    model = UNET(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format = torch.channels_last)
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    model.to(device=device)
    try:
        train_model(
            model,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    
