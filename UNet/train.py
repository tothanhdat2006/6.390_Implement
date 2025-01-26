import logging
import tqdm
import wandb
import matplotlib.pyplot as plt

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from pathlib import Path
from unet.unet import UNET
from utils.load_data import ImageCustomDataset, MaskCustomDataset

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

if __name__ == "__main__":
    print("Training...")

    model = UNET()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    train_model(
        model,
        device,
    )
    
