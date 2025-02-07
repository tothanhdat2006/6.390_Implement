import argparse
import wandb
import logging
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.load_data import load_data
from resnet.resnet import ResNet, ResidualBlock

def train_model(
    model,
    device,
    n_epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    save_checkpoint: bool = True,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
):
    '''
    Resize: shorter side randomly sampled in [256, 480]
    Crop: 224 x 224 crop is randomly sampled
    Flip: Horizontal flip
    Standardize: per-pixel mean subtracted, standardize color augmentation

    Initialize:
    mini-batch size: 256
    lr: 0.1, /10 when error plateaus
    iterations: 60 * 10^4 it
    weight decay: 1e-4
    momentum: 0.9
    dropout: no
    '''
    
    # 1. Load data
    classes_dict, train_dataloader, val_dataloader = load_data(dir_data='data', batch_size=batch_size, num_workers=0, val_percent=0.2)
    


def get_args():
    parser = argparse.ArgumentParser(description='Train ResNet')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', dest='lr', metavar='LR', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from .pth file')
    parser.add_argument('--classes', '-c', dest='classes', metavar='C', type=int, default=2, help='Number of classes')
    return parser.parse_args()


if __name__ == "__main__":
    print('Training...')
    args = get_args()

    n_blocks_list = [2, 2, 2, 2]
    model = ResNet(residual_block=ResidualBlock, 
                   n_blocks_list=n_blocks_list,
                   n_classes=args.classes)
    # logging.info(f'Model trained on {args.classes} classes\n'
    #              f'Number of blocks per conv:\n'
    #              f'\t{n_blocks_list[0]} for conv2_x\n'
    #              f'\t{n_blocks_list[1]} for conv3_x\n'
    #              f'\t{n_blocks_list[2]} for conv4_x\n'
    #              f'\t{n_blocks_list[3]} for conv5_x\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Using device: {device}')
    model.to(device=device)
    try:
        train_model(model,
                    device,
                    n_epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr)
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training.')
        torch.cuda.empty_cache()
        print("Not enough memory to train model")
    
