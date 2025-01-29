import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval() # Set to evaluation mode
    test_loss = 0.0

    # with torch.no_grad():

    net.train() # Set back to training mode
    return test_loss