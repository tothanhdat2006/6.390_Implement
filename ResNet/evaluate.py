import torch
from torchvision import datasets, transforms

import torch.nn as nn

def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            label_pred = model(image)
            loss = criterion(label_pred, label)
            losses.append(loss.item())
            _, predicted = torch.max(label_pred, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    avg_loss = sum(losses) / len(losses)
    accuracy = correct / total
    return avg_loss, accuracy
