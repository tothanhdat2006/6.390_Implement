import os

from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dir_images: str, filepaths: list, labels: list, transform=None):
        self.dir_images = dir_images
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        # Get image path and label from the CSV file
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        image = Image.open(filepath).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def encode_labels(labels):
    """
    Encode categorical labels into numeric values.

    Args:
        labels (list): List of categorical labels.
    Returns:
        tuple: A tuple (label_to_index, numeric_labels) where:
            - label_to_index (dict): Mapping from label to numeric value.
            - numeric_labels (list): List of numeric labels corresponding to the input labels.
    """
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_index[label] for label in labels]
    return label_to_index, numeric_labels


def load_data(dir_data: str, val_percent: int = 0.2, batch_size: int = 1, num_workers: int = 0) -> dict | DataLoader | DataLoader:
    '''
    Load data from directory. Expects the following structure:
      dir_data/
          train/
          class.csv   (columns: filename, label class)

    Args:
        dir_data (str): Path to the data directory.
        val_percent (int): Percentage of validation data.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
    Returns:
        dict: Mapping from label to index.
        DataLoader: DataLoader for training data.
        DataLoader: DataLoader for validation data.
    '''

    # Define paths
    dir_csv = os.path.join(dir_data, 'class.csv')
    dir_train = os.path.join(dir_data, 'train')

    # Read class names from class.csv
    class_csv = pd.read_csv(dir_csv)
    filepaths = []
    labels = []
    for i in range(len(class_csv)):
        filepaths.append(os.path.join(dir_train, class_csv['filename'][i]))
        labels.append(class_csv['label'][i])

    labels_dict, labels_encoded = encode_labels(labels)

    # Split data into training set and validation set
    X_train, X_val, y_train, y_val = train_test_split(filepaths, labels_encoded, test_size=val_percent, random_state=86)

    # Define transformation for image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(dir_train, X_train, y_train, transform=transform)
    val_dataset = CustomDataset(dir_train, X_val, y_val, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return labels_dict, train_dataloader, val_dataloader
