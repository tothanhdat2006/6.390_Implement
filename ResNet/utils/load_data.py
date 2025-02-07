import os

import cv2
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, class_csv, dir_images, transform=None):
        self.csv_data = pd.read_csv(class_csv)  # Expects two columns: filename, label
        self.dir_images = dir_images
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        # Get image path and label from the CSV file
        file_name = self.csv_data.iloc[idx, 0]
        label = self.csv_data.iloc[idx, 1]
        img_path = os.path.join(self.dir_images, file_name)
        image = cv2.imread(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def load_data(dir_data, batch_size, num_workers):
    '''
    Load data from directory. Expects the following structure:
      dir_data/
          train/
          class.csv   (columns: filename, label class)
    '''
    train_dir = os.path.join(dir_data)

    # Define default transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ])
    
    dataset = CustomDataset(train_dir, class_csv='class.csv', transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return dataloader
