import cv2
import torchvision.datasets as datasets

class CustomDataset(Dataset):
    def __init__(self, csv_file, dir_images, transform=None):
        self.csv_data = pd.read_csv(csv_file)  # Expects two columns: filename, label
        self.dir_images = dir_images
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        # Get image path and label from the CSV file
        file_name = self.csv_data.iloc[idx, 0]
        label = self.csv_data.iloc[idx, 1]
        img_path = os.path.join(self.dir_images, file_name)
        image = cv2.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


def load_data(dir_data, batch_size, num_workers):
    '''
    Load data from directory. Expects the following structure:
      dir_data/
          train/
          test/
          Training_set.csv   (columns: filename, label class)
          Testing_set.csv    (columns: filename, label class)
    '''
    train_dir = os.path.join(dir_data, 'train')
    test_dir = os.path.join(dir_data, 'test')
    train_csv = os.path.join(dir_data, 'Training_set.csv')
    test_csv = os.path.join(dir_data, 'Testing_set.csv')
    
    # Define default transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dataset = CustomDataset(train_csv, train_dir, transform=transform)
    test_dataset = CustomDataset(test_csv, test_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, test_loader
