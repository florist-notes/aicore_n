import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import yaml
import glob
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

time1 = time.time()

def pad_labels(labels, max_num_objects):
    padded_labels = F.pad(labels, (0, 0, 0, max_num_objects - labels.size(0)))
    return padded_labels

def custom_collate(batch):
    images, labels = zip(*batch)
    max_num_objects = max(label.size(0) for label in labels)
    padded_labels = [pad_labels(label, max_num_objects) for label in labels]
    return torch.stack(images, 0), torch.stack(padded_labels, 0)

class YOLODataset(Dataset):
    def __init__(self, config_file, transform=None, img_size=(416, 416)):
        with open(config_file, 'r') as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.root_dir = cfg['path']
        self.img_dir = os.path.join(self.root_dir, cfg['train'])  # assuming training data
        self.label_dir = os.path.join(self.root_dir, 'labels/train2017')  # assuming labels are here
        self.img_files = glob.glob(os.path.join(self.img_dir, '*.jpg'))  # Assuming images are in .jpg format
        self.transform = transforms.Compose([
            transforms.Resize(img_size),  # This line resizes images
            transforms.ToTensor(),  # This line converts PIL Images to PyTorch tensors
            transform
        ]) if transform is not None else transforms.Compose([
            transforms.Resize(img_size),  # This line resizes images
            transforms.ToTensor()  # This line converts PIL Images to PyTorch tensors
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)  # This line applies the transform to the image
        with open(label_path, 'r') as file:
            labels = file.read().strip().split('\n')
        labels = [list(map(float, label.split())) for label in labels]  # Convert string labels to float
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels


def load_data(config_file, batch_size_train, batch_size_infer):
    dataset = YOLODataset(config_file)
    train_size = int(0.8 * len(dataset))
    infer_size = len(dataset) - train_size
    train_dataset, infer_dataset = random_split(dataset, [train_size, infer_size])    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4, collate_fn=custom_collate)
    infer_loader = DataLoader(infer_dataset, batch_size=batch_size_infer, shuffle=False, num_workers=4, collate_fn=custom_collate)
    return train_loader, infer_loader

config_file = 'coco25.yaml'  # Path to your .yaml configuration file
train_loader, infer_loader = load_data(config_file, 4, 2)

for batch_idx, (images, labels) in enumerate(infer_loader):
    pass

res = os.popen("free -h")
print(res.read())

print("Time taken :",time.time() - time1)
