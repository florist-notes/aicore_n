import torch
from PIL import Image
from torchvision.datasets import ImageNet,ImageFolder
import time
from torchvision import transforms
import multiprocessing
import logging
import sys
import time
import csv
import pandas as pd
import os
import numpy as np

os.system('sudo bash ./drop_caches.sh')

time_start = time.time()

data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

infer_data = ImageFolder(root="/media/ssd/ILSVRC/val",
                          transform=data_transforms)

infer_loader = torch.utils.data.DataLoader(
        infer_data,
        batch_size=16,
        num_workers=4
    )

for batch_idx, (images, labels) in enumerate(infer_loader):
    pass


res = os.popen("free -h")
print(res.read())

print("Time taken: ", time.time() - time_start)
