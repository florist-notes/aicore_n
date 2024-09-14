import torch
import collections
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

from landmark_dataset import Landmarks
from torch.utils.data import DataLoader

os.system('sudo bash ./drop_caches.sh')
outer_folder = "/media/ssd/data"
time1 = time.time()
class Cutout(object):
    def __init__(self, length):
        self.length = length
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_landmarks():
    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]
    image_size = 224
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_transform.transforms.append(Cutout(16))
    valid_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, valid_transform


def _read_csv(path):
  with open(path, 'r') as f:
    return list(csv.DictReader(f))
  
def get_mapping_per_user(fn):
    mapping_table = _read_csv(fn)
    expected_cols = ['user_id', 'image_id', 'class']
    if not all(col in mapping_table[0].keys() for col in expected_cols):
        logger.error('%s has wrong format.', mapping_file)
        raise ValueError(
            'The mapping file must contain user_id, image_id and class columns. '
            'The existing columns are %s' % ','.join(mapping_table[0].keys()))
    data_local_num_dict = dict()
    mapping_per_user = collections.defaultdict(list)
    data_files = []
    net_dataidx_map = {}
    sum_temp = 0
    for row in mapping_table:
        user_id = row['user_id']
        mapping_per_user[user_id].append(row)
    for user_id, data in mapping_per_user.items():
        num_local = len(mapping_per_user[user_id])
        # net_dataidx_map[user_id]= (sum_temp, sum_temp+num_local)
        # data_local_num_dict[user_id] = num_local
        net_dataidx_map[int(user_id)]= (sum_temp, sum_temp+num_local)
        data_local_num_dict[int(user_id)] = num_local
        sum_temp += num_local
        data_files += mapping_per_user[user_id]
    assert sum_temp == len(data_files)
    return data_files, data_local_num_dict, net_dataidx_map


def get_dataloader_Landmarks(datadir, train_files, test_files, train_bs, test_bs, dataidxs=None, num_w=0, pf=2):
    dl_obj = Landmarks
    transform_train, transform_test = _data_transforms_landmarks()
    train_ds = dl_obj(datadir, train_files, 0, dataidxs=None, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, test_files, 1, dataidxs=None, train=False, transform=transform_test, download=True)
    train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False, num_workers=num_w, prefetch_factor=pf)
    test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, num_workers=num_w, prefetch_factor=pf)
    return train_dl, test_dl

def load_partition_data_landmarks(dataset, data_dir, fed_train_map_file, fed_test_map_file, partition_method=None, partition_alpha=None, client_number=1, batch_size=16, num_w=0, pf=2):
    train_files, data_local_num_dict, net_dataidx_map = get_mapping_per_user(fed_train_map_file)
    test_files = _read_csv(fed_test_map_file)
    class_num = len(np.unique([item['class'] for item in train_files]))
    train_data_num = len(train_files)
    train_data_global, test_data_global = get_dataloader_Landmarks(data_dir, train_files, test_files, batch_size,batch_size,None, num_w, pf)
    test_data_num = len(test_files)
    return train_data_global, test_data_global, class_num

def load_data(global_dir, num_w, pf_factor):
    fed_train_map_file = os.path.join(global_dir, 'data_dict/data_user_dict/gld23k_user_dict_train.csv')
    fed_test_map_file = os.path.join(global_dir, 'data_dict/data_user_dict/gld23k_user_dict_test.csv')
    data_dir = os.path.join(global_dir,'landmark_dataset/images/') # change line 547 accordingly if you change this line
    trainloader, testloader, class_num = load_partition_data_landmarks('Landmarks', data_dir,fed_train_map_file,fed_test_map_file,None, None,1, 16, num_w, pf_factor)
    return trainloader, testloader, class_num


trainloader, infer_loader, class_num = load_data(outer_folder, 4,2)
for batch_idx, (images, labels) in enumerate(trainloader):
    pass

res = os.popen("free -h")
print(res.read())

print("Time taken: ",time.time()-time1)
