from collections import OrderedDict
import sys
# import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#import torchvision.models as models
import numpy as np
import csv
import time
import os
import collections
from jtop import jtop
import pandas as pd
import time
#import wandb
import multiprocessing
import logging

from landmark_dataset import Landmarks

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
formatter = logging.Formatter('%(message)s')
#from torch.profiler import profile, record_function, ProfilerActivity

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def start_logging(filename, iostats_filename, memstats_filename, swapstats_filename, cpufreqstats_filename, gpufreqstats_filename, emcstats_filename, ramstats_filename, external_device, reference_time):
    print("jtop logging started")
    output = pd.DataFrame()
    output_cpufreqstats = pd.DataFrame()
    output_gpufreqstats = pd.DataFrame()
    output_emcstats = pd.DataFrame()
    output_ramstats = pd.DataFrame()
    with jtop() as jetson:
        tegrastats_entry = jetson.stats
        tegrastats_entry['log_time'] = str(time.time() - reference_time)
        output = pd.concat([output,pd.DataFrame([tegrastats_entry])], ignore_index=True)
        output.to_csv(filename, index=False)

        cpufreqstats_entry = jetson.cpu
        cpufreqstats_entry['log_time'] = str(time.time() - reference_time)
        output_cpufreqstats = pd.concat([output_cpufreqstats,pd.DataFrame([cpufreqstats_entry])], ignore_index=True)
        output_cpufreqstats.to_csv(cpufreqstats_filename, index=False)

        gpufreqstats_entry = jetson.gpu
        gpufreqstats_entry['log_time'] = str(time.time() - reference_time)
        output_gpufreqstats = pd.concat([output_gpufreqstats,pd.DataFrame([gpufreqstats_entry])], ignore_index=True)
        output_gpufreqstats.to_csv(gpufreqstats_filename, index=False)

        emcstats_entry = jetson.emc
        emcstats_entry['log_time'] = str(time.time() - reference_time)
        output_emcstats = pd.concat([output_emcstats,pd.DataFrame([emcstats_entry])], ignore_index=True)
        output_emcstats.to_csv(emcstats_filename, index=False)

        ramstats_entry = jetson.ram
        ramstats_entry['log_time'] = str(time.time() - reference_time)
        output_ramstats = pd.concat([output_ramstats,pd.DataFrame([ramstats_entry])], ignore_index=True)
        output_ramstats.to_csv(ramstats_filename, index=False)

    with jtop() as jetson:
        while jetson.ok():
            tegrastats_entry = jetson.stats
            tegrastats_entry['log_time'] = str(time.time() - reference_time)
            output = pd.concat([output, pd.DataFrame([tegrastats_entry])], ignore_index=True)

            cpufreqstats_entry = jetson.cpu
            cpufreqstats_entry['log_time'] = str(time.time() - reference_time)
            output_cpufreqstats = pd.concat([output_cpufreqstats, pd.DataFrame([cpufreqstats_entry])],
                                            ignore_index=True)

            gpufreqstats_entry = jetson.gpu
            gpufreqstats_entry['log_time'] = str(time.time() - reference_time)
            output_gpufreqstats = pd.concat([output_gpufreqstats, pd.DataFrame([gpufreqstats_entry])],
                                            ignore_index=True)

            emcstats_entry = jetson.emc
            emcstats_entry['log_time'] = str(time.time() - reference_time)
            output_emcstats = pd.concat([output_emcstats, pd.DataFrame([emcstats_entry])], ignore_index=True)

            ramstats_entry = jetson.ram
            ramstats_entry['log_time'] = str(time.time() - reference_time)
            output_ramstats = pd.concat([output_ramstats, pd.DataFrame([ramstats_entry])], ignore_index=True)

            io_output = os.popen("iostat -xy 1 1 -d " + external_device +
                                 " | awk 'NR>3{ for (x=2; x<=16; x++) {  printf\"%s \", $x}}' | sed 's/ /,/g'| sed 's/,*$//g'")
            io_output = io_output.read()+","+str(time.time() - reference_time)
            iostats_filename.info(io_output)

            mem_output = os.popen(
                "free -mh | awk 'NR==2{for (x=2;x<=7;x++){printf\"%s \", $x}}' | sed 's/ /,/g'| sed 's/,*$//g'")
            mem_output = mem_output.read()+","+str(time.time() - reference_time)
            memstats_filename.info(mem_output)

            swap_output = os.popen(
                "free -mh | awk 'NR==3{for (x=2;x<=4;x++){printf\"%s \", $x}}' | sed 's/ /,/g'| sed 's/,*$//g'")
            swap_output = swap_output.read()+","+str(time.time() - reference_time)
            swapstats_filename.info(swap_output)

            output.to_csv(filename,index=False, mode='a', header=False)
            output = pd.DataFrame()

            output_cpufreqstats.to_csv(cpufreqstats_filename,index=False, mode='a', header=False)
            output_cpufreqstats = pd.DataFrame()
            
            output_gpufreqstats.to_csv(gpufreqstats_filename,index=False, mode='a', header=False)
            output_gpufreqstats = pd.DataFrame()

            output_emcstats.to_csv(emcstats_filename,index=False, mode='a', header=False)
            output_emcstats = pd.DataFrame()

            output_ramstats.to_csv(ramstats_filename,index=False, mode='a', header=False)
            output_ramstats = pd.DataFrame()

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
    # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
    # IMAGENET_STD = [0.2673, 0.2564, 0.2762]

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
  """Reads a csv file, and returns the content inside a list of dictionaries.
  Args:
    path: The path to the csv file.
  Returns:
    A list of dictionaries. Each row in the csv file will be a list entry. The
    dictionary is keyed by the column names.
  """
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
    #print(num_w, pf)

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
    #logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = len(train_files)

    train_data_global, test_data_global = get_dataloader_Landmarks(data_dir, train_files, test_files, batch_size,batch_size,None, num_w, pf)
    # logging.info("train_dl_global number = " + str(len(train_data_global)))
    # logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_files)
    return train_data_global, test_data_global, class_num

    # logging("data_local_num_dict: %s" % data_local_num_dict)
    

def main(dataset_outer_folder, no_workers, pf_factor, epoch_fname, fetch_fname, compute_fname, vmtouch_fname, reference_time):
    """Create model, load data, define Flower client, start Flower client."""
    
    # Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
    def get_model_parameters(model):
        total_parameters = 0
        for layer in list(model.parameters()):
            layer_parameter = 1
            for l in list(layer.size()):
                layer_parameter *= l
            total_parameters += layer_parameter
        return total_parameters


    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


    class h_sigmoid(nn.Module):
        def __init__(self, inplace=True):
            super(h_sigmoid, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            return F.relu6(x + 3., inplace=self.inplace) / 6.


    class h_swish(nn.Module):
        def __init__(self, inplace=True):
            super(h_swish, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            out = F.relu6(x + 3., self.inplace) / 6.
            return out * x


    def _make_divisible(v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


    class SqueezeBlock(nn.Module):
        def __init__(self, exp_size, divide=4):
            super(SqueezeBlock, self).__init__()
            self.dense = nn.Sequential(
                nn.Linear(exp_size, exp_size // divide),
                nn.ReLU(inplace=True),
                nn.Linear(exp_size // divide, exp_size),
                h_sigmoid()
            )

        def forward(self, x):
            batch, channels, height, width = x.size()
            out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
            out = self.dense(out)
            out = out.view(batch, channels, 1, 1)
            # out = hard_sigmoid(out)

            return out * x


    class MobileBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size):
            super(MobileBlock, self).__init__()
            self.out_channels = out_channels
            self.nonLinear = nonLinear
            self.SE = SE
            padding = (kernal_size - 1) // 2

            self.use_connect = stride == 1 and in_channels == out_channels

            if self.nonLinear == "RE":
                activation = nn.ReLU
            else:
                activation = h_swish

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(exp_size),
                activation(inplace=True)
            )
            self.depth_conv = nn.Sequential(
                nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size),
                nn.BatchNorm2d(exp_size),
            )

            if self.SE:
                self.squeeze_block = SqueezeBlock(exp_size)

            self.point_conv = nn.Sequential(
                nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                activation(inplace=True)
            )

        def forward(self, x):
            # MobileNetV2
            out = self.conv(x)
            out = self.depth_conv(out)

            # Squeeze and Excite
            if self.SE:
                out = self.squeeze_block(out)

            # point-wise conv
            out = self.point_conv(out)

            # connection
            if self.use_connect:
                return x + out
            else:
                return out


    class MobileNetV3(nn.Module):
        def __init__(self, model_mode="LARGE", num_classes=1000, multiplier=1.0, dropout_rate=0.0):
            super(MobileNetV3, self).__init__()
            self.num_classes = num_classes

            if model_mode == "LARGE":
                layers = [
                    [16, 16, 3, 1, "RE", False, 16],
                    [16, 24, 3, 2, "RE", False, 64],
                    [24, 24, 3, 1, "RE", False, 72],
                    [24, 40, 5, 2, "RE", True, 72],
                    [40, 40, 5, 1, "RE", True, 120],

                    [40, 40, 5, 1, "RE", True, 120],
                    [40, 80, 3, 2, "HS", False, 240],
                    [80, 80, 3, 1, "HS", False, 200],
                    [80, 80, 3, 1, "HS", False, 184],
                    [80, 80, 3, 1, "HS", False, 184],

                    [80, 112, 3, 1, "HS", True, 480],
                    [112, 112, 3, 1, "HS", True, 672],
                    [112, 160, 5, 1, "HS", True, 672],
                    [160, 160, 5, 2, "HS", True, 672],
                    [160, 160, 5, 1, "HS", True, 960],
                ]
                init_conv_out = _make_divisible(16 * multiplier)
                self.init_conv = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(init_conv_out),
                    h_swish(inplace=True),
                )

                self.block = []
                for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
                    in_channels = _make_divisible(in_channels * multiplier)
                    out_channels = _make_divisible(out_channels * multiplier)
                    exp_size = _make_divisible(exp_size * multiplier)
                    self.block.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
                self.block = nn.Sequential(*self.block)

                out_conv1_in = _make_divisible(160 * multiplier)
                out_conv1_out = _make_divisible(960 * multiplier)
                self.out_conv1 = nn.Sequential(
                    nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1),
                    nn.BatchNorm2d(out_conv1_out),
                    h_swish(inplace=True),
                )

                out_conv2_in = _make_divisible(960 * multiplier)
                out_conv2_out = _make_divisible(1280 * multiplier)
                self.out_conv2 = nn.Sequential(
                    nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1),
                    h_swish(inplace=True),
                    nn.Dropout(dropout_rate),
                    nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1),
                )

            elif model_mode == "SMALL":
                layers = [
                    [16, 16, 3, 2, "RE", True, 16],
                    [16, 24, 3, 2, "RE", False, 72],
                    [24, 24, 3, 1, "RE", False, 88],
                    [24, 40, 5, 2, "RE", True, 96],
                    [40, 40, 5, 1, "RE", True, 240],
                    [40, 40, 5, 1, "RE", True, 240],
                    [40, 48, 5, 1, "HS", True, 120],
                    [48, 48, 5, 1, "HS", True, 144],
                    [48, 96, 5, 2, "HS", True, 288],
                    [96, 96, 5, 1, "HS", True, 576],
                    [96, 96, 5, 1, "HS", True, 576],
                ]

                init_conv_out = _make_divisible(16 * multiplier)
                self.init_conv = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(init_conv_out),
                    h_swish(inplace=True),
                )

                self.block = []
                for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
                    in_channels = _make_divisible(in_channels * multiplier)
                    out_channels = _make_divisible(out_channels * multiplier)
                    exp_size = _make_divisible(exp_size * multiplier)
                    self.block.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
                self.block = nn.Sequential(*self.block)

                out_conv1_in = _make_divisible(96 * multiplier)
                out_conv1_out = _make_divisible(576 * multiplier)
                self.out_conv1 = nn.Sequential(
                    nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1),
                    SqueezeBlock(out_conv1_out),
                    nn.BatchNorm2d(out_conv1_out),
                    h_swish(inplace=True),
                )

                out_conv2_in = _make_divisible(576 * multiplier)
                out_conv2_out = _make_divisible(1280 * multiplier)
                self.out_conv2 = nn.Sequential(
                    nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1),
                    h_swish(inplace=True),
                    nn.Dropout(dropout_rate),
                    nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1),
                )

            self.apply(_weights_init)

        def forward(self, x):
            out = self.init_conv(x)
            out = self.block(out)
            out = self.out_conv1(out)
            batch, channels, height, width = out.size()
            out = F.avg_pool2d(out, kernel_size=[height, width])
            out = self.out_conv2(out).view(batch, -1)
            return out


    # temp = torch.zeros((1, 3, 224, 224))
    # model = MobileNetV3(model_mode="LARGE", num_classes=1000, multiplier=1.0)
    # print(model(temp).shape)
    # print(get_model_parameters(model))


    #net = models.mobilenet_v3_large(pretrained=False)
    #net.train(mode=True)
    
    trainloader, testloader, class_num = load_data(dataset_outer_folder, no_workers, pf_factor)


    net = MobileNetV3(model_mode='LARGE',num_classes=class_num)
    temp = torch.zeros((1, 3, 224, 224))
    print('Shape = '+str(net(temp).shape))
    print('No. of model parameters = ' + str(get_model_parameters(net)))
    print('No. of training batches = ' + str(len(trainloader)))
    print('No. of testing batches = ' + str(len(testloader)))
    print('No of workers is ' +str(no_workers))
    print('Prefetch factor is ' + str(pf_factor))
    net = net.to(DEVICE)
    train(net, trainloader,testloader, epoch_fname, fetch_fname, compute_fname, vmtouch_fname, dataset_outer_folder, reference_time, epochs=2)


def train(net, trainloader,testloader, epoch_fname, fetch_fname, compute_fname, vmtouch_fname, dataset_outer_folder, reference_time, epochs):
    print('Training')
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    vmtouch_dir = os.path.join(dataset_outer_folder,'landmark_dataset/images/train')
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start3 = torch.cuda.Event(enable_timing=True)
    end3 = torch.cuda.Event(enable_timing=True)
    #start_time = time.time()
    epoch_count = 0

    vmtouch_output = os.popen(r"vmtouch -f " + vmtouch_dir + r" | sed 's/^.*://' | sed -z 's/\n/,/g' | sed 's/\s\+/,/g' | sed 's/,\{2,\}/,/g' | sed -e 's/^.\(.*\).$/\1/'")
    vmtouch_output = vmtouch_output.read()+","+str(time.time() - reference_time)
    vmtouch_fname.info(vmtouch_output)
    # with open(vmtouchlogtime_fname, 'a', newline='') as file:
    #     writer = csv.writer(file, delimiter=' ',escapechar=' ', quoting=csv.QUOTE_NONE)
    #     writer.writerow([str(time.time() - reference_time)])
    os.system('sudo bash ./drop_caches.sh')
    img_idx = batch_size - 1
    e2e_first_batch = 0
    dataloader_start_time = time.time()
    stabilization_time = 30
    for _ in range(1):       
      
        #start1.record()
        vmtouch_output = os.popen(r"vmtouch -f " + vmtouch_dir + r" | sed 's/^.*://' | sed -z 's/\n/,/g' | sed 's/\s\+/,/g' | sed 's/,\{2,\}/,/g' | sed -e 's/^.\(.*\).$/\1/'")
        vmtouch_output = vmtouch_output.read()+","+str(time.time() - reference_time)
        vmtouch_fname.info(vmtouch_output)

        print('Epoch: ' + str(_) + ' Begins')
        # start_time = time.time() # DEPRECATED
        start1.record() # e2e epoch time starts
        # start2_time = time.time() # DEPRECATED
        start2.record() # fetch time per batch starts
        start_time = time.time()
        print("Start time :", start_time)
        batch_count = 0        
        for batch_idx, (images, labels) in enumerate(trainloader):
            if time.time() - start_time > stabilization_time:
                batch_count += 1
            if(batch_count == 50):
                break
            print('Batch index = '+ str(batch_idx))
            images, labels = images.to(DEVICE), labels.to(DEVICE) # won't be part of fetch + preprocess
            # end2_time = time.time() - start2_time # DEPRECATED
            end2.record() # fetch time per batch ends
            start3.record() # compute time per batch starts
            # start_time1 = time.time() # DEPRECATED
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            end3.record()  # compute time per batch ends
            torch.cuda.synchronize()
            end1.record()
            torch.cuda.synchronize()
            fetch_fname.info(str(_) + "," + str(batch_idx) + "," + "null" + "," + str(
                start2.elapsed_time(end2)) + "," + str(time.time() - reference_time))
            compute_fname.info(str(_) + "," + str(batch_idx) + "," + "null" + "," + str(
                start3.elapsed_time(end3)) + "," + str(time.time() - reference_time))
            epoch_fname.info(str(_) + "," + "null" + "," + "\"" + ":" + "\"" + "," + "null" +
                "," + str(start1.elapsed_time(end1)) + "," + str(time.time() - reference_time))

            start1.record()
            start2.record()  # fetch time per batch starts
            img_idx += batch_size
        print('Epoch: ' + str(_) + ' Ends')
        epoch_count += 1

    vmtouch_output = os.popen(r"vmtouch -f " + vmtouch_dir + r" | sed 's/^.*://' | sed -z 's/\n/,/g' | sed 's/\s\+/,/g' | sed 's/,\{2,\}/,/g' | sed -e 's/^.\(.*\).$/\1/'")
    vmtouch_output = vmtouch_output.read()+","+str(time.time() - reference_time)
    vmtouch_fname.info(vmtouch_output)

        
def test(net, testloader):
    """Validate the network on the entire test se."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def load_data(global_dir, num_w, pf_factor):

    print(global_dir)
    """
    Below paths need to be changed as per the requirement.
    """
    # fed_train_map_file = os.path.join(global_dir, 'data_dict/data_user_dict/gld23k_user_dict_train_4x.csv')
    fed_train_map_file = os.path.join(global_dir, 'data_dict/data_user_dict/gld23k_user_dict_train.csv')
    fed_test_map_file = os.path.join(global_dir, 'data_dict/data_user_dict/gld23k_user_dict_test.csv')
    # data_dir = os.path.join(global_dir,'landmark_dataset/images_4x/')
    data_dir = os.path.join(global_dir,'landmark_dataset/images/') # change line 547 accordingly if you change this line

    #train_data_num, test_data_num, train_data_global, test_data_global, \
    #train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    trainloader, testloader, class_num = load_partition_data_landmarks('Landmarks', data_dir,fed_train_map_file,fed_test_map_file,None, None,1, 16, num_w, pf_factor)
    return trainloader, testloader, class_num

if __name__ == "__main__":

    dataset_outer_folder=sys.argv[1]
    num_workers=int(sys.argv[2])
    prefetch_factor=int(sys.argv[3])
    external_device = sys.argv[4]
    file_prefix='mn_'+'nw'+ str(num_workers) + '_pf'+str(prefetch_factor)
    batch_size = 16
    
    logger_fetch = setup_logger("logger_fetch", file_prefix + "_fetch.csv")
    logger_compute = setup_logger("logger_compute", file_prefix + "_compute.csv")
    logger_e2e = setup_logger("logger_e2e", file_prefix + "_epoch_stats.csv")
    logger_vmtouch = setup_logger("logger_vmtouch", file_prefix+"_vmtouch_stats.csv")
    
    logger_iostats = setup_logger("logger_iostats", file_prefix+"_io_stats.csv")
    logger_memstats = setup_logger("logger_memstats", file_prefix+"_mem_stats.csv")
    logger_swapstats = setup_logger("logger_swapstats", file_prefix+"_swap_stats.csv")
    
    logger_fetch.info('epoch,batch_idx,fetchtime,fetchtime_ms,log_time')
    logger_compute.info('epoch,batch_idx,computetime,computetime_ms,log_time')
    logger_e2e.info('epoch,time,loss,accuracy,epochtime_ms,log_time')
    logger_vmtouch.info('files,directories,resident_pages,resident_pages_size,resident_pages_%,elapsed,redundant,log_time')
    
    logger_iostats.info('r/s,w/s,rkB/s,wkB/s,rrqm/s,wrqm/s,%rrqm,%wrqm,r_await,w_await,aqu-sz,rareq-sz,wareq-sz,svctm,%util,log_time')
    logger_memstats.info('total,used,free,shared,buff/cache,available,log_time')
    logger_swapstats.info('total,used,free,log_time')

    reference_time = time.time()
    p2 = multiprocessing.Process(target=start_logging, args=[file_prefix+'_tegrastats.csv', logger_iostats, logger_memstats, logger_swapstats, file_prefix+'_cpufreq_stats.csv', file_prefix+'_gpufreq_stats.csv', file_prefix+'_emc_stats.csv', file_prefix+'_ram_stats.csv', external_device, reference_time])
    p2.start()
    try:
        main(dataset_outer_folder, num_workers, prefetch_factor, logger_e2e, logger_fetch, logger_compute, logger_vmtouch, reference_time)
    except():
        print("hit an exception")
        print(e)
        p2.terminate()

    p2.terminate()
    
