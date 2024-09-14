import torch
from jtop import jtop
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
from torchvision import models
formatter = logging.Formatter('%(message)s')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#from torchsummary import summary
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

def train(model, train_loader, epoch_fname, fetch_fname, compute_fname, vmtouch_fname, dataset_outer_folder, reference_time, epochs):
    model.train()
 #   print(summary(model, (3,28,28), 16))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    vmtouch_dir = os.path.join(dataset_outer_folder, 'vmtouch_output/')
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start3 = torch.cuda.Event(enable_timing=True)
    end3 = torch.cuda.Event(enable_timing=True)
    epoch_count = 0

    vmtouch_output = os.popen(r"vmtouch -f " + vmtouch_dir +
                              r" | sed 's/^.*://' | sed -z 's/\n/,/g' | sed 's/\s\+/,/g' | sed 's/,\{2,\}/,/g' | sed -e 's/^.\(.*\).$/\1/'")
    vmtouch_output = vmtouch_output.read()+","+str(time.time() - reference_time)
    vmtouch_fname.info(vmtouch_output)
    #os.system('sudo bash ./drop_caches.sh')
    img_idx = batch_size - 1
    e2e_first_batch = 0
    dataloader_start_time = time.time()
    stabilization_time = 30
    for _ in range(1):
        print('Epoch: ' + str(_) + ' Begins')
        start1.record()  # mb epoch time starts
        start2.record()  # fetch time per batch starts
        start_time = time.time()
        print("Start time :", start_time)
        batch_count = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            print("Current time :",time.time())
            if time.time() - start_time > stabilization_time:
                batch_count += 1
            if(batch_count == 50):
                break
            print('Batch index = ' + str(batch_idx))
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            end2.record()  # fetch time per batch ends
            start3.record()  # compute time per batch starts
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
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

    vmtouch_output = os.popen(r"vmtouch -f " + vmtouch_dir +
                              r" | sed 's/^.*://' | sed -z 's/\n/,/g' | sed 's/\s\+/,/g' | sed 's/,\{2,\}/,/g' | sed -e 's/^.\(.*\).$/\1/'")
    vmtouch_output = vmtouch_output.read()+","+str(time.time() - reference_time)
    vmtouch_fname.info(vmtouch_output)

def main(dataset_outer_folder, no_workers, pf_factor, epoch_fname, fetch_fname, compute_fname, vmtouch_fname, reference_time, batch_size):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    model = models.resnet18()
    train_data = ImageFolder(root = dataset_outer_folder + "/ILSVRC/val",
                          transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=no_workers
    )
    print('Reference Time is', str(reference_time))
    print('No. of testing batches = ' + str(len(train_loader)))
    print('No of workers is ' + str(no_workers))
    print('Prefetch factor is ' + str(pf_factor))
    print('Batch size is ', str(batch_size))
    model.to(DEVICE)
    train(model, train_loader, epoch_fname, fetch_fname, compute_fname,
              vmtouch_fname, dataset_outer_folder, reference_time, epochs=3)

if __name__ == "__main__":
    dataset_outer_folder = sys.argv[1]
    num_workers = int(sys.argv[2])
    prefetch_factor = int(sys.argv[3])
    external_device = sys.argv[4]
    batch_size = 16
    file_prefix = 'mn_'+'nw' + str(num_workers) + '_pf'+str(prefetch_factor)

    logger_fetch = setup_logger("logger_fetch", file_prefix + "_fetch.csv")

    logger_compute = setup_logger(
        "logger_compute", file_prefix + "_compute.csv")
    logger_e2e = setup_logger("logger_e2e", file_prefix + "_epoch_stats.csv")
    logger_vmtouch = setup_logger(
        "logger_vmtouch", file_prefix+"_vmtouch_stats.csv")

    logger_iostats = setup_logger(
        "logger_iostats", file_prefix+"_io_stats.csv")
    logger_memstats = setup_logger(
        "logger_memstats", file_prefix+"_mem_stats.csv")
    logger_swapstats = setup_logger(
        "logger_swapstats", file_prefix+"_swap_stats.csv")

    logger_fetch.info('epoch,batch_idx,fetchtime,fetchtime_ms,log_time')
    logger_compute.info('epoch,batch_idx,computetime,computetime_ms,log_time')
    logger_e2e.info('epoch,time,loss,accuracy,epochtime_ms,log_time')
    logger_vmtouch.info(
        'files,directories,resident_pages,resident_pages_size,resident_pages_%,elapsed,redundant,log_time')

    logger_iostats.info(
        'r/s,w/s,rkB/s,wkB/s,rrqm/s,wrqm/s,%rrqm,%wrqm,r_await,w_await,aqu-sz,rareq-sz,wareq-sz,svctm,%util,log_time')
    logger_memstats.info(
        'total,used,free,shared,buff/cache,available,log_time')
    logger_swapstats.info('total,used,free,log_time')

    with open(file_prefix+'_io_stats.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['r/s', 'w/s', 'rkB/s', 'wkB/s', 'rrqm/s', 'wrqm/s', '%rrqm', '%wrqm',
                        'r_await', 'w_await', 'aqu-sz', 'rareq-sz', 'wareq-sz', 'svctm', '%util', 'log_time'])
    with open(file_prefix+'_mem_stats.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total', 'used', 'free', 'shared',
                        'buff/cache', 'available', 'log_time'])
    with open(file_prefix+'_swap_stats.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total', 'used', 'free', 'log_time'])
    reference_time = time.time()
    p2 = multiprocessing.Process(target=start_logging, args=[file_prefix+'_tegrastats.csv', logger_iostats, logger_memstats, logger_swapstats, file_prefix +
                                 '_cpufreq_stats.csv', file_prefix+'_gpufreq_stats.csv', file_prefix+'_emc_stats.csv', file_prefix+'_ram_stats.csv', external_device, reference_time])
    p2.start()
    try:
        main(dataset_outer_folder, num_workers, prefetch_factor, logger_e2e,
             logger_fetch, logger_compute, logger_vmtouch, reference_time, batch_size)
    except ():
        print("hit an exception")
        print()
        p2.terminate()

    p2.terminate()
