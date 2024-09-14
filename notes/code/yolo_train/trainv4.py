from ultralytics import YOLO
import time
import pandas as pd
import os
import logging
formatter = logging.Formatter('%(message)s')
from jtop import jtop
import sys
import csv
import multiprocessing

os.environ['WANDB_DISABLED'] = 'true'

# Initialize batch count variable
batch_count = 0

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


def train(batch_size,dataset_outer_folder, file_prefix):

    path = dataset_outer_folder+"coco25.yaml"
    # Initialize an empty list for each column
    start_times = []
    stop_times = []



    # Custom exception to stop training
    class StopTrainingException(Exception):
        pass

    def on_train_batch_start(trainer):
        start_times.append(time.time())

    def on_train_batch_end(trainer):
        global batch_count
        stop_times.append(time.time())
        batch_count += 1
        
        # Check if we've reached 50 batches
        if batch_count > 50:
            raise StopTrainingException("Training completed after 10 batches")

    model = YOLO('yolov8n.pt')
    model.add_callback("on_train_batch_start", on_train_batch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)

    try:
        model.train(data=path, epochs=1, imgsz=320, batch=batch_size, workers=0, val=False, save=False, amp=False, device=0, pretrained=False)
    except StopTrainingException as e:
        print(f"Training stopped: {e}")

    # Create a DataFrame using the lists and calculate Minibatch_time
    df = pd.DataFrame({'Start_time': start_times, 'Stop_time': stop_times})
    df['ignore1'] = 0
    df['ignore2'] = 0
    df['epochtime_ms'] = (df['Stop_time'] - df['Start_time'])*1000
    df['log_time'] = df['Stop_time'] - reference_time
    print(df)
    df.to_csv(file_prefix+"_epoch_stats.csv")

def main(dataset_outer_folder, num_workers, prefetch_factor, reference_time, batch_size, file_prefix):
    print('Reference Time is', str(reference_time))
    # print('No. of testing batches = ' + str(len(train_loader)))
    print('No of workers is ' + str(num_workers))
    print('Prefetch factor is ' + str(prefetch_factor))
    print('Batch size is ', str(batch_size))
    train(batch_size, dataset_outer_folder, file_prefix)

if __name__ == "__main__":
    dataset_outer_folder = sys.argv[1]
    num_workers = int(sys.argv[2])
    prefetch_factor = int(sys.argv[3])
    external_device = sys.argv[4]
    batch_size = 16
    pass

    file_prefix = 'mn_'+'nw' + str(num_workers) + '_pf'+str(prefetch_factor)

    # logger_fetch = setup_logger("logger_fetch", file_prefix + "_fetch.csv")

    # logger_compute = setup_logger(
    #     "logger_compute", file_prefix + "_compute.csv")
    # logger_e2e = setup_logger("logger_e2e", file_prefix + "_epoch_stats.csv")
    # logger_vmtouch = setup_logger(
    #     "logger_vmtouch", file_prefix+"_vmtouch_stats.csv")

    logger_iostats = setup_logger(
        "logger_iostats", file_prefix+"_io_stats.csv")
    logger_memstats = setup_logger(
        "logger_memstats", file_prefix+"_mem_stats.csv")
    logger_swapstats = setup_logger(
        "logger_swapstats", file_prefix+"_swap_stats.csv")

    # logger_fetch.info('epoch,batch_idx,fetchtime,fetchtime_ms,log_time')
    # logger_compute.info('epoch,batch_idx,computetime,computetime_ms,log_time')
    # logger_e2e.info('epoch,time,loss,accuracy,epochtime_ms,log_time')
    # logger_vmtouch.info(
    #     'files,directories,resident_pages,resident_pages_size,resident_pages_%,elapsed,redundant,log_time')

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
        main(dataset_outer_folder, num_workers, prefetch_factor, reference_time, batch_size, file_prefix)
    except ():
        print("hit an exception")
        print()
        p2.terminate()

    p2.terminate()
