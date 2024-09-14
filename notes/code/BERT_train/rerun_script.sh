#!/bin/bash

prefetch_factor=2
no_workers=(4)
external_device="nvme0n1"
batch_size=16

rm mn_nw*
rm log_file_*

# Read parameter combinations from CSV and store in an array
IFS=$'\n' combinations=($(awk -F"," 'NR > 1 {print $1,$2,$3,$4}' /home/dream-orin3/BERT/rerun.csv))

# Iterate over each combination and execute commands
for combo in "${combinations[@]}"; do
    IFS=' ' read -ra params <<< "$combo"
    cpu_cores="${params[0]}"
    cpu_frq="${params[1]}"
    gpu_frq="${params[2]}"
    mem_frq="${params[3]}"

    python3 generate_nvpmodel.py $cpu_cores $cpu_frq $gpu_frq $mem_frq
    sudo nvpmodel -m 14 &> log_file_pm_14
    sudo jetson_clocks --fan
    sudo jetson_clocks --show &>> log_file_pm_14
    python3 bert_finetune_logging.py "/home/dream-orin3/BERT" $no_workers $prefetch_factor $external_device $batch_size > log_file_$batch_size
    sudo pkill python

    mkdir "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
    mv mn_nw* "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
    mv log_file_* "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
    mv train_log_file_* "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
done