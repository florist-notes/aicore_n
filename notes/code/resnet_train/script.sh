#!/bin/bash

prefetch_factor=2
no_workers=(4)
external_device="nvme0n1"
batch_size=16

rm mn_nw*
rm log_file_*

#cpu_cores_values=(2 4 6 8 10 12)
#cpu_frq_values=(268800 422400 576000 729600 883200 1036800 1190400 1344000 1497600 1651200 1804800 1958400 2112000 2201600)
#gpu_frq_values=(114750000 216750000 318750000 420750000 522750000 624750000 726750000 828750000 930750000 1032750000 1134750000 1236750000 1300500000)
#mem_frq_values=(204000000 2133000000 3199000000 665600000)

#cpu completed 2201 2112 1958 
#stablization time = 10s from 1958
cpu_cores_values=(2 4 6 8 10 12)
cpu_frq_values=(1804800)
gpu_frq_values=(1300500000 1236750000 1134750000 1032750000 930750000 828750000 726750000 624750000 522750000 420750000 318750000 216750000 114750000)
mem_frq_values=(204000000 2133000000 3199000000 665600000)
#python3 dataload_resnet.py

for cpu_frq in "${cpu_frq_values[@]}"; do
    for gpu_frq in "${gpu_frq_values[@]}"; do
        for cpu_cores in "${cpu_cores_values[@]}"; do
            for mem_frq in "${mem_frq_values[@]}"; do
                python3 generate_nvpmodel.py $cpu_cores $cpu_frq $gpu_frq $mem_frq
                sudo nvpmodel -m 14 &> log_file_pm_14
                sudo jetson_clocks --fan
                sudo jetson_clocks --show &>> log_file_pm_14
                python3 resnet18.py "/media/ssd/" $no_workers $prefetch_factor $external_device > train_log_file_pm_14
                sudo pkill python

                mkdir "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
                mv mn_nw* "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
                mv log_file_* "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
                mv train_log_file_* "pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
            done
        done
    done
done
