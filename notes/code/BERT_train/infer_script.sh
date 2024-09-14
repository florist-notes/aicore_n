prefetch_factor=2
no_workers=4
external_device="nvme0n1"
#batch_size_list=(32)
batch_size_list=(4)
#batch_size_list=(1 4 16 32 64 256)
#batch_size_list=(32 64 256)


rm mn_nw*
rm log_file_*

# cpu_cores_values=(4 8 12)
# cpu_frq_values=(422400 729600 1036800 1344000 1651200 1958400 2201600)
# gpu_frq_values=(114750000 318750000 522750000 726750000 930750000 1134750000 1300500000)
# mem_frq_values=(2133000000 3199000000 665600000)


cpu_cores_values=(4 8 12)
cpu_frq_values=(422400)
gpu_frq_values=(114750000 318750000 522750000 726750000 930750000 1134750000 1300500000)
mem_frq_values=(2133000000 3199000000 665600000)


for batch_size in ${batch_size_list[@]}; do
	mkdir -p "bs_${batch_size}"
		for cpu_frq in "${cpu_frq_values[@]}"; do
	        for gpu_frq in "${gpu_frq_values[@]}"; do
    	        for cpu_cores in "${cpu_cores_values[@]}"; do
        	        for mem_frq in "${mem_frq_values[@]}"; do
						python3 generate_nvpmodel.py $cpu_cores $cpu_frq $gpu_frq $mem_frq
                    	sudo nvpmodel -m 14 &> log_file_pm_14
                    	sudo jetson_clocks --fan
                    	sudo jetson_clocks --show &>> log_file_pm_14
                    	python3 inferLogNew.py "/home/dream-orin3/BERT" $no_workers $prefetch_factor $external_device $batch_size > log_file_$batch_size
                    	sudo pkill python

                    folder_name="pm_${cpu_cores}_${cpu_frq}_${gpu_frq}_${mem_frq}"
					mkdir $folder_name                    
                    mv $folder_name "bs_${batch_size}/$folder_name"
                    mv log_file_* "bs_${batch_size}/$folder_name"
                    mv train_log_file_* "bs_${batch_size}/$folder_name"
                    mv mn_nw* "bs_${batch_size}/$folder_name"
                done
            done
        done
    done
done

