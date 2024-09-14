prefetch_factor=2
no_workers=4
external_device="nvme0n1"
batch_size_list=(16)
batch_size=16

rm mn_nw*
rm log_file_*

for batch_size in ${batch_size_list[@]};
do
	python3 bert_finetune_logging.py "/home/dream-orin3/BERT" $no_workers $prefetch_factor $external_device $batch_size > log_file_$batch_size
	pkill python
	mkdir bs$batch_size
	mv mn_nw* bs$batch_size
	mv log_file_* bs$batch_size
	sleep 3s
done

