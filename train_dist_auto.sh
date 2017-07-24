#!/bin/bash
n=$1
export CUDA_VISIBLE_DEVICES=-1
python train_dist.py --job_name "ps" --task_index $n &

for i in {0..7}
do
	export CUDA_VISIBLE_DEVICES=$i
	t=$(($n * 8 + $i))
	python train_dist.py --job_name "worker" --task_index $t &
done
