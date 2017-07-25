#!/bin/bash
n=$1
export CUDA_VISIBLE_DEVICES=-1
python train_dist.py --job_name "ps" --task_index $n &

for i in {0..3}
do
	export CUDA_VISIBLE_DEVICES=$i
	t=$(($n * 4 + $i))
	python train_dist.py --job_name "worker" --task_index $t &
done
