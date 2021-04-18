#!/bin/bash
batch_list=(
1
2
4
8
16
32
64
128
256
512
640
768
896
)
net=vgg16
for batch in ${batch_list[@]};
do
    python ManyModel.py --lms --model ${net} --epochs 5 --image_size 32 --batch_size ${batch} 2>&1 | tee ${net}-batch-${batch}-throughput.log
done
