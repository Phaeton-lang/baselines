#!/bin/bash
batch_list=(
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
)
net=inception
for batch in ${batch_list[@]};
do
    python ManyModel.py --no-lms --model ${net} --epochs 4 --image_size 299 --batch_size ${batch} 2>&1 | tee nolms-turing-${net}-batch-${batch}-throughput.log
done
