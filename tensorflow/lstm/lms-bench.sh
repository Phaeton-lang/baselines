#!/bin/bash

#python lstm.py --lms 256 32 10

net=lstm
SRC_FILE=lstm.py
data_scale_list=(
32
#224
#300
)
batch_list=(
#1
#2
#4
#8
#16
#24
256
512
864
#1024
#2048
#3072
#4096
#6144
#6656
)
for ds in ${data_scale_list[@]};
do
    for bs in ${batch_list[@]};
    do
        python lstm.py --lms ${bs} ${ds} 10 2>&1 | tee lms-${net}-${ds}-${bs}.log
    done
done
