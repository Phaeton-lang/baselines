#!/bin/bash

#python gan.py --no-lms 256 32 20

net=gan
data_scale_list=(
#32
224
#300
)
batch_list=(
#1
#2
#4
8
16
24
#256
#512
#864
#1024
#2048
#4096
#6144
#6656
)
for ds in ${data_scale_list[@]};
do
    for bs in ${batch_list[@]};
    do
        python gan.py --no-lms ${bs} ${ds} 20 2>&1 | tee nolms-${net}-${ds}-${bs}.log
    done
done
