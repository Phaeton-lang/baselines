#!/bin/bash
for i in `seq 1 20`;
do
    echo ${i}
    ./a.out 2>&1 | tee round-${i}.log
done

for i in `seq 1 20`;
do
    cat round-${i}.log >> swapinout-ns.txt
done
