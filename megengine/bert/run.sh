#!/bin/bash
python train.py --do_lower_case --max_seq_length 128 --data_dir ./glue_data/MRPC --pretrained_bert uncased_L-12_H-768_A-12 --learning_rate 2e-5 --num_train_epochs 2
