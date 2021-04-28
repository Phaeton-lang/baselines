#!/bin/bash
python train.py --do_lower_case --max_seq_length 128 --data_dir ./glue_data/MRPC --learning_rate 2e-5 --num_train_epochs 1
