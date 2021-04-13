#!/bin/bash
# Plz download bert
if [ ! -f ${dir_name} ]; then
    echo 'Plz download bert-base first!!!'
    echo 'Plz download it from <https://github.com/google-research/bert/blob/master/README.md>'
    echo 'Download "12/768 (BERT-Base)" from <https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip>'
    echo 'Plz prepare BERT-Base with the following cmds!'
    #bert_model_name="uncased_L-12_H-768_A-12"
    #dir_name=".model"
    #mkdir -p ${dir_name} ${dir_name}/${bert_model_name}
    #wget -c https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip -P ${dir_name}/${bert_model_name}
    #unzip ${dir_name}/${bert_model_name}/uncased_L-12_H-768_A-12.zip
    #rm -f ${dir_name}/${bert_model_name}/uncased_L-12_H-768_A-12.zip
fi
python gpu_movie_reviews.py
