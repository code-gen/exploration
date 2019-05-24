#!/bin/bash

base_dir=../embeddings
exp_name=2019-05-14_18-19-19-python-so-200

python emb_fine_tune/test_fine_tune.py \
    -pt_emb_file ${base_dir}/glove.6B.200d.txt \
    -vocab_file ${base_dir}/${exp_name}/*.vocab \
    -ft_emb_file ${base_dir}/${exp_name}/*.emb \
    -pt_factor 0.05 \
    -ft_factor 0.95 \
    -n 10
