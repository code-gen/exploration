#!/bin/bash

base_dir=../embeddings
exp_name=python-so-in-emb

python emb_fine_tune/test-fine-tune.py \
    -pt_emb_file ${base_dir}/glove.6B.50d.txt \
    -vocab_file ${base_dir}/${exp_name}/*.vocab \
    -ft_emb_file ${base_dir}/${exp_name}/*.emb \
    -pt_factor 0.0 \
    -ft_factor 1.0
