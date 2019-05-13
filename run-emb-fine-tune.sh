#!/bin/bash

base_dir=$(dirname "$(readlink -f $0)")

python emb_fine_tune/glove-fine-tune.py \
    -root_dir ${base_dir}/../embeddings \
    -data_source ${base_dir}/../corpus/simple \
    -name simple \
    -num_ft_iter 5000 \
    -vocab_size 10 \
    -window_size 3 \
    -pt_emb_file ${base_dir}/../embeddings/glove.840B.300d.txt
