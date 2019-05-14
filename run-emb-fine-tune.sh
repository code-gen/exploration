#!/bin/bash

base_dir=$(dirname "$(readlink -f $0)")

python emb_fine_tune/glove-fine-tune.py \
    -root_dir ${base_dir}/../embeddings \
    -data_source ${base_dir}/../corpus/python-stackoverflow/question_words_clean.pickle \
    -exp_name python-so \
    -pt_emb_file ${base_dir}/../embeddings/glove.6B.50d.txt \
    -num_ft_iter 2000 \
    -vocab_size 10000 \
    -window_size 7 \
    -min_freq 100 \
    -mu 0.1 \
    -only_in_emb
