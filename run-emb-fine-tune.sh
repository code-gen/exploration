#!/bin/bash

base_dir=$(dirname "$(readlink -f $0)")

python emb_fine_tune/glove-fine-tune.py \
    -root_dir ${base_dir}/../embeddings \
    -data_source ${base_dir}/../corpus/python-stackoverflow/question_words_clean.pickle \
    -name python-so-in-emb-6B.50d \
    -num_ft_iter 2000 \
    -vocab_size 10000 \
    -window_size 7 \
    -pt_emb_file ${base_dir}/../embeddings/glove.6B.50d.txt
