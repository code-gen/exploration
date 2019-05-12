#!/bin/bash

base_dir=../embeddings

python emb-fine-tune/test-fine-tune.py \
    -vocab_file ${base_dir}/pydoc-vocab-20000-window-5.vocab \
    -pt_emb_file ${base_dir}/glove.840B.300d.txt.pickle \
    -ft_emb_file ${base_dir}/pydoc-glove-fine-tuned-vocab-20000-window-5-iter-5000 \
    -pt_factor 0.5 \
    -ft_factor 0.5
