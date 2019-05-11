#!/bin/bash

base_dir=$(dirname "$(readlink -f $0)")

python glove-fine-tune.py \
    -root_dir ${base_dir}/data-out/ \
    -data_source ${base_dir}/data-out/python-3.7.3-docs-text/all_files_listing.txt \
    -name pydoc \
    -num_ft_iter 5000 \
    -pt_emb_file ${base_dir}/data-out/glove.840B.300d.txt
