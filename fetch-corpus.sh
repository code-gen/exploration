#!/bin/bash

root_dir=$1

get_python_docs() {
    aria2c -s16 -x16 https://docs.python.org/3/archives/python-3.7.3-docs-text.zip -d ./ --continue=true
    unzip ./python-3.7.3-docs-text.zip
    rm -f ./python-3.7.3-docs-text.zip
}


pushd . && mkdir -p ${root_dir} && cd ${root_dir}

for d in "${@:2}"; do echo "Getting $d" && get_${d} && echo; done

popd
