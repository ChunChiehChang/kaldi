#!/bin/bash

decomposition_table=data/local/dict/cj5-cc.txt
lang_dir=data/lang
tree_dir=exp/e2e_monotree

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

pdf-to-phone $tree_dir/tree $lang_dir/topo $tree_dir/phones.txt > $tree_dir/pdf2phone
local/create_decomposition_matrix.py $tree_dir/pdf2phone $decomposition_table $tree_dir | \
    copy-matrix - $tree_dir/decomp.mat
