#!/bin/bash

stage=0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh
fi

if [ $stage -le 1 ]; then
  image/validate_image_dir.sh data/train --is-square-image false
  image/validate_image_dir.sh data/test
fi

if [ $stage -le 2 ]; then
  image/nnet3/get_egs.sh --egs-per-archive 1000 --train-subset-egs 600 --cmd "$cmd" \
    --crop true --crop-size 144 \
    --crop-scale-min 188 \
    --crop-scale-max 256 \
    data/train data/test exp/egs
fi

