#!/bin/bash

# Copyright 2017 Johns Hopkins University (author: Chun-Chieh "Jonathan" Chang)

# This script loads the training and test data for
# Imagenet 2012 Task 1 or 3 Classification

# Currently the script is set to run the one for Task 3
# To change modify the paths to the correct directories

[ -f ./path.sh ] && . ./path.sh;

# Path to imagenet directory
# Requires the datasets for Task 1:
# "development kit", "training images", "validation images", and "test images"
dl_dir=/export/b18/imagenet_2012/
local_dir=data/download/

# Various other paths
devkit_dir=$local_dir/devkit_t3
train_dir=$local_dir/train_t3
train_bbox_dir=$local_dir/train_t3_bbox
val_dir=$local_dir/val
val_bbox_dir=$local_dir/val_bbox
test_dir=$local_dir/test

# Various tar files
devkit_tar=ILSVRC2012_devkit_t3.tar.gz
train_tar=ILSVRC2012_img_train_t3.tar
train_bbox_tar=ILSVRC2012_bbox_train_dogs.tar.gz
val_tar=ILSVRC2012_img_val.tar
val_bbox_tar=ILSVRC2012_bbox_val_v3.tgz
test_tar=ILSVRC2012_img_test.tar

# Extra
# For when running the task 3
# the devkit used for classes.txt still needs to be from task 1

# Check if dataset is downloaded 
if [ ! -d $dl_dir ] || \
     [ ! -f $dl_dir/$devkit_tar ] || \
     [ ! -f $dl_dir/$train_tar ] || \
     [ ! -f $dl_dir/$val_tar ] || \
     [ ! -f $dl_dir/$test_tar ]; then
  echo Need to download ImageNet2012 dataset first. Need tar for devkit train val and test data.
  exit 1
else
  if [ ! -d $devkit_dir ]; then
    mkdir -p $devkit_dir
    tar -xzf $dl_dir/$devkit_tar -C $devkit_dir || exit 1
    # echo Missing devkit
    # exit 1
  fi

  if [ ! -d $train_dir ]; then
    mkdir -p $train_dir
    tar -xf $dl_dir/$train_tar -C $train_dir || exit 1
    find $train_dir -name "*.tar" | \
    while read name; do mkdir -p "${name%.tar}"; tar -xf "${name}" -C "${name%.tar}";done
    # echo Missing train
    # exit 1
  fi

  if [ ! -d $train_bbox_dir ]; then
    mkdir -p $train_bbox_dir
    tar -xzf $dl_dir/$train_bbox_tar -C $train_bbox_dir || exit 1
    # echo Missing train bbox
    # exit 1
  fi

  if [ ! -d $val_dir ]; then
    mkdir -p $val_dir
    tar -xf $dl_dir/$val_tar -C $val_dir || exit 1
    # echo Missing val
    # exit 1
  fi

  if [ ! -d $val_bbox_dir ]; then
    mkdir -p $val_bbox_dir
    tar -xzf $dl_dir/$val_bbox_tar -C $val_bbox_dir || exit 1
    # echo Missing val bbox
    # exit 1
  fi

  if [ ! -d $test_dir ]; then
    mkdir -p $test_dir
    tar -xf $dl_dir/$test_tar -C $test_dir || exit 1
    # echo Missing test
    # exit 1
  fi
fi

mkdir -p data/{train,val,test}/data

# Retrieve all the possible classes from the devkit .mat file
local/process_classes.py $devkit_dir $devkit_tar data --task 3 || exit 1

cp data/classes.txt data/train/classes.txt
cp data/classes.txt data/val/classes.txt
cp data/classes.txt data/test/classes.txt

echo 3 > data/train/num_channels
echo 3 > data/test/num_channels

# Process training data
local/process_data.py $train_dir $train_bbox_dir $devkit_dir $devkit_tar data/train --dataset train | \
  copy-feats --compress=true --compression-method=7 \
  ark:- ark,scp:data/train/data/images.ark,data/train/images.scp || exit 1

# Process testing data
# Using validation data instead because testing data does not include ground truth
local/process_data.py $val_dir $val_bbox_dir $devkit_dir $devkit_tar data/test \
  --dataset test --scale-size 224 --crop-size 144 | \
  copy-feats --compress=true --compression-method=7 \
  ark:- ark,scp:data/test/data/images.ark,data/test/images.scp || exit 1
