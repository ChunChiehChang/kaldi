#!/bin/bash


# aug_1a is as 1a but with data augmentation
# accuracy 79.5% (1a had accuracy 69%)

# steps/info/nnet3_dir_info.pl exp/cnn_aug_1a_cifar10
# exp/cnn_aug_1a_cifar10: num-iters=60 nj=1..2 num-params=0.2M dim=96->10 combine=-0.61->-0.58 loglike:train/valid[39,59,final]=(-0.60,-0.49,-0.57/-0.68,-0.60,-0.67) accuracy:train/valid[39,59,final]=(0.79,0.83,0.81/0.76,0.79,0.77)

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail



# training options
stage=0
train_stage=-10
dataset=imagenet2012_t3
srand=0
reporting_email=
affix=_aug_1a


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi



dir=exp/cnn${affix}_${dataset}

egs=exp/egs

if [ ! -d $egs ]; then
  echo "$0: expected directory $egs to exist.  Run the get_egs.sh commands in the"
  echo "    run.sh before this script."
  exit 1
fi

# check that the expected files are in the egs directory.

for f in $egs/egs.1.ark $egs/train_diagnostic.egs $egs/valid_diagnostic.egs $egs/combine.egs \
         $egs/info/feat_dim $egs/info/left_context $egs/info/right_context \
         $egs/info/output_dim; do
  if [ ! -e $f ]; then
    echo "$0: expected file $f to exist."
    exit 1;
  fi
done


mkdir -p $dir/log


if [ $stage -le 1 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(cat $egs/info/output_dim)

  # Note: we hardcode in the CNN config that we are dealing with 32x3x color
  # images.

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=672 name=input
  conv-relu-batchnorm-layer name=cnn1 height-in=224 height-out=224 time-offsets=-3,-2,-1,0,1,2,3 required-time-offsets=0 height-offsets=-3,-2,-1,0,1,2,3 num-filters-out=96
  conv-relu-batchnorm-layer name=cnn2 height-in=224 height-out=112 time-offsets=-3,-2,-1,0,1,2,3 required-time-offsets=0 height-offsets=-3,-2,-1,0,1,2,3 num-filters-out=96 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn3 height-in=112 height-out=112 time-offsets=-2,-1,0,1,2 required-time-offsets=0 height-offsets=-2,-1,0,1,2 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn4 height-in=112 height-out=56  time-offsets=-2,-1,0,1,2 required-time-offsets=0 height-offsets=-2,-1,0,1,2 num-filters-out=256 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn5 height-in=56 height-out=56   time-offsets=-1,0,1 required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=512
  conv-relu-batchnorm-layer name=cnn6 height-in=56 height-out=28   time-offsets=-1,0,1 required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=512 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn7 height-in=28 height-out=28   time-offsets=-1,0,1 required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=512
  relu-batchnorm-layer name=fully_connected1 input=Append(0,16,32,48,64,80,96,112,128,144,160,176,192,208) dim=256
  relu-batchnorm-layer name=fully_connected2 dim=512
  output-layer name=output dim=$num_targets
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 2 ]; then

  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$cmd" \
    --image.augmentation-opts="--horizontal-flip-prob=0.5 --num-channels=3 \
                               --horizontal-shift=0.1 --vertical-shift=0.1 \
                               --crop=true --crop-size=224 --crop-scale-min=256 --crop-scale-max=256" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=30 \
    --egs.frames-per-eg=1 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=5 \
    --trainer.optimization.initial-effective-lrate=0.0003 \
    --trainer.optimization.final-effective-lrate=0.00003 \
    --trainer.optimization.minibatch-size=8,4 \
    --compute-prob-minibatch-size="1:16" \
    --trainer.shuffle-buffer-size=2000 \
    --egs.dir="$egs" \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi


exit 0;
