#!/usr/bin/env bash

# Copyright      2017  Hossein Hadian
#                2017  Chun Chieh Chang
#                2017  Ashish Arora

# This script prepares the dictionary.

set -e
dir=data/local/dict
lm_dir=data/local/lm
. ./utils/parse_options.sh || exit 1;

base_dir=$(echo "$DIRECTORY" | cut -d "/" -f2)

mkdir -p $dir

paste -d' ' \
    <(sed 's/^\([^[:space:]]*\)[[:space:]]*\(.*\)/\1/' $lm_dir/librispeech-vocab.txt) \
    <(sed 's/^\([^[:space:]]*\)[[:space:]]*\(.*\)/\1/' $lm_dir/librispeech-vocab.txt | \
        sed 's/\(.\)/\1 /g') \
    > $dir/lexicon.txt

cut -d' ' -f2- $dir/lexicon.txt | sed 's/SIL//g' | tr ' ' '\n' | sort -u | sed '/^$/d' >$dir/nonsilence_phones.txt || exit 1;

echo '<sil> SIL' >> $dir/lexicon.txt
echo '<SPOKEN_NOISE> SPN' >> $dir/lexicon.txt
echo '<UNK> SPN' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt
echo SPN >> $dir/silence_phones.txt

echo SIL > $dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
