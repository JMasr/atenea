#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey) 2013.  Apache 2.0.

# This script segments speech data based on some kind of decoding of
# whole recordings (e.g. whole conversation sides.  See 
# egs/swbd/s5b/local/run_resegment.sh for an example of usage.
# You'll probably want to use the script resegment_text.sh


# begin configuration section.
stage=0
cmd=run.pl
cleanup=true
segmentation_opts=  # E.g. set this as --segmentation-opts "--silence-proportion 0.2 --max-segment-length 10"

#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 6 ]; then
  echo "Usage: $0 [options] <in-data-dir> <lang> <decode-dir|ali-dir> <model-dir> <out-data-dir> <temp/log-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --segmentation-opts '--opt1 opt1val --opt2 opt2val' # options for segmentation.pl"
  echo "e.g.:"
  echo "$0 data/train_unseg exp/tri3b/decode_train_unseg data/train_seg exp/tri3b_resegment"
  exit 1;
fi

data=$1
lang=$2
alidir=$3 # may actually be decode-dir.
modeldir=$4
data_out=$5
dir=$6

mkdir -p $data_out || exit 1;
rm $data_out/* 2>/dev/null # Old stuff that's partial can cause problems later if
                           # we call fix_data_dir.sh; it will cause things to be 
                           # thrown out.
mkdir -p $dir/log || exit 1;

for f in $data/feats.scp $lang/phones.txt $alidir/ali.1.gz $alidir/num_jobs; do
  if [ ! -f $f ]; then 
    echo "$0: no such file $f"
    exit 1;
    fi
done

if [ -f $modeldir/final.mdl ]; then
  model=$modeldir/final.mdl
else
  if [ ! -f $modeldir/../final.mdl ]; then
    echo "$0: found no model in $modeldir/final.mdl or $modeldir/../final.mdl"
    exit 1;
  fi
  model=$modeldir/../final.mdl
fi

# get lists of sil,noise,nonsil phones
# convert *.ali.gz to *.ali.gz with 0,1,2.
# run perl script..
# output segments?


if ! [ `cat $lang/phones/optional_silence.txt | wc -w` -eq 1 ]; then
  echo "Error: this script only works if $lang/phones/optional_silence.txt contains exactly one entry.";
  echo "You'd have to modify the script to handle other cases."
  exit 1;
fi

silphone=`cat $lang/phones/optional_silence.txt` 
# silphone will typically be "sil" or "SIL". 

# 3 sets of phones: 0 is silence, 1 is noise, 2 is speech.,
(
 echo "$silphone 0"
 grep -v -w $silphone $lang/phones/silence.txt | awk '{print $1, 1;}'
 cat $lang/phones/nonsilence.txt | awk '{print $1, 2;}'
) > $dir/phone_map.txt


nj=`cat $alidir/num_jobs` || exit 1;

if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/resegment.JOB.log \
    ali-to-phones --per-frame=true "$model" "ark:gunzip -c $alidir/ali.JOB.gz|" ark,t:- \| \
    utils/int2sym.pl -f 2- $lang/phones.txt \| \
    utils/apply_map.pl -f 2- $dir/phone_map.txt \| \
    utils/segmentation.pl $segmentation_opts \| \
    gzip -c '>' $dir/segments.JOB.gz
fi

if [ $stage -le 1 ]; then
  if [ -f $data/reco2file_and_channel ]; then
    cp $data/reco2file_and_channel $data_out/reco2file_and_channel
  fi
  if [ -f $data/wav.scp ]; then
    cp $data/wav.scp $data_out/wav.scp
  else
    echo "Expected file $data/wav.scp to exist" # or there is really nothing to copy.
    exit 1
  fi
  for f in glm stm; do 
    if [ -f $data/$f ]; then
      cp $data/$f $data_out/$f
    fi
  done

  for n in `seq $nj`; do gunzip -c $dir/segments.$n.gz; done | \
    local/convert_segments.pl $data/segments | sed 's/_[0-9][0-9][0-9][0-9]-/_/' | sort > $data_out/segments || exit 1;

  [ ! -s $data_out/segments ] && echo "No data produced" && exit 1;

  cat $data/utt2spk | sed 's/^\([^ ]*\) \([^ ]*\)/s\/ \1$\/ \2\//' > $data_out/sed
  cat $data_out/segments | awk '{print $1, $2}' | sed -f $data_out/sed > $data_out/utt2spk || exit 1
  utils/utt2spk_to_spk2utt.pl $data_out/utt2spk > $data_out/spk2utt || exit 1
  rm -f $data/sed

  if $cleanup; then
    rm $dir/segments.*.gz
  fi
fi

cat $data_out/segments | awk '{num_secs += $4 - $3;} END{print "Number of hours of data is " (num_secs/3600);}'

