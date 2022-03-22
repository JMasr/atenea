#!/bin/bash

# Copyright     2013  Daniel Povey
# Apache 2.0.

# This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=20 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
posterior_scale=1.0 # This scale helps to control for successve features being highly
                    # correlated.  E.g. try 0.1 or 0.3.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 9 ]; then
  echo "Usage: $0 <extractor-dir> <data> <ivector-dir> <file-list>"
  echo " e.g.: $0 exp/extractor_2048_male data/train_male exp/ivectors_male"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters|10>                          # Number of iterations of E-M"
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --num-threads <n|8>                              # Number of threads for each process"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <min-post|0.025>                      # Pruning threshold for posteriors"
  exit 1;
fi

data=$1
dir=$2
listaFicheros=$3
listaVAD=$4
inputUBM=$5
UBMtype=$6
ivExtractor=$7
outdir=$8
filename=$9

# Set various variables.
mkdir -p $dir/log
sdata=$data/split$nj;

for n in `seq $nj`; do
echo "mkdir -p $data/split$nj/$n"
   mkdir -p $data/split$nj/$n
   feats="$feats $data/split$nj/$n/feats.scp"
done

split_scp.pl $utt2spk_opt ${listaFicheros} $feats || exit 1

for n in `seq $nj`; do
   vads="$vads $data/split$nj/$n/vad.scp"
done

split_scp.pl $utt2spk_opt ${listaVAD} $vads || exit 1

## Set up features.
feats="ark:add-deltas scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp:$sdata/JOB/vad.scp ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"
#  ubm="gmm-global-to-fgmm $dubm -|"
if [ "$UBMtype" == "full" ];then
ubm=$inputUBM
dubm=$dir/final.dubm
fgmm-global-to-gmm $ubm $dubm
elif [ "$UBMtype" == "diag" ];then
dubm=$inputUBM
ubm=$dir/final.ubm
gmm-global-to-fgmm $dubm $ubm
else
echo "Error"
exit -1
fi
#  $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
#    gmm-gselect --n=$num_gselect "$dubm" "$feats" ark:- \| \
#    fgmm-global-gselect-to-post --min-post=$min_post $ubm "$feats" \
#       ark,s,cs:- ark:- \| scale-post ark:- $posterior_scale ark:- \| \
#    ivector-extract --verbose=2 $ivExtractor "$feats" ark,s,cs:- \
#      ark,scp,t:$dir/ivector.JOB.ark,$dir/ivector.JOB.scp || exit 1;
  $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
    gmm-gselect --n=$num_gselect "$dubm" "$feats" ark:- \| \
    fgmm-global-gselect-to-post --min-post=$min_post $ubm "$feats" \
       ark:- ark:- \| scale-post ark:- $posterior_scale ark:- \| \
    ivector-extract --verbose=2 $ivExtractor "$feats" ark:- \
      ark,scp,t:$outdir/ivector.JOB.ark,$outdir/ivector.JOB.scp || exit 1;

fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $outdir/ivector.$j.scp; done >$outdir/ivector.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  # Be careful here: the speaker-level iVectors are now length-normalized,
  # even if they are otherwise the same as the utterance-level ones.
  echo "$0: computing mean of iVectors for each speaker and length-normalizing"
#  $cmd $dir/log/speaker_mean.log \
#    ivector-normalize-length scp:$dir/ivector.scp  ark:- \| \
#    ivector-mean ark:$data/spk2utt ark:- ark:- ark,t:$dir/num_utts.ark \| \
#    ivector-normalize-length ark:- ark,scp:$dir/spk_ivector.ark,$dir/spk_ivector.scp || exit 1;
  $cmd $dir/log/speaker_mean.log \
    ivector-normalize-length scp:$outdir/ivector.scp ark,t,scp:$outdir/$filename.ark,$outdir/$filename.scp
fi
