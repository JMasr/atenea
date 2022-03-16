#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# begin configuration section.
cmd=run.pl
stage=0
decode_mbr=true
beam=10
word_ins_penalty=0.0
min_lmwt=7
max_lmwt=22
model=

#end configuration section.

#debugging stuff
echo $0 $@

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 [options] <dataDir> <langDir|graphDir> <acoustic-model-dir> <decodeDir>" && exit;
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1)                 # (createCTM | filterCTM )."
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
modeldir=$3
dir=$4

if [ -z "$model" ] ; then
  model=${modeldir}/final.mdl # Relative path does not work in some cases
  #model=$dir/../final.mdl # assume model one level up from decoding dir.
  #[ ! -f $model ] && model=`(set +P; cd $dir/../; pwd)`/final.mdl
fi


name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log


wip=0.0
if [ $stage -le 0 ]; then
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/get_ctm.LMWT.log \
      mkdir -p $dir/score_LMWT/wip${wip}/  '&&' \
      lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
      lattice-to-ctm-conf --decode-mbr=$decode_mbr ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt  \| tee $dir/score_LMWT/wip${wip}/${name}.utt.ctm 
fi


