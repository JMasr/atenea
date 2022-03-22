#!/bin/bash
. ./cmd.sh
. ./path.sh

# Directory with the xvector model 
voxcelebdir=./voxceleb/v2/
nnet_dir=${voxcelebdir}0007_voxceleb_v2_1a/exp/xvector_nnet_1a

modelsname=0007_voxceleb_v2_1a

execClustering=true
stage=1

datafilesDir=$1
fileId=$2
th1CW=$3  # Chinese Whispers Threshold (default value 13.5 ) 
window=$4 # Windows size for X-vector  (default value  5.0s) 
period=$5 # Shifting windows period    (default value  0.5s)
outDir=$6

mkdir -p $outDir
mfccdir=${outDir}/mfcc
vaddir=${outDir}/mfcc


# Prepare features
if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset. 30 MFCCs. Window=25ms, frame-shift=10ms
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config ${voxcelebdir}/conf/mfcc.conf --nj 1 --cmd "$train_cmd" \
    ${datafilesDir} ${outDir}/make_mfcc $mfccdir
  utils/fix_data_dir.sh ${datafilesDir}
  local/sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
    ${datafilesDir} ${outDir}/make_vad $vaddir
  utils/fix_data_dir.sh ${datafilesDir}

  # This writes features to disk after applying the sliding window CMN.
  # Although this is somewhat wasteful in terms of disk space, for diarization
  # it ends up being preferable to performing the CMN in memory.  If the CMN
  # were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.
  local/nnet3/xvector/prepare_feats.sh --nj 1 --cmd "$train_cmd" \
    ${datafilesDir} ${datafilesDir}_cmn ${datafilesDir}_cmn
  cp ${datafilesDir}/vad.scp ${datafilesDir}_cmn/
  if [ -f ${datafilesDir}/segments ]; then
    cp ${datafilesDir}/segments ${datafilesDir}_cmn/
  fi
  utils/fix_data_dir.sh ${datafilesDir}_cmn
  echo "0.01" > ${datafilesDir}_cmn/frame_shift

fi

if [ $stage -le 2 ]; then
  # Extract x-vectors used in the evaluation.
  local/diarization/nnet3/xvector/extract_xvectors_laura.sh --cmd "$train_cmd" --nj 1 \
    --window $window --period $period --apply-cmn false --min-segment 1.5 $nnet_dir \
    ${datafilesDir}_cmn ${outDir}/${modelsname}/xvector_nnet_1a/xvectors_subseg_w${window}s${period}ms1.5_segm_cmn_${fileId}
fi


if [ $stage -le 3 ]; then
  if [ $execClustering ]; then

  ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:${outDir}/${modelsname}/xvector_nnet_1a/xvectors_subseg_w${window}s${period}ms1.5_segm_cmn_${fileId}/xvector.scp ark:-  | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:${outDir}/${modelsname}/xvector_nnet_1a/xvectors_subseg_w${window}s${period}ms1.5_segm_cmn_${fileId}/xvector_lda_normlen.ark
  mkdir -p ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm
  rm ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/*
  python3 exec_chinesewhispers_clustering_speechdiar.py ${outDir}/${modelsname}/xvector_nnet_1a/xvectors_subseg_w${window}s${period}ms1.5_segm_cmn_${fileId}/xvector_lda_normlen.ark ${th1CW} ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm

  fi

fi


# Get RTTM files
if [ $stage -le 4 ]; then
  if [ $execClustering ]; then

  for file in ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/xvector_lda_normlen_clustCW_th*.txt; do
    f=$(basename $file)
    f1=${f%%.*}
    f2=${f#*.}
    #threshold=${f1##*_}.${f2%%_*}
    threshold=${f1##*_}.${f2%%.*}
    awk '{print $5,$1,$2}' $file | sed 's/ID /ID/g' >  ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/clustersId_${threshold}.txt
    join -j 1 -o 1.1,1.2,1.3,1.4 ${outDir}/${modelsname}/xvector_nnet_1a/xvectors_subseg_w${window}s${period}ms1.5_segm_cmn_${fileId}/segments ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/clustersId_${threshold}.txt > ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/segments
    local/diarization/make_rttm.py ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/segments ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/clustersId_${threshold}.txt ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/rttm_${threshold}

  done

  fi

fi


