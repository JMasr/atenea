#!/bin/bash
. ./cmd.sh
. ./path.sh

# Directory with the xvector model
voxcelebdir=./voxceleb/v2/0007_voxceleb_v2_1a
nnet_dir=${voxcelebdir}/exp/xvector_nnet_1a

modelsname=0007_voxceleb_v2_1a

execClustering=true
stage=0

audio_file=$1
rttm_file=$2
th1CW=$3 # Chinese Whispers Threshold
window=$4
period=$5
outDir=$6

mkdir -p $outDir
mfccdir=${outDir}/mfcc
vaddir=${outDir}/mfcc


audio_dir=$(dirname $1)
audio_fname=$(basename $1)
fileId=${audio_fname%.*}


# Prepare data dir
if [ $stage -le 0 ]; then
  audio_dir=$(dirname $1)
  audio_fname=$(basename $1)
  fileId=${audio_fname%.*}

  mkdir -p ${outDir}/data/${fileId}_segm
  echo $fileId $audio_fname | awk -v dir=$audio_dir '{printf "%s %s%s/%s%s\n", $1,"sox ",dir,$2," -G -r 16000 -e signed-integer -b 16 -c 1 -t wav - |"}' > ${outDir}/data/${fileId}_segm/wav.scp
  totalDur=`soxi -D $1`

  # Get the speaker segmentation, and the segments, from the RTTM file
  ./rttmSort.pl ${rttm_file} | grep SPEAKER | awk -v var="$fileId" '{utt=sprintf("spk%05d-%s-%07d-%07d",NR,var,$4*100,($4+$5)*100); recId=sprintf("spk%05d-%s",NR,var); print utt,recId}' > ${outDir}/data/${fileId}_segm/utt2spk
  ./rttmSort.pl ${rttm_file} | grep SPEAKER | awk -v var="$fileId" '{utt=sprintf("spk%05d-%s-%07d-%07d",NR,var,$4*100,($4+$5)*100); recId=sprintf("%s",var); print utt,recId,$4,$4+$5;}' > ${outDir}/data/${fileId}_segm/segments

  utils/utt2spk_to_spk2utt.pl ${outDir}/data/${fileId}_segm/utt2spk > ${outDir}/data/${fileId}_segm/spk2utt
  utils/data/get_utt2dur.sh ${outDir}/data/${fileId}_segm
  utils/fix_data_dir.sh ${outDir}/data/${fileId}_segm

fi

# Prepare features
if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset. 30 MFCCs. Window=25ms, frame-shift=10ms
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config ${voxcelebdir}/../conf/mfcc.conf --nj 1 --cmd "$train_cmd" \
    ${outDir}/data/${fileId}_segm ${outDir}/make_mfcc $mfccdir
  utils/fix_data_dir.sh ${outDir}/data/${fileId}_segm
  local/sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
    ${outDir}/data/${fileId}_segm ${outDir}/make_vad $vaddir
  utils/fix_data_dir.sh ${outDir}/data/${fileId}_segm

  # This writes features to disk after applying the sliding window CMN.
  # Although this is somewhat wasteful in terms of disk space, for diarization
  # it ends up being preferable to performing the CMN in memory.  If the CMN
  # were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.
  local/nnet3/xvector/prepare_feats.sh --nj 1 --cmd "$train_cmd" \
    ${outDir}/data/${fileId}_segm ${outDir}/data/${fileId}_segm_cmn ${outDir}/${fileId}_segm_cmn
  cp ${outDir}/data/${fileId}_segm/vad.scp ${outDir}/data/${fileId}_segm_cmn/
  if [ -f ${outDir}/data/${fileId}_segm/segments ]; then
    cp ${outDir}/data/${fileId}_segm/segments ${outDir}/data/${fileId}_segm_cmn/
  fi
  utils/fix_data_dir.sh ${outDir}/data/${fileId}_segm_cmn
  echo "0.01" > ${outDir}/data/${fileId}_segm_cmn/frame_shift

fi



if [ $stage -le 2 ]; then

  # Extract x-vectors used in the evaluation.
  local/diarization/nnet3/xvector/extract_xvectors_laura.sh --cmd "$train_cmd" --nj 1 \
    --window $window --period $period --apply-cmn false --min-segment 1.5 $nnet_dir \
    ${outDir}/data/${fileId}_segm_cmn ${outDir}/${modelsname}/xvector_nnet_1a/xvectors_subseg_w${window}s${period}ms1.5_segm_cmn_${fileId}

fi

if [ $stage -le 3 ]; then
  if [ $execClustering ]; then

  ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:${outDir}/${modelsname}/xvector_nnet_1a/xvectors_subseg_w${window}s${period}ms1.5_segm_cmn_${fileId}/xvector.scp ark:-  | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:${outDir}/${modelsname}/xvector_nnet_1a/xvectors_subseg_w${window}s${period}ms1.5_segm_cmn_${fileId}/xvector_lda_normlen.ark
  mkdir -p ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm
  rm ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5segm/*
  python3 exec_chinesewhispers_clustering_speechdiar.py ${outDir}/${modelsname}/xvector_nnet_1a/xvectors_subseg_w${window}s${period}ms1.5_segm_cmn_${fileId}/xvector_lda_normlen.ark ${th1CW} ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm

  fi
fi



if [ $stage -le 4 ]; then
  if [ $execClustering ]; then

  for file in ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/xvector_lda_normlen_clustCW_th*.txt; do
    f=$(basename $file)
    f1=${f%%.*}
    f2=${f#*.}
    threshold=${f1##*_}.${f2%%.*}
    awk '{print $5,$1,$2}' $file | sed 's/ID /ID/g' >  ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/clustersId_${threshold}.txt
    join -j 1 -o 1.1,1.2,1.3,1.4 ${outDir}/${modelsname}/xvector_nnet_1a/xvectors_subseg_w${window}s${period}ms1.5_segm_cmn_${fileId}/segments ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/clustersId_${threshold}.txt > ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/segments
    local/diarization/make_rttm.py ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/segments ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/clustersId_${threshold}.txt ${outDir}/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/rttm_${threshold}

  done

  fi

fi


