#!/bin/bash

if [ -f path.sh ]; then . ./path.sh; fi

fichero=$1
fileId=$2
fileRttm=$3
fileOut=$4

dirModelos=./modelos_SAD/
dOut=$(dirname $fileRttm)

mkdir -p ${dOut}

  dirTmp=${dOut}/tmp/
  mkdir -p ${dirTmp}
  rm -r ${dirTmp}/*
  #Feature extraction
  echo "Feature extraction..."
  echo "${fileId} ${fichero}" > ${dirTmp}featureExtraction.scp

  #Obtengo el fichero segments a partir del RTTM
  cat ${fileRttm} | awk -v file=$fileId '{printf "%s_%.5i %s %f %f\n",file,NR,file,$4,$4+$5}' > ${dirTmp}segments
  cat ${fileRttm} | awk '{printf "%f %f %s\n",$4,$5,$8}' > ${dirTmp}segments_SS.lab
  awk '{printf "%s SPK_%.5i\n", $1,NR}' ${dirTmp}segments >${dirTmp}utt2spk
  steps/segmentation/convert_utt2spk_and_segments_to_rttm.py ${dirTmp}utt2spk ${dirTmp}segments ${dirTmp}segments.rttm

  #Audio segmentation
  plp_feats="ark:extract-segments scp:${dirTmp}featureExtraction.scp  ${dirTmp}segments ark:- | compute-plp-feats --verbose=2 --config=conf/plp.conf ark:- ark:- |"
  pitch_feats="ark:extract-segments scp:${dirTmp}featureExtraction.scp  ${dirTmp}segments ark:- | compute-kaldi-pitch-feats --verbose=2 --config=conf/pitch.conf ark:- ark:- | process-kaldi-pitch-feats ark:- ark:- |"
  paste-feats --length-tolerance=2 "$plp_feats" "$pitch_feats" ark:- | copy-feats --compress=true ark:- ark,scp:${dirTmp}plp_pitch.ark,${dirTmp}plp_pitch.scp

  compute-vad --config=conf/vad1.conf scp:${dirTmp}plp_pitch.scp ark,scp:${dirTmp}vad.ark,${dirTmp}vad.scp

  ./local/extract_ivectors_vad.sh --cmd "run.pl" --nj 1 ${dirTmp} ${dirTmp} ${dirTmp}plp_pitch.scp ${dirTmp}vad.scp ${dirModelos}UBM512_plp_pitch full ${dirModelos}IVextractor_plp_pitch ${dirTmp} ivectors_plp_pitch

  logistic-regression-eval ${dirModelos}logistic_regression_rebalanced_5classes "ark:ivector-normalize-length scp:${dirTmp}ivectors_plp_pitch.scp ark:- |" ark,t:${dirTmp}class_posteriors
  cat ${dirTmp}class_posteriors | cut -d " " -f 4-8 > ${dirTmp}probs
  ./local/eliminaMusica.pl ${dirTmp}segments_SS.lab ${dirTmp}probs ${fileOut}.lab
  cat ${fileOut}.lab | awk -v file=$fileId '{printf "SPEAKER %s 1 %f %f <NA> <NA> %s <NA>\n",file,$1,$2,$3}' > ${fileOut}.rttm

  cat ${dirTmp}class_posteriors | \
  awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max)
                          { max=$f; argmax=f; }}
                          print $1, (argmax - 3); }' | \
  utils/int2sym.pl -f 2 ${dirModelos}/classes5.txt \
    >${dirTmp}${fileId}.scores5classes

  ./local/processOuputIVectors.pl ${dirTmp}segments ${dirTmp}${fileId}.scores5classes ${fileOut}_AS.lab

