#!/bin/bash

path1=$(pwd)
cd $path1/pykaldi/drz

. ./path.sh
. ./cmd.sh

stage=0

fichero=$1
outDir=$2
th1aCW=$3 # 1st step Chinese Whispers Threshold
th1bCW=$4 # 2nd step Chinese Whispers Threshold
window=$5 # Windows Size
period=$6 # Shifting windows period

fileId=$(basename $fichero '.wav')
fileDir=$(dirname $fichero)

mkdir -p ${outDir}

echo '1st Stage INIT'
if [ $stage -le 0 ]; then

  mkdir -p ${outDir}/data/${fileId}
  echo $fileId ${fileDir}/${fileId}.wav > ${outDir}/data/${fileId}/wav.scp
  awk -v name=$fileId '{printf "%s SPK_%s\n", $1,name}' ${outDir}/data/${fileId}/wav.scp > ${outDir}/data/${fileId}/utt2spk
  utils/utt2spk_to_spk2utt.pl < ${outDir}/data/${fileId}/utt2spk > ${outDir}/data/${fileId}/spk2utt
  cat ${outDir}/data/${fileId}/wav.scp | awk '{print $1,$1,"1"}' > ${outDir}/data/${fileId}/reco2file_and_channel
  read_entire_file=true
  num_utts=$(wc -l <${outDir}/data/${fileId}/utt2spk)
  wav-to-duration --read-entire-file=$read_entire_file scp:${outDir}/data/${fileId}/wav.scp ark,t:${outDir}/data/${fileId}/wav2dur
  paste -d' ' ${outDir}/data/${fileId}/utt2spk ${outDir}/data/${fileId}/wav2dur | awk '{ print $1, $3, 0, $4 }' > ${outDir}/data/${fileId}/segments
  paste -d' ' ${outDir}/data/${fileId}/utt2spk ${outDir}/data/${fileId}/wav2dur | awk '{ print $1, $4 }' > ${outDir}/data/${dataDir}/utt2dur
  utils/fix_data_dir.sh ${outDir}/data/${fileId}

  ./run_clustering_allvideo_CW.sh ${outDir}/data/${fileId} ${fileId} ${th1aCW} ${window} ${period} ${outDir}/ClusterSlWinSegm
fi

echo '1st Stage Done!'

echo '2nd Stage: Speaker Diarization.'
if [ $stage -le 1 ]; then
  ## Con el resultado de la segmentacion temporal obtenida tras aplicar Chinese Whispers con ventana deslizante, vuelvo a realizar segmentacion y clustering.
  ## Tomo primero la configuracion Chinese Whispers 01
  rttmFile=${outDir}/ClusterSlWinSegm/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/rttm_th${th1aCW}
  mkdir -p ${outDir}/ClusterSlWinSADSegm
  cp $rttmFile ${outDir}/ClusterSlWinSADSegm/${fileId}_segmentacion1.rttm
  rttmFile=${outDir}/ClusterSlWinSADSegm/${fileId}_segmentacion1.rttm
  ./run_clustering_CW.sh ${fileDir}/${fileId}.wav $rttmFile ${th1bCW} ${window} ${period} ${outDir}/ClusterSlWinSADSegm

fi
echo '2nd Stage Done!'

echo '3rd Stage: Remove Music using the SAD based on ivectors.'
if [ $stage -le 2 ]; then
  rttmFile=${outDir}/ClusterSlWinSADSegm/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/rttm_th${th1bCW}
  ./postprocSegmentsRttm.pl ${rttmFile} ${rttmFile}.temp 
  ./run_eliminaMusica_1file.sh ${fichero} ${fileId} ${rttmFile}.temp ${outDir}/ClusterSlWinSADSegm/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/${fileId}_SD_${th1bCW}

fi
echo '3rd Stage Done!'

cp ${outDir}/ClusterSlWinSADSegm/resultsClustering_CW/${fileId}/subseg_w${window}s${period}ms1.5_segm/${fileId}_SD_${th1bCW}.rttm ${outDir}/${fileId}_drz.rttm
# rm -r ${outDir}/data/ ${outDir}/ClusterSlWinSADSegm/ ${outDir}/ClusterSlWinSegm/