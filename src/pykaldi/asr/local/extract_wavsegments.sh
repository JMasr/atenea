#!/bin/bash

if [ $# -ne 3 ]; then
  echo "Usage: extract_wavsegments.sh <wav file> <segments file> <output-dir>"
  exit 1;
fi

wavfile=$1
segmentsfile=$2
outdir=$3

mkdir -p ${outdir}
rm ${outdir}/segments

audio_dir=$(dirname $wavfile)
audio_name=$(basename $wavfile)
fname=${audio_name%.*}
audio_ext=$(echo ${audio_name#*.} | awk '{print tolower($0)}')

n=1
while read -r line; do

  VADindex=$(printf "%03d" $n)
  UTTID=$(echo $line | awk '{print $1}')
  WAVID=$(echo $line | awk '{print $2}')
  ST=$(echo $line | awk '{print $3}')
  ET=$(echo $line | awk '{print $4}')
  #echo "sox ${wavfile} -c 1 -r 16000 -s -2 ${outdir}/${fname}_${VADindex}.wav trim $ST =$ET"
  sox ${wavfile} -G -r 16000 -e signed-integer -b 16 -c 1 ${outdir}/${fname}_${VADindex}.wav trim ${ST} =${ET}
  
  echo "${fname}_${VADindex} ${fname} ${ST} ${ET}" >> ${outdir}/segments
  n=$((n+1))
done < ${segmentsfile}

