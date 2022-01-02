#!/bin/bash

fileIn=$1
outdir=$2
dirRelative=$3

. ${dirRelative}/conf/path.sh; # source the path.
. ${dirRelative}/conf/cmd.sh; # source train and decode cmds.
. ${dirRelative}/conf/parse_options.sh


mkdir -p ${outdir}

cmd=run.pl

dirModelos=${dirRelative}/models/diarization
dirConfig=${dirRelative}/conf
exeFeatDir=${dirRelative}/kaldi/src/featbin
exeSpkSeg=${dirRelative}/bin/SpeakerSegmentation
exeLittleEndian=${dirRelative}/bin/htkBigEndian2LittleEndian


fichero=$fileIn
fileId=$(basename $fichero '.wav')

# Make a temporal file and copy the audio file
rm -r ${outdir}
echo "Making temporal folder..."
dirTmp=${outdir}/${fileId}
mkdir -p ${dirTmp}
cp $fichero ${dirTmp}/${fileId}.wav

#Feature extraction
echo "Feature extraction..."
echo "${fileId} ${fichero}" > ${dirTmp}/fileId.scp
dirScp=${dirTmp}/fileId.scp

${exeFeatDir}/compute-mfcc-feats --allow-downsample=true --verbose=2 --config=${dirConfig}/mfcc_19features.conf \
scp:${dirScp} ark:- | \
${exeFeatDir}/copy-feats-to-htk --output-dir=${dirTmp}/ --output-ext=htk.tmp --sample-period=100000 ark:-
${exeLittleEndian} -e ${dirTmp}/${fileId}.htk.tmp -s ${dirTmp}/${fileId}.htk

#Speaker segmentation
echo "Speaker segmentation..."
echo "${exeSpkSeg} ${dirTmp}/${fileId}.wav ${dirTmp}/${fileId}.htk ${dirTmp}/${fileId}_SS.lab 1.5 0.0816 300.0"
${exeSpkSeg} ${dirTmp}/${fileId}.wav ${dirTmp}/${fileId}.htk ${dirTmp}/${fileId}_SS.lab 1.5 0.0816 300.0

# Se mueven las fronteras de los segmentos de voz para ampliarlos 0.3 hacia cada lado 
echo "Post-processing time steps +-0.3"
cat ${dirTmp}/${fileId}_SS.lab | awk -v file=$fileId '{printf "%s_%.5i %s %f %f\n",file,NR,file,$1,$1+$2}' > ${dirTmp}/segments.temp
${dirRelative}/utils/extend_segment_times.py --start-padding=0.3 --end-padding=0.3 < ${dirTmp}/segments.temp >${dirTmp}/segments
awk '{printf "%s SPK_%.5i\n", $1,NR}' ${dirTmp}/segments >${dirTmp}/utt2spk
${dirRelative}/utils/convert_utt2spk_and_segments_to_rttm.py ${dirTmp}/utt2spk ${dirTmp}/segments ${dirTmp}/segments.rttm

#Audio segmentation
plp_feats="ark:extract-segments scp:${dirScp} ${dirTmp}/segments ark:- | compute-plp-feats --allow-downsample=true --verbose=2 --config=${dirConfig}/plp.conf ark:- ark:- |"
pitch_feats="ark:extract-segments scp:${dirScp} ${dirTmp}/segments ark:- | compute-kaldi-pitch-feats --verbose=2 --config=${dirConfig}/pitch.conf ark:- ark:- | process-kaldi-pitch-feats ark:- ark:- |"
paste-feats --length-tolerance=2 "$plp_feats" "$pitch_feats" ark:- | copy-feats --compress=true ark:- ark,scp:${dirTmp}/plp_pitch.ark,${dirTmp}/plp_pitch.scp

compute-vad --config=${dirConfig}/vad1.conf scp:${dirTmp}/plp_pitch.scp ark,scp:${dirTmp}/vad.ark,${dirTmp}/vad.scp
${dirRelative}/utils/extract_ivectors_vad.sh --cmd "${dirConfig}/run.pl" --nj 1 ${dirTmp}/ ${dirTmp}/ ${dirTmp}/plp_pitch.scp ${dirTmp}/vad.scp ${dirModelos}/UBM512_plp_pitch full ${dirModelos}/IVextractor_plp_pitch ${dirTmp}/ ivectors_plp_pitch

logistic-regression-eval ${dirModelos}/logistic_regression_rebalanced_5classes "ark:ivector-normalize-length scp:${dirTmp}/ivectors_plp_pitch.scp ark:- |" ark,t:${dirTmp}/class_posteriors
cat ${dirTmp}/class_posteriors | cut -d " " -f 4-8 > ${dirTmp}/probs
${dirRelative}/utils/eliminaMusica.pl ${dirTmp}/${fileId}_SS.lab ${dirTmp}/probs ${dirTmp}/${fileId}_AS.lab

cat ${dirTmp}/class_posteriors | \
awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max)
                        { max=$f; argmax=f; }}
                        print $1, (argmax - 3); }' | \
${dirRelative}/utils/int2sym.pl -f 2 ${dirModelos}/classes5.txt \
  >${dirTmp}/${fileId}.scores5classes

${dirRelative}/utils/processOuputIVectors.pl ${dirTmp}/segments ${dirTmp}/${fileId}.scores5classes ${dirTmp}/${fileId}.sad
