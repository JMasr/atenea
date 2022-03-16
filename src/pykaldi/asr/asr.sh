#!/bin/bash

# Estos ficheros son necesarios para poner el el PATH los ejecutables de Kaldi


path1=$(pwd)
cd $path1/pykaldi/asr
pathModels=./models
confDir=./conf

. ./path.sh; # source the Kaldi path.
. ./cmd.sh; # source train and decode cmds.


dir_out=./tmp
tmp_dir=
decode_extra_opts=( --num-threads 1 )
decode_nj=1
beam=16
latticebeam=10
lmwt=13
umbral=1

. utils/parse_options.sh


if [ $# -le 1 ]; then
  echo "Usage: reconoceGrabacion.sh <wav file> <output-dir>"
  echo "e.g.:  $0 <wav file> <output dir>"
  echo "main options (for others, see top of script file)"
  echo "  --tmp_dir <tmp_dir>                              # temporary dir"
  echo "  --beam <beam>                                    # decode beam"
  echo "  --latticebeam <latticebeam>                      # decode latticebeam"
  echo "  --lmwt <lmwt>                                    # language model weight for ctm extraction"
  echo "  --umbral <umbral>                                # threshold for the generation of the confidence (CONF) line"
  echo "  --num_tiers <num_tiers>                          # number of the tiers into output .eaf"
  exit 1;
fi

wavfile=$1
outdir=$2
dirRelative=$3

# Se obtiene el nombre del fichero sin extensión, el directorio en el que está, etc
audio_dir=$(dirname $wavfile)
audio_name=$(basename $wavfile)
file_name=${audio_name%.*}
audio_ext=$(echo ${audio_name#*.} | awk '{print tolower($0)}')

# Creación de un directorio tmp donde se van a guardar los resultados
if [ -z $tmp_dir ]; then
    tmp_dir=${outdir}/tmp/${file_name}
fi
mkdir -p ${tmp_dir}
echo $$ > ${tmp_dir}/${file_name}".PID"  #Se guarda el PID del proceso
mkdir -p ${tmp_dir}/data
logfile=${tmp_dir}/log_kaldi.txt

##### 0. Definición de directorios
# Directorio donde están todos los modelos para un idioma en concreto
lang=gl
srcModelsDir=${pathModels}/exp_${lang}

# Directorios donde están el modelo de lenguaje (graph_dir), el lexicon (lang_dir), el extractor de i-vectors, los modelos acústicos
gmmdir=${srcModelsDir}/tri4b
graph_dir=${srcModelsDir}/tri4b/graph_ML_CORGA_GEV_DUVI_all_mix05 # Directorio con el Grafo, i.e., modelo de lenguaje
lang_dir=${srcModelsDir}/lang_ML_CORGA_GEV_DUVI_all_mix05 # Directorio con el lexicon
amkind=nnet3_d1 # "Tipo" de modelos acústicos
ivectorextractor_dir=${srcModelsDir}/${amkind}/extractor # Directorio con el extractor de i-vectors
acoustModel_dir=${srcModelsDir}/${amkind}/tdnn_sp # Direcorio con los modelos acústicos
# TDNN => original + low-resolution speed-perturbed data, input MFCCs (40 dim) + ivector (100 dim), left-context: 16, right-context: 12, 1 LDA node, 6 hidden layers, RELU 1024

###### 1. Creación de los ficheros wav.scp, utt2spk y spk2utt
# echo ${file_name} $audio_name | awk -v dir=$audio_dir '{printf "%s %s%s/%s%s\n", $1,"sox ",dir,$2," -G -r 16000 -e signed-integer -b 16 -c 1 -t wav - |"}' > ${tmp_dir}/data/wav.scp

echo "${file_name} ${wavfile}" > ${tmp_dir}/data/wav.scp


awk '{printf "%s SPK\n", $1}' ${tmp_dir}/data/wav.scp > ${tmp_dir}/data/utt2spk
utils/utt2spk_to_spk2utt.pl < ${tmp_dir}/data/utt2spk > ${tmp_dir}/data/spk2utt
cat ${tmp_dir}/data/wav.scp | awk '{print $1,$1,"A"}' > ${tmp_dir}/data/reco2file_and_channel
# Creo el fichero segments
utils/get_utt2dur.sh ${tmp_dir}/data 1>&2 || exit 1;
awk '{ print $1, $1, 0, $2 }' ${tmp_dir}/data/utt2dur > ${tmp_dir}/data/segments

## Parametrizacion:
steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 ${tmp_dir}/data ${tmp_dir}/tmp ${tmp_dir}/mfcc
steps/compute_cmvn_stats.sh ${tmp_dir}/data ${tmp_dir}/tmp ${tmp_dir}/mfcc

## Pase de reconocimiento para segmentar el audio
# Directorio donde están todos los modelos para un idioma en concreto
lang=gl
srcModelsDir=${pathModels}/exp_${lang}
graph_dir=${srcModelsDir}/tri4b/graph_lmtg # Directorio con el Grafo, i.e., modelo de lenguaje
lang_dir=${pathModels}/exp_${lang}/lang_lmtg
local/decode_nolats_cmnsliding.sh --write-words false --write-alignments true \
    --cmd "run.pl" --nj 1 --beam 7.0 --max-active 1000 \
    $graph_dir ${tmp_dir}/data ${srcModelsDir}/tri4b ${tmp_dir}/decode_tri4b_nolats_${file_name}

local/resegment_data.sh --cmd "$train_cmd" --segmentation-opts "--silence-proportion 0.2 --max-segment-length 30 --hard-max-segment-length 30" \
        ${tmp_dir}/data $lang_dir \
        ${tmp_dir}/decode_tri4b_nolats_${file_name} ${srcModelsDir}/tri4b ${tmp_dir}/data/rec_segments ${tmp_dir}/tmp/tri4b_resegment_rec
awk '{printf "%s SPK_%.4i\n", $1,NR}' ${tmp_dir}/data/rec_segments/utt2spk > ${tmp_dir}/data/rec_segments/utt2spk2
mv -f $tmp_dir/data/rec_segments/utt2spk2 $tmp_dir/data/rec_segments/utt2spk
utils/utt2spk_to_spk2utt.pl < $tmp_dir/data/rec_segments/utt2spk > $tmp_dir/data/rec_segments/spk2utt
cat $tmp_dir/data/rec_segments/wav.scp | awk '{print $1,$1,"A"}' > $tmp_dir/data/rec_segments/reco2file_and_channel

cp $tmp_dir/data/rec_segments/segments $tmp_dir/data/rec_segments/segments_ini

## Reconocimiento de la segmentación de "rec" utilizando mfcchires + iVector + NNET3:
#Calcúlanse o número de segmentos en función do número de liñas do ficheiro
# nj=$(wc -l $tmp_dir/data/rec_segments/utt2spk | awk '{print $1}')
nj=1

steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj  --mfcc-config ${confDir}/mfcc_hires.conf $tmp_dir/data/rec_segments $tmp_dir/tmp/OUT $tmp_dir/mfcc >> $logfile || exit 1
steps/compute_cmvn_stats.sh $tmp_dir/data/rec_segments $tmp_dir/tmp/OUT $tmp_dir/mfcc >> $logfile
# dump iVectors for the testing data.
steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj $tmp_dir/data/rec_segments ${ivectorextractor_dir} ${tmp_dir}/ivectors >> $logfile 

# Reconocimiento
decode_dir=${tmp_dir}/decode_${amkind}
graph_dir=${srcModelsDir}/tri4b/graph_ML_CORGA_GEV_DUVI_all_mix05
lang_dir=${srcModelsDir}/lang_ML_CORGA_GEV_DUVI_all_mix05
amkind=nnet3_d1 # "Tipo" de modelos acústicos
ivectorextractor_dir=${srcModelsDir}/${amkind}/extractor # Directorio con el extractor de i-vectors
acoustModel_dir=${srcModelsDir}/${amkind}/tdnn_sp # Direcorio con los modelos acústicos
# TDNN => original + low-resolution speed-perturbed data, input MFCCs (40 dim) + ivector (100 dim), left-context: 16, right-context: 12, 1 LDA node, 6 hidden layers, RELU 1024

local/nnet3/decode.sh --skip-scoring true --skip_diagnostics true --nj $nj --cmd "$decode_cmd"  --online-ivector-dir ${tmp_dir}/ivectors --beam $beam --lattice-beam $latticebeam $graph_dir $tmp_dir/data/rec_segments $acoustModel_dir $decode_dir >> $logfile || exit 1

# 4. Extracción de ctms y generación de .eaf
local/lattice_to_ctm_laura.sh --min_lmwt $lmwt --max_lmwt $lmwt  ${tmp_dir}/data $graph_dir $acoustModel_dir $decode_dir >> $logfile || exit 1

cat $decode_dir/score_${lmwt}/wip0.0/data.utt.ctm | utils/convert_ctm.pl $tmp_dir/data/rec_segments/segments $tmp_dir/data/rec_segments/reco2file_and_channel > $decode_dir/score_${lmwt}/wip0.0/${file_name}.ctm
cp $decode_dir/score_${lmwt}/wip0.0/data.utt.ctm $tmp_dir/
cp $decode_dir/score_${lmwt}/wip0.0/${file_name}.ctm $tmp_dir/

cat ${decode_dir}/score_${lmwt}/wip0.0/data.utt.ctm | cut -d' ' -f1 | sort -u > $tmp_dir/data/rec_segments/id_segments_rec
awk -F' ' 'FNR==NR{a[$1];next} ($1 in a)' $tmp_dir/data/rec_segments/id_segments_rec $tmp_dir/data/rec_segments/segments > $tmp_dir/data/rec_segments/segments2

awk 'NF > 5 {print}' ${decode_dir}/score_${lmwt}/wip0.0/data.utt.ctm | awk '{print $3, $4, $5, $6}' ${decode_dir}/score_${lmwt}/wip0.0/data.utt.ctm > ${tmp_dir}/${file_name}.wordconfid.txt

#php local/generaEAF.php $tmp_dir/data/rec_segments/segments2 ${decode_dir}/score_${lmwt}/wip0.0/data.utt.ctm $tmp_dir/${file_name}

mv $tmp_dir/${file_name}.ctm $outdir/
rm -rf ${outdir}/tmp/
