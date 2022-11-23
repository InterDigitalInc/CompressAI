#!/bin/bash

BDIR=$(dirname $(realpath $0));

MODEL=""
VERBOSE=0;
WITH_QUANT=0;
DATASET="";
LATENT_SIZE="16 16 192"; # to extract from model


if (( VERBOSE )); then
  MESSAGE="/dev/stdout";
  OPT="--verbose";
else
  MESSAGE="/dev/null";
  OPT="";
fi;

while [[ $# -gt 0 ]]
do
 key="$1"
 case $key in
  -h|--help)
   cat << _EOF_
build_codec.sh --model model.pth --training_dataset training.npy [--model_int16] 
 --model model.pth: contains a g_s g_a model with layers compatible with onnx and sadl
 --model_int16: will also generate a int16 decoder (requires the quantizers value inside the pth)
 --training_dataset: uses the patch to recompute the cdfs and channel activation. Format is raw npy in uint8 RGB 256x256 (see dataset2latent.py for details)
_EOF_
   exit 1;
;;
  --model)
   shift;
   MODEL=$(realpath "$1");
   shift;
   ;;

  --training_dataset)
   shift;
   DATASET=$1;
   shift;
   ;;

  --model_int16)
   shift;
   WITH_QUANT=1;
   ;;
   
  *)
  echo "[ERROR] unknown prm $1";
  exit 1;
  ;;
esac;
done;

if [ -z "$MODEL"  -o -z "$DATASET" ]; then
 echo "Missing parameters, see --help";
 exit -1;
fi;

echo "[STEP 0] convert model: pth => onnx, (+info in pickle if needed)";
python3 ${BDIR}/extract_codec.py  --model $MODEL --output model  >$MESSAGE;

DEC=model_dec.onnx;
DECINFO=model_info.pkl;
echo -e "\n[STEP 1] convert decoder: onnx => sadl float";
python3 ${BDIR}/../../sadl/converter/main.py --input_onnx $DEC --output decoder_float.sadl $OPT >$MESSAGE;


# copy src
if [ ! -d src/compressai/sadl_codec ]; then
 mkdir -p src/compressai;
 cd src;
 cp -R ${BDIR}/../sadl_codec compressai/;
 ln -s ${BDIR}/../../sadl .;
 ln -s ${BDIR}/../../third_party .;
 cd ..;
fi;

echo -e "\n[STEP 2] extract decoder information";
echo " [INFO] recompute cdf from trainingset";
echo "  [INFO] compute training set output (long)";
if [ ! -f output_tempo.npy ]; then
  python3 ${BDIR}/../sadl_codec/dataset2latent.py --model ${MODEL} --input $DATASET --output output_tempo.npy $OPT >$MESSAGE;
else
  echo "  [INFO] CACHED"; 
fi;


echo "  [INFO] build cdf extractor";
mkdir -p build/sadl_codec
cd build/sadl_codec;
cmake --log-level=ERROR ../../src/compressai/sadl_codec >$MESSAGE;
make extract_cdf >$MESSAGE;
cd ../..;

echo "  [INFO] extract cdf";
if [ ! -f model_cdfs.h ]; then
  build/sadl_codec/extract_cdf ${LATENT_SIZE} output_tempo.npy;
else
  echo "  [INFO] CACHED"; 
fi;

# quantify the decoder
if (( WITH_QUANT )); then 
    echo -e "\n[STEP 3] build utils for quantization";
    mkdir -p build/utils;
    cd build/utils;
    cmake --log-level=ERROR ../../src/sadl/sample >$MESSAGE;
    make naive_quantization >$MESSAGE;
    cd ../..;
    echo -e "\n[STEP 4] model quantization";
    echo " [INFO] convert sadl model: float => int16";
    if [ ! -f decoder_quantizers.txt ]; then
      echo "[ERROR] missing file decoder_quantizers.txt";
    fi;
    cat decoder_quantizers.txt | build/utils/naive_quantization decoder_float.sadl decoder_int16.sadl >$MESSAGE;
else
    echo -e "\n[STEP 3] skipped";
    echo -e "\n[STEP 4] skipped";
fi;


echo -e "\n[STEP 5] build decoder float";
bin2c --const --name decoderSadl --static --type int decoder_float.sadl > decoder_float.h
cp decoder_float.h model_cdfs.h src/compressai/sadl_codec/
mkdir -p build/sadl_codec
cd build/sadl_codec;
make -j8 decoder_sadl_float_simd512 >$MESSAGE;
cd ../..;

if (( WITH_QUANT )); then 
 echo " [STEP5b] build decoder int16";
 bin2c --const --name decoderSadl --static --type int decoder_int16.sadl > decoder_int16.h
 cp decoder_int16.h src/compressai/sadl_codec;
 mkdir -p build/sadl_codec
 cd build/sadl_codec;
 make -j8 decoder_sadl_int16_simd512 >$MESSAGE;
 cd ../..;
fi;

echo -e "\n[STEP 6] convert encoder: onnx => sadl float";
python3 ${BDIR}/../../sadl/converter/main.py --input_onnx model_enc.onnx --output encoder_float.sadl $OPT >$MESSAGE;

echo -e "\n[STEP 6b] build encoder float";
bin2c --const --name encoderSadl --static --type int encoder_float.sadl > encoder_float.h
cp encoder_float.h src/compressai/sadl_codec/
mkdir -p build/sadl_codec
cd build/sadl_codec;
# cmake --log-level=ERROR ../../src/compressai/sadl_codec >$MESSAGE;
make -j8 encoder_sadl_float_simd512 >$MESSAGE;
cd ../..;

if (( WITH_QUANT )); then 
    echo -e "\n[STEP 6c] build encoder int16";
    if [ ! -f encoder_quantizers.txt ]; then
      echo "[ERROR] missing file decoder_quantizers.txt";
    fi;
    cat encoder_quantizers.txt | build/utils/naive_quantization encoder_float.sadl encoder_int16.sadl >$MESSAGE;
    bin2c --const --name encoderSadl --static --type int encoder_int16.sadl > encoder_int16.h
    cp encoder_int16.h src/compressai/sadl_codec/
    mkdir -p build/sadl_codec
    cd build/sadl_codec;
    make -j8 encoder_sadl_int16_simd512 >$MESSAGE;
    cd ../..;
fi



