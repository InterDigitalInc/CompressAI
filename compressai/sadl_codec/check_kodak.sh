#!/bin/bash

BDIR=$(dirname $(realpath $0));

DIR=""
LAMBDA="";
INT16=0;
FLOATENC=0;

while [[ $# -gt 0 ]]
do
 key="$1"
 case $key in
  -h|--help)
   cat << _EOF_
check_kodak.sh --dir DIRKODAK [--rdoq lambda] [--int16]
# assume build_codec.sh has been run in the same directory
--rdoq lambda: lambda used to build the model
--int16: use quantized decoder
_EOF_
   exit 1;
;;
  --dir)
   shift;
   DIR=$(realpath "$1");
   shift;
   ;;
      
  --rdoq)
   shift;
   LAMBDA=$1;
   shift;
   ;;
   
  --int16)
   shift;
   INT16=1;
   ;;   
   
  --float_enc)
   shift;
   FLOATENC=1;
   ;;   
  *)
  echo "[ERROR] unknown prm $1";
  exit 1;
  ;;
esac;
done;

if [ -z "$DIR"  ]; then
 echo "Missing parameters, see --help";
 exit -1;
fi;

if (( INT16 )); then
 ENC=build/sadl_codec/encoder_sadl_int16_simd512;
 DEC=build/sadl_codec/decoder_sadl_int16_simd512;
else
 ENC=build/sadl_codec/encoder_sadl_float_simd512;
 DEC=build/sadl_codec/decoder_sadl_float_simd512;
fi;
if (( FLOATENC )); then
 ENC=build/sadl_codec/encoder_sadl_float_simd512;
fi;
  
rm -f kodak_results.txt kodak_py.log;
for((i=1;i<=24;i++)); do
 IMAGE=$(printf "$DIR/kodim%02d.png" $i);
 convert $IMAGE im.ppm > /dev/null;
 ${ENC} im.ppm image.sadlbs $LAMBDA >> kodak_py.log;
 bpp=$($DEC image.sadlbs image_rec_sadl.ppm | grep bpp | cut -d'=' -f2);
 psnr=$(python3 -c "import cv2;import sys;print(\"{:.3f} dB\".format(cv2.PSNR(cv2.imread(sys.argv[1]),cv2.imread(sys.argv[2]))))" $IMAGE image_rec_sadl.ppm | cut -d' ' -f1);
 I=$(basename $IMAGE);
 echo "$I $psnr $bpp" | tee -a kodak_results.txt;
done
echo "[INFO] results in kodak_results.txt";


python - <<EOF
import re
f = open("kodak_results.txt")
psnr = 0.
bpp =0.
cnt = 0.
for line in f:
 m = re.match("[^ ]*\s([^ ]*)\s([^ ]*)",line)
 if m is not None: 
   psnr += float(m.group(1))
   bpp  += float(m.group(2))
   cnt += 1
print("Average PSNR/BPP:\n{:.3f} {:.3f}".format(psnr/cnt, bpp/cnt))
EOF

