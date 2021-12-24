#!/usr/bin/env bash

# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set -e

err_report() {
    echo "Error on line $1"
    echo "check codec path"
}
trap 'err_report $LINENO' ERR

NJOBS=30

usage() {
    echo "usage: $(basename $0) dataset CODECS"
    echo "supported codecs: [jpeg, jpeg2000, webp, bpg, hm, vtm, av1, bmshj2018-factorized-mse, bmshj2018-hyperprior-mse, mbt2018-mean-mse]"
}

if [[ $1 == "-h" || $1 == "--help" ]]; then
    usage
    exit 1
fi

if [[ $# -lt 2 ]]; then
    echo "Error: missing arguments"
    usage
    exit 1
fi

dataset="$1"
shift

dataset_name=$(basename "${dataset}")
if [[ ${dataset_name} == "val"* ]] || [[ ${dataset_name} == "train" ]] || [[ ${dataset_name} == "test" ]]; then
  dataset_name=$(basename $(dirname "${dataset}"))
fi


# libpng
BPGENC="$(which bpgenc)"
BPGDEC="$(which bpgdec)"

# Tensorflow Compression script
# https://github.com/tensorflow/compression
# edit path below or uncomment locate function
# TFCI_SCRIPT="${HOME}/tensorflow-compression/compression/models/tfci.py"

# VTM
# edit below to provide the path to the chosen version of VTM
# _VTM_SRC_DIR="${HOME}/vvc/vtm-9.1"
# VTM_BIN_DIR="$(dirname "$(locate '*release/EncoderApp' | grep "$_VTM_SRC_DIR")")"
# uncomment below and provide bin directory if not found
# VTM_BIN_DIR="${_VTM_SRC_DIR}/bin/umake/clang-11.0/x86_64/release/"
# VTM_CFG="${_VTM_SRC_DIR}/cfg/encoder_intra_vtm.cfg"
# VTM_VERSION_FILE="${_VTM_SRC_DIR}/source/Lib/CommonLib/version.h"
# VTM_VERSION="$(sed -n -e 's/^#define VTM_VERSION //p' ${VTM_VERSION_FILE})"

# HM
# edit below to provide the path to the chosen version of HM
# _HM_SRC_DIR="${HOME}/hevc/HM-16.20+SCM-8.8"
# HM_BIN_DIR="${_HM_SRC_DIR}/bin/"
# HM_CFG="${_HM_SRC_DIR}/cfg/encoder_intra_main_rext.cfg"
# HM_VERSION_FILE="${_HM_SRC_DIR}/source/Lib/TLibCommon/CommonDef.h"
# HM_VERSION="$(sed -n -e 's/^#define NV_VERSION \(.*\)\/\/\/< Current software version/\1/p' ${HM_VERSION_FILE})"

# AV1
# edit below to provide the path to the chosen version of VTM
AV1_BIN_DIR="${HOME}/av1/aom/build_gcc"

jpeg() {
    python3 -m compressai.utils.bench jpeg "$dataset"            \
        -q $(seq 5 5 95) -j "$NJOBS" > "results/${dataset_name}/jpeg.json"
}

jpeg2000() {
    python3 -m compressai.utils.bench jpeg2000 "$dataset"        \
        -q $(seq 5 5 95) -j "$NJOBS" > "results/${dataset_name}/jpeg2000.json"
}

webp() {
    python3 -m compressai.utils.bench webp "$dataset"            \
        -q $(seq 5 5 95) -j "$NJOBS" > "results/${dataset_name}/webp.json"
}

bpg() {
    if [ -z ${BPGENC+x} ] || [ -z ${BPGDEC+x} ]; then echo "install libBPG"; exit 1; fi
    python3 -m compressai.utils.bench bpg "$dataset"             \
        -q $(seq 47 -5 12) -m "$1" -e "$2" -c "$3"               \
        --encoder-path "$BPGENC"                                \
        --decoder-path "$BPGDEC"                                \
        -j "$NJOBS" > "results/${dataset_name}/$4"
}

hm() {
    if [ -z ${HM_BIN_DIR+x} ]; then echo "set HM bin directory HM_BIN_DIR"; exit 1; fi
    echo "using HM version $HM_VERSION"
    python3 -m compressai.utils.bench hm "$dataset"             \
        -q $(seq 47 -5 12) -b "$HM_BIN_DIR" -c "$HM_CFG"         \
        -j "$NJOBS" > "results/${dataset_name}/hm.json"
}

vtm() {
    if [ -z ${VTM_BIN_DIR+x} ]; then echo "set VTM bin directory VTM_BIN_DIR"; exit 1; fi
    echo "using VTM version $VTM_VERSION"
    python3 -m compressai.utils.bench vtm "$dataset"            \
        -q $(seq 47 -5 12) -b "$VTM_BIN_DIR" -c "$VTM_CFG"       \
        -j "$NJOBS" > "results/${dataset_name}/vtm.json"
}

av1() {
    if [ -z ${AV1_BIN_DIR+x} ]; then echo "set AV1 bin directory AV1_BIN_DIR"; exit 1; fi
    python3 -m compressai.utils.bench av1 "$dataset"            \
        -q $(seq 62 -5 7) -b "${AV1_BIN_DIR}"       \
        -j "$NJOBS" > "results/${dataset_name}/av1.json"
}

tfci() {
    if [ -z ${TFCI_SCRIPT+x} ]; then echo "set TFCI_SCRIPT bin path"; exit 1; fi
    python3 -m compressai.utils.bench tfci "$dataset"           \
        --path "$TFCI_SCRIPT" --model "$1"                      \
        -q $(seq 1 8) -j "$NJOBS" > "results/${dataset_name}/official-$1.json"
}

mkdir -p "results/${dataset_name}"

for i in "$@"; do
    case $i in
        "jpeg")
            jpeg
            ;;
        "jpeg2000")
            jpeg2000
            ;;
        "webp")
            webp
            ;;
        "bpg")
            # bpg "420" "x265" "rgb" bpg_420_x265_rgb.json
            # bpg "420" "x265" "ycbcr" bpg_420_x265_ycbcr.json
            # bpg "444" "x265" "rgb" bpg_444_x265_rgb.json
            bpg "444" "x265" "ycbcr" bpg_444_x265_ycbcr.json

            # bpg "420" "jctvc" "rgb" bpg_420_jctvc_rgb.json
            # bpg "420" "jctvc" "ycbcr" bpg_420_jctvc_ycbcr.json
            # bpg "444" "jctvc" "rgb" bpg_444_jctvc_rgb.json
            # bpg "444" "jctvc" "ycbcr" bpg_444_jctvc_ycbcr.json
            ;;
        "hm")
            hm
            ;;
        "vtm")
            vtm
            ;;
        "av1")
            av1
            ;;
        'bmshj2018-factorized-mse')
            tfci 'bmshj2018-factorized-mse'
            ;;
        'bmshj2018-hyperprior-mse')
            tfci 'bmshj2018-hyperprior-mse'
            ;;
        'mbt2018-mean-mse')
            tfci 'mbt2018-mean-mse'
            ;;
        *)
            echo "Error: unknown option $i"
            exit 1
            ;;
    esac
done
