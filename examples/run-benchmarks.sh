#!/usr/bin/env bash

# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Do not forget to 
# - set paths to codec bins and sources below
# - activate the virtual environment containing CompressAI

set -e

err_report() {
    echo "Error on line $1"
    echo "check codec path"
}
trap 'err_report $LINENO' ERR

NJOBS=${NJOBS:-4}

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
# AV1_BIN_DIR="${HOME}/av1/aom/build_gcc"

jpeg() {
    python3 -m compressai.utils.bench jpeg "$dataset"            \
        -q $(seq 5 5 95) -j "$NJOBS" > benchmarks/jpeg.json
}

jpeg2000() {
    python3 -m compressai.utils.bench jpeg2000 "$dataset"        \
        -q $(seq 5 5 95) -j "$NJOBS" > benchmarks/jpeg2000.json
}

webp() {
    python3 -m compressai.utils.bench webp "$dataset"            \
        -q $(seq 5 5 95) -j "$NJOBS" > benchmarks/webp.json
}

bpg() {
    if [ -z ${BPGENC+x} ] || [ -z ${BPGDEC+x} ]; then echo "install libBPG"; exit 1; fi
    python3 -m compressai.utils.bench bpg "$dataset"             \
        -q $(seq 47 -5 2) -m "$1" -e "$2" -c "$3"               \
        --encoder-path "$BPGENC"                                \
        --decoder-path "$BPGDEC"                                \
        -j "$NJOBS" > "benchmarks/$4"
}

hm() {
    if [ -z ${HM_BIN_DIR+x} ]; then echo "set HM bin directory HM_BIN_DIR"; exit 1; fi
    echo "using HM version $HM_VERSION"
    python3 -m compressai.utils.bench hm "$dataset"             \
        -q $(seq 47 -5 2) -b "$HM_BIN_DIR" -c "$HM_CFG"         \
        -j "$NJOBS" > "benchmarks/hm.json"
}

vtm() {
    if [ -z ${VTM_BIN_DIR+x} ]; then echo "set VTM bin directory VTM_BIN_DIR"; exit 1; fi
    echo "using VTM version $VTM_VERSION"
    python3 -m compressai.utils.bench vtm "$dataset"            \
        -q $(seq 47 -5 2) -b "$VTM_BIN_DIR" -c "$VTM_CFG"       \
        -j "$NJOBS" > "benchmarks/vtm.json"
}

av1() {
    if [ -z ${AV1_BIN_DIR+x} ]; then echo "set AV1 bin directory AV1_BIN_DIR"; exit 1; fi
    python3 -m compressai.utils.bench av1 "$dataset"            \
        -q $(seq 62 -5 2) -b "${AV1_BIN_DIR}"       \
        -j "$NJOBS" > "benchmarks/av1.json"
}

tfci() {
    if [ -z ${TFCI_SCRIPT+x} ]; then echo "set TFCI_SCRIPT bin path"; exit 1; fi
    python3 -m compressai.utils.bench tfci "$dataset"           \
        --path "$TFCI_SCRIPT" --model "$1"                      \
        -q $(seq 1 8) -j "$NJOBS" > "benchmarks/$1.json"
}

mkdir -p "benchmarks"

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
