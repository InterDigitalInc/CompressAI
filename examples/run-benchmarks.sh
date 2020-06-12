#!/usr/bin/env bash

set -e

err_report() {
    echo "Error on line $1"
}
trap 'err_report $LINENO' ERR

NJOBS=${NJOBS:-4}

BPGENC="$(which bpgenc)"
BPGDEC="$(which bpgdec)"
TFCI_SCRIPT="$(locate tfci.py)"

_VTM_SRC_DIR="$(locate '*VVCSoftware_VTM')"

VTM_BIN_DIR="$(dirname "$(locate '*release/EncoderApp' | grep "$_VTM_SRC_DIR")")"
VTM_CFG="$(locate encoder_intra_vtm.cfg | grep "$_VTM_SRC_DIR")"

usage() {
    echo "usage: $(basename $0) dataset CODECS"
}

jpeg() {
    python -m compressai.utils.bench jpeg "$dataset"        \
        -q $(seq 5 5 95) -j "$NJOBS" > benchmarks/jpeg.json
}

jpeg2000() {
    python -m compressai.utils.bench jpeg2000 "$dataset"    \
        -q $(seq 5 5 95) -j "$NJOBS" > benchmarks/jpeg2000.json
}

webp() {
    python -m compressai.utils.bench webp "$dataset"        \
        -q $(seq 5 5 95) -j "$NJOBS" > benchmarks/webp.json
}

bpg() {
    python -m compressai.utils.bench bpg "$dataset"         \
        -q $(seq 47 -5 2) -m "$1" -e "$2" -c "$3"           \
        --encoder-path "$BPGENC"                            \
        --decoder-path "$BPGDEC"                            \
        -j "$NJOBS" > "benchmarks/$4"
}

vtm() {
    python -m compressai.utils.bench vtm "$dataset"         \
        -q $(seq 47 -5 2) -b "$VTM_BIN_DIR" -c "$VTM_CFG" \
        -j "$NJOBS" > "benchmarks/vtm.json"
}

tfci() {
    python -m compressai.utils.bench tfci "$dataset"        \
        --path "$TFCI_SCRIPT" --model "$1"                  \
        -q $(seq 1 8) -j "$NJOBS" > "benchmarks/tfci_$1.json"
}

if [[ $# -lt 2 ]]; then
    echo "Error: missing arguments"
    usage
    exit 1
fi

dataset="$1"
shift

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
        "vtm")
            vtm
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
