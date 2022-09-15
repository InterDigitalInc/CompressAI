# Results

## Evaluate and plot rate-distortion curves
To evaluate and compare your model with existing methods:
- use compressai.utils.eval_model to evaluate your model
- check (and modify) the script examples/run-benchmarks.sh to run encode/decode with existing methods
- use compressai.utils.plot to obtain the rate-distortion curves (matplootlib or plotly)

## Note on runtimes
The provided results using Compressai implementation contain runtimes that correspond to the average encode/decode time for each image of the corresponding test set. They have been obtained on a server equipped with Nvidia Quadro RTX 8000 and 80 Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz.

Please note that runtimes are provided as an indication. Direct comparisons with traditional codecs and standard reference models that run on 1 CPU only may not be relevant.

## Note on reconstructed images for metrics computations
The original PSNR and MS-SSIM results were computed on floating point reconstructed pictures (for images models only).
To make the comparisons fairer with traditional codecs, a folder "8bit-decoded" has been added, where json files contain metrics computed on rescaled reconstructed images on 8bit, like the inputs.
The current implementation of compressai.utils.eval_model now includes the computation of metrics on 8bit reconstructed images.
Note that we found there are negligible differences in PSNR and MS-SSIM on average for the considered datasets, but could be sensible for a particular image at higher bpp.