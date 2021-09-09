# Results

## Evaluate and plot rate-distortion curves
To evaluate and compare your model with existing methods:
- use compressai.utils.eval_model to evaluate your model
- check (and modify) the script examples/run-benchmarks.sh to run encode/decode with existing methods
- use compressai.utils.plot to obtain the rate-distortion curves (matplootlib or plotly)

## Note on runtimes
The provided results using Compressai implementation contain runtimes that correspond to the average encode/decode time for each image of the corresponding test set. They have been obtained on a server equipped with Nvidia Quadro RTX 8000 and 80 Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz.

Please note that runtimes are provided as an indication. Direct comparisons with traditional codecs and standard reference models that run on 1 CPU only may not be relevant.
