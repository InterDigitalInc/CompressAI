# Convert a compressAI model into a standalone C++ encoder/decoder

## Workflow

1- compressAI:
 - train a model convertible to SADL (for example FactorizedPriorBasic), see below for details
 - dump a pth containing the model (g\_a, g\_s, possibly the cdfs and quantizers if needed)
 - prepare a training set of raw patch 256x256x3 in npy format (uint8) to be used to recompute statistics of the latents
 

2- Conversion
 - initialize the SADL submodule
 - run the script build\_codec.sh in a working directory:
 ```shell
 compressai/sadl_codec/build_codec.sh --model model.pth --training_dataset trainingdataset.npy 
 ```
Note: it can take a while to generate the first time as it has to perform an inference on the whole training set.

 It will run the following steps:
 - STEP 0: extract the codec from the pth and save it to onnx format, using extract\_codec.py
 - STEP 1: convert the onnx models => sadl float, using the SADL converter
 - STEP 2: extract cdfs values from the trainingdataset using extract\_cdf.py.
 - OPTIONAL STEP 3/4: extract quantizers information and create a sadl int16 decoder.
 - STEP 5: build the C++ decoder
 - STEP 6: extract the encoder, build the C++ encoder
 

3- Run on kodak dataset
```shell
compressai/sadl_codec/check_kodak.sh --dir kodak_dir
```
or for better performance (longer to generate):
```shell
compressai/sadl_codec/check_kodak.sh --dir kodak_dir --rdoq lambda
```

It will run the evaluation using the C++ codec on kodak dataset:
- convert image to PPM
- encode image with encoder\_sadl\_float\_simd512 (optionnaly performs an rdoq on the latent)
- decode the bitstream using decoder\_sadl\_float\_simd512
- compute PSNR on reconstructed images

Several version of the encoder/decoder are available (float version, non SIMD etc.)

4- Details

File details:
- model\_dec.onnx: contains just the network part with the deconvolution and activation compatible with SADL (e.g. ReLU). 
- model\_info.pkl: contains information beside the network itself: 
the cdfs, cdfs length and cdfs offset and the quantizers for each deconv layers parameters inside a dict { 'cdfs': nparray, 'cdflen': nparray, 'cdfoff': nparray, 'quantizers': '0.weight': 8,  '0.bias': 10, ...} }
- model_enc.onnx: contains just the network part with the convolution and activation compatible with SADL (e.g. ReLU). 


5- Train a model
The model should be trained with the following constraints:
* decoder uses ReLU (or other SADL compatible activation like leakyReLU)
* quantization:
  - the decoder weights and bias have been dynamically quantized (i.e. the parameters are representable on a 16bits signed int) if plan to use the quantized model
  - all the latents in the decoder have been dynamically quantized (i.e. the parameters are representable on a 16bits signed int)
  - A straightforward dynamic quantization method consists in adding a quantization -> clip -> dequantization module after each operation and on each parameters.
* the final pth contains information on cdf, quantizers etc.

