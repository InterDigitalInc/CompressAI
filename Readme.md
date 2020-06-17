# CompressAI

CompressAI (_compress-ay_) is built on top of PyTorch and provides:

* custom operations, layers and models for deep learning based data compression
* a partial port of the official [TensorFlow compression
  ](https://github.com/tensorflow/compression>) library
* pre-trained end-to-end compression models for learned image compression
* evaluation scripts to compare learned models against classical image/video
  compression codecs

## Installation

CompressAI only supports python 3.6+ and pytorch 1.4+. A C++17 compiler, a
recent version of pip (19.0+), and some python dependencies (see `setup.py`)
are also required.

To get started and install CompressAI, run the following commands in a virtual
environment:

```bash
git clone https://github.com/InterDigitalInc/CompressAI
cd compressai
pip install -e .
```

For a custom installation, you can also run one of the following commands:
* `pip install -e '.[dev]'`: install the packages required for development (testing, linting, docs)
* `pip install -e '.[tutorials]'`: install the packages required for the tutorials (notebooks)
* `pip install -e '.[all]'`: install all the optional packages

This is the currently recommended installation method. Docker images and pypi
packages will be released in the future. Conda environments are not officially
supported.


## Documentation

* [Installation]()
* [CompressAI API]()
* [Training your own model]()
* [List of available models (model zoo)]()


## Usage

### Examples

Scripts and notebooks examples can be found in the `examples/` directory.

To encode/decode images with the provided pre-trained models, run the
`codec.py` example:

```bash
python examples/codec.py --help
```

An example training script with a rate-distortion loss is provided in
`examples/train.py`. You can replace the example model in the training script
with your own model implemented with CompressAI, and then run the script for a
simple training pipeline:

```bash
python example/train.py -d /path/to/my/image/dataset/ --epochs 300 -lr 1e-4 --batch-size 16 --cuda --save
```

A jupyter notebook illustrating a pre-trained model for learned image
compression is also provided in the `examples` directory:

```bash
pip install -U ipython jupyter ipywidgets matplotlib
jupyter notebook examples/
```

### Evaluation

To evaluate a pre-trained model on your own dataset, CompressAI provides an
evaluation script:

```bash
python -m compressai.utils.eval_model MODEL_NAME /path/to/images/folder/
```

To evaluate published classical or machine-learning based image/video
codec solutions:

```bash
python3 -m compressai.utils.bench --help
python3 -m compressai.utils.bench bpg --help
python3 -m compressai.utils.bench vtm --help
```

## License

CompressAI is licensed under under the Apache License, Version 2.0

## Contributing

We welcome feedback and contributions. Please open a GitHub issue to report
bugs, request enhancements or if you have questions. 

Before contributing, please read the CONTRIBUTING.md file.

## Authors

* Jean Bégaint, Fabien Racapé, Simon Feltman and Akshay Pushparaja, from the InterDigital AI Lab.
* *Contact*: firstname.lastname@interdigital.com

## Citation

If you use this project, please cite the relevant publications for the
original models and datasets, and cite this project as:

```
@misc{CompressAI,
	title = {{CompressAI}: A PyTorch library and evaluation platform for end-to-end compression research},
	author = "{Jean Bégaint, Fabien Racapé, Simon Feltman, Akshay Pushparaja}",
	howpublished = {\url{https://github.com/InterDigitalInc/CompressAI}},
	url = "https://github.com/InterDigitalInc/CompressAI",
	year = 2020,
	note = "[Online; accessed 20-June-2020]"
}

```

## Related links
 * Tensorflow compression library by _Ballé et al._: https://github.com/tensorflow/compression
 * Range Asymmetric Numeral System code from _Fabian 'ryg' Giesen_: https://github.com/rygorous/ryg_rans
 * BPG image format by _Fabrice Bellard_: https://bellard.org/bpg
 * HEVC HM reference software: https://hevc.hhi.fraunhofer.de
 * VVC VTM reference software: https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM
 * Kodak image dataset: http://r0k.us/graphics/kodak/
