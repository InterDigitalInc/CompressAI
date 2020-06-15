# CompressAI

CompressAI (_compress-ay_) provides custom operations, layers, models and tools to research,
develop and evaluate end-to-end image and video compression codecs.

CompressAI is built on top of PyTorch and provides:
* a partial port of the official [TensorFlow
  implementation](https://github.com/tensorflow/compression) of _Ballé et al._
  research
* pre-trained end-to-end compression models from the learned image compression
  state-of-the-art 
* evaluation scripts to compare learned models against classical image/video compression codecs

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
	- [Quickstart](#quickstart)
	- [Evaluation](#evaluation)
	- [Training](#training)
- [License](#license)
- [Contributing](#contributing)
- [Citation](#citation)

## Installation

### Requirements

* python 3.6 or later (`python3-dev`, `python3-pip`, `python3-venv`)
* pip 19.0 or later
* a `C++17` compiler
* python packages: `numpy`, `scipy`, `torch`, `torchvision`

### Virtual environment

We recommend using a virtual environment to isolate project packages
installation from the base system:

* `python3 -m venv venv`
* `source ./venv/bin/activate`

### Using pip

* `git clone https://github.com/InterDigitalInc/CompressAI`
* `pip install -e ./compressai`

You can also build your own pip package:

* `git clone https://github.com/InterDigitalInc/CompressAI`
* `cd compressai`
* `python3 setup.py bdist_wheel --dist-dir dist/`
* `pip install dist/compressai-*.whl`

> **Note**: on MacOS you might want to use `CC=clang CXX=clang++ pip install...`
to compile with clang instead of gcc

We are also planning to provide a Docker image in the future.

## Usage

`import compressai`

### Quickstart

Take a look at the examples:

* an example training script `examples/train.py`
* an example codec pipeline `examples/codec.py`

### Evaluation

`python3 -m compressai.utils.bench --help`

### Training

An example training script is provided in the `examples` directory.

```python3 examples/train.py --help```

Training usually take between one or two weeks, depending on the model, the
number of channels and the GPU architecture used.

Pretrained models were learned with a batch size of 16 or 32, a learning rate
of 1e-4, for approximately 1-2M steps. The learning rate is divided by 2 when
the loss reaches a plateau (we use a patience of 20 epochs).

## License

CompressAI is licensed under under the Apache License, Version 2.0

## Contributing

See CONTRIBUTING.md

## Authors
* Jean Bégaint
* Fabien Racapé
* Simon Feltman

InterDigital, AI Lab.

*Contact*: firstname.lastname@interdigital.com

## Citation

If you use this project, please cite the relevant publications for the
original models and datasets, and cite this project as:

```
@misc{CompressAI,
	title = {{CompressAI}: A ML library for end-to-end data compression research},
	author = "{Jean Bégaint, Fabien Racapé, Simon Feltman}",
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
