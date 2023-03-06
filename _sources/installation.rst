Installation
============

CompressAI supports python 3.6+ and PyTorch 1.7+.

From PyPI
~~~~~~~~~~~

This is the recommended method to get started.

.. code-block:: bash

   pip install compressai


From source
~~~~~~~~~~~

We recommend to use a virtual environment to isolate project packages from the
base system installation.

Requirements
------------

* pip 19.0 or later
* a C++17 compiler (tested with `gcc` and `clang`)
* python packages: `numpy`, `scipy`, `torch`, `torchvision`


Virtual environment
-------------------

.. code-block:: bash

   python3 -m venv venv
   source ./venv/bin/activate
   pip install -U pip


Using pip
---------

1. Clone the CompressAI repository:

.. code-block:: bash

   git clone https://github.com/InterDigitalInc/CompressAI compressai

2. Install CompressAI:

.. code-block:: bash

  cd compressai
  pip install -e .

3. Custom installation

You can also run one of the following commands:

* :code:`pip install -e '.[dev]'`: install the packages required for development (testing, linting, docs)
* :code:`pip install -e '.[tutorials]'`: install the packages required for the tutorials (notebooks)
* :code:`pip install -e '.[all]'`: install all the optional packages


Build your own package
----------------------

You can also build your own pip package:

.. code-block:: bash

   git clone https://github.com/InterDigitalInc/CompressAI compressai
   cd compressai
   python3 setup.py bdist_wheel --dist-dir dist/
   pip install dist/compressai-*.whl

.. note::
   on MacOS you might want to use :code:`CC=clang CXX=clang++ pip install ...` to
   compile with clang instead of gcc.


Docker
~~~~~~

We are planning to publish docker images in the future.

For now, a Makefile is provided to build docker images locally.
Run :code:`make help` in the source code directory to list the available options.
