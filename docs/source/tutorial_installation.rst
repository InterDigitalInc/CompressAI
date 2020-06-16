Installation
============

CompressAI only supports python3 . We also recommend to using a virtual 
environment to isolate project packages from the base system installation.


Requirements
~~~~~~~~~~~~

* python 3.6 or later (`python3-dev`, `python3-pip`, `python3-venv`)
* pip 19.0 or later
* a `C++17` compiler
* python packages: `numpy`, `scipy`, `torch`, `torchvision`


Virtual environment
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -m venv venv
   source ./venv/bin/activate


Using pip
---------

1. Clone the CompressAI repository:

.. code-block:: bash

   git clone https://github.com/InterDigitalInc/CompressAI compressai

2. Install CompressAI:

.. code-block:: bash

  pip install -e ./compressai


Building your own package
-------------------------

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
------

We are planning to provide docker images in the future.
