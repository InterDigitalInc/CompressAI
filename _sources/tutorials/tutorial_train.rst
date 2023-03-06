Training
========

An example training script :code:`train.py` is provided script in the
:code:`examples/` folder of the CompressAI source tree.

Example:

.. code-block:: bash

    python3 examples/train.py -m mbt2018-mean -d /path/to/image/dataset \
        --batch-size 16 -lr 1e-4 --save --cuda

Run `train.py --help` to list the available options. See also the model zoo
:ref:`training <zoo-training>` section to reproduce the performances of the
pretrained models.

Model update
------------------

Once a model has been trained, you need to run the :code:`update_model` script
to update the internal parameters of the entropy bottlenecks:

.. code-block:: bash

   python -m compressai.utils.update_model --architecture ARCH checkpoint_best_loss.pth.tar

This will modify the buffers related to the learned cumulative distribution
functions (CDFs) required to perform the actual entropy coding.


You can run :code:`python -m compressai.utils.update_model --help` to get the
complete list of options.


Alternatively, you can call the :meth:`~compressai.models.CompressionModel.update`
method of a :mod:`~compressai.models.CompressionModel` or
:mod:`~compressai.entropy_models.EntropyBottleneck` instance at the end of your
training script, before saving the model checkpoint.

Model evaluation
--------------------

Once a model checkpoint has been updated, you can use :code:`eval_model` to get
its performances on an image dataset:

.. code-block:: bash

   python -m compressai.utils.eval_model checkpoint /path/to/image/dataset \
       -a ARCH -p path/to/checkpoint-xxxxxxxx.pth.tar

You can run :code:`python -m compressai.utils.eval_model --help` to get the
complete list of options.

Entropy coding
--------------

By default CompressAI uses a range Asymmetric Numeral Systems (ANS) entropy
coder. You can use :meth:`compressai.available_entropy_coders()` to get a list
of the implemented entropy coders and change the default entropy coder via
:meth:`compressai.set_entropy_coder()`.


1. Compress an image tensor to a bit-stream:

.. code-block:: python

    x = torch.rand(1, 3, 64, 64)
    y = net.encode(x)
    strings = net.entropy_bottleneck.compress(y)


2. Decompress a bit-stream to an image tensor:

.. code-block:: python

    shape = y.size()[2:]
    y_hat = net.entropy_bottleneck.decompress(strings, shape)
    x_hat = net.decode(y_hat)
