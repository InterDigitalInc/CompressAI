Training
========


Defining a model
----------------


Loss functions
--------------


Updating the model
------------------

Once a model has been trained, you need to run the :code:`update_model` script
to update the internal parameters of the entropy bottlenecks:

.. code-block:: bash

   python -m compressai.utils.update_model -n final-model --arch ARCH model_checkpoint.pth.tar

This will modify the buffers related to the learned cumulative distribution
functions (CDFs) required to perform the actual entropy coding.

You can run :code:`python -m compressai.utils.update_model --help` to get the
complete list of options.

