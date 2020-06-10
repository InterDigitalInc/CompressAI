Image compression
=================

This is the list of the pre-trained models available in CompressAI.

Training
~~~~~~~~

Unless specified otherwise, networks were the trained for 4-5M steps, on images 
extracted from the `Vime90K <http://toflow.csail.mit.edu/>`_ dataset 
[#fvimeo90lcitation]_.

Models are trained with a batch size of 16 or 32, and an initial learning rate
of 1e-4 for approximately 1-2M steps. The learning rate is then divided by 2 
when the evaluation loss reaches a plateau (we use a patience of 20 epochs).

Training usually take between one or two weeks, depending on the model, the
number of channels and the GPU architecture used.

.. rubric:: Footnotes

.. [#fvimeo90lcitation]
    Please cite the original paper if you use this dataset:

    .. code-block:: bibtex

        @article{xue2019video,
            title={Video Enhancement with Task-Oriented Flow},
            author={Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and 
            Freeman, William T},
            journal={International Journal of Computer Vision (IJCV)},
            volume={127},
            number={8},
            pages={1106--1125},
            year={2019},
            publisher={Springer}
        }


