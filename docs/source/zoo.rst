Image compression
=================

.. currentmodule:: compressai.zoo

This is the list of the pre-trained models for end-to-end image compression
available in CompressAI.

Currently, only models optimized w.r.t to the mean square error (*mse*) computed
on the RGB channels are available. We expect to release models fine-tuned with
other metrics in the future.

Training
~~~~~~~~

Unless specified otherwise, networks were trained for 4-5M steps on *256x256*
image patches randomly cropped and extracted from the `Vime90K
<http://toflow.csail.mit.edu/>`_ dataset [xue2019video]_.

Models are trained with a batch size of 16 or 32, and an initial learning rate
of 1e-4 for approximately 1-2M steps. The learning rate is then divided by 2
when the evaluation loss reaches a plateau (we use a patience of 20 epochs).

Training usually take between one or two weeks to reach state-of-the-art
performances, depending on the model, the number of channels and the GPU
architecture used.

....

Models
~~~~~~

bmshj2018_factorized
--------------------
Original paper: [bmshj2018]_

.. autofunction:: bmshj2018_factorized


bmshj2018_hyperprior
--------------------
Original paper: [bmshj2018]_

.. autofunction:: bmshj2018_hyperprior


mbt2018_mean
------------
Original paper: [mbt2018]_

.. autofunction:: mbt2018_mean


mbt2018
-------
Original paper: [mbt2018]_

.. autofunction:: mbt2018


....


.. rubric:: Citations

.. [bmshj2018]

    .. code-block:: bibtex

        @inproceedings{ballemshj18,
          author    = {Johannes Ball{\'{e}} and
                       David Minnen and
                       Saurabh Singh and
                       Sung Jin Hwang and
                       Nick Johnston},
          title     = {Variational image compression with a scale hyperprior},
          booktitle = {6th International Conference on Learning Representations, {ICLR} 2018,
                       Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings},
          publisher = {OpenReview.net},
          year      = {2018},
        }


.. [mbt2018]

    .. code-block:: bibtex

        @inproceedings{minnenbt18,
          author    = {David Minnen and
                       Johannes Ball{\'{e}} and
                       George Toderici},
          editor    = {Samy Bengio and
                       Hanna M. Wallach and
                       Hugo Larochelle and
                       Kristen Grauman and
                       Nicol{\`{o}} Cesa{-}Bianchi and
                       Roman Garnett},
          title     = {Joint Autoregressive and Hierarchical Priors for Learned Image Compression},
          booktitle = {Advances in Neural Information Processing Systems 31: Annual Conference
                       on Neural Information Processing Systems 2018, NeurIPS 2018, 3-8 December
                       2018, Montr{\'{e}}al, Canada},
          pages     = {10794--10803},
          year      = {2018},
        }


.. [xue2019video]
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
