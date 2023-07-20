compressai.latent_codecs
========================

.. currentmodule:: compressai.latent_codecs


A :py:class:`~LatentCodec` is an abstraction for compressing a latent space using some entropy modeling technique.
A :py:class:`~LatentCodec` can be thought of as a miniature :py:class:`~compressai.models.base.CompressionModel`.
In fact, it implements some of the same methods: ``forward``, ``compress``, and ``decompress``, as described in :ref:`define-custom-latent-codec`.
By composing latent codecs, we can easily create more complex entropy models.

CompressAI provides the following predefined :py:class:`~LatentCodec` subclasses:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module name
     - Description
   * - :py:class:`~EntropyBottleneckLatentCodec`
     - Uses an :py:class:`~compressai.entropy_models.EntropyBottleneck` to encode ``y``.
   * - :py:class:`~GaussianConditionalLatentCodec`
     - Uses a :py:class:`~compressai.entropy_models.GaussianConditional` to encode ``y`` using ``(scale, mean)`` parameters.
   * - :py:class:`~HyperLatentCodec`
     - Uses an :py:class:`~compressai.entropy_models.EntropyBottleneck` to encode ``z``, with surrounding ``h_a`` and ``h_s`` transforms.
   * - :py:class:`~HyperpriorLatentCodec`
     - Uses an e.g. :py:class:`~GaussianConditionalLatentCodec` or :py:class:`~RasterScanLatentCodec` to encode ``y``, using ``(scale, mean)`` parameters generated from an e.g. :py:class:`~HyperLatentCodec`.
   * - :py:class:`~RasterScanLatentCodec`
     - Encodes ``y`` in raster-scan order using a PixelCNN-style autoregressive context model.
   * - :py:class:`~GainHyperLatentCodec`
     - Like :py:class:`~HyperLatentCodec`, but with trainable gain vectors for ``z``.
   * - :py:class:`~GainHyperpriorLatentCodec`
     - Like :py:class:`~HyperpriorLatentCodec`, but with trainable gain vectors for ``y``.
   * - :py:class:`~ChannelGroupsLatentCodec`
     - Encodes ``y`` in multiple chunked groups, each group conditioned on previously encoded groups.
   * - :py:class:`~CheckerboardLatentCodec`
     - Encodes ``y`` in two passes in checkerboard order.


Diagrams for some of the above predefined latent codecs:

.. code-block:: none

    HyperLatentCodec:

               ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
        y ──►──┤h_a├──►──┤ Q ├───►───····───►───┤h_s├──►── params
               └───┘     └───┘        EB        └───┘

        Entropy bottleneck codec with surrounding `h_a` and `h_s` transforms.

.. code-block:: none

    GaussianConditionalLatentCodec:

                          ctx_params
                              │
                              ▼
                              │
                           ┌──┴──┐
                           │  EP │
                           └──┬──┘
                              │
               ┌───┐  y_hat   ▼
        y ──►──┤ Q ├────►────····──►── y_hat
               └───┘          GC

        Gaussian conditional for compressing latent `y` using `ctx_params`.

.. code-block:: none

    HyperpriorLatentCodec:

                 ┌──────────┐
            ┌─►──┤ lc_hyper ├──►─┐
            │    └──────────┘    │
            │                    ▼ params
            │                    │
            │                 ┌──┴───┐
        y ──┴───────►─────────┤ lc_y ├───►── y_hat
                              └──────┘

        Composes a HyperLatentCodec and a "lc_y" latent codec such as
        GaussianConditionalLatentCodec or RasterScanLatentCodec.

.. code-block:: none

    RasterScanLatentCodec:

                         ctx_params
                             │
                             ▼
                             │ ┌───◄───┐
                           ┌─┴─┴─┐  ┌──┴──┐
                           │  EP │  │  CP │
                           └──┬──┘  └──┬──┘
                              │        │
                              │        ▲
               ┌───┐  y_hat   ▼        │
        y ──►──┤ Q ├────►────····───►──┴──►── y_hat
               └───┘          GC


Rationale
---------

This abstraction makes it easy to swap between different entropy models such as "factorized", "hyperprior", "raster scan autoregressive", "checkerboard", or "channel conditional groups".

It also aids in composition: we may now easily take any complicated composition of the above :py:class:`~LatentCodec` subclasses. For example, we may create models containing multiple hyperprior branches (`Hu et al., 2020`_), or a "channel conditional group" context model which encodes each group using "raster-scan" (`Minnen et al., 2020`_) or "checkerboard" (`He et al., 2022`_) autoregression, and so on.

Lastly, it reduces code duplication, and favors `composition instead of inheritance`_.

.. _Hu et al., 2020: https://huzi96.github.io/coarse-to-fine-compression.html
.. _Minnen et al., 2020: https://arxiv.org/abs/2007.08739
.. _He et al., 2022: https://arxiv.org/abs/2203.10886
.. _composition instead of inheritance: https://en.wikipedia.org/wiki/Composition_over_inheritance


Example models
--------------

A simple VAE model with an arbitrary latent codec can be implemented as follows:

.. code-block:: python

    class SimpleVAECompressionModel(CompressionModel):
        """Simple VAE model with arbitrary latent codec.

        .. code-block:: none

                   ┌───┐  y  ┌────┐ y_hat ┌───┐
            x ──►──┤g_a├──►──┤ lc ├───►───┤g_s├──►── x_hat
                   └───┘     └────┘       └───┘
        """

        g_a: nn.Module
        g_s: nn.Module
        latent_codec: LatentCodec

        def forward(self, x):
            y = self.g_a(x)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]
            x_hat = self.g_s(y_hat)
            return {
                "x_hat": x_hat,
                "likelihoods": y_out["likelihoods"],
            }

        def compress(self, x):
            y = self.g_a(x)
            outputs = self.latent_codec.compress(y)
            return outputs

        def decompress(self, strings, shape):
            y_out = self.latent_codec.decompress(strings, shape)
            y_hat = y_out["y_hat"]
            x_hat = self.g_s(y_hat).clamp_(0, 1)
            return {
                "x_hat": x_hat,
            }

This pattern is so common that CompressAI provides it via the import:

.. code-block:: python

    from compressai.models.base import SimpleVAECompressionModel

Using :py:class:`~compressai.models.base.SimpleVAECompressionModel`, some Google-style VAE models may be implemented as follows:

.. code-block:: python

    @register_model("bmshj2018-factorized")
    class FactorizedPrior(SimpleVAECompressionModel):
        def __init__(self, N, M, **kwargs):
            super().__init__(**kwargs)

            self.g_a = nn.Sequential(...)
            self.g_s = nn.Sequential(...)

            self.latent_codec = EntropyBottleneckLatentCodec(channels=M)


.. code-block:: python

    @register_model("mbt2018-mean")
    class MeanScaleHyperprior(SimpleVAECompressionModel):
        def __init__(self, N, M, **kwargs):
            super().__init__(**kwargs)

            self.g_a = nn.Sequential(...)
            self.g_s = nn.Sequential(...)
            h_a = nn.Sequential(...)
            h_s = nn.Sequential(...)

            self.latent_codec = HyperpriorLatentCodec(
                # A HyperpriorLatentCodec is made of "hyper" and "y" latent codecs.
                latent_codec={
                    # Side-information branch with entropy bottleneck for "z":
                    "hyper": HyperLatentCodec(
                        h_a=h_a,
                        h_s=h_s,
                        entropy_bottleneck=EntropyBottleneck(N),
                    ),
                    # Encode y using GaussianConditional:
                    "y": GaussianConditionalLatentCodec(),
                },
            )


.. code-block:: python

    @register_model("mbt2018")
    class JointAutoregressiveHierarchicalPriors(SimpleVAECompressionModel):
        def __init__(self, N, M, **kwargs):
            super().__init__(**kwargs)

            self.g_a = nn.Sequential(...)
            self.g_s = nn.Sequential(...)
            h_a = nn.Sequential(...)
            h_s = nn.Sequential(...)

            self.latent_codec = HyperpriorLatentCodec(
                # A HyperpriorLatentCodec is made of "hyper" and "y" latent codecs.
                latent_codec={
                    # Side-information branch with entropy bottleneck for "z":
                    "hyper": HyperLatentCodec(
                        h_a=h_a,
                        h_s=h_s,
                        entropy_bottleneck=EntropyBottleneck(N),
                    ),
                    # Encode y using autoregression in raster-scan order:
                    "y": RasterScanLatentCodec(
                        entropy_parameters=nn.Sequential(...),
                        context_prediction=MaskedConv2d(
                            M, M * 2, kernel_size=5, padding=2, stride=1
                        ),
                    ),
                },
            )


.. _define-custom-latent-codec:

Defining a custom latent codec
------------------------------

Latent codecs should inherit from the abstract base class :py:class:`~LatentCodec`, which is defined as:

.. code-block:: python

    class LatentCodec(nn.Module, _SetDefaultMixin):
        def forward(self, y: Tensor, *args, **kwargs) -> Dict[str, Any]:
            raise NotImplementedError

        def compress(self, y: Tensor, *args, **kwargs) -> Dict[str, Any]:
            raise NotImplementedError

        def decompress(
            self, strings: List[List[bytes]], shape: Any, *args, **kwargs
        ) -> Dict[str, Any]:
            raise NotImplementedError


Like :py:class:`~compressai.models.base.CompressionModel`, a subclass of :py:class:`~LatentCodec` should implement:

- ``forward``: differentiable function for training, returning a ``dict`` in the form of:

  .. code-block:: python

      {
          "likelihoods": {
              "y": y_likelihoods,
              ...
          },
          "y_hat": y_hat,
      }

- ``compress``: compressor to generate bitstreams from input tensor, returning a ``dict`` in the form of:

  .. code-block:: python

      {
          "strings": [y_strings, z_strings],
          "shape": ...,
      }

- ``decompress``: decompressor to reconstruct the input tensors using the bitstreams, returning a ``dict`` in the form of:

  .. code-block:: python

      {
          "y_hat": y_hat,
      }

Please refer to any of the predefined latent codecs for more concrete examples.


----


EntropyBottleneckLatentCodec
----------------------------
.. autoclass:: EntropyBottleneckLatentCodec


GaussianConditionalLatentCodec
------------------------------
.. autoclass:: GaussianConditionalLatentCodec


HyperLatentCodec
----------------
.. autoclass:: HyperLatentCodec


HyperpriorLatentCodec
---------------------
.. autoclass:: HyperpriorLatentCodec


RasterScanLatentCodec
---------------------
.. autoclass:: RasterScanLatentCodec


GainHyperLatentCodec
--------------------
.. autoclass:: GainHyperLatentCodec


GainHyperpriorLatentCodec
-------------------------
.. autoclass:: GainHyperpriorLatentCodec


ChannelGroupsLatentCodec
------------------------
.. autoclass:: ChannelGroupsLatentCodec


CheckerboardLatentCodec
-----------------------
.. autoclass:: CheckerboardLatentCodec

