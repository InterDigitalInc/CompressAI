Train your own model
====================

In this tutorial we are going to implement a custom auto encoder architecture
by using some modules and layers pre-defined in CompressAI.

For a complete runnable example, check out the :code:`train.py` script in the
:code:`examples/` folder of the CompressAI source tree.


Defining a custom model
-----------------------

Let's build a simple auto encoder with an
:mod:`~compressai.entropy_models.EntropyBottleneck` module, 3 convolutions at
the encoder, 3 transposed deconvolutions for the decoder, and
:mod:`~compressai.layers.GDN` activation functions:

.. code-block:: python

   import torch.nn as nn

   from compressai.entropy_models import EntropyBottleneck
   from compressai.layers import GDN

   class Network(nn.Module):
       def __init__(self, N=128):
           super().__init__()
           self.entropy_bottleneck = EntropyBottleneck(N)
           self.encode = nn.Sequential(
               nn.Conv2d(3, N, stride=2, kernel_size=5, padding=2),
               GDN(N)
               nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
               GDN(N)
               nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
           )

           self.decode = nn.Sequential(
               nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2)
               GDN(N, inverse=True),
               nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2)
               GDN(N, inverse=True),
               nn.ConvTranspose2d(N, 3, kernel_size=5, padding=2, output_padding=1, stride=2)
           )

      def forward(self, x):
          y = self.encode(x)
          y_hat, y_likelihoods = self.entropy_bottleneck(y)
          x_hat = self.decode(y_hat)
          return x_hat, y_likelihoods


The convolutions are strided to reduce the spatial dimensions of the tensor,
while increasing the number of channels (which helps to learn better latent
representation). The bottleneck module is used to obtain a differentiable
entropy estimation of the latent tensors while training.

.. note::

   See the original paper: `"Variational image compression with a scale
   hyperprior" <https://arxiv.org/abs/1802.01436>`_, and the **tensorflow/compression**
   `documentation <https://github.com/tensorflow/compression/blob/v1.3/docs/entropy_bottleneck.md>`_
   for a detailed explanation of the EntropyBottleneck module.


Loss functions
--------------

1. Rate distortion loss
~~~~~~~~~~~~~~~~~~~~~~~

We are going to define a simple rate-distortion loss, which maximizes the
PSNR reconstruction (RGB) and minimizes the length (in bits) of the quantized
latent tensor (:code:`y_hat`).

A scalar is used to balance between the reconstruction quality and the
bit-rate (like the JPEG quality parameter, or the QP with HEVC):

.. math::

       \mathcal{L} = \mathcal{D} + \lambda * \mathcal{R}

.. code-block:: python

      import math
      import torch.nn as nn
      import torch.nn.functional as F

      x = torch.rand(1, 3, 64, 64)
      net = Network()
      x_hat, y_likelihoods = net(x)

      # bitrate of the quantized latent
      N, _, H, W = x.size()
      num_pixels = N * H * W
      bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)

      # mean square error
      mse_loss = F.mse_loss(x, x_hat)

      # final loss term
      loss = mse_loss + lmbda * bpp_loss


.. note::

    It's possible to train architectures that can handle multiple bit-rate
    distortion points but that's outside the scope of this tutorial. See this
    paper: `"Variable Rate Deep Image Compression With a Conditional Autoencoder"
    <http://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Variable_Rate_Deep_Image_Compression_With_a_Conditional_Autoencoder_ICCV_2019_paper.pdf>`_
    for a good example.


2. Auxiliary loss
~~~~~~~~~~~~~~~~~

The entropy bottleneck parameters need to be trained to minimize the density
model evaluation of the latent elements. The auxiliary loss is accessible
through the :code:`entropy_bottleneck` layer:

.. code-block:: python

    aux_loss = net.entropy_bottleneck.loss()

The auxiliary loss must be minimized during or after the training of the
network.


Optimizers
----------

To train both the compression network and the entropy bottleneck densities
estimation, we will thus need two optimizers. To simplify the implementation,
CompressAI provides a :mod:`~compressai.models.CompressionModel` base class,
that includes an :mod:`~compressai.entropy_models.EntropyBottleneck` module
and some helper methods, let's rewrite our network:

.. code-block:: python

     from compressai.models import CompressionModel
     from compressai.models.utils import conv, deconv

     class Network(CompressionModel):
         def __init__(self, N=128):
             super().__init__()
             self.encode = nn.Sequential(
                 conv(3, N),
                 GDN(N)
                 conv(N, N),
                 GDN(N)
                 conv(N, N),
             )

             self.decode = nn.Sequential(
                 deconv(N, N),
                 GDN(N, inverse=True),
                 deconv(N, N),
                 GDN(N, inverse=True),
                 deconv(N, 3),
             )

        def forward(self, x):
            y = self.encode(x)
            y_hat, y_likelihoods = self.entropy_bottleneck(y)
            x_hat = self.decode(y_hat)
            return x_hat, y_likelihoods


Now, we can simply access the two sets of trainable parameters:

.. code-block:: python

    import torch.optim as optim

    parameters = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
    aux_parameters = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
    optimizer = optim.Adam(parameters, lr=1e-4)
    aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)

.. note::

    You can also use :code:`torch.optim.Optimizer` `parameter groups <https://pytorch.org/docs/stable/optim.html#per-parameter-options>`_ to define a single optimizer.

Training loop
-------------

And write a training loop:

.. code-block:: python

    x = torch.rand(1, 3, 64, 64)
    for i in range(10):
      optimizer.zero_grad()
      aux_optimizer.zero_grad()

      x_hat, y_likelihoods = net(x)

      # ...
      # compute loss as before
      # ...

      loss.backward()
      optimizer.step()

      aux_loss = net.aux_loss()
      aux_loss.backward()
      aux_optimizer.step()
