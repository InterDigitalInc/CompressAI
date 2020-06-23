# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.hub import load_state_dict_from_url

from compressai.models import (FactorizedPrior, ScaleHyperprior,
                               MeanScaleHyperprior,
                               JointAutoregressiveHierarchicalPriors,
                               Cheng2020Anchor, Cheng2020Attention)

__all__ = [
    'bmshj2018_factorized',
    'bmshj2018_hyperprior',
    'mbt2018',
    'mbt2018_mean',
    'cheng2020_anchor',
    'cheng2020_attn',
]

model_architectures = {
    'bmshj2018-factorized': FactorizedPrior,
    'bmshj2018-hyperprior': ScaleHyperprior,
    'mbt2018-mean': MeanScaleHyperprior,
    'mbt2018': JointAutoregressiveHierarchicalPriors,
    'cheng2020-anchor': Cheng2020Anchor,
    'cheng2020-attn': Cheng2020Attention,
}

root_url = 'https://compressai.s3.amazonaws.com/models/v1'
model_urls = {
    'bmshj2018-factorized': {
        'mse': {
            1: f'{root_url}/bmshj2018-factorized-prior-1-446d5c7f.pth.tar',
            2: f'{root_url}/bmshj2018-factorized-prior-2-87279a02.pth.tar',
            3: f'{root_url}/bmshj2018-factorized-prior-3-5c6f152b.pth.tar',
            4: f'{root_url}/bmshj2018-factorized-prior-4-1ed4405a.pth.tar',
            5: f'{root_url}/bmshj2018-factorized-prior-5-866ba797.pth.tar',
            6: f'{root_url}/bmshj2018-factorized-prior-6-9b02ea3a.pth.tar',
            7: f'{root_url}/bmshj2018-factorized-prior-7-6dfd6734.pth.tar',
            8: f'{root_url}/bmshj2018-factorized-prior-8-5232faa3.pth.tar',
        },
    },
    'bmshj2018-hyperprior': {
        'mse': {
            1: f'{root_url}/bmshj2018-hyperprior-1-7eb97409.pth.tar',
            2: f'{root_url}/bmshj2018-hyperprior-2-93677231.pth.tar',
            3: f'{root_url}/bmshj2018-hyperprior-3-6d87be32.pth.tar',
            4: f'{root_url}/bmshj2018-hyperprior-4-de1b779c.pth.tar',
            5: f'{root_url}/bmshj2018-hyperprior-5-f8b614e1.pth.tar',
            6: f'{root_url}/bmshj2018-hyperprior-6-1ab9c41e.pth.tar',
            7: f'{root_url}/bmshj2018-hyperprior-7-3804dcbd.pth.tar',
            8: f'{root_url}/bmshj2018-hyperprior-8-a583f0cf.pth.tar',
        },
    },
    'mbt2018-mean': {
        'mse': {
            1: f'{root_url}/mbt2018-mean-1-e522738d.pth.tar',
            2: f'{root_url}/mbt2018-mean-2-e54a039d.pth.tar',
            3: f'{root_url}/mbt2018-mean-3-723404a8.pth.tar',
            4: f'{root_url}/mbt2018-mean-4-6dba02a3.pth.tar',
            5: f'{root_url}/mbt2018-mean-5-d504e8eb.pth.tar',
            6: f'{root_url}/mbt2018-mean-6-a19628ab.pth.tar',
            7: f'{root_url}/mbt2018-mean-7-d5d441d1.pth.tar',
            8: f'{root_url}/mbt2018-mean-8-8089ae3e.pth.tar',
        },
    },
    'mbt2018': {
        'mse': {
            1: f'{root_url}/mbt2018-1-3f36cd77.pth.tar',
            2: f'{root_url}/mbt2018-2-43b70cdd.pth.tar',
            3: f'{root_url}/mbt2018-3-22901978.pth.tar',
            4: f'{root_url}/mbt2018-4-456e2af9.pth.tar',
            5: f'{root_url}/mbt2018-5-b4a046dd.pth.tar',
            6: f'{root_url}/mbt2018-6-7052e5ea.pth.tar',
            7: f'{root_url}/mbt2018-7-8ba2bf82.pth.tar',
            8: f'{root_url}/mbt2018-8-dd0097aa.pth.tar',
        },
    },
    'cheng2020-anchor': {
        'mse': {},
    },
    'cheng2020-attn': {
        'mse': {},
    },
}

cfgs = {
    'bmshj2018-factorized': {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    'bmshj2018-hyperprior': {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    'mbt2018-mean': {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (192, 320),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    'mbt2018': {
        1: (192, 192),
        2: (192, 192),
        3: (192, 192),
        4: (192, 192),
        5: (192, 320),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    'cheng2020-anchor': {
        1: (128, ),
        2: (128, ),
        3: (128, ),
        4: (192, ),
        5: (192, ),
        6: (192, ),
    },
    'cheng2020-attn': {
        1: (128, ),
        2: (128, ),
        3: (128, ),
        4: (192, ),
        5: (192, ),
        6: (192, ),
    },
}


def _load_model(architecture,
                metric,
                quality,
                pretrained=False,
                progress=True,
                **kwargs):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if quality not in cfgs[architecture]:
        raise ValueError(f'Invalid quality value "{quality}"')

    if pretrained:
        if architecture not in model_urls or \
                metric not in model_urls[architecture] or \
                quality not in model_urls[architecture][metric]:
            raise RuntimeError('Pre-trained model not yet available')

        url = model_urls[architecture][metric][quality]
        state_dict = load_state_dict_from_url(url, progress=progress)
        model = model_architectures[architecture].from_state_dict(state_dict)
        return model

    model = model_architectures[architecture](*cfgs[architecture][quality],
                                              **kwargs)
    return model


def bmshj2018_factorized(quality,
                         metric='mse',
                         pretrained=False,
                         progress=True,
                         **kwargs):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ('mse', ):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model('bmshj2018-factorized', metric, quality, pretrained,
                       progress, **kwargs)


def bmshj2018_hyperprior(quality,
                         metric='mse',
                         pretrained=False,
                         progress=True,
                         **kwargs):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ('mse', ):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model('bmshj2018-hyperprior', metric, quality, pretrained,
                       progress, **kwargs)


def mbt2018_mean(quality,
                 metric='mse',
                 pretrained=False,
                 progress=True,
                 **kwargs):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ('mse', ):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model('mbt2018-mean', metric, quality, pretrained, progress,
                       **kwargs)


def mbt2018(quality, metric='mse', pretrained=False, progress=True, **kwargs):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ('mse', ):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model('mbt2018', metric, quality, pretrained, progress,
                       **kwargs)


def cheng2020_anchor(quality,
                     metric='mse',
                     pretrained=False,
                     progress=True,
                     **kwargs):
    r"""Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ('mse', ):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 6)')

    return _load_model('cheng2020-anchor', metric, quality, pretrained,
                       progress, **kwargs)


def cheng2020_attn(quality,
                   metric='mse',
                   pretrained=False,
                   progress=True,
                   **kwargs):
    r"""Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ('mse', ):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(
            f'Invalid quality "{quality}", should be between (1, 6)')

    return _load_model('cheng2020-attn', metric, quality, pretrained, progress,
                       **kwargs)
