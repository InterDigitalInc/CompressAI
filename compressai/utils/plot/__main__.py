#- Copyright 2020 InterDigital Communications, Inc.
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
"""
Simple plotting utility to display Rate-Distortion curves (RD) comparison
between codecs.
"""
import argparse
import json
import sys

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_backends = ['matplotlib']

try:
    import plotly.graph_objs as go
    import plotly.offline
    _backends.append('plotly')
except ImportError:
    pass


def parse_json_file(filepath, metric):
    filepath = Path(filepath)
    name = filepath.name.split('.')[0]
    with filepath.open('r') as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as err:
            print(f'Error reading file "{filepath}"')
            raise err

    if 'results' not in data or \
            'bpp' not in data['results']:
        raise ValueError(f'Invalid file "{filepath}"')

    if metric not in data['results']:
        raise ValueError(
            f'Error: metric "{metric}" not available.'
            f' Available metrics: {", ".join(data["results"].keys())}')

    if metric == 'ms-ssim':
        # Convert to db
        values = np.array(data['results'][metric])
        data['results'][metric] = -10 * np.log10(1 - values)

    return {
        'name': data.get('name', name),
        'xs': data['results']['bpp'],
        'ys': data['results'][metric]
    }


def matplotlib_plt(scatters,
                   title,
                   ylabel,
                   output_file,
                   limits=None,
                   show=False,
                   figsize=None):
    if figsize is None:
        figsize = (9, 6)
    fig, ax = plt.subplots(figsize=figsize)
    for sc in scatters:
        ax.plot(sc['xs'], sc['ys'], '.-', label=sc['name'])

    ax.set_xlabel('Bit-rate [bpp]')
    ax.set_ylabel(ylabel)
    ax.grid()
    if limits is not None:
        ax.axis(limits)
    ax.legend(loc='lower right')

    if title:
        ax.title.set_text(title)

    if show:
        plt.show()

    if output_file:
        fig.savefig(output_file, dpi=300)


def plotly_plt(scatters,
               title,
               ylabel,
               output_file,
               limits=None,
               show=False,
               figsize=None):
    del figsize
    scatters = [
        go.Scatter(x=sc['xs'], y=sc['ys'], name=sc['name']) for sc in scatters
    ]
    plotly.offline.plot(
        {
            "data":
            scatters,
            "layout":
            go.Layout(
                title=title,
                legend={
                    'font': {
                        'size': 14,
                    },
                },
                xaxis={
                    'title': 'Bit-rate [bpp]',
                    'range': [limits[0], limits[1]]
                },
                yaxis={
                    'title': ylabel,
                    'range': [limits[2], limits[3]]
                },
            )
        },
        auto_open=show,
        filename=output_file or 'plot.html')


def setup_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f',
                        '--results-file',
                        metavar='',
                        default='',
                        type=str,
                        nargs='*',
                        required=True)
    parser.add_argument('-m',
                        '--metric',
                        metavar='',
                        type=str,
                        default='psnr',
                        help='Metric (default: %(default)s)')
    parser.add_argument('-t',
                        '--title',
                        metavar='',
                        type=str,
                        help='Plot title')
    parser.add_argument('-o',
                        '--output',
                        metavar='',
                        type=str,
                        help='Output file name')
    parser.add_argument(
        '--figsize',
        metavar='',
        type=float,
        nargs=2,
        default=(9, 6),
        help='Figure relative size (width, height), default: %(default)s')
    parser.add_argument(
        '--axes',
        metavar='',
        type=float,
        nargs=4,
        default=(0, 2, 28, 43),
        help='Axes limit (xmin, xmax, ymin, ymax), default: %(default)s')
    parser.add_argument('--backend',
                        type=str,
                        metavar='',
                        default=_backends[0],
                        choices=_backends,
                        help='Change plot backend (default: %(default)s)')
    parser.add_argument('--show',
                        action='store_true',
                        help='Open plot figure')
    return parser


def main(argv):
    args = setup_args().parse_args(argv)

    scatters = []
    for f in args.results_file:
        rv = parse_json_file(f, args.metric)
        scatters.append(rv)

    ylabel = args.metric
    if ylabel == 'psnr':
        ylabel = 'PSNR [dB]'
    func_map = {
        'matplotlib': matplotlib_plt,
        'plotly': plotly_plt,
    }
    func_map[args.backend](scatters,
                           args.title,
                           ylabel,
                           args.output,
                           limits=args.axes,
                           figsize=args.figsize,
                           show=args.show)


if __name__ == '__main__':
    main(sys.argv[1:])
