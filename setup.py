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

import os

from pathlib import Path

from torch.utils.cpp_extension import BuildExtension, CppExtension

from setuptools import find_packages, setup

cwd = Path(__file__).resolve().parent
package_name = 'compressai'
version = '1.0.0'


def write_version_file():
    path = cwd / package_name / 'version.py'
    with path.open('w') as f:
        f.write(f'__version__ = "{version}"\n')


write_version_file()


def get_extensions():
    ext_dirs = cwd / package_name / 'cpp_exts'
    ext_modules = []

    # Add rANS module
    rans_lib_dir = cwd / 'third_party/ryg_rans'
    rans_ext_dir = ext_dirs / 'rans'

    extra_compile_args = ['-std=c++17']
    if os.getenv('DEBUG_BUILD', None):
        extra_compile_args += ['-O0', '-g', '-UNDEBUG']
    else:
        extra_compile_args += ['-O3']
    ext_modules.append(
        CppExtension(name=f'{package_name}.ans',
                     sources=[str(s) for s in rans_ext_dir.glob('*.cpp')],
                     language='c++',
                     include_dirs=[rans_lib_dir, rans_ext_dir],
                     extra_compile_args=extra_compile_args))

    # Add ops
    ops_ext_dir = ext_dirs / 'ops'
    ext_modules.append(
        CppExtension(name=f'{package_name}._CXX',
                     sources=[str(s) for s in ops_ext_dir.glob('*.cpp')],
                     language='c++',
                     extra_compile_args=extra_compile_args))

    return ext_modules


TEST_REQUIRES = ['pytest', 'pytest-cov']
DEV_REQUIRES = TEST_REQUIRES + ['pylint', 'yapf', 'sphinx']

def get_extra_requirements():
    extras_require = {
        'test': TEST_REQUIRES,
        'dev': DEV_REQUIRES,
        'tutorials': ['jupyter', 'ipywidgets'],
    }
    extras_require['all'] = set(req for reqs in extras_require.values()
                                for req in reqs)
    return extras_require


setup(
    name=package_name,
    version=version,
    description='A PyTorch library and evaluation platform for end-to-end compression research',
    packages=find_packages(exclude=('tests', )),
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'torch>=1.4.0',
        'torchvision>=0.5.0',
        'pytorch-msssim==0.2.0',
    ],
    extras_require=get_extra_requirements(),
    license='Apache-2',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension,
    },
)
