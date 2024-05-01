# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import subprocess

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

cwd = Path(__file__).resolve().parent

package_name = "compressai"
version = "1.2.6.dev0"
git_hash = "unknown"


try:
    git_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
    )
except (FileNotFoundError, subprocess.CalledProcessError):
    pass


def write_version_file():
    path = cwd / package_name / "version.py"
    with path.open("w") as f:
        f.write(f'__version__ = "{version}"\n')
        f.write(f'git_version = "{git_hash}"\n')


write_version_file()


def get_extensions():
    ext_dirs = cwd / package_name / "cpp_exts"
    ext_modules = []

    # Add rANS module
    rans_lib_dir = cwd / "third_party/ryg_rans"
    rans_ext_dir = ext_dirs / "rans"

    extra_compile_args = ["-std=c++17"]
    if os.getenv("DEBUG_BUILD", None):
        extra_compile_args += ["-O0", "-g", "-UNDEBUG"]
    else:
        extra_compile_args += ["-O3"]
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}.ans",
            sources=[str(s) for s in rans_ext_dir.glob("*.cpp")],
            language="c++",
            include_dirs=[rans_lib_dir, rans_ext_dir],
            extra_compile_args=extra_compile_args,
        )
    )

    # Add ops
    ops_ext_dir = ext_dirs / "ops"
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}._CXX",
            sources=[str(s) for s in ops_ext_dir.glob("*.cpp")],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    )

    return ext_modules


TEST_REQUIRES = ["pytest", "pytest-cov", "plotly"]
DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    "flake8",
    "flake8-bugbear",
    "flake8-comprehensions",
    "isort",
    "mypy",
]
POINTCLOUD_REQUIRES = [
    "pointops-yoda",
    "pyntcloud-yoda",  # Patched version of pyntcloud.
]


def get_extra_requirements():
    extras_require = {
        "test": TEST_REQUIRES,
        "dev": DEV_REQUIRES,
        "doc": ["sphinx", "sphinx-book-theme", "Jinja2<3.1"],
        "tutorials": ["jupyter", "ipywidgets"],
        "pointcloud": POINTCLOUD_REQUIRES,
    }
    extras_require["all"] = {req for reqs in extras_require.values() for req in reqs}
    return extras_require


setup(
    name=package_name,
    version=version,
    description="A PyTorch library and evaluation platform for end-to-end compression research",
    url="https://github.com/InterDigitalInc/CompressAI",
    author="InterDigital AI Lab",
    author_email="compressai@interdigital.com",
    packages=find_packages(exclude=("tests",)),
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "einops",
        "numpy>=1.21.0",
        "pandas",
        "scipy",
        "matplotlib",
        "torch>=1.7.1, <2.3",
        "torch-geometric>=2.3.0",
        "typing-extensions>=4.0.0",
        "torchvision",
        "pytorch-msssim",
        "tqdm",
    ],
    extras_require=get_extra_requirements(),
    license="BSD 3-Clause Clear License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": build_ext},
)
