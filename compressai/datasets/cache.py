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

import json
import os
import os.path

from pathlib import Path

import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm


class CacheDataset(Dataset):
    def __init__(
        self,
        cache_root=None,
        pre_transform=None,
        transform=None,
    ):
        self.__cache_root = Path(cache_root)
        self.pre_transform = pre_transform
        self.transform = transform
        self._store = {}

    def __len__(self):
        return len(self._store[next(iter(self._store))])

    def __getitem__(self, index):
        data = {k: v[index].copy() for k, v in self._store.items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

    def _ensure_cache(self):
        try:
            self._load_cache(mode="r")
        except FileNotFoundError:
            self._generate_cache()
            self._load_cache(mode="r")

    def _load_cache(self, mode):
        with open(self.__cache_root / "info.json", "r") as f:
            info = json.load(f)

        self._store = {
            k: np.memmap(
                self.__cache_root / f"{k}.npy",
                mode=mode,
                dtype=settings["dtype"],
                shape=tuple(settings["shape"]),
            )
            for k, settings in info.items()
        }

    def _generate_cache(self, verbose=True):
        if verbose:
            print(f"Generating cache at {self.__cache_root}...")

        items = self._get_items()

        if verbose:
            items = tqdm(items)

        for i, item in enumerate(items):
            data = self._load_item(item)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if not self._store:
                self._write_cache_info(len(items), data)
                self._load_cache(mode="w+")

            for k, v in data.items():
                self._store[k][i] = v

    def _write_cache_info(self, num_samples, data):
        info = {
            k: {
                "dtype": _removeprefix(str(v.dtype), "torch."),
                "shape": (num_samples, *v.shape),
            }
            for k, v in data.items()
        }
        os.makedirs(self.__cache_root, exist_ok=True)
        with open(self.__cache_root / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def _get_items(self):
        raise NotImplementedError

    def _load_item(self, item):
        raise NotImplementedError


def _removeprefix(s: str, prefix: str) -> str:
    return s[len(prefix) :] if s.startswith(prefix) else s
