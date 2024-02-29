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

from __future__ import annotations

import json
import os
import re
import shutil

from pathlib import Path

import numpy as np
import pandas as pd

from compressai.datasets.cache import CacheDataset
from compressai.datasets.utils import download_url, hash_file
from compressai.registry import register_dataset


@register_dataset("ShapeNetCorePartDataset")
class ShapeNetCorePartDataset(CacheDataset):
    """ShapeNet-Part dataset.

    The ShapeNet dataset of 3D CAD models of objects was introduced by
    [Yi2016]_, consisting of over 3000000 models.
    The ShapeNetCore (v2) dataset is a "clean" subset of ShapeNet,
    consisting of 51127 aligned items from 55 object categories.
    The ShapeNet-Part dataset is a further subset of this dataset,
    consisting of 16881 items from 16 object categories.
    See page 2 of [Yi2017]_ for additional description.

    Object categories are labeled with two to six segmentation parts
    each, as shown in the image below.
    (Purple represents a "miscellaneous" part.)

    .. image:: https://cs.stanford.edu/~ericyi/project_page/part_annotation/figures/categoriesNumbers.png

    [ProjectPage_ShapeNetPart]_ also releases a processed version of
    ShapeNet-Part containing point cloud and normals with
    expert-verified segmentations, which we use here.

    The ``semantic_index`` is a number between 0 and 49 (inclusive),
    which can be used as the semantic label for each point.

    See also: [PapersWithCode_ShapeNetPart]_ (benchmarks).

    References:

        .. [Yi2016] `"A scalable active framework for region annotation
            in 3D shape collections,"
            <https://dl.acm.org/doi/10.1145/2980179.2980238>`_,
            by Li Yi, Vladimir G. Kim, Duygu Ceylan, I-Chao Shen,
            Mengyan Yan, Hao Su, Cewu Lu, Qixing Huang, Alla Sheffer,
            and Leonidas Guibas, ACM Transactions on Graphics, 2016.

        .. [Yi2017] `"Large-scale 3D shape reconstruction and
            segmentation from ShapeNet Core55,"
            <https://arxiv.org/pdf/1710.06104.pdf>`_,
            by Li Yi et al. (total 50 authors), ICCV 2017.

        .. [ProjectPage_ShapeNetPart] `Project page (ShapeNet-Part)
            <https://cs.stanford.edu/~ericyi/project_page/part_annotation/>`_

        .. [PapersWithCode_ShapeNetPart] `PapersWithCode: ShapeNet-Part Benchmark
            (3D Part Segmentation)
            <https://paperswithcode.com/sota/3d-part-segmentation-on-shapenet-part>`_
    """

    URLS = {
        "shapenetcore_partanno_segmentation_benchmark_v0": "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip",
        "shapenetcore_partanno_segmentation_benchmark_v0_normal": "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip",
    }
    # Related: https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip

    HASHES = {
        "shapenetcore_partanno_segmentation_benchmark_v0": "f1dc7bad73237060946f13e1fa767b40d9adba52a79d42d64de31552b8c0b65e",
        "shapenetcore_partanno_segmentation_benchmark_v0_normal": "0e26411700bae2da38ee8ecc719ba4db2e6e0133486e258665952ad5dfced0fe",
    }

    CATEGORY_ID_TO_CATEGORY_STR = {
        "02691156": "Airplane",
        "02773838": "Bag",
        "02954340": "Cap",
        "02958343": "Car",
        "03001627": "Chair",
        "03261776": "Earphone",
        "03467517": "Guitar",
        "03624134": "Knife",
        "03636649": "Lamp",
        "03642806": "Laptop",
        "03790512": "Motorbike",
        "03797390": "Mug",
        "03948459": "Pistol",
        "04099429": "Rocket",
        "04225987": "Skateboard",
        "04379243": "Table",
    }

    NUM_PARTS = {
        "02691156": 4,  # Airplane
        "02773838": 2,  # Bag
        "02954340": 2,  # Cap
        "02958343": 4,  # Car
        "03001627": 4,  # Chair
        "03261776": 3,  # Earphone
        "03467517": 3,  # Guitar
        "03624134": 2,  # Knife
        "03636649": 4,  # Lamp
        "03642806": 2,  # Laptop
        "03790512": 6,  # Motorbike
        "03797390": 2,  # Mug
        "03948459": 3,  # Pistol
        "04099429": 3,  # Rocket
        "04225987": 3,  # Skateboard
        "04379243": 3,  # Table
    }

    def __init__(
        self,
        root=None,
        cache_root=None,
        split="train",
        split_name=None,
        pre_transform=None,
        transform=None,
        name="shapenetcore_partanno_segmentation_benchmark_v0_normal",
        download=True,
    ):
        if cache_root is None:
            assert root is not None
            cache_root = f"{str(root).rstrip('/')}_cache"

        self.root = Path(root) if root else None
        self.cache_root = Path(cache_root)
        self.split = split
        self.split_name = split if split_name is None else split_name
        self.name = name

        if download and self.root:
            self.download()

        self._verify_category_ids()

        self.category_id_info = {
            category_id: {
                "category_str": category_str,
                "category_index": category_index,
            }
            for category_index, (category_id, category_str) in enumerate(
                self.CATEGORY_ID_TO_CATEGORY_STR.items()
            )
        }

        self.category_offsets = np.cumsum([0] + list(self.NUM_PARTS.values()))

        super().__init__(
            cache_root=self.cache_root / self.split_name,
            pre_transform=pre_transform,
            transform=transform,
        )

        self._ensure_cache()

    def download(self, force=False):
        if not force and self.root.exists():
            return
        tmpdir = self.root.parent / "tmp"
        os.makedirs(tmpdir, exist_ok=True)
        filepath = download_url(
            self.URLS[self.name], tmpdir, check_certificate=False, overwrite=force
        )
        assert self.HASHES[self.name] == hash_file(filepath, method="sha256")
        shutil.unpack_archive(filepath, tmpdir)
        shutil.move(tmpdir / f"{self.name}", self.root)

    def _verify_category_ids(self):
        with open(self.root / "synsetoffset2category.txt") as f:
            pairs = [line.split() for line in f.readlines()]
        category_id_to_category_str = {
            category_id: category_str for category_str, category_id in pairs
        }
        assert category_id_to_category_str == self.CATEGORY_ID_TO_CATEGORY_STR

    def _get_items(self):
        file_list = f"shuffled_{self.split}_file_list.json"
        with open(self.root / "train_test_split" / file_list) as f:
            paths = json.load(f)
        return paths

    def _load_item(self, path):
        category_id, file_hash = self._parse_path(path)
        category_index = self.category_id_info[category_id]["category_index"]
        category_offset = self.category_offsets[category_index]
        read_csv_kwargs = {"sep": " ", "header": None, "index_col": False}

        if self.name == "shapenetcore_partanno_segmentation_benchmark_v0_normal":
            names = ["x", "y", "z", "nx", "ny", "nz", "semantic_index"]
            df = pd.read_csv(
                f"{self.root}/{category_id}/{file_hash}.txt",
                names=names,
                dtype={k: np.float32 for k in names},
                **read_csv_kwargs,
            )
            df["semantic_index"] = df["semantic_index"].astype(np.uint8)
            df["part_index"] = df["semantic_index"] - category_offset

        elif self.name == "shapenetcore_partanno_segmentation_benchmark_v0":
            df_points = pd.read_csv(
                f"{self.root}/{category_id}/points/{file_hash}.pts",
                names=["x", "y", "z"],
                dtype={k: np.float32 for k in ["x", "y", "z"]},
                **read_csv_kwargs,
            )
            df_points_label = pd.read_csv(
                f"{self.root}/{category_id}/points_label/{file_hash}.seg",
                names=["part_index"],
                dtype={"part_index": np.uint8},
                **read_csv_kwargs,
            )
            df = pd.concat([df_points, df_points_label], axis="columns")
            assert df["part_index"].min() >= 1
            df["part_index"] -= 1
            df["semantic_index"] = category_offset + df["part_index"]

        else:
            raise ValueError(f"Unknown name: {self.name}")

        data = {
            "category_index": np.array([category_index], dtype=np.uint8),
            "part_index": df["part_index"].values,
            "semantic_index": df["semantic_index"].values,
            "pos": df[["x", "y", "z"]].values,
        }

        if self.name == "shapenetcore_partanno_segmentation_benchmark_v0_normal":
            data["normal"] = df[["nx", "ny", "nz"]].values

        return data

    def _parse_path(self, path):
        pattern = r"^.*?/?(?P<category_id>\d+)/(?P<file_hash>[-a-fu\d]+)$"
        match = re.match(pattern, str(path))
        if match is None:
            raise ValueError(f"Could not parse path: {path}")
        category_id = match.group("category_id")
        file_hash = match.group("file_hash")
        return category_id, file_hash
