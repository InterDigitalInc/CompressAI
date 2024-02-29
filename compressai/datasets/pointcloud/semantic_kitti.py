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
import re
import shutil

from pathlib import Path

import numpy as np

from compressai.datasets.cache import CacheDataset
from compressai.datasets.utils import download_url, hash_file
from compressai.registry import register_dataset


@register_dataset("SemanticKittiDataset")
class SemanticKittiDataset(CacheDataset):
    """SemanticKITTI dataset.

    The KITTI dataset, introduced by [Geiger2012]_, contains 3D point
    clouds sequences (i.e. video) of LiDAR sensor data from the
    perspective of a driving vehicle.
    The SemanticKITTI dataset, introduced by [Behley2019]_ and
    [Behley2021]_, provides semantic annotation of all 22 sequences from
    the odometry task [Odometry_KITTI]_ of KITTI.
    See the [ProjectPage_SemanticKITTI]_ for a visualization.
    Note that the test set is unlabelled, and must be evaluated on the
    server, as mentioned at [ProjectPageTasks_SemanticKITTI]_.

    The ``semantic_index`` is a number between 0 and 33 (inclusive),
    which can be used as the semantic label for each point.

    See also: [PapersWithCode_SemanticKITTI]_.

    References:

        .. [Geiger2012] `"Are we ready for Autonomous Driving? The KITTI
            Vision Benchmark Suite,"
            <https://www.cvlibs.net/publications/Geiger2012CVPR.pdf>`_,
            by Andreas Geiger, Philip Lenz, and Raquel Urtasun,
            CVPR 2012.

        .. [Behley2019] `"SemanticKITTI: A Dataset for Semantic Scene
            Understanding of LiDAR Sequences,"
            <https://arxiv.org/abs/1904.01416>`_,
            by Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel,
            Sven Behnke, Cyrill Stachniss, and Juergen Gall, ICCV 2019.

        .. [Behley2021] `"Towards 3D LiDAR-based semantic scene
            understanding of 3D point cloud sequences: The SemanticKITTI
            Dataset,"
            <https://journals.sagepub.com/doi/10.1177/02783649211006735>`_,
            by Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel,
            Sven Behnke, Jürgen Gall, and Cyrill Stachniss, IJRR 2021.

        .. [ProjectPage_SemanticKITTI] `Project page (SemanticKITTI)
            <http://www.semantic-kitti.org/>`_

        .. [ProjectPageTasks_SemanticKITTI] `Project page: Tasks
            (SemanticKITTI)
            <http://www.semantic-kitti.org/tasks.html>`_

        .. [Odometry_KITTI] `"Visual Odometry / SLAM Evaluation 2012"
            <https://www.cvlibs.net/datasets/kitti/eval_odometry.php>`_

        .. [PapersWithCode_SemanticKITTI] `PapersWithCode: SemanticKITTI
            <https://paperswithcode.com/dataset/semantickitti>`_
    """

    URLS = [
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip",
        "http://www.semantic-kitti.org/assets/data_odometry_labels.zip",
        "http://www.semantic-kitti.org/assets/data_odometry_voxels_all.zip",
        "http://www.semantic-kitti.org/assets/data_odometry_voxels.zip",
    ]

    HASHES = [
        "fa45d2bbff828776e6df689b161415fb7cd719345454b6d3567c2ff81fa4d075",  # data_odometry_calib.zip
        "062a45667bec6874ac27f733bd6809919f077265e7ac0bb25ac885798fa85ab5",  # data_odometry_velodyne.zip
        "408ec524636a393bae0288a0b2f48bf5418a1af988e82dee8496f89ddb7e6dda",  # data_odometry_labels.zip
        "10f333faa63426a519a573fbf0b4e3b56513511af30583473fa6a5782e037f3a",  # data_odometry_voxels_all.zip
        "d92c253e88e5e30c0a0b88f028510760e1db83b7e262d75c5931bf9b8d6dd51b",  # data_odometry_voxels.zip
    ]

    # Suggested splits:
    SEQUENCES = {
        "train": (0, 1, 2, 3, 4, 5, 6, 7, 9, 10),
        "valid": (8,),
        "infer": (8,),
        "test": (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),  # Unlabelled.
    }

    # fmt: off
    NUM_SAMPLES_PER_SEQUENCE = [
        4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201,
        921, 1061, 3281, 631, 1901, 1731, 491, 1801, 4981, 831, 2721
    ]
    # fmt: on

    RAW_SEMANTIC_INDEX_TO_LABEL = {
        0: "unlabeled",
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }

    RAW_SEMANTIC_INDEX_TO_SEMANTIC_INDEX = {
        idx: i for i, idx in enumerate(RAW_SEMANTIC_INDEX_TO_LABEL)
    }

    def __init__(
        self,
        root=None,
        cache_root=None,
        split="train",
        split_name=None,
        sequences=SEQUENCES["train"],
        pre_transform=None,
        transform=None,
        download=True,
    ):
        if cache_root is None:
            assert root is not None
            cache_root = f"{str(root).rstrip('/')}_cache"

        self.root = Path(root) if root else None
        self.cache_root = Path(cache_root)
        self.split = split
        self.split_name = split if split_name is None else split_name
        self.sequences = sequences

        if download and self.root:
            self.download()

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
        for expected_hash, url in zip(self.HASHES, self.URLS):
            filepath = download_url(
                url, tmpdir, check_certificate=False, overwrite=force
            )
            shutil.unpack_archive(filepath, tmpdir)
            assert expected_hash == hash_file(filepath, method="sha256")
        shutil.move(tmpdir / "dataset", self.root)

    def _get_items(self):
        return sorted(
            x
            for i in self.sequences
            for x in self.root.glob(f"**/{i:02}/velodyne/*.bin")
        )

    def _load_item(self, path):
        path_prefix, sequence_index, file_index = self._parse_path(path)
        assert str(path) == f"{path_prefix}{sequence_index}/velodyne/{file_index}.bin"
        point_data = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        label_data = (
            np.fromfile(
                f"{path_prefix}{sequence_index}/labels/{file_index}.label", dtype="<u2"
            )
            .reshape(-1, 2)
            .astype(np.int16)
        )

        return {
            "file_index": np.array([file_index], dtype=np.int32),
            "sequence_index": np.array([sequence_index], dtype=np.int32),
            "raw_semantic_index": label_data[:, 0],
            "semantic_index": np_remap(
                label_data[:, 0], self.RAW_SEMANTIC_INDEX_TO_SEMANTIC_INDEX
            ),
            "instance_index": label_data[:, 1],
            "pos": point_data[:, :3],
            "remission": point_data[:, 3, None],
        }

    def _parse_path(self, path):
        pattern = (
            r"^(?P<path_prefix>.*?/?)"
            r"(?P<sequence_index>\d+)/"
            r"velodyne/"
            r"(?P<file_index>\d{6})\.\w+$"
        )
        match = re.match(pattern, str(path))
        if match is None:
            raise ValueError(f"Could not parse path: {path}")
        path_prefix = match.group("path_prefix")
        sequence_index = match.group("sequence_index")
        file_index = match.group("file_index")
        return path_prefix, sequence_index, file_index


def np_remap(arr, d):
    values, inverse = np.unique(arr, return_inverse=True)
    values = np.array([d[x] for x in values], dtype=arr.dtype)
    return values[inverse].reshape(arr.shape)
