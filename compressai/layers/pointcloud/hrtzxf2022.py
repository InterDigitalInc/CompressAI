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

# Code adapted from https://github.com/yunhe20/D-PCC

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from pointops.functions import pointops
except ImportError:
    pass  # NOTE: Optional dependency.

from .utils import index_points


class DownsampleLayer(nn.Module):
    """Downsampling layer used in [He2022pcc]_.

    Downsamples positions into a smaller number of centroids.
    Each centroid is grouped with nearby points,
    and the local point density is estimated for that group.
    Then, the positions, features, and density for the group
    are embedded into a single aggregate vector from which the
    group of points may later be reconstructed.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, downsample_rate, dim, hidden_dim, k, ngroups):
        super().__init__()
        self.k = k
        self.downsample_rate = downsample_rate
        self.pre_conv = nn.Conv1d(dim, dim, 1)
        self.embed_features = PointTransformerLayer(dim, dim, hidden_dim, ngroups)
        self.embed_positions = PositionEmbeddingLayer(hidden_dim, dim, ngroups)
        self.embed_densities = DensityEmbeddingLayer(hidden_dim, dim, ngroups)
        self.post_conv = nn.Conv1d(dim * 3, dim, 1)

    def get_density(self, downsampled_xyzs, input_xyzs):
        # downsampled_xyzs: (b, 3, m)
        # input_xyzs: (b, 3, n)
        # nn_idx: (b, n, 1)
        # downsample_num: (b, m)
        # knn_idx: (b, m, k)
        # mask: (b, m, k)
        # distance: (b, m)
        # mean_distance: (b, m)
        _, _, n = input_xyzs.shape
        distance, mask, knn_idx, _ = nearby_distance_sum(
            downsampled_xyzs, input_xyzs, min(self.k, n)
        )
        downsample_num = mask.sum(dim=-1).float()
        mean_distance = distance / downsample_num
        return downsample_num, mean_distance, mask, knn_idx

    def forward(self, xyzs, feats):
        # xyzs: (b, 3, n)
        # features: (b, cin, n)
        # sample_idx: (b, m)
        # sampled_xyzs: (b, 3, m)
        # sampled_feats: (b, c, m)

        # Downsample positions into a smaller number of centroids.
        sampled_xyzs, sample_idx = self.downsample_positions(xyzs, feats)

        # For each centroid, form a group with nearby points.
        # Also, estimate local point density ("mean distance") for each centroid.
        downsample_num, mean_distance, mask, knn_idx = self.get_density(
            sampled_xyzs, xyzs
        )

        # Embed features, positions, and density for each downsampled
        # point group into a single aggregate vector for that group.
        sampled_feats = self.downsample_features(
            sampled_xyzs, xyzs, feats, downsample_num, sample_idx, knn_idx, mask
        )

        return sampled_xyzs, sampled_feats, downsample_num, mean_distance

    def downsample_positions(self, xyzs, sample_num):
        _, _, n = xyzs.shape
        sample_num = round(n * self.downsample_rate)
        xyzs_tr = xyzs.permute(0, 2, 1).contiguous()
        sample_idx = pointops.furthestsampling(xyzs_tr, sample_num).long()
        sampled_xyzs = index_points(xyzs, sample_idx)
        return sampled_xyzs, sample_idx

    def downsample_features(
        self, sampled_xyzs, xyzs, feats, downsample_num, sample_idx, knn_idx, mask
    ):
        # sampled_xyzs: (b, 3, m)
        # sampled_feats: (b, c, m)

        identity = index_points(feats, sample_idx)

        feats = self.pre_conv(feats)
        sampled_feats = index_points(feats, sample_idx)
        embeddings = [
            self.embed_features(
                sampled_xyzs, xyzs, sampled_feats, feats, feats, knn_idx, mask
            ),
            self.embed_positions(sampled_xyzs, xyzs, knn_idx, mask),
            self.embed_densities(downsample_num.unsqueeze(1)),
        ]
        agg_embedding = self.post_conv(torch.cat(embeddings, dim=1))

        sampled_feats_new = agg_embedding + identity
        return sampled_feats_new


class PointTransformerLayer(nn.Module):
    """Point Transformer layer introduced by [Zhao2021]_.

    References:

        .. [Zhao2021] `"Point Transformer"
            <https://arxiv.org/abs/2012.09164>`_, by Hengshuang Zhao,
            Li Jiang, Jiaya Jia, Philip Torr, and Vladlen Koltun,
            CVPR 2021.
    """

    def __init__(self, in_fdim, out_fdim, hidden_dim, ngroups):
        super().__init__()

        self.w_qs = nn.Conv1d(in_fdim, hidden_dim, 1)
        self.w_ks = nn.Conv1d(in_fdim, hidden_dim, 1)
        self.w_vs = nn.Conv1d(in_fdim, hidden_dim, 1)

        self.conv_delta = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 1),
            nn.GroupNorm(ngroups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )

        self.conv_gamma = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GroupNorm(ngroups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )

        self.post_conv = nn.Conv1d(hidden_dim, out_fdim, 1)

    def forward(self, q_xyzs, k_xyzs, q_feats, k_feats, v_feats, knn_idx, mask):
        # q: (b, c, m)
        # k: (b, c, n)
        # knn_idx: (b, m, k)
        # mask: (b, m, k)
        # knn_xyzs: (b, 3, m, k)
        # query: (b, c, m)
        # key: (b, c, m, k)
        # pos_enc: (b, c, m, k)
        # attn: (b, c, m, k)

        knn_xyzs = index_points(k_xyzs, knn_idx)

        # NOTE: it's q_feats, not v_feats!
        identity = q_feats

        query = self.w_qs(q_feats)
        key = index_points(self.w_ks(k_feats), knn_idx)
        value = index_points(self.w_vs(v_feats), knn_idx)

        pos_enc = self.conv_delta(q_xyzs.unsqueeze(-1) - knn_xyzs)

        attn = self.conv_gamma(query.unsqueeze(-1) - key + pos_enc)
        attn = attn / math.sqrt(key.shape[1])
        mask_value = -(torch.finfo(attn.dtype).max)
        attn.masked_fill_(~mask[:, None], mask_value)
        attn = F.softmax(attn, dim=-1)

        result = torch.einsum("bcmk, bcmk -> bcm", attn, value + pos_enc)
        result = self.post_conv(result) + identity

        return result


class PositionEmbeddingLayer(nn.Module):
    """Position embedding for downsampling, as introduced in [He2022pcc]_.

    For each group of feature vectors (f₁, ..., fₖ) with centroid fₒ,
    represents the offsets (f₁ - fₒ, ..., fₖ - fₒ) as
    magnitude-direction vectors, then applies an MLP to each vector,
    then takes a softmax self-attention over the resulting vectors,
    and finally reduces the vectors via a sum,
    resulting in a single embedded vector for the group.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, hidden_dim, dim, ngroups):
        super().__init__()

        self.embed_positions = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 1),
            nn.GroupNorm(ngroups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GroupNorm(ngroups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1),
        )

    def forward(self, q_xyzs, k_xyzs, knn_idx, mask):
        # q_xyzs: (b, 3, m)
        # k_xyzs: (b, 3, n)
        # knn_idx: (b, m, k)
        # mask: (b, m, k)
        # knn_xyzs: (b, 3, m, k)
        # repeated_xyzs: (b, 3, m, k)
        # direction: (b, 3, m, k)
        # distance: (b, 1, m, k)
        # local_pattern: (b, 4, m, k)
        # position_embedding_expanded: (b, c, m, k)
        # attn: (b, c, m, k)
        # position_embedding: (b, c, m)

        # "query" (q_xyzs) points are the centroids of each point group.
        # "key" (k_xyzs) points are points in the neighborhood of each centroid.

        _, _, k = knn_idx.shape
        knn_xyzs = index_points(k_xyzs, knn_idx)
        repeated_xyzs = q_xyzs[..., None].repeat(1, 1, 1, k)

        # Represent points within a point group as (direction, distance)
        # of offsets from the group centroid.
        offset_xyzs = knn_xyzs - repeated_xyzs
        direction = F.normalize(offset_xyzs, p=2, dim=1)
        distance = torch.linalg.norm(offset_xyzs, dim=1, keepdim=True)
        local_pattern = torch.cat((direction, distance), dim=1)

        # Apply a pointwise MLP to each point.
        position_embedding_expanded = self.embed_positions(local_pattern)

        # Compute self-attention, ignoring points that are not in the
        # neighborhood of the centroid.
        attn = self.attention(position_embedding_expanded)
        mask_value = -(torch.finfo(attn.dtype).max)
        attn.masked_fill_(~mask[:, None], mask_value)
        attn = F.softmax(attn, dim=-1)
        position_embedding = (position_embedding_expanded * attn).sum(dim=-1)

        return position_embedding


class DensityEmbeddingLayer(nn.Module):
    """Density embedding for downsampling, as introduced in [He2022pcc]_.

    Applies an embedding ℝ → ℝᶜ to the local point density (scalar).
    The local point density is measured using the mean distance of the
    points within the neighborhood of a "downsampled" centroid.
    This information is useful when upsampling from the single centroid.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, hidden_dim, dim, ngroups):
        super().__init__()
        self.embed_densities = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 1),
            nn.GroupNorm(ngroups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, dim, 1),
        )

    def forward(self, downsample_num):
        # downsample_num: (b, 1, n)
        # density_embedding: (b, c, n)
        density_embedding = self.embed_densities(downsample_num)
        return density_embedding


class UpsampleLayer(nn.Module):
    """Upsampling layer used in [He2022pcc]_.

    Upsamples many candidate points from a smaller number of centroids.
    (Not all candidate upsampled points will be kept; some will be
    thrown away to match the predicted local point density.)

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, dim, hidden_dim, k, sub_point_conv_mode, upsample_rate):
        super().__init__()
        self.xyzs_upsample_nn = XyzsUpsampleLayer(
            dim, hidden_dim, k, sub_point_conv_mode, upsample_rate
        )
        self.feats_upsample_nn = FeatsUpsampleLayer(
            dim, hidden_dim, k, sub_point_conv_mode, upsample_rate
        )

    def forward(self, xyzs, feats):
        upsampled_xyzs = self.xyzs_upsample_nn(xyzs, feats)
        upsampled_feats = self.feats_upsample_nn(feats)
        return upsampled_xyzs, upsampled_feats


class UpsampleNumLayer(nn.Module):
    """Predicts local point density while upsampling, as used in [He2022pcc]_.

    Extracts the number of candidate points to keep after upsampling
    from a given "centroid" feature vector.
    (Some candidate upsampled points will be thrown away to match the
    predicted local point density.)

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, dim, hidden_dim, upsample_rate):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.upsample_num_nn = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feats):
        # upsample_num: (b, n)
        upsample_frac = self.upsample_num_nn(feats).squeeze(1)
        upsample_num = upsample_frac * (self.upsample_rate - 1) + 1
        return upsample_num


class RefineLayer(nn.Module):
    """Refines upsampled points, as used in [He2022pcc]_.

    After the centroids are upsampled, there may be overlapping
    point groups between nearby centroids, and other artifacts.
    Refinement should help correct various such artifacts.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, dim, hidden_dim, k, sub_point_conv_mode, decompress_normal):
        super().__init__()

        self.xyzs_refine_nn = XyzsUpsampleLayer(
            dim,
            hidden_dim,
            k,
            sub_point_conv_mode,
            upsample_rate=1,
        )

        self.feats_refine_nn = FeatsUpsampleLayer(
            dim,
            hidden_dim,
            k,
            sub_point_conv_mode,
            upsample_rate=1,
            decompress_normal=decompress_normal,
        )

    def forward(self, xyzs, feats):
        # refined_xyzs: (b, 3, n, 1)
        # refined_xyzs: (b, 3, n)  [after rearrange]
        # refined_feats: (b, c, n, 1)
        # refined_feats: (b, c, n)  [after rearrange]

        refined_xyzs = self.xyzs_refine_nn(xyzs, feats)
        refined_xyzs = rearrange(refined_xyzs, "b c n u -> b c (n u)")

        refined_feats = self.feats_refine_nn(feats)
        refined_feats = rearrange(refined_feats, "b c n u -> b c (n u)")

        return refined_xyzs, refined_feats


class XyzsUpsampleLayer(nn.Module):
    """Position upsampling layer used in [He2022pcc]_.

    Upsamples many positions from each "centroid" feature vector.
    Each feature vector is upsampled into various offsets represented as
    magnitude-direction vectors, where each direction is determined by a
    weighted sum of various fixed hypothesized directions.
    From this, the candidate upsampled positions are simply the
    the offset vectors plus their original centroid position.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, dim, hidden_dim, k, sub_point_conv_mode, upsample_rate):
        super().__init__()

        self.upsample_rate = upsample_rate

        # The hypothesis is a basis of 43 candidate direction vectors.
        hypothesis, _ = icosahedron2sphere(1)
        hypothesis = np.append(np.zeros((1, 3)), hypothesis, axis=0)
        self.hypothesis = torch.from_numpy(hypothesis).float().cuda()

        self.weight_nn = SubPointConv(
            hidden_dim, k, sub_point_conv_mode, dim, 43 * upsample_rate, upsample_rate
        )

        self.scale_nn = SubPointConv(
            hidden_dim, k, sub_point_conv_mode, dim, 1 * upsample_rate, upsample_rate
        )

    def forward(self, xyzs, feats):
        # xyzs: (b, 3, n)
        # feats: (b, c, n)
        # weights: (b, 43, n, u)
        # weights: (b, 43, 1, n, u)  [after unsqueeze]
        # hypothesis: (b, 43, 3, n, u)
        # directions: (b, 3, n, u)
        # scales: (b, 1, n, u)
        # deltas: (b, 3, n, u)
        # repeated_xyzs: (b, 3, n, u)

        batch_size = xyzs.shape[0]
        points_num = xyzs.shape[2]

        weights = self.weight_nn(feats)
        weights = weights.unsqueeze(2)
        weights = F.softmax(weights, dim=1)

        hypothesis = repeat(
            self.hypothesis,
            "h c -> b h c n u",
            b=batch_size,
            n=points_num,
            u=self.upsample_rate,
        )
        weighted_hypothesis = weights * hypothesis
        directions = torch.sum(weighted_hypothesis, dim=1)
        directions = F.normalize(directions, p=2, dim=1)

        scales = self.scale_nn(feats)

        deltas = directions * scales

        repeated_xyzs = repeat(xyzs, "b c n -> b c n u", u=self.upsample_rate)
        upsampled_xyzs = repeated_xyzs + deltas

        return upsampled_xyzs


class FeatsUpsampleLayer(nn.Module):
    """Feature upsampling layer used in [He2022pcc]_.

    Upsamples many features from each "centroid" feature vector.
    The feature vector associated with each centroid is upsampled
    into various candidate upsampled feature vectors.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(
        self,
        dim,
        hidden_dim,
        k,
        sub_point_conv_mode,
        upsample_rate,
        decompress_normal=False,
    ):
        super().__init__()

        self.upsample_rate = upsample_rate
        self.decompress_normal = decompress_normal
        out_fdim = (3 if decompress_normal else dim) * upsample_rate

        self.feats_nn = SubPointConv(
            hidden_dim, k, sub_point_conv_mode, dim, out_fdim, upsample_rate
        )

    def forward(self, feats):
        # upsampled_feats: (b, c, n, u)
        upsampled_feats = self.feats_nn(feats)
        if not self.decompress_normal:
            repeated_feats = repeat(feats, "b c n -> b c n u", u=self.upsample_rate)
            upsampled_feats = upsampled_feats + repeated_feats
        return upsampled_feats


class SubPointConv(nn.Module):
    """Sub-point convolution for upsampling, as introduced in [He2022pcc]_.

    Each feature vector (representing a "centroid" point) is sliced
    into g feature vectors, where each feature vector represents a
    point that has been upsampled from the original centroid point.
    Then, an MLP is applied to each slice individually.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, hidden_dim, k, mode, in_fdim, out_fdim, group_num):
        super().__init__()

        self.mode = mode
        self.group_num = group_num
        group_in_fdim = in_fdim // group_num
        group_out_fdim = out_fdim // group_num

        if self.mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Conv2d(group_in_fdim, hidden_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, group_out_fdim, 1),
            )
        elif self.mode == "edge_conv":
            self.edge_conv = EdgeConv(in_fdim, out_fdim, hidden_dim, k)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, feats):
        # feats: (b, cin * g, n)
        # expanded_feats: (b, cout, n, g)

        if self.mode == "mlp":
            feats = rearrange(
                feats, "b (c g) n -> b c n g", g=self.group_num
            ).contiguous()
            expanded_feats = self.mlp(feats)
        elif self.mode == "edge_conv":
            expanded_feats = self.edge_conv(feats)
            expanded_feats = rearrange(
                expanded_feats, "b (c g) n -> b c n g", g=self.group_num
            ).contiguous()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return expanded_feats


class EdgeConv(nn.Module):
    """EdgeConv introduced by [Wang2019dgcnn]_.

    First, groups similar feature vectors together via k-nearest neighbors
    using the following distance metric between feature vectors fᵢ and fⱼ:
    distance[i, j] = 2fᵢᵀfⱼ - ||fᵢ||² - ||fⱼ||².

    Then, for each group of feature vectors (f₁, ..., fₖ) with centroid fₒ,
    the residual feature vectors are each concatenated with the centroid,
    then an MLP is applied to each resulting vector individually,
    i.e., (MLP(f₁ - fₒ, fₒ), ..., MLP(fₖ - fₒ, fₒ)),
    and finally the elementwise max is taken across the resulting vectors,
    resulting in a single vector fₘₐₓ for the group.

    Original code located at [DGCNN]_ under MIT License.

    References:

        .. [Wang2019dgcnn] `"Dynamic Graph CNN for Learning on Point Clouds"
            <https://arxiv.org/abs/1801.07829>`_, by Yue Wang, Yongbin
            Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein,
            Justin M. Solomon, ACM Transactions on Graphics 2019.

        .. [DGCNN] `DGCNN
            <https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py>`_
    """

    def __init__(self, in_fdim, out_fdim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_fdim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_fdim, 1),
        )

    # WARN: This requires at least O(n^2) memory.
    def knn(self, feats, k):
        # feats: (b, c, n)
        # sq_norm: (b, 1, n)
        # pairwise_dot: (b, n, n)
        # pairwise_distance: (b, n, n)
        # knn_idx: (b, n, k)

        sq_norm = (feats**2).sum(dim=1, keepdim=True)

        # Pairwise dot product fᵢᵀfⱼ between feature vectors fᵢ and fⱼ.
        pairwise_dot = torch.matmul(feats.transpose(2, 1), feats)

        # pairwise_distance[i, j] = 2fᵢᵀfⱼ - ||fᵢ||² - ||fⱼ||²
        pairwise_distance = 2 * pairwise_dot - sq_norm - sq_norm.transpose(2, 1)

        _, knn_idx = pairwise_distance.topk(k=k, dim=-1)
        return knn_idx

    def get_graph_features(self, feats, k):
        # knn_feats: (b, c, n, k)
        # graph_feats: (b, 2c, n, k)
        dim = feats.shape[1]
        if dim == 3:
            feats_tr = feats.permute(0, 2, 1).contiguous()
            knn_idx = pointops.knnquery_heap(k, feats_tr, feats_tr).long()
        else:
            knn_idx = self.knn(feats, k)
        torch.cuda.empty_cache()
        knn_feats = index_points(feats, knn_idx)
        repeated_feats = repeat(feats, "b c n -> b c n k", k=k)
        graph_feats = torch.cat((knn_feats - repeated_feats, repeated_feats), dim=1)
        return graph_feats

    def forward(self, feats):
        # feats: (b, c, n)
        # graph_feats: (b, 2c, n, k)
        # expanded_feats: (b, cout*g, n, k)
        # feats_new: (b, cout*g, n)
        _, _, n = feats.shape
        graph_feats = self.get_graph_features(feats, k=min(self.k, n))
        expanded_feats = self.conv(graph_feats)
        feats_new, _ = expanded_feats.max(dim=-1)
        return feats_new


def icosahedron2sphere(level):
    """Samples uniformly on a sphere using a icosahedron.

    Code adapted from [IcoSphere_MATLAB]_ and [IcoSphere_Python]_,
    from paper [Xiao2009]_.

    References:

        .. [Xiao2009] `"Image-based street-side city modeling"
            <https://dl.acm.org/doi/10.1145/1618452.1618460>`_,
            by Jianxiong Xiao, Tian Fang, Peng Zhao, Maxime Lhuillier,
            and Long Quan, ACM Transactions on Graphics, 2009.

        .. [IcoSphere_MATLAB] https://github.com/jianxiongxiao/ProfXkit/blob/master/icosahedron2sphere/icosahedron2sphere.m

        .. [IcoSphere_Python] https://github.com/23michael45/PanoContextTensorflow/blob/master/PanoContextTensorflow/icosahedron2sphere.py
    """
    a = 2 / (1 + np.sqrt(5))

    # fmt: off
    M = np.array([
         0,  a, -1,  a,  1,  0, -a,  1,  0,  # noqa: E241, E126
         0,  a,  1, -a,  1,  0,  a,  1,  0,  # noqa: E241
         0,  a,  1,  0, -a,  1, -1,  0,  a,  # noqa: E241
         0,  a,  1,  1,  0,  a,  0, -a,  1,  # noqa: E241
         0,  a, -1,  0, -a, -1,  1,  0, -a,  # noqa: E241
         0,  a, -1, -1,  0, -a,  0, -a, -1,  # noqa: E241
         0, -a,  1,  a, -1,  0, -a, -1,  0,  # noqa: E241
         0, -a, -1, -a, -1,  0,  a, -1,  0,  # noqa: E241
        -a,  1,  0, -1,  0,  a, -1,  0, -a,  # noqa: E241, E131
        -a, -1,  0, -1,  0, -a, -1,  0,  a,  # noqa: E241
         a,  1,  0,  1,  0, -a,  1,  0,  a,  # noqa: E241
         a, -1,  0,  1,  0,  a,  1,  0, -a,  # noqa: E241
         0,  a,  1, -1,  0,  a, -a,  1,  0,  # noqa: E241
         0,  a,  1,  a,  1,  0,  1,  0,  a,  # noqa: E241
         0,  a, -1, -a,  1,  0, -1,  0, -a,  # noqa: E241
         0,  a, -1,  1,  0, -a,  a,  1,  0,  # noqa: E241
         0, -a, -1, -1,  0, -a, -a, -1,  0,  # noqa: E241
         0, -a, -1,  a, -1,  0,  1,  0, -a,  # noqa: E241
         0, -a,  1, -a, -1,  0, -1,  0,  a,  # noqa: E241
         0, -a,  1,  1,  0,  a,  a, -1,  0,  # noqa: E241
    ])
    # fmt: on

    coor = M.reshape(60, 3)
    coor, idx = np.unique(coor, return_inverse=True, axis=0)
    tri = idx.reshape(20, 3)

    # extrude
    coor_norm = np.linalg.norm(coor, axis=1, keepdims=True)
    coor = list(coor / np.tile(coor_norm, (1, 3)))

    for _ in range(level):
        tris = []

        for t in range(len(tri)):
            n = len(coor)
            coor.extend(
                [
                    (coor[tri[t, 0]] + coor[tri[t, 1]]) / 2,
                    (coor[tri[t, 1]] + coor[tri[t, 2]]) / 2,
                    (coor[tri[t, 2]] + coor[tri[t, 0]]) / 2,
                ]
            )
            tris.extend(
                [
                    [n, tri[t, 0], n + 2],
                    [n, tri[t, 1], n + 1],
                    [n + 1, tri[t, 2], n + 2],
                    [n, n + 1, n + 2],
                ]
            )

        tri = np.array(tris)

        # uniquefy
        coor, idx = np.unique(coor, return_inverse=True, axis=0)
        tri = idx[tri]

        # extrude
        coor_norm = np.linalg.norm(coor, axis=1, keepdims=True)
        coor = list(coor / np.tile(coor_norm, (1, 3)))

    return np.array(coor), np.array(tri)


def nearby_distance_sum(a_xyzs, b_xyzs, k):
    """Computes sum of nearby distances to B for each point in A.

    Partitions a point set B into non-intersecting sets
    C(a_1), ..., C(a_m) where each C(a_i) contains points that are
    nearest to a_i ∈ A.
    For each a_i ∈ A, computes the total distance from a_i to C(a_i).
    (Note that C(a_1), ..., C(a_m) may not cover all of B.)

    In more precise terms:
    For each a ∈ A, let C(a) ⊆ B denote its "collapsed point set" s.t.
    (i)   b ∈ C(a)  ⇒  min_{a' ∈ A} ||a' - b|| = ||a - b||,
    (ii)  ⋃ _{a ∈ A} C(a) ⊆ B,
    (iii) ⋂ _{a ∈ A} C(a) = ∅, and
    (iv)  |C(a)| ≤ k.
    For each a ∈ A, we then compute d(a) = ∑_{b ∈ C(a)} ||a - b||.

    Args:
        a_xyzs: (b, 3, m) Input point set A.
        b_xyzs: (b, 3, n) Input point set B.
        k: Maximum number of points in each collapsed point set C(a_i).

    Returns:
        distance: (b, m) Sum of distances from each point in A to its
            collapsed point set.
        mask: (b, m, k) Mask indicating which points in the ``knn_idx``
            belong to the collapsed point set of each point in A.
        knn_idx: (b, m, k) Indices of the points in B that are nearest
            to each point in A.
        nn_idx: (b, n, 1) Indices of the point in A that is nearest
            to each point in B.
    """
    # expect_idx: (b, m, k)
    # actual_idx: (b, m, k)
    # knn_xyzs: (b, 3, m, k)
    # knn_distances: (b, m, k)

    device = a_xyzs.device
    _, _, m = a_xyzs.shape
    a_xyzs_tr = a_xyzs.permute(0, 2, 1).contiguous()
    b_xyzs_tr = b_xyzs.permute(0, 2, 1).contiguous()

    # Determine which point in A each point in B is closest to.
    nn_idx = pointops.knnquery_heap(1, a_xyzs_tr, b_xyzs_tr)
    nn_idx_tr = nn_idx.permute(0, 2, 1).contiguous()

    # Determine k nearest neighbors in B for each point in A.
    knn_idx = pointops.knnquery_heap(k, b_xyzs_tr, a_xyzs_tr).long()
    torch.cuda.empty_cache()

    # Mask points that do not belong to the collapsed points set C(a).
    expect_idx = torch.arange(m, device=device)[None, :, None]
    actual_idx = index_points(nn_idx_tr, knn_idx).squeeze(1)
    mask = expect_idx == actual_idx

    # Compute the distance from each A point to its k nearest neighbors in B.
    knn_xyzs = index_points(b_xyzs, knn_idx)
    knn_distances = torch.linalg.norm(knn_xyzs - a_xyzs[..., None], dim=1)

    # Zero away the distances for points that are not in C(a).
    # knn_distances.masked_fill_(~mask, 0)
    knn_distances = knn_distances * mask.float()

    # Compute masked distances.
    # Notably, distance.sum(-1) is upper bounded by the sum of the
    # distances from each point in B to its nearest point in A.
    distance = knn_distances.sum(dim=-1)

    return distance, mask, knn_idx, nn_idx
