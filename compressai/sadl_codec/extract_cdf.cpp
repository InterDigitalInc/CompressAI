/* Copyright (c) 2021-2024, InterDigital Communications, Inc
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of InterDigital Communications, Inc nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "range_coder.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sadl/tensor.h>
#include <tuple>
#include <vector>

using namespace std;
namespace { //
[[maybe_unused]] const int verbose = 1;
using LatentType = int16_t;
constexpr int means_quantizer=10;
constexpr int input_quantizer=5; // will shift input by 1<<5
int maxTensor = -1;

constexpr int kNbCtxSpatial = 3;
constexpr int kNbCtxChannel = 2;
constexpr int kNbCtxSpatialChannel = kNbCtxSpatial + kNbCtxChannel - 1;

using CdfSpatialCtx = std::array<Cdf, kNbCtxSpatial>;
using CdfSpatialChannelCtx = std::array<Cdf, kNbCtxSpatialChannel>;

constexpr int thDefault = 0;

sadl::Tensor<LatentType> readTensor(const std::string &filename, int h, int w, int d) {
  ifstream file(filename, ios::binary);
  file.seekg(0, file.end);
  auto size = file.tellg();
  file.seekg(0, file.beg);
  // assume int16
  sadl::Tensor<LatentType> t;
  sadl::Dimensions dims;
  dims.resize(4);
  int n = size / (d * h * w * sizeof(LatentType));
  dims[0] = n;
  dims[1] = d;
  dims[2] = h;
  dims[3] = w;
  cout << "[INFO] " << n << " tensors" << endl;
  if (maxTensor > 0)
    dims[0] = min((int)dims[0], maxTensor);
  cout << "[INFO] reduced to " << dims[0] << " tensors" << endl;
  t.resize(dims);
  cout << t.dims() << ' ' << t.size() << endl;
  file.read((char *)t.data(), t.size() * sizeof(LatentType));
  return t;
}

sadl::Tensor<LatentType> slice(const sadl::Tensor<LatentType> &t, int c) {
  sadl::Tensor<LatentType> t2;
  sadl::Dimensions dims{t.dims()[0], t.dims()[2], t.dims()[3]};
  t2.resize(dims);
  for (int n = 0; n < t.dims()[0]; ++n) {
    for (int i = 0; i < t.dims()[2]; ++i)
      for (int j = 0; j < t.dims()[3]; ++j) {
        t2(n, i, j) = t(n, c, i, j);
      }
  }
  return t2;
}

std::tuple<int, int> stat(const sadl::Tensor<LatentType> &t) {
  int m = numeric_limits<int>::max();
  int M = -numeric_limits<int>::max();
  for (auto x : t) {
    if (x < m)
      m = x;
    if (x > M)
      M = x;
  }
  return {m, M};
}

Cdf computeCdf(const sadl::Tensor<LatentType> &t, int m, int M) {
  Cdf cdf;
  cdf.min_val = m - 1;
  cdf.max_val = M + 1;
  if (cdf.min_val <= -(1 << (Cdf::precision - 1))) {
    cout << "[ERROR] values too large [" << m << ' ' << M << "]" << endl;
    exit(0);
  }
  if (cdf.max_val >= (1 << (Cdf::precision - 1))) {
    cout << "[ERROR] values too large [" << m << ' ' << M << "]" << endl;
    exit(0);
  }
  std::vector<double> proba;
  proba.resize(cdf.max_val - cdf.min_val + 1, 0);
  double s = 0.;
  for (auto x : t) {
    int i = x - cdf.min_val + 1;
    proba[i]++;
    s += x;
  }

  const double f = ((1 << Cdf::precision) - 1) / (double)t.size();
  for (auto &x : proba) {
    x = round(x * f);
  }
  cdf.cproba.resize(proba.size(), 0);
  for (int k = 1; k < (int)proba.size(); ++k) {
    cdf.cproba[k] = cdf.cproba[k - 1] + max(1, (int)proba[k]);
  }
  cdf.cproba.back() = (1 << Cdf::precision);
  for (int k = (int)cdf.cproba.size() - 2; k > 0; --k) {
    cdf.cproba[k] = min(cdf.cproba[k], cdf.cproba[k + 1] - 1);
  }
  int max_proba = -1;
  int max_idx = -1;
  for (int k = 0; k < (int)proba.size() - 1; ++k) {
    int p = cdf.cproba[k + 1] - cdf.cproba[k];
    if (p > max_proba) {
      max_proba = p;
      max_idx = k;
    }
  }
  cdf.probable = max_idx + cdf.min_val;
  if (cdf.cproba[1] <= 0) {
    cout << "[ERROR] logical error" << endl;
    exit(0);
  }
  //  cout<<cdf.min_val<<' '<<cdf.max_val<<' '<<cdf.mean<<' '<<s/t.size()<<endl;
  return cdf;
}

int getSpatialContext(const sadl::Tensor<LatentType> &t, int n, int i, int j, int mean, int th) {
  int ctx = 0;
  if (i > 0 && abs(t(n, i - 1, j) - mean) > th)
    ctx++;
  if (j > 0 && abs(t(n, i, j - 1) - mean) > th)
    ctx++;
  return ctx;
}


int getSpatialChannelContext(const sadl::Tensor<LatentType> &cur, const sadl::Tensor<LatentType> &prev, int n, int i, int j, int mean,
                             int mean_prev, int th, int th_prev) {
  int ctx = 0;
  if (i > 0 && abs(cur(n, i - 1, j) - mean) > th)
    ++ctx;
  if (j > 0 && abs(cur(n, i, j - 1) - mean) > th)
    ++ctx;
  if (prev.size() > 0) {
    if (th_prev >= 0 && abs(prev(n, i, j) - mean_prev) > th_prev)
      ++ctx;
    else if (th_prev < 0 && abs(prev(n, i, j) - mean_prev) > th)
      ++ctx;
  }
  return ctx;
}

CdfSpatialCtx computeCdfSpatialCond(const sadl::Tensor<LatentType> &t, int th, const Cdf &cdf) {
  CdfSpatialCtx cdfs_cond;
  auto min_val = cdf.min_val;
  auto max_val = cdf.max_val;
  auto probable = cdf.probable;
  auto mean_offset = cdf.mean_offset;
  for (auto &cdf : cdfs_cond) {
    cdf.min_val = min_val;
    cdf.max_val = max_val;
    cdf.probable = probable;
    cdf.mean_offset = mean_offset;
  }

  std::array<std::vector<double>, kNbCtxSpatial> probas;
  for (auto &proba : probas)
    proba.resize(max_val - min_val + 1, 0);
  double sum[kNbCtxSpatial] = {};

  for (int n = 0; n < (int)t.dims()[0]; ++n)
    for (int i = 0; i < (int)t.dims()[1]; ++i)
      for (int j = 0; j < (int)t.dims()[2]; ++j) {
        int ctx = getSpatialContext(t, n, i, j, probable, th);
        int x = t(n, i, j);
        int idx = x - cdfs_cond[ctx].min_val + 1;
        probas[ctx][idx]++;
        sum[ctx]++;
      }

  for (int ctx = 0; ctx < kNbCtxSpatial; ++ctx) {
    if (sum[ctx] > 0) {
      const double f = ((1 << Cdf::precision) - 1) / sum[ctx];
      for (auto &x : probas[ctx]) {
        x = round(x * f);
      }
    }
  }
  for (int ctx = 0; ctx < kNbCtxSpatial; ++ctx) {
    auto &cdf = cdfs_cond[ctx];
    const auto &proba = probas[ctx];
    cdf.cproba.resize(probas[ctx].size(), 0);
    for (int k = 1; k < (int)proba.size(); ++k)
      cdf.cproba[k] = cdf.cproba[k - 1] + max(1, (int)proba[k]);
    cdf.cproba.back() = (1 << Cdf::precision);
    for (int k = (int)cdf.cproba.size() - 2; k > 0; --k)
      cdf.cproba[k] = min(cdf.cproba[k], cdf.cproba[k + 1] - 1);
    if (cdf.cproba[1] <= 0) {
      cout << "[ERROR] logical error" << endl;
      exit(0);
    }
  }
  return cdfs_cond;
}



CdfSpatialChannelCtx computeCdfSpatialChannelCond(const sadl::Tensor<LatentType> &t, const sadl::Tensor<LatentType> &prev, const Cdf &cdf,
                                                  const Cdf &cdf_prev, int th, int th_prev = -1) {
  CdfSpatialChannelCtx cdfs_cond;
  auto min_val = cdf.min_val;
  auto max_val = cdf.max_val;
  auto mean = cdf.probable;
  auto mean_offset = cdf.mean_offset;
  for (auto &cdf : cdfs_cond) {
    cdf.min_val = min_val;
    cdf.max_val = max_val;
    cdf.probable = mean;
    cdf.mean_offset = mean_offset;
  }

  std::array<std::vector<double>, kNbCtxSpatialChannel> probas;
  for (auto &proba : probas)
    proba.resize(max_val - min_val + 1, 0);
  double sum[kNbCtxSpatialChannel] = {};

  int mean_prev = cdf_prev.probable;

  for (int n = 0; n < (int)t.dims()[0]; ++n)
    for (int i = 0; i < (int)t.dims()[1]; ++i)
      for (int j = 0; j < (int)t.dims()[2]; ++j) {
        int ctx = getSpatialChannelContext(t, prev, n, i, j, mean, mean_prev, th, th_prev);
        int x = t(n, i, j);
        int idx = x - cdfs_cond[ctx].min_val + 1;
        probas[ctx][idx]++;
        sum[ctx]++;
      }

  for (int ctx = 0; ctx < kNbCtxSpatialChannel; ++ctx) {
    if (sum[ctx] > 0) {
      const double f = ((1 << Cdf::precision) - 1) / sum[ctx];
      for (auto &x : probas[ctx]) {
        x = round(x * f);
      }
    }
  }
  for (int ctx = 0; ctx < kNbCtxSpatialChannel; ++ctx) {
    auto &cdf = cdfs_cond[ctx];
    const auto &proba = probas[ctx];
    cdf.cproba.resize(probas[ctx].size(), 0);
    for (int k = 1; k < (int)proba.size(); ++k)
      cdf.cproba[k] = cdf.cproba[k - 1] + max(1, (int)proba[k]);
    cdf.cproba.back() = (1 << Cdf::precision);
    for (int k = (int)cdf.cproba.size() - 2; k > 0; --k)
      cdf.cproba[k] = min(cdf.cproba[k], cdf.cproba[k + 1] - 1);
    if (cdf.cproba[1] <= 0) {
      cout << "[ERROR] logical error" << endl;
      exit(0);
    }
  }
  return cdfs_cond;
}

double entropyCondSpatialChannel(const sadl::Tensor<LatentType> &t, const sadl::Tensor<LatentType> &prev, const CdfSpatialChannelCtx &cdfs,
                                 const Cdf &cdf, const Cdf &cdf_prev, int th, int th_prev) {
  double H = 0.;
  constexpr double M = (1 << Cdf::precision);
  const int mean = cdf.probable;
  const int mean_prev = cdf_prev.probable;

  for (int n = 0; n < (int)t.dims()[0]; ++n)
    for (int i = 0; i < (int)t.dims()[1]; ++i)
      for (int j = 0; j < (int)t.dims()[2]; ++j) {
        int ctx = getSpatialChannelContext(t, prev, n, i, j, mean, mean_prev, th, th_prev);
        int x = t(n, i, j);
        const Cdf &cdf = cdfs[ctx];
        int idx = x - cdf.min_val;
        H += log2(M) - log2(cdf.cproba[idx + 1] - cdf.cproba[idx]);
      }

  return H / t.size();
}

double entropyCondSpatial(const sadl::Tensor<LatentType> &t, const CdfSpatialCtx &cdfs, const Cdf &cdf, int thh) {
  double H = 0.;
  constexpr double M = (1 << Cdf::precision);
  const int mean = cdf.probable;

  for (int n = 0; n < (int)t.dims()[0]; ++n)
    for (int i = 0; i < (int)t.dims()[1]; ++i)
      for (int j = 0; j < (int)t.dims()[2]; ++j) {
        int ctx = getSpatialContext(t, n, i, j, mean, thh);
        int x = t(n, i, j);
        const Cdf &cdf = cdfs[ctx];
        int idx = x - cdf.min_val;
        H += log2(M) - log2(cdf.cproba[idx + 1] - cdf.cproba[idx]);
      }

  return H / t.size();
}

double entropy(const sadl::Tensor<LatentType> &t, const Cdf &cdf) {
  double H = 0.;
  constexpr double M = (1 << Cdf::precision);

  for (auto x : t) {
    int i = x - cdf.min_val;
    H += log2(M) - log2(cdf.cproba[i + 1] - cdf.cproba[i]);
  }
  return H / t.size();
}

vector<int> getChannelOrderEntropyBased(const sadl::Tensor<LatentType> &t, const vector<int> &best_th, const vector<double> &spatial_cost,
                                        const vector<Cdf> &cdfs) {

  vector<int> order;
  order.reserve(best_th.size());

  constexpr int kMinRange = 4;
  int best_range = -1;
  int best_idx = -1;
  for (int i = 0; i < (int)cdfs.size(); ++i) {
    if (cdfs[i].max_val - cdfs[i].min_val > best_range) {
      best_range = cdfs[i].max_val - cdfs[i].min_val;
      best_idx = i;
    }
  }
  order.push_back(best_idx);
  cout << "[INFO] order by cost: ";
  sadl::Tensor<LatentType> prev_tc = slice(t, best_idx);
  while ((int)order.size() != t.dims()[1]) {
    const int k_prev = order.back();
    double bestCost = -1.;
    int bestIdx = -1;
    for (int k_cur = 0; k_cur < (int)t.dims()[1]; ++k_cur) {
      int range = cdfs[k_cur].max_val - cdfs[k_cur].min_val;
      if (range > kMinRange && find(order.begin(), order.end(), k_cur) == order.end()) {
        auto tc = slice(t, k_cur);
        const auto &cdf = cdfs[k_cur];
        const auto &cdf_prev = cdfs[k_prev];
        auto cdfs_cond = computeCdfSpatialChannelCond(tc, prev_tc, cdf, cdf_prev, best_th[k_cur], best_th[k_prev]);
        double dcost =
            spatial_cost[k_cur] - entropyCondSpatialChannel(tc, prev_tc, cdfs_cond, cdf, cdf_prev, best_th[k_cur], best_th[k_prev]);
        if (dcost > bestCost) {
          bestCost = dcost;
          bestIdx = k_cur;
        }
      }
    }
    if (bestIdx >= 0) {
      cout << bestIdx << ' ';
      cout.flush();
      order.push_back(bestIdx);
      prev_tc = slice(t, bestIdx);
    } else
      break;
  }
  for (int k = 0; k < (int)t.dims()[1]; ++k) {
    int range = cdfs[k].max_val - cdfs[k].min_val;
    if (range <= kMinRange && find(order.begin(), order.end(), k) == order.end()) {
      order.push_back(k);
    }
  }
  cout << endl;
  return order;
}



vector<int> channelActivationProba(const sadl::Tensor<LatentType> &t, const vector<Cdf> &cdfs) {
  const double f = (double)(1 << Cdf::precision) / (t.dims()[0]);
  vector<int> v;
  v.resize(t.dims()[1], 0);
  for (int k = 0; k < (int)t.dims()[1]; ++k) {
    double s = 0.;
    const int mean = cdfs[k].probable;
    for (int n = 0; n < (int)t.dims()[0]; ++n) {
      bool zero = true;
      for (int i = 0; i < (int)t.dims()[2] && zero; ++i)
        for (int j = 0; j < (int)t.dims()[3] && zero; ++j) {
          int x = t(n, k, i, j);
          zero = (x == mean);
        }
      s += !zero;
    }
    v[k] = max(1, min((1 << Cdf::precision) - 1, (int)round(s * f)));
  }
  return v;
}



void writeCdfs(const vector<array<Cdf, kNbCtxSpatialChannel>> &cdfscond, const vector<int> &best_th, const vector<int> &order,const vector<int> &channel_proba,
               const string &filename) {
  ofstream file(filename, ios::binary);
  const int N=cdfscond.size();
  file << "#pragma once\n"
       << "\n"
          "#include \"range_coder.h\"\n"
          "#include <sadl/tensor.h>\n"
          "static constexpr int kNbCdfs="
       << N << ";\n";
  file << "static const int kOrder[kNbCdfs]={\n";
  for (auto x : order)
    file << x << ", ";
  file << "};\n";

  file << "static constexpr int kNbChannels="<< N << ";\n"
          "static const int kChannelsProba[kNbChannels]={\n";
  for (auto p : channel_proba)
    file << p << ", ";
  file << "};\n";

  file << "static const int kThreholds[kNbCdfs]={\n";
  for (auto x : best_th)
    file << x << ", ";
  file << "};\n";

  file << "static constexpr int kNbCtx=" << kNbCtxSpatialChannel << ";\n";
  file << "static constexpr int kMeansQuantizerShift=" << means_quantizer << ";\n";
  file << "static constexpr int kInputQuantizerShift=" << input_quantizer << ";\n";
  file << "static const Cdf kCdfs[kNbCdfs][kNbCtx]={\n";
  for (const auto &cdfcond : cdfscond) {
    file << "{\n";
    for (const auto &cdf : cdfcond) {
      file << " {" << cdf.min_val << ", " << cdf.max_val << ", " << cdf.probable << ", " << cdf.mean_offset << " ,{";
      for (auto x : cdf.cproba)
        file << x << ", ";
      file << "} },\n";
    }
    file << "},\n";
  }
  file << "};\n";


  file <<
          R"cpp(
          template<typename T>
          int getContext(const sadl::Tensor<T> &t,int n_cur,int n_prev,int i,int j) {
          int ctx=0;
          int th=kThreholds[n_cur];
          int mean=kCdfs[n_cur][0].probable;
          if (i>0&&abs(t(0,i-1,j,n_cur)-mean)>th)      ++ctx;
          if (j>0&&abs(t(0,i,j-1,n_cur)-mean)>th)      ++ctx;
          if (n_prev>=0&&abs(t(0,i,j,n_prev)-kCdfs[n_prev][0].probable)>kThreholds[n_prev]) ++ctx;
          return ctx;
          }
          )cpp";

}

void help() {
  cout << "extract_cdf latent_h latent_w latent_d outputs.bin\n"
          " OUTPUT: model_cdfs.h\n"
       << endl;
  exit(0);
}

double sum(const vector<double> &v) { return accumulate(v.begin(), v.end(), 0.); }

} // namespace

int main(int argc, char **argv) {
  if (argc != 5)
    help();
  const int h = atoi(argv[1]);
  const int w = atoi(argv[2]);
  const int d = atoi(argv[3]);
  const string filename_latent = argv[4];
  const auto t = readTensor(filename_latent, h, w, d);

  vector<Cdf> cdfs(d);
  vector<double> defaultH(d);
  vector<float> means_float(d);
  { // load means
    ifstream file(filename_latent + ".means");
    if (!file) {
      cerr << "[ERROR] " << filename_latent + ".means"
           << " is missing" << endl;
      exit(0);
    }
    file.read((char *)means_float.data(), means_float.size() * sizeof(float));
  }
  // first compute normal cdfs
  cout << "\n[INFO] ===== Default cdfs ======" << endl;
  for (int c = 0; c < d; ++c) {
    auto tc = slice(t, c);
    auto [m, M] = stat(tc);
    cdfs[c] = computeCdf(tc, m, M);
    cdfs[c].mean_offset = round(means_float[c]*(1<<means_quantizer));
    auto H = entropy(tc, cdfs[c]);
    defaultH[c] = H;
    cout << " [INFO] channel " << c << ": " << cdfs[c].probable << " [" << m << ' ' << M << "] " << H << " bpp" << endl;
  }
  double s_default = sum(defaultH);
  cout << "[INFO] CDF: Total: " << s_default << " bpp" << endl;

  vector<CdfSpatialCtx> cdfs_spatial_cond(d);
  vector<int> best_th(d);
  vector<double> spatialH(d);
  double s_spatial = 0.;
  { // we need to compute the thresholds on spatial consideration
    cout << "\n[INFO] ===== Spatial cond. cdfs ======" << endl;
    for (int c = 0; c < d; ++c) {
      int m = cdfs[c].min_val;
      int M = cdfs[c].max_val;
      int probable = cdfs[c].probable;
      int th_max = thDefault;

      th_max = max(1, min(abs(m - probable), abs(M - probable)));

      auto tc = slice(t, c);
      double bestH = numeric_limits<double>::max();
      for (int th = 0; th < th_max; ++th) {
        auto cdfs_cur = computeCdfSpatialCond(tc, th, cdfs[c]);
        auto H = entropyCondSpatial(tc, cdfs_cur, cdfs[c], th);
        if (H < bestH) {
          bestH = H;
          cdfs_spatial_cond[c] = move(cdfs_cur);
          best_th[c] = th;
        }
      }
      spatialH[c] = bestH;
      cout << " [INFO] " << c << ": " << defaultH[c] << " => " << bestH << " bpp, th=" << best_th[c] << endl;
    }
    s_spatial = sum(spatialH);
    cout << "[INFO] CDF spatial cond: Total: " << s_default << " => " << s_spatial << " (" << 100. * s_spatial / s_default << "%)" << endl;
  }

  //

  vector<int> order = getChannelOrderEntropyBased(t, best_th, defaultH, cdfs);
  vector<CdfSpatialChannelCtx> cdfs_spatial_channel_cond(d);

  vector<double> withChannelH(d);
  {
    cout << "\n[INFO] ===== spatial/channel cond. cdfs ======" << endl;
    sadl::Tensor<LatentType> tc_prev;
    int c_prev = -1;
    for (int c0 = 0; c0 < d; ++c0) {
      int c = order[c0];
      auto tc = slice(t, c);

      int th_cur = best_th[c];
      int th_prev = (c_prev >= 0) ? best_th[c_prev] : -1;
      const auto &cdf = cdfs[c];
      const auto &cdf_prev = (c_prev >= 0) ? cdfs[c_prev] : cdfs[0];

      th_prev = (c_prev >= 0) ? best_th[c_prev] : -1;
      if (c_prev >= 0) {
        th_prev = best_th[c_prev];
      }

      cdfs_spatial_channel_cond[c] = computeCdfSpatialChannelCond(tc, tc_prev, cdf, cdf_prev, th_cur, th_prev);
      withChannelH[c] = entropyCondSpatialChannel(tc, tc_prev, cdfs_spatial_channel_cond[c], cdf, cdf_prev, th_cur, th_prev);

      cout << " [INFO] " << c << ": " << defaultH[c] << " => " << withChannelH[c] << " bpp" << endl;
      swap(tc_prev, tc);
      c_prev = c;
    }
    double s_wchannel = sum(withChannelH);
    cout << "[INFO] CDF spatial/channel cond: Total: " << sum(defaultH)
         << " => " << s_spatial << " (" << 100. * s_spatial / s_default << "%) => " << s_wchannel << " (" << 100. * s_wchannel / s_default
         << "%)" << endl;

    auto channel_proba = channelActivationProba(t, cdfs);

    writeCdfs(cdfs_spatial_channel_cond, best_th, order, channel_proba, "model_cdfs.h");
  }
}
