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

#include <chrono>
#include <fstream>
//#define DEBUG_VALUES        1 // show values
//#define DEBUG_MODEL         1 // show pb with model
//#define DEBUG_COUNTERS      1 // print overflow, MAC etc.
//#define DEBUG_PRINT         1 // print model info
#include "model_cdfs.h"
#include "common.h"
#include "range_coder.h"
#include <filesystem>
#include <sadl/model.h>
#include <tuple>

int verbose = 1;
using namespace std;

namespace { //

// file format
// SADLE2E01
// W [uint16] image width
// H [uint16] image height
// D [uint16] depth latent
// C [uint6]  downsampling factor
// T [uint2] 0:uncompressed, 1: cdfs
// channelactivation [C bits entropy coding]
// W*H*D [float|int16]
template <typename T> std::tuple<sadl::Tensor<T>, Size> decompressTensor(const std::string &filename) {
  ifstream file(filename, ios::binary);
  if (!file) {
    cerr << "[ERROR] no bitstream file " << filename << endl;
    exit(-1);
  }
  char magic[10];
  file.read(magic, 9);
  magic[9] = '\0';
  std::string magic_s = magic;
  if (magic_s != "SADLE2E01") {
    cout << "[ERROR] bad magic number for " << filename << endl;
    exit(-1);
  }
  uint16_t s[3];
  file.read((char *)s, sizeof(uint16_t) * 3);
  uint8_t tempo;
  file.read((char *)&tempo, sizeof(tempo));
  uint8_t down = (tempo >> 2);
  Size size = {s[1], s[0]};
  // deduce tensor size
  const int padding = (1 << down);
  s[0] = (((size[0] + padding - 1) / padding) * padding) >> down;
  s[1] = (((size[1] + padding - 1) / padding) * padding) >> down;

  uint8_t mode = (tempo & 3);
  if (verbose)
    std::cout << "[INFO] down: " << (int)down << ", mode=" << (mode == 0 ? "uncompressed" : (mode == 1 ? "entropy" : "cond_entropy"))
              << endl
              << "[INFO] image HxW:" << size[0] << 'x' << size[1] << "\n[INFO] tensor: {" << s[0] << ',' << s[1] << ',' << s[2] << "}"
              << endl;

  sadl::Tensor<int16_t> t;
  sadl::Dimensions dims;
  dims.resize(4);
  dims[0] = 1;
  dims[1] = s[0];
  dims[2] = s[1];
  dims[3] = s[2];
  t.resize(dims);
  // tempo
  if (mode == 0) {
    if (verbose) {
      cout << "[INFO] uncompressed tensor read" << endl;
    }
    file.read((char *)t.data(), t.dims().nbElements() * sizeof(int16_t));
  }
  if (mode == 1 ) {
    if (verbose) {
      cout << "[INFO] compressed tensor read" << endl;
    }

    RangeDecoder dec;
    std::vector<char> channel_activation;
    channel_activation.resize(s[2]);
    Cdf channel_cdf;
    channel_cdf.min_val = 0;
    channel_cdf.max_val = 2;
    channel_cdf.cproba.resize(3);
    channel_cdf.cproba[0] = 0;
    channel_cdf.cproba[2] = (1 << Cdf::precision);
    for (int k = 0; k < dims[3]; ++k) {
      channel_cdf.cproba[1] = (1 << Cdf::precision) - kChannelsProba[k];
      channel_activation[k] = dec.decode(file, channel_cdf);
    }

    int kprev = -1;
    for (int k0 = 0; k0 < dims[3]; ++k0) {
      int k = kOrder[k0]; // default
      if (channel_activation[k]) {
        for (int i = 0; i < dims[1]; ++i)
          for (int j = 0; j < dims[2]; ++j) {
            int x;
            if (mode == 1) {
              const auto &cdf = getCdf(t, k, kprev, i, j);
              x = dec.decode(file, cdf);
            }
            assert(x >= std::numeric_limits<int16_t>::min());
            assert(x <= std::numeric_limits<int16_t>::max());
            t(0, i, j, k) = x;
          }
      } else {
        for (int i = 0; i < dims[1]; ++i)
          for (int j = 0; j < dims[2]; ++j) {
            t(0, i, j, k) = kCdfs[k][0].probable;
          }
      }
      kprev = k;
    }
  } else {
  }
  if (verbose) {
    cout << "[INFO] input tensor " << t.dims() << endl;
  }
  if constexpr (is_same<T, int16_t>::value) {
    t.quantizer=kInputQuantizerShift;
    for (int i = 0; i < dims[1]; ++i)
      for (int j = 0; j < dims[2]; ++j)
        for (int k = 0; k < dims[3]; ++k)
          t(0, i, j, k) = (t(0, i, j, k)<<kInputQuantizerShift) + (kCdfs[k][0].mean_offset>>(kMeansQuantizerShift-kInputQuantizerShift)); // todo: take into account quantizer
    return {t, size};
  } else if constexpr (is_same<T, float>::value) {
    sadl::Tensor<float> t2;
    t2.resize(t.dims());
    constexpr float Q=(1<<kMeansQuantizerShift);
    for (int i = 0; i < dims[1]; ++i)
      for (int j = 0; j < dims[2]; ++j)
        for (int k = 0; k < dims[3]; ++k)
          t2(0, i, j, k) = t(0, i, j, k) + (kCdfs[k][0].mean_offset/Q);
    return {t2, size};
  } else {
    exit(-1);
  }
}

sadl::Tensor<Type> decode(sadl::Model<Type> &model, sadl::Tensor<Type> &t) {
  std::vector<sadl::Tensor<Type>> inputs;
  inputs.resize(1);

  inputs[0] = std::move(t);
  if (!model.init(inputs)) {
    cerr << "[ERROR] Pb init" << endl;
    exit(-1);
  }
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  // inputs already shifted with mean_offset
  if (!model.apply(inputs)) {
    cerr << "[ERROR] Pb inference" << endl;
    exit(-1);
  }
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> cold = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  if (verbose) {
    std::cout << "[INFO] decompress in " << cold.count() * 1000. << " ms" << endl;
  }

  if (verbose) {
    cout << "[INFO] output tensor " << model.result().dims() << endl;
  }
  return model.result();
}

} // namespace

int main(int argc, char **argv) {

  if (argc != 3) {
    cout << argv[0] << " bitstream.bin image.ppm" << endl;
    return 0;
  }
  const string filename_bs = argv[1];
  const string filename_out = argv[2];
  auto [t, size] = decompressTensor<Type>(filename_bs);
  auto model = loadDecoder();
  assert(model.getInputsTemplate()[0].quantizer==kInputQuantizerShift);
  auto out = decode(model, t);
  auto im = toImage(out, size);
  savePPM(im, filename_out);
  std::uintmax_t bssize = std::filesystem::file_size(filename_bs);
  int nb_pix = im.size[0] * im.size[1];
  cout << " bpp=" << (double)bssize * 8. / nb_pix << std::endl;
  return 0;
}
