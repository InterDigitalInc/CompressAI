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
#include <tuple>
#include <vector>

const int verbose = 1;
using LatentType = int16_t;

#include "model_cdfs.h"
#include "common.h"
#include "rdoq.h"

sadl::Tensor<LatentType> asInt16(const sadl::Tensor<Type> &t) {
  sadl::Tensor<LatentType> t2;
  t2.resize(t.dims());
  for (int k = 0; k < t.size(); ++k)
    t2[k] = (int)round(t[k]);
  return t2;
}

using namespace std;

constexpr int kMaxImageSize = 8192;

CodecProperties getProperties(const sadl::Model<Type> & /*model*/) {
  CodecProperties p;
  // TODO
  p.down = 4;
  p.receiptive_field = 30; // to check
  p.lambda = 0.0130;
  return p;
}

std::vector<char> computeChannelActivation(const sadl::Tensor<LatentType> &t) {
  std::vector<char> v;
  v.resize(t.dims()[3], 0);
  for (int k = 0; k < t.dims()[3]; ++k) {
    const int mean = kCdfs[k][0].probable;
    char x = 0;
    for (int i = 0; i < t.dims()[1]; ++i)
      for (int j = 0; j < t.dims()[2]; ++j)
        if (t(0, i, j, k) != mean) {
          x = 1;
          break;
        }
    v[k] = x;
  }
  return v;
}

std::string compress(const sadl::Tensor<LatentType> &t, Size size) {
  ostringstream oss;
  RangeCoder re;
  std::vector<char> channel_activation = computeChannelActivation(t);
  if (kNbChannels > 0) {
    Cdf channel_cdf;
    channel_cdf.min_val = 0;
    channel_cdf.max_val = 2;
    channel_cdf.cproba.resize(3);
    channel_cdf.cproba[0] = 0;
    channel_cdf.cproba[2] = (1 << Cdf::precision);
    for (int k = 0; k < t.dims()[3]; ++k) {
      channel_cdf.cproba[1] = (1 << Cdf::precision) - kChannelsProba[k];
      re.encode((int)channel_activation[k], channel_cdf, oss);
    }
    if (verbose)
      std::cout << "[INFO] header activation: " << 8 * oss.str().size() << " bytes" << std::endl;
  } else {
    std::fill(channel_activation.begin(), channel_activation.end(), 1);
    if (verbose)
      std::cout << "[INFO] no header acitvation" << endl;
  }
  int kprev = -1;
  for (int k0 = 0; k0 < t.dims()[3]; ++k0) {
    int k = kOrder[k0];
    if (channel_activation[k]) {
      for (int i = 0; i < t.dims()[1]; ++i)
        for (int j = 0; j < t.dims()[2]; ++j) {
          const Cdf &cdf = getCdf(t, k, kprev, i, j);
          int v = t(0, i, j, k);
          if (v <= cdf.min_val || v >= cdf.max_val) {
            cout << "[WARNING] value out of range: " << v << " [" << cdf.min_val << ' ' << cdf.max_val << "]" << endl;
            v = max(cdf.min_val + 1, min(cdf.max_val - 1, v)); // exit(-1);
          }
          re.encode(v, cdf, oss);
        }
    }
    kprev = k;
  }
  re.finalize(oss);
  if (verbose) {
    std::cout << "[INFO] entropy=" << re.entropy << " bits, " << re.entropy / 8. << " bytes => " << re.entropy / (size[0] * size[1])
              << " bpp" << endl;
  }
  return oss.str();
}

// file format
// SADLE2E01
// W [uint16] image width
// H [uint16] image height
// D [uint16] depth latent
// C [uint6]  downsampling factor
// T [uint2] 0:uncompressed, 1: compressed, 2: cond cdfs
// channelactivation [C bits entropy coding]
// W*H*D [float|int16]
void compressTensor(const std::string &filename, const sadl::Tensor<LatentType> &t, Size size, int mode) {
  ofstream file(filename, ios::binary);
  if (!file) {
    cerr << "[ERROR] unable to open bitstream " << filename << endl;
    exit(-1);
  }
  char magic[10] = "SADLE2E01";
  magic[9] = '\0';
  file.write((const char *)magic, 9);
  uint16_t s[3] = {size[1] /* width */, size[0] /* height */, (uint16_t)t.dims()[3]};
  file.write((const char *)s, sizeof(uint16_t) * 3);
  uint8_t down = 0;
  while (down < (63) && (t.dims()[1] << down) < s[0] && (t.dims()[2] << down) < s[1])
    ++down;
  uint8_t type = mode;
  uint8_t tempo = (down << 2) | type;
  file.write((const char *)&tempo, sizeof(tempo));

  if (verbose)
    cout << "[INFO] mode=" << (mode == 0 ? "uncompressed" : "cond_entropy") << endl
         << "[INFO] downsampling: " << (int)down << ", type=" << (int)type << endl
         << "[INFO] input tensor " << t.size() * sizeof(int16_t) << " bytes " << t.dims() << endl
         << "[INFO] image: " << size[0] << ',' << size[1] << " => downsampling " << (int)down << endl;
  if (mode > 0) {
    if (kNbCdfs != t.dims()[3]) {
      cout << "[ERROR] cdfs size != tensor depth " << endl;
      exit(-1);
    }
    auto s = compress(t, size);
    if (verbose)
      cout << "[INFO] compressed: " << s.size() << " " << (100. * s.size()) / (t.size() * sizeof(int16_t)) << "%" << std::endl;
    file.write((const char *)s.c_str(), s.size());
  } else {
    file.write((const char *)t.data(), sizeof(LatentType) * t.size());
    if (verbose)
      cout << "[INFO] wrote uncompressed tensor" << endl;
  }
}

Image loadImage(const string &filename) {
  ifstream file(filename, ios::binary);
  if (!file) {
    cerr << "[ERROR] unable to open image filename" << endl;
    exit(-1);
  }
  string line;
  file >> line;
  if (line != "P6") {
    cerr << "[ERROR] image not a PPM" << endl;
    exit(-1);
  }
  Image im;
  file >> im.size[1] >> im.size[0];
  if (verbose)
    cout << "[INFO] image size: " << im.size[1] << 'x' << im.size[0] << endl;
  if (im.size[0] > kMaxImageSize || im.size[1] > kMaxImageSize) {
    cerr << "[ERROR] image too large" << endl;
    exit(-1);
  }
  int L;
  file >> L;
  if (L != 255) {
    cerr << "[ERROR] image to a valid PPM" << endl;
    exit(-1);
  }
  file.ignore(1024, '\n');
  im.data_.resize(3 * im.size[0] * im.size[1]);
  file.read((char *)im.data_.data(), im.data_.size());
  return im;
}

sadl::Model<Type> loadEncoder() {
  sadl::Model<Type> model;
  istringstream file_model(string((const char *)encoderSadl, sizeof(encoderSadl)), ios::binary);

  if (!model.load(file_model)) {
    cerr << "[ERROR] Unable to read model " << endl;
    exit(-1);
  }
  return model;
}

Image pad(const Image &im, int d) {
  Image im2;
  int p = (1 << d);
  im2.size[0] = ((im.size[0] + p - 1) / p) * p;
  im2.size[1] = ((im.size[1] + p - 1) / p) * p;
  im2.data_.resize(im2.size[0] * im2.size[1] * 3, 0);
  for (int i = 0; i < im2.size[0]; ++i)
    for (int j = 0; j < im2.size[1]; ++j)
      for (int k = 0; k < 3; ++k)
        im2.data_[3 * (i * im2.size[1] + j) + k] = im.data_[3 * (i * im.size[1] + j) + k];
  return im2;
}

sadl::Tensor<Type> image2tensor(const Image &im,int shift) {
  sadl::Tensor<Type> t;
  sadl::Dimensions dims({1, im.size[0], im.size[1], 3});
  t.resize(dims);
  if constexpr (is_same<Type,float>::value) {
  constexpr float Q = 1.f / 255.f; // 256
  for (int k = 0; k < t.size(); ++k)
    t[k] = im.data_[k] * Q;
  } else {
      for (int k = 0; k < t.size(); ++k)
          t[k] = (((int)im.data_[k])<<shift)+(1<<(shift-1)); // half?
  }
  return t;
}




sadl::Tensor<LatentType> encode(sadl::Model<Type> &model, const Image &im, const CodecProperties &properties) {

  Image im2 = pad(im, properties.down);

  std::vector<sadl::Tensor<Type>> inputs;
  inputs.resize(1);
  int shift=model.getInputsTemplate()[0].quantizer-8;
  inputs[0] = image2tensor(im2,shift);
  if (!model.init(inputs)) {
    cerr << "[ERROR] Pb init" << endl;
    exit(-1);
  }
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  if (!model.apply(inputs)) {
    cerr << "[ERROR] Pb inference" << endl;
    exit(-1);
  }
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> cold = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  if (verbose) {
    std::cout << "[INFO] encode in " << cold.count() * 1000. << " ms" << endl;
  }
  auto res = model.result();
  const int s=kMeansQuantizerShift-res.quantizer;
  const int half=(1<<(res.quantizer-1));
  (void)s;
  (void)half;
  // modif latent to take into account mean offset
  for (int i = 0; i < res.dims()[1]; ++i)
    for (int j = 0; j < res.dims()[2]; ++j) {
      for (int k = 0; k < res.dims()[3]; ++k) {
          if constexpr (is_same<Type,float>::value) {
            const float off= (float)kCdfs[k][0].mean_offset/(1<<kMeansQuantizerShift);
            res(0, i, j, k) -= off; // add the difference to the float mean
          } else if constexpr (is_same<Type,int16_t>::value) {
            res(0, i, j, k) = ((int)res(0, i, j, k)-(kCdfs[k][0].mean_offset>>s)+half)>>res.quantizer; // emulate round
          }
      }
    }
  if constexpr (is_same<Type,float>::value) {
      return asInt16(res);
  } else if constexpr (is_same<Type,int16_t>::value) {
      res.quantizer=0;
      return asInt16(res);
  }
}

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4) {
    cout << argv[0]
         << " image.ppm bitstream.bin [lambda]\n"
            " if lambda, perform RDOQ\n"
         << endl;
    return 0;
  }
  const string filename_image = argv[1];
  const string filename_out = argv[2];

  bool rdoq_on = (argc==4);
  bool mt =false;
  bool approx = false;
  int pass = 0;
  if (rdoq_on) {
      mt = true;
      approx = true;
      pass = 3;
  }
  if (mt) {
    cout << "[INFO] multithread" << endl;
  }
  Image im = loadImage(filename_image);
  sadl::Model<Type> model = loadEncoder();
  auto prop = getProperties(model);
  if (approx) {
    prop.receiptive_field *= 0.5;
    cout << "[INFO] approx RF" << endl;
  }
  if (rdoq_on) {
    prop.lambda = atof(argv[3]);
    cout << "[INFO] lambda=" << prop.lambda << endl;
  }
  auto latent = encode(model, im, prop);
  auto decoder = loadDecoder();
  vector<sadl::Tensor<Type>> inputs{1};
  inputs[0].resize(latent.dims());
  if constexpr (is_same<Type,int16_t>::value) inputs[0].quantizer=kInputQuantizerShift;
  if (!decoder.init(inputs)) {
    cerr << "[ERROR] Pb init" << endl;
    exit(-1);
  }
  cout << "[INFO] PSNR=" << psnrReconstructed(decoder, latent, im) << " dB" << endl;
  if (rdoq_on) {
    if (mt)
      latent = rdoq_mt(decoder, latent, im, prop, pass);
    else
      latent = rdoq(decoder, latent, im, prop, pass);
  }
  compressTensor(filename_out, latent, im.size, 1);
  if (verbose) {
    cout << "[INFO] wrote " << filename_out << endl;
  }
}
