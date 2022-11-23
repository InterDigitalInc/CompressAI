#pragma once
#include <iostream>
#include <sadl/model.h>
#include <string>
#include <vector>
#include "range_coder.h"

using Size = std::array<uint16_t, 2>; // h w : uint16_t because use to write header
struct Image {
  Size size;
  std::vector<uint8_t> data_;
  uint8_t operator()(int i, int j, int k) const { return data_[3 * (i * size[1] + j) + k]; }
  uint8_t &operator()(int i, int j, int k) { return data_[3 * (i * size[1] + j) + k]; }
};

template <typename T> int clip(int m, int M, T x) {
  if (x < m)
    return 0;
  if (x > M)
    return M;
  return (int)x;
}

Image toImage(const sadl::Tensor<float> &t, Size s) {
  Image im;
  constexpr float f = 255.f;
  im.size = s;
  im.data_.resize(im.size[0] * im.size[1] * 3);
  for (int i = 0; i < s[0]; ++i)
    for (int j = 0; j < s[1]; ++j) {
      for (int k = 0; k < 3; ++k)
        im(i, j, k) = clip(0, 255, round(f * t(0, i, j, k)));
    }
  return im;
}

Image toImage(const sadl::Tensor<int16_t> &t, Size s) {
  Image im;
  const int shift = t.quantizer - 8;
  im.size = s;
  im.data_.resize(im.size[0] * im.size[1] * 3);
  for (int i = 0; i < s[0]; ++i)
    for (int j = 0; j < s[1]; ++j) {
      for (int k = 0; k < 3; ++k)
        im(i, j, k) = clip(0, 255, (t(0, i, j, k) >> shift));
    }
  return im;
}

void savePPM(const Image &im, const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  file << "P6\n" << im.size[1] << ' ' << im.size[0] << "\n255\n";
  file.write((const char *)im.data_.data(), im.data_.size());
}

sadl::Model<Type> loadDecoder() {
  sadl::Model<Type> model;
  std::istringstream file_model(std::string((const char *)decoderSadl, sizeof(decoderSadl)), std::ios::binary);

  if (!model.load(file_model)) {
    std::cerr << "[ERROR] Unable to read model " << std::endl;
    exit(-1);
  }
  return model;
}

const Cdf &getCdf(const sadl::Tensor<int16_t> &t, int k, int kprev, int i, int j) {
  const Cdf &cdf = kCdfs[k][getContext(t, k, kprev, i, j)];
  return cdf;
}

const Cdf &getCdf(const sadl::Tensor<float> &t, int k, int kprev, int i, int j) {
  const Cdf &cdf = kCdfs[k][getContext(t, k, kprev, i, j)];
  return cdf;
}


void offsetValue(float &x,int c) {
  constexpr float Q=(1<<kMeansQuantizerShift);
  x+= kCdfs[c][0].mean_offset/Q; // add back the mean in float
}

void offsetValue(int16_t &x,int c) {
  constexpr int s=kMeansQuantizerShift-kInputQuantizerShift;
  x= (x<<kInputQuantizerShift)+(kCdfs[c][0].mean_offset>>s); // add back
}


void offsetLatent(sadl::Tensor<float> &t) {
  auto dims = t.dims();
  constexpr float Q=(1<<kMeansQuantizerShift);
  for (int i = 0; i < dims[1]; ++i)
    for (int j = 0; j < dims[2]; ++j) {
      for (int k = 0; k < dims[3]; ++k) {
        t(0, i, j, k) += kCdfs[k][0].mean_offset/Q; // add back the mean in float
      }
    }
}

void offsetLatent(sadl::Tensor<int16_t> &t) {
  auto dims = t.dims();
  constexpr int s=kMeansQuantizerShift-kInputQuantizerShift;
  t.quantizer=kInputQuantizerShift;
  for (int i = 0; i < dims[1]; ++i)
    for (int j = 0; j < dims[2]; ++j) {
      for (int k = 0; k < dims[3]; ++k) {
        t(0, i, j, k) = (t(0, i, j, k)<<kInputQuantizerShift)+((kCdfs[k][0].mean_offset+(1<<(s-1)))>>s); //  add back
      }
    }
}
