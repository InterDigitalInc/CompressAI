/* Copyright 2020 InterDigital Communications, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rans64.h"

namespace py = pybind11;

struct RansSymbol {
  uint16_t start;
  uint16_t range;
  bool bypass; // bypass flag to write raw bits to the stream
};

/* NOTE: Warning, we buffer everything for now... In case of large files we
 * should split the bitstream into chunks... Or for a memory-bounded encoder
 **/
class BufferedRansEncoder {
public:
  BufferedRansEncoder() = default;

  BufferedRansEncoder(const BufferedRansEncoder &) = delete;
  BufferedRansEncoder(BufferedRansEncoder &&) = delete;
  BufferedRansEncoder &operator=(const BufferedRansEncoder &) = delete;
  BufferedRansEncoder &operator=(BufferedRansEncoder &&) = delete;

  void encode_with_indexes(const std::vector<int32_t> &symbols,
                           const std::vector<int32_t> &indexes,
                           const std::vector<std::vector<int32_t>> &cdfs,
                           const std::vector<int32_t> &cdfs_sizes,
                           const std::vector<int32_t> &offsets);
  py::bytes flush();

private:
  std::vector<RansSymbol> _syms;
};

class RansEncoder {
public:
  RansEncoder() = default;

  RansEncoder(const RansEncoder &) = delete;
  RansEncoder(RansEncoder &&) = delete;
  RansEncoder &operator=(const RansEncoder &) = delete;
  RansEncoder &operator=(RansEncoder &&) = delete;

  py::bytes encode_with_indexes(const std::vector<int32_t> &symbols,
                                const std::vector<int32_t> &indexes,
                                const std::vector<std::vector<int32_t>> &cdfs,
                                const std::vector<int32_t> &cdfs_sizes,
                                const std::vector<int32_t> &offsets);
};

class RansDecoder {
public:
  RansDecoder() = default;

  RansDecoder(const RansDecoder &) = delete;
  RansDecoder(RansDecoder &&) = delete;
  RansDecoder &operator=(const RansDecoder &) = delete;
  RansDecoder &operator=(RansDecoder &&) = delete;

  std::vector<int32_t>
  decode_with_indexes(const std::string &encoded,
                      const std::vector<int32_t> &indexes,
                      const std::vector<std::vector<int32_t>> &cdfs,
                      const std::vector<int32_t> &cdfs_sizes,
                      const std::vector<int32_t> &offsets);

  void set_stream(const std::string &stream);

  std::vector<int32_t>
  decode_stream(const std::vector<int32_t> &indexes,
                const std::vector<std::vector<int32_t>> &cdfs,
                const std::vector<int32_t> &cdfs_sizes,
                const std::vector<int32_t> &offsets);

private:
  Rans64State _rans;
  std::string _stream;
  uint32_t *_ptr;
};
