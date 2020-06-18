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

class rANSEncoderInterface {
public:
  rANSEncoderInterface() = default;

  rANSEncoderInterface(const rANSEncoderInterface &) = delete;
  rANSEncoderInterface(rANSEncoderInterface &&) = delete;
  rANSEncoderInterface &operator=(const rANSEncoderInterface &) = delete;
  rANSEncoderInterface &operator=(rANSEncoderInterface &&) = delete;

  py::bytes encode_with_indexes(const std::vector<int32_t> &symbols,
                                const std::vector<int32_t> &indexes,
                                const std::vector<std::vector<int32_t>> &cdfs,
                                const std::vector<int32_t> &cdfs_sizes,
                                const std::vector<int32_t> &offsets);
};

class rANSDecoderInterface {
public:
  rANSDecoderInterface() = default;

  rANSDecoderInterface(const rANSDecoderInterface &) = delete;
  rANSDecoderInterface(rANSDecoderInterface &&) = delete;
  rANSDecoderInterface &operator=(const rANSDecoderInterface &) = delete;
  rANSDecoderInterface &operator=(rANSDecoderInterface &&) = delete;

  std::vector<int32_t>
  decode_with_indexes(const std::string &encoded,
                      const std::vector<int32_t> &indexes,
                      const std::vector<std::vector<int32_t>> &cdfs,
                      const std::vector<int32_t> &cdfs_sizes,
                      const std::vector<int32_t> &offsets);

  void init_decode(const std::string &encoded);

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
