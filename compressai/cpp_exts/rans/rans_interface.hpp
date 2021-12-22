/* Copyright (c) 2021-2022, InterDigital Communications, Inc
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of InterDigital Communications, Inc nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
