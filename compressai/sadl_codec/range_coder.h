#pragma once
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

#include "range_coder_impl.h"

struct Cdf {
  static constexpr int precision = 16;
  static constexpr int overflow_width = 4;
  static constexpr int max_overflow = (1 << overflow_width) - 1;
  int min_val, max_val;    // assume the cdf represents value in [min,max[
  int probable;            // value with max proba
  int mean_offset;        //  mean offset
  std::vector<int> cproba; // per construction, proba.size()=max-min+1 +1 (the last proba represent the proba for value outside the range)
};

class RangeCoder {
public:
  void encode(int value, const Cdf &cdf, std::ostream &out);
  void finalize(std::ostream &out) { coder_.finalize(out); }
  double entropy = 0.;

private:
  RangeEncoderImpl coder_;
};

class RangeDecoder {
public:
  int decode(std::istream &in, const Cdf &cdf);

private:
  RangeDecoderImpl decoder_;
};
