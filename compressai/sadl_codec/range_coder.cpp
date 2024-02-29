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
#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

using namespace std;

void RangeCoder::encode(int value, const Cdf &cdf, std::ostream &out) {
  value -= cdf.min_val;

  if (value < 0 || value > (int)cdf.cproba.size() - 1) {
    cout << "[ERROR] overflow support deactivated " << value << "[" << cdf.min_val << ' ' << cdf.max_val << "] n=" << cdf.cproba.size()
         << endl;
    exit(-1);
  }
  entropy += log2(1 << Cdf::precision) - log2(cdf.cproba[value + 1] - cdf.cproba[value]);
  coder_.encode(cdf.cproba[value], cdf.cproba[value + 1], Cdf::precision, out);
}

int RangeDecoder::decode(std::istream &in, const Cdf &cdf) {

  int value = decoder_.decode(in, cdf.cproba.data(), cdf.cproba.data() + cdf.cproba.size(), Cdf::precision);

  if (value == (int)cdf.cproba.size() - 1) {
    cout << "[ERROR] overflow support deactivated " << endl;
    exit(-1);
  }
  return value + cdf.min_val;
}
