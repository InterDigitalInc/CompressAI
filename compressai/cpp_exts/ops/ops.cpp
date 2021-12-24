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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float> &pmf,
                                           int precision) {
  /* NOTE(begaintj): ported from `ryg_rans` public implementation. Not optimal
   * although it's only run once per model after training. See TF/compression
   * implementation for an optimized version. */

  for (float p : pmf) {
    if (p < 0 || !std::isfinite(p)) {
      throw std::domain_error(
          std::string("Invalid `pmf`, non-finite or negative element found: ") +
          std::to_string(p));
    }
  }

  std::vector<uint32_t> cdf(pmf.size() + 1);
  cdf[0] = 0; /* freq 0 */

  std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1,
                 [=](float p) { return std::round(p * (1 << precision)); });

  const uint32_t total = std::accumulate(cdf.begin(), cdf.end(), 0);
  if (total == 0) {
    throw std::domain_error("Invalid `pmf`: at least one element must have a "
                            "non-zero probability.");
  }

  std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                 [precision, total](uint32_t p) {
                   return ((static_cast<uint64_t>(1 << precision) * p) / total);
                 });

  std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
  cdf.back() = 1 << precision;

  for (int i = 0; i < static_cast<int>(cdf.size() - 1); ++i) {
    if (cdf[i] == cdf[i + 1]) {
      /* Try to steal frequency from low-frequency symbols */
      uint32_t best_freq = ~0u;
      int best_steal = -1;
      for (int j = 0; j < static_cast<int>(cdf.size()) - 1; ++j) {
        uint32_t freq = cdf[j + 1] - cdf[j];
        if (freq > 1 && freq < best_freq) {
          best_freq = freq;
          best_steal = j;
        }
      }

      assert(best_steal != -1);

      if (best_steal < i) {
        for (int j = best_steal + 1; j <= i; ++j) {
          cdf[j]--;
        }
      } else {
        assert(best_steal > i);
        for (int j = i + 1; j <= best_steal; ++j) {
          cdf[j]++;
        }
      }
    }
  }

  assert(cdf[0] == 0);
  assert(cdf.back() == (1 << precision));
  for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
    assert(cdf[i + 1] > cdf[i]);
  }

  return cdf;
}

PYBIND11_MODULE(_CXX, m) {
  m.attr("__name__") = "compressai._CXX";

  m.doc() = "C++ utils";

  m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf,
        "Return quantized CDF for a given PMF");
}
