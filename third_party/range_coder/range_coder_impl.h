#pragma once

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

class RangeEncoderImpl {
public:
  void encode(int32_t lower, int32_t upper, int precision, std::ostream &sink);
  void finalize(std::ostream &sink);
  void reset() {
    _base = 0;
    _size = std::numeric_limits<uint32_t>::max();
    _delay = 0;
  }

private:
  uint32_t _base = 0;
  uint32_t _size = std::numeric_limits<uint32_t>::max();
  uint64_t _delay = 0;
};

class RangeDecoderImpl {
public:
  void reset() {
    _size = std::numeric_limits<uint32_t>::max();
    _base = 0;
    _value = 0;
    init_ = false;
  }
  int32_t decode(std::istream &source, const int32_t *const cdf_begin, const int32_t *const cdf_end, int precision);

private:
  void read16bitvalue(std::istream &source);
  uint32_t _base = 0;
  uint32_t _size = std::numeric_limits<uint32_t>::max();
  uint32_t _value = 0;
  bool init_ = false;
};
