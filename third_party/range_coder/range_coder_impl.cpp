#include "range_coder_impl.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

void RangeEncoderImpl::encode(int32_t lower, int32_t upper, int precision, std::ostream &sink) {
  assert(precision > 0);
  assert(precision <= 16);
  assert(lower >= 0);
  assert(upper > lower);
  assert((1 << precision) >= upper);

  const uint64_t size = static_cast<uint64_t>(_size) + 1;
  assert((size >> 16) != 0);

  const uint32_t a = (size * static_cast<uint64_t>(lower)) >> precision;
  const uint32_t b = ((size * static_cast<uint64_t>(upper)) >> precision) - 1;
  assert(a <= b);

  _base += a;
  _size = b - a;
  const bool base_overflow = (_base < a);

  if ((_base + _size) < _base) {
    assert((((_base - a) + size) >> 32) != 0);
    assert((_delay & 0xFFFF) != 0);

    if ((_size >> 16) == 0) {
      assert((_base >> 16) == 0xFFFF);

      _base <<= 16;
      _size <<= 16;
      _size |= 0xFFFF;

      assert(_delay < (static_cast<uint64_t>(1) << 62));
      _delay += 0x20000;
    }
    return;
  }

  if (_delay != 0) {
    if (base_overflow) {
      assert(((static_cast<uint64_t>(_base - a) + a) >> 32) != 0);
      char c[2] = {static_cast<char>(_delay >> 8), static_cast<char>(_delay >> 0)}; // just delay &FFFF ?
      sink.write(c, 2);
      c[0] = 0;
      for (int i = 0; i < (int)(_delay >> 16); ++i)
        sink.write(c, 1);
    } else {
      assert((static_cast<uint64_t>(_base + _size) >> 32) == 0);
      --_delay;
      char c[2] = {static_cast<char>(_delay >> 8), static_cast<char>(_delay >> 0)}; // just delay &FFFF ?
      sink.write(c, 2);
      c[0] = static_cast<unsigned char>(0xFF);
      for (int i = 0; i < (int)(_delay >> 16); ++i)
        sink.write(c, 1);
    }
    _delay = 0;
  }

  if ((_size >> 16) == 0) {
    const uint32_t top = _base >> 16;
    _base <<= 16;
    _size <<= 16;
    _size |= 0xFFFF;
    if (_base <= (_base + _size)) {
      char c[2] = {static_cast<char>(top >> 8), static_cast<char>(top)};
      sink.write(c, 2);
    } else {
      assert(top < 0xFFFF);
      _delay = top + 1;
    }
  }
}

void RangeEncoderImpl::finalize(std::ostream &sink) {
  char c[1];
  if (_delay != 0) {
    c[0] = static_cast<char>(_delay >> 8);
    sink.write(c, 1);
    if ((_delay & 0xFF) != 0) {
      c[0] = static_cast<char>(_delay);
      sink.write(c, 1);
    }
  } else if (_base != 0) {
    const uint32_t mid = ((_base - 1) >> 16) + 1;
    assert((mid & 0xFFFF) == mid);
    c[0] = static_cast<char>(mid >> 8);
    sink.write(c, 1);
    if ((mid & 0xFF) != 0) {
      c[0] = static_cast<char>(mid >> 0);
      sink.write(c, 1);
    }
  }
  _base = 0;
  _delay = 0;
  _size = std::numeric_limits<uint32_t>::max();
}

void RangeDecoderImpl::read16bitvalue(std::istream &in) {
  _value <<= 8;
  if (in) {
    uint8_t c;
    in.read((char *)&c, 1);
    if (in)
      _value |= c;
  }
  _value <<= 8;
  if (in && !in.eof()) {
    uint8_t c;
    in.read((char *)&c, 1);
    if (in)
      _value |= c;
  }
}

int32_t RangeDecoderImpl::decode(std::istream &source, const int32_t *const cdf_begin, const int32_t *const cdf_end, int precision) {
  assert(precision > 0);
  assert(precision <= 16);
  if (!init_) {
    read16bitvalue(source);
    read16bitvalue(source);
    init_ = true;
  }

  const uint64_t size = static_cast<uint64_t>(_size) + 1;
  const uint64_t offset = ((static_cast<uint64_t>(_value - _base) + 1) << precision) - 1;

  const int32_t *pv = cdf_begin + 1;
  const auto cdf_size = cdf_end - cdf_begin;
  auto len = cdf_size - 1;
  assert(len > 0);

  do {
    const auto half = len / 2;
    const int32_t *mid = pv + half;
    assert(*mid >= 0);
    assert(*mid <= (1 << precision));
    if ((size * static_cast<uint64_t>(*mid)) <= offset) {
      pv = mid + 1;
      len -= half + 1;
    } else {
      len = half;
    }
  } while (len > 0);

  assert(pv < (cdf_begin + cdf_size));

  const uint32_t a = (size * static_cast<uint64_t>(*(pv - 1))) >> precision;
  const uint32_t b = ((size * static_cast<uint64_t>(*pv)) >> precision) - 1;
  assert(a <= (offset >> precision));
  assert(b >= (offset >> precision));

  _base += a;
  _size = b - a;

  if ((_size >> 16) == 0) {
    _base <<= 16;
    _size <<= 16;
    _size |= 0xFFFF;

    read16bitvalue(source);
  }

  return pv - cdf_begin - 1;
}
