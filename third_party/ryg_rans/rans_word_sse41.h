// Word-aligned SSE 4.1 rANS encoder/decoder - public domain - Fabian 'ryg' Giesen
//
// This implementation has a regular rANS encoder and a 4-way interleaved SIMD
// decoder. Like rans_byte.h, it's intended to illustrate the idea, not to
// be used as a drop-in arithmetic coder.

#ifndef RANS_WORD_SSE41_HEADER
#define RANS_WORD_SSE41_HEADER

#include <stdint.h>
#include <smmintrin.h>

// READ ME FIRST:
//
// The intention in this version is to demonstrate a design where the decoder
// is made as fast as possible, even when it makes the encoder slightly slower
// or hurts compression a bit. (The code in rans_byte.h, with the 31-bit
// arithmetic to allow for faster division by constants, is a more "balanced"
// approach).
//
// This version is intended to be used with relatively low-resolution
// probability distributions (scale_bits=12 or less). In these regions, the
// "fully unrolled" table-based approach shown here (suggested by "enotuss"
// on my blog) is optimal; for larger scale_bits, other approaches are more
// favorable. It also only assumes an 8-bit symbol alphabet for simplicity.
//
// Unlike rans_byte.h, this file needs to be compiled as C++.

// --------------------------------------------------------------------------

// This coder uses L=1<<16 and B=1<<16 (16-bit word based renormalization).
// Since we still continue to use 32-bit words, this means we require
// scale_bits <= 16; on the plus side, renormalization never needs to
// iterate.
#define RANS_WORD_L (1u << 16)

#define RANS_WORD_SCALE_BITS 12
#define RANS_WORD_M (1u << RANS_WORD_SCALE_BITS)

#define RANS_WORD_NSYMS 256

typedef uint32_t RansWordEnc;
typedef uint32_t RansWordDec;

typedef union {
    __m128i simd;
    uint32_t lane[4];
} RansSimdDec;

union RansWordSlot {
    uint32_t u32;
    struct {
        uint16_t freq;
        uint16_t bias;
    };
};

struct RansWordTables {
    RansWordSlot slots[RANS_WORD_M];
    uint8_t slot2sym[RANS_WORD_M];
};

// Initialize slots for a symbol in the table
static inline void RansWordTablesInitSymbol(RansWordTables* tab, uint8_t sym, uint32_t start, uint32_t freq)
{
    for (uint32_t i=0; i < freq; i++) {
        uint32_t slot = start + i;
        tab->slot2sym[slot] = sym;
        tab->slots[slot].freq = (uint16_t)freq;
        tab->slots[slot].bias = (uint16_t)i;
    }
}

// Initialize a rANS encoder
static inline RansWordEnc RansWordEncInit()
{
    return RANS_WORD_L;
}

// Encodes a single symbol with range "start" and frequency "freq".
static inline void RansWordEncPut(RansWordEnc* r, uint16_t** pptr, uint32_t start, uint32_t freq)
{
    // renormalize
    uint32_t x = *r;
    if (x >= ((RANS_WORD_L >> RANS_WORD_SCALE_BITS) << 16) * freq) {
        *pptr -= 1;
        **pptr = (uint16_t) (x & 0xffff);
        x >>= 16;
    }

    // x = C(s,x)
    *r = ((x / freq) << RANS_WORD_SCALE_BITS) + (x % freq) + start;
}

// Flushes the rANS encoder
static inline void RansWordEncFlush(RansWordEnc* r, uint16_t** pptr)
{
    uint32_t x = *r;
    uint16_t* ptr = *pptr;

    ptr -= 2;
    ptr[0] = (uint16_t) (x >> 0);
    ptr[1] = (uint16_t) (x >> 16);

    *pptr = ptr;
}

// Initializes a rANS decoder.
static inline void RansWordDecInit(RansWordDec* r, uint16_t** pptr)
{
    uint32_t x;
    uint16_t* ptr = *pptr;

    x  = ptr[0] << 0;
    x |= ptr[1] << 16;
    ptr += 2;

    *pptr = ptr;
    *r = x;
}

// Decodes a symbol using the given tables.
static inline uint8_t RansWordDecSym(RansWordDec* r, RansWordTables const* tab)
{
    uint32_t x = *r;
    uint32_t slot = x & (RANS_WORD_M - 1);

    // s, x = D(x)
    *r = tab->slots[slot].freq * (x >> RANS_WORD_SCALE_BITS) + tab->slots[slot].bias;
    return tab->slot2sym[slot];
}

// Renormalize after decoding a symbol.
static inline void RansWordDecRenorm(RansWordDec* r, uint16_t** pptr)
{
    uint32_t x = *r;
    if (x < RANS_WORD_L) {
        *r = (x << 16) | **pptr;
        *pptr += 1;
    }
}

// Initializes a SIMD rANS decoder.
static inline void RansSimdDecInit(RansSimdDec* r, uint16_t** pptr)
{
    r->simd = _mm_loadu_si128((const __m128i*)*pptr);
    *pptr += 2*4;
}

// Decodes a four symbols in parallel using the given tables.
static inline uint32_t RansSimdDecSym(RansSimdDec* r, RansWordTables const* tab)
{
    __m128i freq_bias_lo, freq_bias_hi, freq_bias;
    __m128i freq, bias;
    __m128i xscaled;
    __m128i x = r->simd;
    __m128i slots = _mm_and_si128(x, _mm_set1_epi32(RANS_WORD_M - 1));
    uint32_t i0 = (uint32_t) _mm_cvtsi128_si32(slots);
    uint32_t i1 = (uint32_t) _mm_extract_epi32(slots, 1);
    uint32_t i2 = (uint32_t) _mm_extract_epi32(slots, 2);
    uint32_t i3 = (uint32_t) _mm_extract_epi32(slots, 3);

    // symbol
    uint32_t s = tab->slot2sym[i0] | (tab->slot2sym[i1] << 8) | (tab->slot2sym[i2] << 16) | (tab->slot2sym[i3] << 24);

    // gather freq_bias
    freq_bias_lo = _mm_cvtsi32_si128(tab->slots[i0].u32);
    freq_bias_lo = _mm_insert_epi32(freq_bias_lo, tab->slots[i1].u32, 1);
    freq_bias_hi = _mm_cvtsi32_si128(tab->slots[i2].u32);
    freq_bias_hi = _mm_insert_epi32(freq_bias_hi, tab->slots[i3].u32, 1);
    freq_bias = _mm_unpacklo_epi64(freq_bias_lo, freq_bias_hi);

    // s, x = D(x)
    xscaled = _mm_srli_epi32(x, RANS_WORD_SCALE_BITS);
    freq = _mm_and_si128(freq_bias, _mm_set1_epi32(0xffff));
    bias = _mm_srli_epi32(freq_bias, 16);
    r->simd = _mm_add_epi32(_mm_mullo_epi32(xscaled, freq), bias);
    return s;
}

// Renormalize after decoding a symbol.
static inline void RansSimdDecRenorm(RansSimdDec* r, uint16_t** pptr)
{
    static ALIGNSPEC(int8_t const, shuffles[16][16], 16) = {
#define _ -1 // for readability
        { _,_,_,_, _,_,_,_, _,_,_,_, _,_,_,_ }, // 0000
        { 0,1,_,_, _,_,_,_, _,_,_,_, _,_,_,_ }, // 0001
        { _,_,_,_, 0,1,_,_, _,_,_,_, _,_,_,_ }, // 0010
        { 0,1,_,_, 2,3,_,_, _,_,_,_, _,_,_,_ }, // 0011
        { _,_,_,_, _,_,_,_, 0,1,_,_, _,_,_,_ }, // 0100
        { 0,1,_,_, _,_,_,_, 2,3,_,_, _,_,_,_ }, // 0101
        { _,_,_,_, 0,1,_,_, 2,3,_,_, _,_,_,_ }, // 0110
        { 0,1,_,_, 2,3,_,_, 4,5,_,_, _,_,_,_ }, // 0111
        { _,_,_,_, _,_,_,_, _,_,_,_, 0,1,_,_ }, // 1000
        { 0,1,_,_, _,_,_,_, _,_,_,_, 2,3,_,_ }, // 1001
        { _,_,_,_, 0,1,_,_, _,_,_,_, 2,3,_,_ }, // 1010
        { 0,1,_,_, 2,3,_,_, _,_,_,_, 4,5,_,_ }, // 1011
        { _,_,_,_, _,_,_,_, 0,1,_,_, 2,3,_,_ }, // 1100
        { 0,1,_,_, _,_,_,_, 2,3,_,_, 4,5,_,_ }, // 1101
        { _,_,_,_, 0,1,_,_, 2,3,_,_, 4,5,_,_ }, // 1110
        { 0,1,_,_, 2,3,_,_, 4,5,_,_, 6,7,_,_ }, // 1111
#undef _
    };
    static uint8_t const numbits[16] = {
        0,1,1,2, 1,2,2,3, 1,2,2,3, 2,3,3,4
    };

    __m128i x = r->simd;

    // NOTE: SSE2+ only offer a signed 32-bit integer compare, while we
    // need unsigned. So we subtract 0x80000000 before the compare,
    // which converts unsigned integers to signed integers in an
    // order-preserving manner.
    __m128i x_biased = _mm_xor_si128(x, _mm_set1_epi32((int) 0x80000000));
    __m128i greater = _mm_cmpgt_epi32(_mm_set1_epi32(RANS_WORD_L - 0x80000000), x_biased);
    unsigned int mask = _mm_movemask_ps(_mm_castsi128_ps(greater));

    // NOTE: this will read slightly past the end of the input buffer.
    // In practice, either pad the input buffer by 8 bytes at the end,
    // or switch to the non-SIMD version once you get close to the end.
    __m128i memvals = _mm_loadl_epi64((const __m128i*)*pptr);
    __m128i xshifted = _mm_slli_epi32(x, 16);
    __m128i shufmask = _mm_load_si128((const __m128i*)shuffles[mask]);
    __m128i newx = _mm_or_si128(xshifted, _mm_shuffle_epi8(memvals, shufmask));
    r->simd = _mm_blendv_epi8(x, newx, greater);
    *pptr += numbits[mask];
}

#endif // RANS_WORD_SSE41_HEADER
