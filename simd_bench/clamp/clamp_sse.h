#pragma once

#include <xmmintrin.h>
#include "util/defs.h"

inline __m128 masked_select(__m128 mask, __m128 a, __m128 b)
{
	return _mm_xor_ps(b, _mm_and_ps(mask, _mm_xor_ps(a, b)));
}

inline __m128 clamp_simd(__m128 x, __m128 min, __m128 max)
{
	__m128 min_mask = _mm_cmplt_ps(x, min);
	__m128 max_mask = _mm_cmpgt_ps(x, max);

	x = masked_select(min_mask, min, x);
	x = masked_select(max_mask, max, x);
	return x;
}

//Base function. Requires buff to be aligned to 16 bytes and len % 4 == 0
no_inline
void clamp_f32_buff_sse(float* __restrict buff, int len)
{
	__m128 min = _mm_setr_ps(1.f, 1.f, 1.f, 1.f);
	__m128 max = _mm_setr_ps(-1.f, -1.f, -1.f, -1.f);

	for (int i = 0; i < len; i += 4)
	{
		__m128 x = _mm_load_ps(&buff[i]);
		x = clamp_simd(x, min, max);
		_mm_store_ps(&buff[i], x);
	}
}

//Still requires len % 4 == 0 but no alignment requirements.
no_inline
void clamp_f32_buff_sse_unaligned(float* __restrict buff, int len)
{
	__m128 min = _mm_setr_ps(1.f, 1.f, 1.f, 1.f);
	__m128 max = _mm_setr_ps(-1.f, -1.f, -1.f, -1.f);

	for (int i = 0; i < len; i += 4)
	{
		__m128 x = _mm_loadu_ps(&buff[i]);
		x = clamp_simd(x, min, max);
		_mm_store_ps(&buff[i], x);
	}
}