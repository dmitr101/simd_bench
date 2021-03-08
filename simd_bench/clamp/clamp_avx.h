#pragma once

#include <immintrin.h>
#include "../util/defs.h"

inline __m256 avx_masked_select(__m256 mask, __m256 a, __m256 b)
{
	return _mm256_xor_ps(b, _mm256_and_ps(mask, _mm256_xor_ps(a, b)));
}

inline __m256 avx_clamp_simd(__m256 x, __m256 min, __m256 max)
{
	__m256 min_mask = _mm256_cmp_ps(x, min, _CMP_LT_OQ);
	__m256 max_mask = _mm256_cmp_ps(x, max, _CMP_GT_OQ);

	x = avx_masked_select(min_mask, min, x);
	x = avx_masked_select(max_mask, max, x);
	return x;
}

//Base function. Requires buff to be aligned to 32 bytes and len % 8 == 0
no_inline
void clamp_f32_buff_avx(float* __restrict buff, int len)
{
	__m256 min = _mm256_setr_ps(1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f);
	__m256 max = _mm256_setr_ps(-1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f);

	for (int i = 0; i < len; i += 8)
	{
		__m256 x = _mm256_load_ps(&buff[i]);
		x = avx_clamp_simd(x, min, max);
		_mm256_store_ps(&buff[i], x);
	}
}

//Still requires len % 8 == 0 but no alignment requirements.
no_inline
void clamp_f32_buff_avx_unaligned(float* __restrict buff, int len)
{
	__m256 min = _mm256_setr_ps(1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f);
	__m256 max = _mm256_setr_ps(-1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f);

	for (int i = 0; i < len; i += 8)
	{
		__m256 x = _mm256_loadu_ps(&buff[i]);
		x = avx_clamp_simd(x, min, max);
		_mm256_storeu_ps(&buff[i], x);
	}
}
