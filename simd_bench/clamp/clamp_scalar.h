#pragma once
#include "util/defs.h"

inline float clamp(float x, float min, float max)
{
	return x < min ? min : (x < max ? x : max);
}

no_inline
void clamp_f32_buff_scalar(float* __restrict buff, int len)
{
	for (int i = 0; i < len; ++i)
	{
		buff[i] = clamp(buff[i], -1.f, 1.f);
	}
}
