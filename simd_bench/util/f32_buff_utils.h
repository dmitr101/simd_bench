#pragma once
#include <random>
#include <algorithm>
#include <gsl/span>

constexpr int SSE_FLOAT_ALIGN = 4;
constexpr int AVX_FLOAT_ALIGN = 8;

std::vector<float> create_floats(int num, int seed, std::pair<float, float> minmax)
{
	std::vector<float> result(num);
	std::mt19937 eng(seed);
	std::uniform_real_distribution<float> distr(minmax.first, minmax.second);
	std::generate_n(result.begin(), num, [&eng, &distr]() {return distr(eng); });
	return result;
}

gsl::span<float> get_simd_aligned_sub_span(std::vector<float>& vec, int float_align)
{
	const int byte_align = float_align * sizeof(float);

	float* data = vec.data();
	size_t num = vec.size();
	int align_rem = (uintptr_t)data % (uintptr_t)byte_align;
	int elems = align_rem / sizeof(float);

	data += ((size_t)float_align - elems);
	num -= (float_align - elems);
	num -= num % 4;

	return { data, num };
}