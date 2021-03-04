#include <random>
#include <algorithm>
#include <xmmintrin.h>

#include <benchmark/benchmark.h>


inline float clamp(float x, float min, float max)
{
	return x < min ? min : (x < max ? x : max);
}

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

void clamp_f32_buff_scalar(float* __restrict buff, int len)
{
	for (int i = 0; i < len; ++i)
	{
		buff[i] = clamp(buff[i], -1.f, 1.f);
	}
}

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

std::vector<float> create_floats(int num, int seed)
{
	std::vector<float> result(num);
	std::mt19937 eng(seed);
	std::uniform_real_distribution<float> distr(-1.3f, 1.3f);
	std::generate_n(result.begin(), num, [&eng, &distr]() {return distr(eng); });
	return result;
}

std::pair<float*, int> get_simd_aligned_sub_span(std::vector<float>& vec)
{
	const int byte_align = 16;
	const int float_align = 4;

	float* data = vec.data();
	int num = static_cast<int>(vec.size());
	int align_rem = (uintptr_t)data % (uintptr_t)byte_align;
	int elms = align_rem / sizeof(float);

	data += ((size_t)float_align - elms);
	num -= (float_align - elms);
	num -= num % 4;

	return { data, num };
}

constexpr int ELEM_COUNT = 2048;
constexpr int RND_SEED = 651987;

static void ScalarBuffClamp(benchmark::State& state) {
	auto v = create_floats(ELEM_COUNT, RND_SEED);
	auto p = get_simd_aligned_sub_span(v);
	for ([[maybe_unused]] auto _ : state) {
		clamp_f32_buff_scalar(p.first, p.second);
		benchmark::DoNotOptimize(p.first);
	}
}
BENCHMARK(ScalarBuffClamp);

static void SIMDBuffClamp(benchmark::State& state) {
	auto v = create_floats(ELEM_COUNT, RND_SEED);
	auto p = get_simd_aligned_sub_span(v);
	for ([[maybe_unused]] auto _ : state) {
		clamp_f32_buff_sse(p.first, p.second);
		benchmark::DoNotOptimize(p.first);
	}
}
BENCHMARK(SIMDBuffClamp);

BENCHMARK_MAIN();
