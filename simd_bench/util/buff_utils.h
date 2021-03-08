#pragma once
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <random>

constexpr size_t FLOAT_ALIGN = 4;
constexpr size_t SSE_ALIGN = 16;
constexpr size_t AVX_ALIGN = 32;

#ifdef _MSC_VER
inline void* aligned_alloc(size_t align, size_t size)
{
	return _aligned_malloc(size, align);
}
#endif

template<size_t Alignment = FLOAT_ALIGN>
struct f32buffer
{
	struct deleter
	{
		void operator()(void* ptr)
		{
#ifdef _MSC_VER
			_aligned_free(ptr);
#else
			free(ptr);
#endif
		}
	};

	std::unique_ptr<float, deleter> data;
	size_t length;

	float* ptr() { return data.get(); }
	size_t len() { return length; }

	static f32buffer create_uninit(size_t len)
	{
		float* mem = static_cast<float*>(aligned_alloc(Alignment, len * sizeof(float)));
		std::uninitialized_default_construct_n(mem, len);
		f32buffer result;
		result.data.reset(mem);
		result.length = len;
		return result;
	}

	static f32buffer create_random(size_t len, size_t seed, float min, float max)
	{
		auto result = create_uninit(len);
		std::mt19937_64 eng(seed);
		std::uniform_real_distribution<float> distr(min, max);
		std::generate_n(result.data.get(), len, [&eng, &distr]() {return distr(eng); });
		return result;
	}
};
