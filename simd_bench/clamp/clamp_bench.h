#pragma once

#include <benchmark/benchmark.h>

#include "util/buff_utils.h"

#include "clamp_scalar.h"
#include "clamp_sse.h"
#include "clamp_avx.h"

constexpr size_t RND_SEED = 651987;
constexpr float DEFAULT_MIN = -1.5f;
constexpr float DEFAULT_MAX = 1.5f;
constexpr size_t COUNT = 40 * 1000 * 1000;
using in_place_buff_op_func = void(*)(float*, int);

template<in_place_buff_op_func Func, size_t Size, size_t Alignment>
static void BenchF32BufferOp(benchmark::State& state)
{
	auto buff = f32buffer<Alignment>::create_random(Size, RND_SEED, DEFAULT_MIN, DEFAULT_MAX);
	for ([[maybe_unused]] auto _ : state) {
		clamp_f32_buff_scalar(buff.ptr(), (int)buff.len());
		benchmark::DoNotOptimize(buff.ptr());
	}
}

#define MAKE_F32BUFF_BENCH(Align, Size, Func) \
	static void Align ## __ ## Size ## __ ## Func (benchmark::State& state) \
	{	\
		BenchF32BufferOp<Func, Size, Align>(state); \
	} \
	BENCHMARK(Align ## __ ## Size ## __ ## Func)


//Unaligned buffer
MAKE_F32BUFF_BENCH(FLOAT_ALIGN, COUNT, clamp_f32_buff_scalar);
MAKE_F32BUFF_BENCH(FLOAT_ALIGN, COUNT, clamp_f32_buff_sse_unaligned);
MAKE_F32BUFF_BENCH(FLOAT_ALIGN, COUNT, clamp_f32_buff_avx_unaligned);

//SSE Aligned
MAKE_F32BUFF_BENCH(SSE_ALIGN, COUNT, clamp_f32_buff_scalar);
MAKE_F32BUFF_BENCH(SSE_ALIGN, COUNT, clamp_f32_buff_sse_unaligned);
MAKE_F32BUFF_BENCH(SSE_ALIGN, COUNT, clamp_f32_buff_sse);
MAKE_F32BUFF_BENCH(SSE_ALIGN, COUNT, clamp_f32_buff_avx_unaligned);

//AVX Aligned
MAKE_F32BUFF_BENCH(AVX_ALIGN, COUNT, clamp_f32_buff_scalar);
MAKE_F32BUFF_BENCH(AVX_ALIGN, COUNT, clamp_f32_buff_sse_unaligned);
MAKE_F32BUFF_BENCH(AVX_ALIGN, COUNT, clamp_f32_buff_sse);
MAKE_F32BUFF_BENCH(AVX_ALIGN, COUNT, clamp_f32_buff_avx_unaligned);
MAKE_F32BUFF_BENCH(AVX_ALIGN, COUNT, clamp_f32_buff_avx);
