#pragma once

#include <benchmark/benchmark.h>

#include "util/f32_buff_utils.h"

#include "clamp_scalar.h"
#include "clamp_sse.h"

constexpr int ELEM_COUNT = 2 * 1024; //8 megs of floats(assuming sizeof(float) == 4)
constexpr int RND_SEED = 651987;
constexpr std::pair<float, float> DEFAULT_RANGE = { -1.5f, 1.5f };

static void ScalarBuffClampAligned(benchmark::State& state) {
	auto v = create_floats(ELEM_COUNT, RND_SEED, DEFAULT_RANGE);
	auto p = get_simd_aligned_sub_span(v, SSE_FLOAT_ALIGN);
	for ([[maybe_unused]] auto _ : state) {
		clamp_f32_buff_scalar(p.first, p.second);
		benchmark::DoNotOptimize(p.first);
	}
}
BENCHMARK(ScalarBuffClampAligned);

static void ScalarBuffClampUnaligned(benchmark::State& state) {
	auto v = create_floats(ELEM_COUNT, RND_SEED, DEFAULT_RANGE);
	for ([[maybe_unused]] auto _ : state) {
		clamp_f32_buff_scalar(v.data(), (int)v.size());
		benchmark::DoNotOptimize(v.data());
	}
}
BENCHMARK(ScalarBuffClampUnaligned);

static void SIMDBuffClampAligned(benchmark::State& state) {
	auto v = create_floats(ELEM_COUNT, RND_SEED, DEFAULT_RANGE);
	auto p = get_simd_aligned_sub_span(v, SSE_FLOAT_ALIGN);
	for ([[maybe_unused]] auto _ : state) {
		clamp_f32_buff_sse(p.first, p.second);
		benchmark::DoNotOptimize(p.first);
	}
}
BENCHMARK(SIMDBuffClampAligned);

static void SIMDBuffClampUnaligned(benchmark::State& state) {
	auto v = create_floats(ELEM_COUNT, RND_SEED, DEFAULT_RANGE);
	auto passed_size = v.size() - (v.size() % 4);
	for ([[maybe_unused]] auto _ : state) {
		clamp_f32_buff_sse_unaligned(v.data(), (int)passed_size);
		benchmark::DoNotOptimize(v.data());
	}
}
BENCHMARK(SIMDBuffClampUnaligned);