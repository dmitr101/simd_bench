cmake_minimum_required(VERSION 3.8)

add_executable(bench "main.cpp"  "clamp/clamp_scalar.h" "clamp/clamp_sse.h" "clamp/clamp_bench.h" "util/f32_buff_utils.h" "util/defs.h")
set_property(TARGET bench PROPERTY CXX_STANDARD 17)
set_property(TARGET bench PROPERTY CXX_EXTENSIONS False)

if (MSVC)
    add_compile_options(/W4 /WX)
    add_compile_options(/arch:AVX2) # A solid guess
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
    add_compile_options(-march=native)
endif()

#google bench lib
find_package(benchmark CONFIG REQUIRED)
find_path(BENCHMARK_INCLUDE_DIR benchmark/benchmark.h)
include_directories(${BENCHMARK_INCLUDE_DIR})
target_link_libraries(bench PRIVATE benchmark::benchmark benchmark::benchmark_main)

