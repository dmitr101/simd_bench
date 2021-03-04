cmake_minimum_required (VERSION 3.8)

find_package(benchmark CONFIG REQUIRED)
find_path(BENCHMARK_INCLUDE_DIR benchmark/benchmark.h)

add_executable (bench "main.cpp" )

include_directories(${BENCHMARK_INCLUDE_DIR})
target_link_libraries(bench PRIVATE benchmark::benchmark benchmark::benchmark_main)

