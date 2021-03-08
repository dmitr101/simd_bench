cmake_minimum_required(VERSION 3.8)

if (MSVC)
    message("Adding MSVC compile options")

    add_compile_options(/W4 /WX)
    add_compile_options(/arch:AVX2) # A solid guess
else()
    message("Adding Clang/GCC compile options")

    add_compile_options(-Wall -Wextra -pedantic -Werror)
    add_compile_options(-mavx2)
endif()

add_executable(bench "main.cpp")
set_property(TARGET bench PROPERTY CXX_STANDARD 17)
set_property(TARGET bench PROPERTY CXX_EXTENSIONS False)

#google bench lib
find_package(benchmark CONFIG REQUIRED)
find_path(BENCHMARK_INCLUDE_DIR benchmark/benchmark.h)
include_directories(${BENCHMARK_INCLUDE_DIR})
target_link_libraries(bench PRIVATE benchmark::benchmark benchmark::benchmark_main)
