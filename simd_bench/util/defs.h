#pragma once

#if defined(_MSC_VER)
#define no_inline __declspec(noinline)
#elif defined(__clang__)
#define no_inline __attribute__((noinline))
#else
#define no_inline
#endif