#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include <chrono>

template <class Function, typename ...Args>
auto benchmark(Function function, Args&& ...args) {
    using std::chrono::steady_clock;
    auto start = steady_clock::now();

    function(std::forward<Args>(args)...);
    
    auto end = steady_clock::now();
    return end - start;
}

/* doNotOptimizeAway from https://stackoverflow.com/a/36781982/1935009 */
#ifdef _MSC_VER

#pragma optimize("", off)

template <class T>
void doNotOptimizeAway(T&& datum) {
  datum = datum;
}

#pragma optimize("", on)

#elif defined(__clang__)

template <class T>
__attribute__((__optnone__)) void doNotOptimizeAway(T&& /* datum */) {}

#else

template <class T>
void doNotOptimizeAway(T&& datum) {
  asm volatile("" : "+r" (datum));
}

#endif

#endif /* BENCHMARK_HPP */