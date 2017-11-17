#pragma once

#include <algorithm>

namespace mathlib {

template <typename T>
constexpr auto saturate(T x) {
    return std::clamp(x, T{0}, T{1});
}

template <typename T>
constexpr auto square(T a) {
    return a * a;
}

template <typename T>
constexpr auto times2(T x) {
    return x + x;
}

template <typename T>
constexpr auto lerp(T a, T b, T t) {
    return a + (b - a) * t;
}

}  // namespace mathlib
