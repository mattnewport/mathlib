#pragma once

namespace mathlib {

template <typename T>
constexpr auto abs(T x) {
    return x < T{ 0 } ? -x : x;
}

template <typename T, typename U, typename V>
constexpr auto clamp(T x, U a, V b) {
    return x < a ? a : (x > b ? b : x);
}

template <typename T>
constexpr auto saturate(T x) {
    return clamp(x, T{0}, T{1});
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
}
