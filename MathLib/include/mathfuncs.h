#pragma once

namespace mathlib {

template<typename T, typename U, typename V>
constexpr auto clamp(T x, U a, V b) {
    return x < a ? a : (x > b ? b : x);
}

template<typename T>
constexpr auto saturate(T x) {
    return clamp(x, T{ 0 }, T{ 1 });
}

template<typename T>
constexpr auto square(T a) { return a * a; }

}
