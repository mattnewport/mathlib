#pragma once

namespace mathlib {

template<typename T>
T clamp(T x, T a, T b) {
    return x < a ? a : (x > b ? b : x);
}

template<typename T>
T saturate(T x) {
    return clamp(x, T{ 0 }, T{ 1 });
}

template<typename T>
T square(T a) { return a * a; }

}
