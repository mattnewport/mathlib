#pragma once

#include "vector.h"
#include "matrix.h"

#include <iostream>
#include <iterator>
#include <string>

namespace mathlib {

template <typename CharT>
struct separator {
    static const CharT* const value;
};
const char* const separator<char>::value = ", ";
const wchar_t* const separator<wchar_t>::value = L", ";

template <typename T, size_t N, typename CharT>
auto& operator<<(std::basic_ostream<CharT>& os, const Vector<T, N>& x) {
    std::copy(x.begin(), std::prev(x.end()),
              std::ostream_iterator<T, CharT>{os << "{", separator<CharT>::value});
    return os << *std::prev(x.end()) << "}";
}

template <typename T, size_t M, size_t N, typename CharT>
auto& operator<<(std::basic_ostream<CharT>& os, const Matrix<T, M, N>& x) {
    return os << x.rows();
}

template <typename T, typename CharT>
auto& operator<<(std::basic_ostream<CharT>& os, const Quaternion<T>& x) {
    return os << x.vec4();
}

}  // namespace mathlib
