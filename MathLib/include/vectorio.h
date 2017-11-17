#pragma once

#include "vector.h"

#include <iostream>
#include <iterator>
#include <string>

namespace mathlib {

template <typename CharT>
struct separator {
    static const CharT* const value;
};
template<>
const char* const separator<char>::value = ", ";
template<>
const wchar_t* const separator<wchar_t>::value = L", ";

template <typename T, size_t N, typename CharT>
auto& operator<<(std::basic_ostream<CharT>& os, const Vector<T, N>& x) {
    std::copy(x.begin(), std::prev(x.end()),
              std::ostream_iterator<T, CharT>{os << "{", separator<CharT>::value});
    return os << *std::prev(x.end()) << "}";
}

}  // namespace mathlib
