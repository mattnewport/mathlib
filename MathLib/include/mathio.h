#pragma once

#include "vector.h"

#include <iostream>
#include <iterator>
#include <string>

namespace mathlib {

template <typename CharType>
struct separator {
    static const CharType* const value;
};
const char* const separator<char>::value = ", ";
const wchar_t* const separator<wchar_t>::value = L", ";

template <typename T, size_t N, typename CharType>
std::basic_ostream<CharType>& operator<<(std::basic_ostream<CharType>& os, const Vector<T, N>& x) {
    using namespace std;
    os << "{";
    copy(cbegin(x), prev(cend(x)), ostream_iterator<T, CharType>{os, separator<CharType>::value});
    copy(prev(cend(x)), cend(x), ostream_iterator<T, CharType>{os});
    os << "}";
    return os;
}

}  // namespace mathlib
