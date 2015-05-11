#pragma once

#include "vector.h"

#include <iostream>
#include <iterator>
#include <string>

namespace mathlib {

template <typename CharType>
std::basic_string<CharType> getSeparator();
template <>
std::string getSeparator<char>() {
    return ", ";
}
template <>
std::wstring getSeparator<wchar_t>() {
    return L", ";
}

template <typename T, size_t N, typename CharType>
std::basic_ostream<CharType>& operator<<(std::basic_ostream<CharType>& os, const Vector<T, N>& x) {
    os << "(";
    copy(std::cbegin(x), std::prev(std::cend(x)),
         std::ostream_iterator<T, CharType>{os, getSeparator<CharType>().c_str()});
    copy(std::prev(std::cend(x)), std::cend(x), std::ostream_iterator<T, CharType>{os});
    os << ")";
    return os;
}

}  // namespace mathlib
