#pragma once

#include "quaternion.h"
#include "vectorio.h"

namespace mathlib {

template <typename T, typename CharT>
auto& operator<<(std::basic_ostream<CharT>& os, const Quaternion<T>& x) {
    return os << x.vec4();
}

}  // namespace mathlib
