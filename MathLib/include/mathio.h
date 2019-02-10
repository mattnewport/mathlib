#pragma once

#include "vectorio.h"
#include "matrix.h"

namespace mathlib {

template <typename T, size_t M, size_t N, typename CharT>
auto& operator<<(std::basic_ostream<CharT>& os, const Matrix<T, M, N>& x) {
    os << "{";
    for (int i = 0; i < M; ++i) {
        os << x.row(i) << (i == M - 1 ? "}" : ", ");
    }
    return os;
}

}  // namespace mathlib
