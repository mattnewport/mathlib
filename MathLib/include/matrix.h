#pragma once

#include <vector>

#include <cstdlib>

#include <DirectXMath.h>

namespace mathlib {

template <typename T, size_t N>
class Matrix {
public:
    static const size_t dimension = N;

    template <typename... Us>
    Matrix(Us&&... us)
        : aw({ std::forward<Us>(us)... }) {}

    Vector<T, N>& row(size_t n) { return aw.rows_[n]; }
    const Vector<T, N>& row(size_t n) const { return aw.rows_[n]; }
    template<size_t... Is>
    auto columnHelper(size_t n, std::index_sequence<Is...>) const { return Vector<T, N>{aw.rows_[Is].e(n)...}; }
    auto column(size_t n) const { return columnHelper(n, std::make_index_sequence<N>{}); }

    T& e(size_t r, size_t c) {
        return row(r).e(c);
    }
    const T& e(size_t r, size_t c) const {
        return row(r).e(c);
    }

private:
    struct ArrayWrapper {
        Vector<T, N> rows_[N];
    } aw;
};

using Mat4f = Matrix<float, 4>;

template<typename T, size_t N, size_t... Is>
auto vecMatMultHelper(const Vector<T, N>& v, const Matrix<T, N>& m, std::index_sequence<Is...>) {
    return Vector<T, N>{dot(v, m.column(Is))...};
}

template<typename T, size_t N>
Vector<T, N> operator*(const Vector<T, N>& v, const Matrix<T, N>& m) {
    return vecMatMultHelper(v, m, std::make_index_sequence<N>{});
}

template<typename T, size_t N>
Matrix<T, N> operator*(const Matrix<T, N>& a, const Matrix<T, N>& b) {
    Matrix<T, N> res;
    for (auto c = 0; c < N; ++c) {
        for (auto r = 0; r < N; ++r) {
            res.e(r, c) = dot(a.row(r), b.column(c));
        }
    }
    return res;
}

Mat4f Mat4fRotationY(float angle) {
    auto xmm = DirectX::XMMatrixRotationY(angle);
    Mat4f m;
    memcpy(&m, &xmm, sizeof(m));
    return m;
}

}  // namespace mathlib
