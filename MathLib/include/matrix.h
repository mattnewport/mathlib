#pragma once

#include "mathfuncs.h"
#include "quaternion.h"
#include "vector.h"

#include <cstdlib>

#include <DirectXMath.h>

namespace mathlib {

template <typename T, size_t N>
class Matrix {
public:
    static const size_t dimension = N;

    Matrix() = default;
    Matrix(const Matrix&) = default;

    template <typename... Us>
    Matrix(const Vector<T, N>& v, Us&&... us)
        : aw({ v, std::forward<Us>(us)... }) {
        static_assert(sizeof...(Us) == N - 1, "Constructor must be passed N row initializers.");
    }

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

    const float* data() const { return &e(0, 0); }

private:
    struct ArrayWrapper {
        Vector<T, N> rows_[N];
    } aw;
};

using Mat4f = Matrix<float, 4>;

template<typename T, size_t N, size_t... Is>
inline auto vecMatMultHelper(const Vector<T, N>& v, const Matrix<T, N>& m, std::index_sequence<Is...>) {
    return Vector<T, N>{dot(v, m.column(Is))...};
}

template<typename T, size_t N>
inline Vector<T, N> operator*(const Vector<T, N>& v, const Matrix<T, N>& m) {
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

inline auto toXMVector(const Vec4f& v) {
    auto res = DirectX::XMVECTOR{};
    memcpy(&res, &v, sizeof(res));
    return res;
}

inline auto toMat4f(const DirectX::XMMATRIX& m) {
    auto res = Mat4f{};
    memcpy(&res, &m, sizeof(res));
    return res;
}

inline auto Mat4fTranslation(const Vec3f& t) {
    return Mat4f{Vec4f{1.0f, 0.0f, 0.0f, 0.0f}, Vec4f{0.0f, 1.0f, 0.0f, 0.0f},
                 Vec4f{0.0f, 0.0f, 1.0f, 0.0f}, Vec4f{t.x(), t.y(), t.z(), 1.0f}};
}

inline auto Mat4fRotationY(float angle) {
    auto xmm = DirectX::XMMatrixRotationY(angle);
    return toMat4f(xmm);
}

inline auto Mat4fLookAtRH(const Vec4f& eye, const Vec4f& at, const Vec4f& up) {
    auto xmm = DirectX::XMMatrixLookAtRH(toXMVector(eye), toXMVector(at), toXMVector(up));
    return toMat4f(xmm);
}

template <typename T>
inline Matrix<T, 4> Mat4FromQuat(const Quaternion<T>& q) {
    using Vec = Vector<T, 4>;
    constexpr auto t0 = T{0};
    constexpr auto t1 = T{1};
    constexpr auto t2 = T{2};
    const auto _2x2 = t2 * square(q.x());
    const auto _2y2 = t2 * square(q.y());
    const auto _2z2 = t2 * square(q.z());
    const auto _2xy = t2 * q.x() * q.y();
    const auto _2zw = t2 * q.z() * q.w();
    const auto _2xz = t2 * q.x() * q.z();
    const auto _2yw = t2 * q.y() * q.w();
    const auto _2yz = t2 * q.y() * q.z();
    const auto _2xw = t2 * q.x() * q.w();
    return {Vec{t1 - _2y2 - _2z2, _2xy + _2zw, _2xz - _2yw, t0},
            Vec{_2xy - _2zw, t1 - _2x2 - _2z2, _2yz + _2xw, t0},
            Vec{_2xz + _2yw, _2yz - _2xw, t1 - _2x2 - _2y2, t0}, Vec{t0, t0, t0, t1}};
}

}  // namespace mathlib
