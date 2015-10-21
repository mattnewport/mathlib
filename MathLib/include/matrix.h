#pragma once

#include "mathfuncs.h"
#include "quaternion.h"
#include "vector.h"

#include <cstdlib>
#include <initializer_list>

#include <DirectXMath.h>

namespace mathlib {

template <typename T, size_t M, size_t N>
class Matrix : private Vector<Vector<T, N>, M> {
public:
    using Vector::Vector;

    Vector<T, N>& row(size_t n) { return Vector::e(n); }
    const Vector<T, N>& row(size_t n) const { return Vector::e(n); }
    template <size_t... Is>
    constexpr auto columnHelper(size_t n, std::index_sequence<Is...>) const {
        return Vector<T, M>{Vector::e(Is).e(n)...};
    }
    constexpr auto column(size_t n) const { return columnHelper(n, std::make_index_sequence<M>{}); }

    T& e(size_t r, size_t c) { return Vector::e(r).e(c); }
    constexpr const T& e(size_t r, size_t c) const { return Vector::e(r).e(c); }

    const float* data() const { return &e(0, 0); }

    const auto& rows() const { return static_cast<const Vector<Vector<T, N>, M>&>(*this); }
};

using Mat4f = Matrix<float, 4, 4>;

template<typename T, size_t M, size_t N, size_t... Is>
constexpr auto transpose(const Matrix<T, M, N>& x, std::index_sequence<Is...>) {
    return Matrix<T, N, M>{x.column(Is)...};
}

template<typename T, size_t M, size_t N, size_t... Is>
constexpr auto transpose(const Matrix<T, M, N>& x) {
    return transpose(x, std::make_index_sequence<N>{});
}

template<typename T, typename U, size_t M, size_t N>
constexpr auto operator==(const Matrix<T, M, N>& x, const Matrix<U, M, N>& y) {
    return x.rows() == y.rows();
}

template <typename T, typename U, size_t M, size_t N>
constexpr auto operator+(const Matrix<T, M, N>& x, const Matrix<U, M, N>& y) {
    return Matrix<decltype(x.e(0, 0) + y.e(0, 0)), M, N>{x.rows() + y.rows()};
}

template <typename T, size_t M, size_t N, size_t... Is>
constexpr auto vecMatMultHelper(const Vector<T, M>& v, const Matrix<T, M, N>& m,
                             std::index_sequence<Is...>) {
    return Vector<T, N>{dot(v, m.column(Is))...};
}

template<typename T, size_t M, size_t N>
constexpr auto operator*(const Vector<T, M>& v, const Matrix<T, M, N>& m) {
    return vecMatMultHelper(v, m, std::make_index_sequence<N>{});
}

template<typename T, typename U, size_t M, size_t N, size_t P>
inline auto operator*(const Matrix<T, M, N>& a, const Matrix<U, N, P>& b) {
    Matrix<std::common_type_t<T, U>, M, P> res;
    for (auto c = 0; c < N; ++c) {
        for (auto r = 0; r < M; ++r) {
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

template <typename T, size_t N, size_t... Js>
constexpr auto makeBasisVector(size_t i, std::index_sequence<Js...>) {
    return Vector<T, N>{T(i == Js)...};
}

template<typename T, size_t N>
constexpr auto makeBasisVector(size_t i) {
    return makeBasisVector<T, N>(i, std::make_index_sequence<N>{});
}

template <typename T, size_t N, size_t... Is>
constexpr auto IdentityMatrix(std::index_sequence<Is...>) {
    return Matrix<T, sizeof...(Is), N>{makeBasisVector<T, N>(Is)...};
}

template <typename T, size_t M, size_t N>
constexpr auto IdentityMatrix() {
    return IdentityMatrix<T, N>(std::make_index_sequence<M>{});
}

inline auto Mat4fScale(float s) {
    return Mat4f{ Vec4f{ s, 0.0f, 0.0f, 0.0f }, Vec4f{ 0.0f, s, 0.0f, 0.0f },
        Vec4f{ 0.0f, 0.0f, s, 0.0f }, Vec4f{ 0.0f, 0.0f, 0.0f, 1.0f } };
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
inline Matrix<T, 4, 4> Mat4FromQuat(const Quaternion<T>& q) {
    using Vec = Vector<T, 4>;
    const auto _2x2 = T{2} * square(q.x());
    const auto _2y2 = T{2} * square(q.y());
    const auto _2z2 = T{2} * square(q.z());
    const auto _2xy = T{2} * q.x() * q.y();
    const auto _2zw = T{2} * q.z() * q.w();
    const auto _2xz = T{2} * q.x() * q.z();
    const auto _2yw = T{2} * q.y() * q.w();
    const auto _2yz = T{2} * q.y() * q.z();
    const auto _2xw = T{2} * q.x() * q.w();
    return {Vec{T{1} - _2y2 - _2z2, _2xy + _2zw, _2xz - _2yw, T{0}},
            Vec{_2xy - _2zw, T{1} - _2x2 - _2z2, _2yz + _2xw, T{0}},
            Vec{_2xz + _2yw, _2yz - _2xw, T{1} - _2x2 - _2y2, T{0}}, Vec{T{0}, T{0}, T{0}, T{1}}};
}

}  // namespace mathlib
