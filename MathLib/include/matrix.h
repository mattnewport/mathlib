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
    // Since we inherit privately from Vector<Vector<T, N>, M>, this gives us constructors taking M
    // Vector<T, N>s representing the rows of the matrix and taking a single Vector<T, N> used to
    // initialize all M rows.
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

    auto begin() { return std::begin(Vector::e(0)); }
    auto end() { return std::end(Vector::e(M - 1)); }
    auto begin() const { return std::begin(Vector::e(0)); }
    auto end() const { return std::end(Vector::e(M - 1)); }

    const float* data() const { return row(0).data(); }

    const auto& rows() const { return static_cast<const Vector<Vector<T, N>, M>&>(*this); }

    template<typename U>
    auto& operator+=(const Matrix<U, M, N>& x) {
        rows() += x.rows();
        return *this;
    }
    template<typename U>
    auto& operator-=(const Matrix<U, M, N>& x) {
        rows() -= x.rows();
        return *this;
    }
    template<typename U>
    auto& operator*=(U s) {
        rows() *= s;
        return *this;
    }
    template<typename U>
    auto& operator/=(U s) {
        rows() /= s;
        return *this;
    }
private:
    auto& rows() { return static_cast<Vector<Vector<T, N>, M>&>(*this); }
};

using Mat4f = Matrix<float, 4, 4>;

template <typename... Us>
constexpr auto MatrixFromRows(const Us&... us) {
    using RowVectorType = std::common_type_t<Us...>;
    return Matrix<VectorElementType_t<RowVectorType>, sizeof...(Us),
                  VectorDimension<RowVectorType>::value>{us...};
}

// Handy to have this concrete version that can deduce the type of the arguments, e.g. you can do:
// auto m = Mat4fFromRows({1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, ...)
// rather than the slightly more verbose:
// auto m = MatrixFromRows(Vec4f{1.0f, 2.0f, 3.0f, 4.0f}, Vec4f{5.0f, 6.0f, 7.0f, 8.0f}, ...)
constexpr auto Mat4fFromRows(Vec4f r0, Vec4f r1, Vec4f r2, Vec4f r3) {
    return MatrixFromRows(r0, r1, r2, r3);
}

template<typename T, size_t M, size_t N, size_t... Is>
constexpr auto transpose(const Matrix<T, M, N>& x, std::index_sequence<Is...>) {
    return Matrix<T, N, M>{x.column(Is)...};
}

template<typename T, size_t M, size_t N, size_t... Is>
constexpr auto transpose(const Matrix<T, M, N>& x) {
    return transpose(x, std::make_index_sequence<N>{});
}

template <typename... Us>
constexpr auto MatrixFromColumns(const Us&... us) {
    return transpose(MatrixFromRows(us...));
}

template<typename T, typename U, size_t M, size_t N>
constexpr auto operator==(const Matrix<T, M, N>& x, const Matrix<U, M, N>& y) {
    return x.rows() == y.rows();
}

template <typename T, typename U, size_t M, size_t N>
constexpr auto operator+(const Matrix<T, M, N>& x, const Matrix<U, M, N>& y) {
    return Matrix<decltype(x.e(0, 0) + y.e(0, 0)), M, N>{x.rows() + y.rows()};
}

template <typename T, typename U, size_t M, size_t N>
constexpr auto operator-(const Matrix<T, M, N>& x, const Matrix<U, M, N>& y) {
    return Matrix<decltype(x.e(0, 0) - y.e(0, 0)), M, N>{x.rows() - y.rows()};
}

template <typename T, typename U, size_t M, size_t N>
constexpr auto operator*(const Matrix<T, M, N>& x, U s) {
    return Matrix<decltype(x.e(0, 0) * s), M, N>{x.rows() * s};
}

template <typename T, typename U, size_t M, size_t N>
constexpr auto operator*(U s, const Matrix<T, M, N>& x) {
    return x * s;
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

template <typename T, size_t M, size_t N>
constexpr auto zeroMatrix() {
    return Matrix<T, M, N>{zeroVector<T, N>()};
}

template <typename T, size_t N, size_t... Is>
constexpr auto identityMatrix(std::index_sequence<Is...>) {
    return Matrix<T, sizeof...(Is), N>{basisVector<T, N>(Is)...};
}

template <typename T, size_t M, size_t N>
constexpr auto identityMatrix() {
    return identityMatrix<T, N>(std::make_index_sequence<M>{});
}

inline auto scaleMat4f(float s) {
    return Mat4f{ Vec4f{ s, 0.0f, 0.0f, 0.0f }, Vec4f{ 0.0f, s, 0.0f, 0.0f },
        Vec4f{ 0.0f, 0.0f, s, 0.0f }, Vec4f{ 0.0f, 0.0f, 0.0f, 1.0f } };
}

inline auto translationMat4f(const Vec3f& t) {
    return Mat4f{Vec4f{1.0f, 0.0f, 0.0f, 0.0f}, Vec4f{0.0f, 1.0f, 0.0f, 0.0f},
                 Vec4f{0.0f, 0.0f, 1.0f, 0.0f}, Vec4f{t.x(), t.y(), t.z(), 1.0f}};
}

inline auto rotationYMat4f(float angle) {
    auto xmm = DirectX::XMMatrixRotationY(angle);
    return toMat4f(xmm);
}

inline auto lookAtRhMat4f(const Vec4f& eye, const Vec4f& at, const Vec4f& up) {
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
