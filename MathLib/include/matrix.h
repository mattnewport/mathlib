#pragma once

#include "mathfuncs.h"
#include "quaternion.h"
#include "vector.h"

#include <cstdlib>
#include <initializer_list>

namespace mathlib {

template <typename T, size_t M, size_t N>
class Matrix : private Vector<Vector<T, N>, M> {
public:
    // Since we inherit privately from Vector<Vector<T, N>, M>, this gives us constructors taking M
    // Vector<T, N>s representing the rows of the matrix and taking a single Vector<T, N> used to
    // initialize all M rows.
    using Vector::Vector;

    // Row and column access
    auto& operator[](size_t i) { return e_[i]; }
    constexpr const auto& operator[](size_t i) const { return e_[i]; }
    auto& row(size_t i) { return e_[i]; }
    constexpr const auto& row(size_t i) const { return return e_[i]; }
    constexpr auto column(size_t i) const { return columnHelper(i, Vector::is{}); }

    // Element access
    T& e(size_t r, size_t c) { return row(r)[c]; }
    constexpr const T& e(size_t r, size_t c) const { return e_[r][c]; }

    // These begin() and end() functions allow a Matrix to be used like a container for element
    // access. Not generally recommended but sometimes useful.
    auto begin() { return std::begin(e_[0]); }
    auto end() { return std::end(e_[M - 1]); }
    auto begin() const { return std::begin(e_[0]); }
    auto end() const { return std::end(e_[M - 1]); }

    // Return a pointer to the raw underlying contiguous element data.
    constexpr auto data() const { return e_[0].data(); }

    // Access the matrix as a const Vector<Vector<T, N>, M>& - not really intended for end user use
    // but helpful to implement freestanding operators and could be useful to users.
    constexpr auto& rows() const { return static_cast<const Vector<Vector<T, N>, M>&>(*this); }

    // @= operators - just delegate to Vector via rows() for implementations
    template <typename U>
    auto& operator+=(const Matrix<U, M, N>& x) {
        rows() += x.rows();
        return *this;
    }
    template <typename U>
    auto& operator-=(const Matrix<U, M, N>& x) {
        rows() -= x.rows();
        return *this;
    }
    template <typename U>
    auto& operator*=(U s) {
        rows() *= s;
        return *this;
    }
    template <typename U>
    auto& operator/=(U s) {
        rows() /= s;
        return *this;
    }

private:
    auto& rows() { return static_cast<Vector<Vector<T, N>, M>&>(*this); }
    template <size_t... Is>
    constexpr auto columnHelper(size_t n, std::index_sequence<Is...>) const {
        return Vector<T, M>{e_[Is][n]...};
    }
};

using Mat4f = Matrix<float, 4, 4>;

// Useful type traits for working with Matrices
template<typename T>
struct IsMatrix : std::false_type {};

template<typename T, size_t M, size_t N>
struct IsMatrix<Matrix<T, M, N>> : std::true_type {};

template<typename T>
struct MatrixRows;

template<typename T, size_t M, size_t N>
struct MatrixRows<Matrix<T, M, N>> : std::integral_constant<size_t, M> {};

template<typename T>
struct MatrixColumns;

template<typename T, size_t M, size_t N>
struct MatrixColumns<Matrix<T, M, N>> : std::integral_constant<size_t, N> {};

template<typename T>
struct MatrixElementType;

template<typename T, size_t M, size_t N>
struct MatrixElementType<Matrix<T, M, N>> {
    using type = T;
};

template<typename T>
using MatrixElementType_t = typename MatrixElementType<T>::type;

template <typename T, size_t M, size_t N>
inline auto MatrixFromDataPointer(const T* p) {
    Matrix<T, M, N> res;
    std::memcpy(&res, p, sizeof(res));
    return res;
}

template <typename M>
inline auto MatrixFromDataPointer(const MatrixElementType_t<M>* p) {
    return MatrixFromDataPointer<MatrixElementType_t<M>, MatrixRows<M>::value,
        MatrixColumns<M>::value>(p);
}

// Implementation helpers for operators and free functions, not part of public API
namespace detail {
template <typename T, size_t M, size_t N, size_t... Is>
constexpr auto vecMatMultHelper(const Vector<T, M, std::index_sequence<Is...>>& v, const Matrix<T, M, N>& m) {
    return Vector<T, N>{(v | m.column(Is))...};
}

} // namespace detail

template <typename... Us>
constexpr auto MatrixFromRows(Us&&... us) {
    using RowVectorType = std::common_type_t<std::remove_reference_t<Us>...>;
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

template <typename T, size_t M, size_t N, size_t... Is>
constexpr auto transpose(const Matrix<T, M, N>& x, std::index_sequence<Is...>) {
    return Matrix<T, N, M>{x.column(Is)...};
}

template <typename T, size_t M, size_t N, size_t... Is>
constexpr auto transpose(const Matrix<T, M, N>& x) {
    return transpose(x, std::make_index_sequence<N>{});
}

template <typename... Us>
constexpr auto MatrixFromColumns(const Us&... us) {
    return transpose(MatrixFromRows(us...));
}

template <typename T, typename U, size_t M, size_t N>
constexpr auto operator==(const Matrix<T, M, N>& x, const Matrix<U, M, N>& y) {
    return x.rows() == y.rows();
}

template <typename T, size_t M, size_t N>
constexpr auto operator+(const Matrix<T, M, N>& x) {
    return x;
}

template <typename T, typename U, size_t M, size_t N>
constexpr auto operator+(const Matrix<T, M, N>& x, const Matrix<U, M, N>& y) {
    return Matrix<decltype(x[0][0] + y[0][0]), M, N>{x.rows() + y.rows()};
}

template <typename T, size_t M, size_t N>
constexpr auto operator-(const Matrix<T, M, N>& x) {
    return Matrix<T, M, N>{-x.rows()};
}

template <typename T, typename U, size_t M, size_t N>
constexpr auto operator-(const Matrix<T, M, N>& x, const Matrix<U, M, N>& y) {
    return Matrix<decltype(x[0][0] - y[0][0]), M, N>{x.rows() - y.rows()};
}

template <typename T, typename U, size_t M, size_t N>
constexpr auto operator*(const Matrix<T, M, N>& x, U s) {
    return Matrix<decltype(x[0][0] * s), M, N>{x.rows() * s};
}

template <typename T, typename U, size_t M, size_t N>
constexpr auto operator*(U s, const Matrix<T, M, N>& x) {
    return x * s;
}

template <typename T, size_t M, size_t N>
constexpr auto operator*(const Vector<T, M>& v, const Matrix<T, M, N>& m) {
    return detail::vecMatMultHelper(v, m);
}

template <typename T, typename U, size_t M, size_t N, size_t P>
inline auto operator*(const Matrix<T, M, N>& a, const Matrix<U, N, P>& b) {
    // should be able to do this but it causes an ICE in VS2015 :(
    // return memberwiseBoundArg(std::multiplies<>{}, a.rows(), b);
    Matrix<std::common_type_t<T, U>, M, P> res;
    for (auto r = 0; r < M; ++r) res[r] = a[r] * b;
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
    return Mat4f{basisVector<Vec4f>(0) * s, basisVector<Vec4f>(1) * s, basisVector<Vec4f>(2) * s,
                 basisVector<Vec4f>(3)};
}

inline auto translationMat4f(const Vec3f& t) {
    return Mat4f{basisVector<Vec4f>(0), basisVector<Vec4f>(1), basisVector<Vec4f>(2),
                 Vec4f{t, 1.0f}};
}

inline auto rotationYMat4f(float angle) {
    const auto sinAngle = std::sin(angle);
    const auto cosAngle = std::cos(angle);
    return Mat4f{Vec4f{cosAngle, 0, -sinAngle, 0}, basisVector<Vec4f>(Y),
                 Vec4f{sinAngle, 0, cosAngle, 0}, basisVector<Vec4f>(W)};
}

// up should be a normalized direction vector
inline auto lookAtRhMat4f(const Vec3f& eye, const Vec3f& at, const Vec3f& up) {
    const auto zAxis = normalize(eye - at);
    const auto xAxis = normalize(cross(up, zAxis));
    const auto yAxis = cross(zAxis, xAxis);
    const auto negEyePos = -eye;
    const auto d = Vec3f{xAxis | negEyePos, yAxis | negEyePos, zAxis | negEyePos};
    return MatrixFromColumns(Vec4f{xAxis, d.x()}, Vec4f{yAxis, d.y()}, Vec4f{zAxis, d.z()},
                             basisVector<Vec4f>(W));
}

template <typename T>
inline auto Mat4FromQuat(const Quaternion<T>& q) {
    const auto sq = memberwiseMultiply(q.v(), q.v());
    const auto sq2 = decltype(q.v()){1} - times2(sq.yzx() + sq.zxy());
    const auto ws = times2(q.v() * q.w());
    const auto x = times2(memberwiseMultiply(q.v(), q.v().yzx()));
    using RowVec = Vector<T, 4>;
    return Matrix<T, 4, 4>{RowVec{sq2.x(), x.x() + ws.z(), x.z() - ws.y(), T(0)},
                           RowVec{x.x() - ws.z(), sq2.y(), x.y() + ws.x(), T(0)},
                           RowVec{x.z() + ws.y(), x.y() - ws.x(), sq2.z(), T(0)},
                           basisVector<T, 4>(W)};
}

}  // namespace mathlib
