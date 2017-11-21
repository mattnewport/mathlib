#pragma once

#include "vector.h"

namespace mathlib {

template <typename T>
class Quaternion : detail::MakeVectorBase_t<Quaternion<T>> {
    using base = detail::MakeVectorBase_t<Quaternion<T>>;
    friend base;
    using IS = std::make_index_sequence<4>;

public:
    constexpr Quaternion() noexcept = default;
    constexpr Quaternion(const T& x, const T& y, const T& z, const T& w) noexcept
        : base{x, y, z, w} {}
    constexpr Quaternion(const Vector<T, 4>& v) noexcept : base{v[0], v[1], v[2], v[3]} {}
    constexpr Quaternion(const Vector<T, 3>& v, const T& s) noexcept : base{v[0], v[1], v[2], s} {}

    using base::operator[];
    constexpr auto x() const noexcept { return this->e[0]; }
    constexpr auto y() const noexcept { return this->e[1]; }
    constexpr auto z() const noexcept { return this->e[2]; }
    constexpr auto w() const noexcept { return this->e[3]; }
    constexpr auto v() const noexcept { return Vector<T, 3>{x(), y(), z()}; }
    constexpr auto s() const noexcept { return w(); }

    using base::data;

    friend constexpr auto norm(const Quaternion& x) noexcept { return x.magnitude(); }

    using base::operator==;
    using base::operator!=;

    using base::operator+;
    using base::operator-;

    static constexpr Quaternion identity() noexcept { return Quaternion{T(0), T(0), T(0), T(1)}; }

    constexpr auto vec4() const noexcept { return Vector<T, 4>{x(), y(), z(), w()}; }
};

// Specializations of type traits for working with Quaternions
template <typename T>
struct VectorDimension<Quaternion<T>> : std::integral_constant<size_t, 4> {};

template <typename T>
struct VectorElementType<Quaternion<T>> {
    using type = T;
};

using Quatf = Quaternion<float>;

template <typename T>
constexpr auto identityQuaternion() noexcept {
    return Quaternion<T>::identity();
}

// Assumes normalized axis
template <typename U, typename V>
inline auto QuaternionFromAxisAngle(const Vector<U, 3>& axis, V angle) noexcept {
    const auto t = angle * V(0.5);
    return Quaternion<U>{axis * std::sin(t), std::cos(t)};
}

// operator~() is used for Quaternion conjugate, seems less potential for confusion than op*()
template <typename T>
constexpr auto operator~(const Quaternion<T>& x) noexcept {
    return Quaternion<T>{-x.v(), x.s()};
}

template <typename T>
constexpr auto operator*(const Quaternion<T>& x, const Quaternion<T>& y) noexcept {
    return Quaternion<T>{x.v() * y.s() + y.v() * x.s() + cross(x.v(), y.v()),
                         x.s() * y.s() - dot(x.v(), y.v())};
}

template <typename T>
inline auto rotate(const Vector<T, 3>& v, const Quaternion<T>& q) noexcept {
    const auto t = times2(cross(q.v(), v));
    return v + q.s() * t + cross(q.v(), t);
}

}  // namespace mathlib
