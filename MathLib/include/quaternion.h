#pragma once

#include "vector.h"

namespace mathlib {

template <typename T>
class Quaternion : private Vector<T, 4> {
public:
    using Vector::Vector;

    constexpr auto x() const noexcept { return Vector::x(); }
    constexpr auto y() const noexcept { return Vector::y(); }
    constexpr auto z() const noexcept { return Vector::z(); }
    constexpr auto w() const noexcept { return Vector::w(); }
    constexpr auto v() const noexcept { return Vector::xyz(); }
    constexpr auto s() const noexcept { return Vector::w(); }

    auto& x() noexcept { return Vector::x(); }
    auto& y() noexcept { return Vector::y(); }
    auto& z() noexcept { return Vector::z(); }
    auto& w() noexcept { return Vector::w(); }

    constexpr auto vec4() const noexcept { return static_cast<const Vector<T, 4>&>(*this); }

    constexpr auto data() const noexcept { return Vector::data(); }
};

using Quatf = Quaternion<float>;

template <typename T>
constexpr auto identityQuaternion() noexcept {
    return Quaternion<T>{basisVector<T, 4>(W)};
}

// Assumes normalized axis
template <typename U, typename V>
inline auto QuaternionFromAxisAngle(const Vector<U, 3>& axis, V angle) noexcept {
    const auto t = angle * V(0.5);
    return Quaternion<U>{axis * std::sin(t), std::cos(t)};
}

template <typename T>
constexpr auto norm(const Quaternion<T>& x) noexcept {
    return magnitude(x.vec4());
}

template<typename T, typename U>
constexpr auto operator==(const Quaternion<T>& x, const Quaternion<U>& y) noexcept {
    return x.vec4() == y.vec4();
}

template <typename T, typename U>
constexpr auto operator+(const Quaternion<T>& x, const Quaternion<U>& y) noexcept {
    return Quaternion<std::common_type_t<T, U>>{x.vec4() + y.vec4()};
}

template <typename T, typename U>
constexpr auto operator-(const Quaternion<T>& x, const Quaternion<U>& y) noexcept {
    return Quaternion<std::common_type_t<T, U>>{x.vec4() - y.vec4()};
}

// operator~() is used for Quaternion conjugate, seems less potential for confusion than op*()
template<typename T>
constexpr auto operator~(const Quaternion<T>& x) noexcept {
    return Quaternion<T>{-x.v(), x.s()};
}

template <typename T, typename U>
constexpr auto operator*(const Quaternion<T>& x, const Quaternion<U>& y) noexcept {
    return Quaternion<std::common_type_t<T, U>>{x.v() * y.s() + y.v() * x.s() + cross(x.v(), y.v()),
                                                x.s() * y.s() - dot(x.v(), y.v())};
}

template<typename T, typename U>
inline auto rotate(const Vector<T, 3>& v, const Quaternion<U>& q) noexcept {
    const auto t = times2(cross(q.v(), v));
    return v + q.s() * t + cross(q.v(), t);
}

} // namespace mathlib
