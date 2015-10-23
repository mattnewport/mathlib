#pragma once

#include "vector.h"

namespace mathlib {

template <typename T>
class Quaternion : private Vector<T, 4> {
public:
    using Vector::Vector;

    constexpr auto x() const { return Vector::x(); }
    constexpr auto y() const { return Vector::y(); }
    constexpr auto z() const { return Vector::z(); }
    constexpr auto w() const { return Vector::w(); }
    constexpr auto xyz() const { return Vector::xyz(); }
    constexpr auto v() const { return Vector::xyz(); }
    constexpr auto s() const { return Vector::w(); }

    auto& x() { return Vector::x(); }
    auto& y() { return Vector::y(); }
    auto& z() { return Vector::z(); }
    auto& w() { return Vector::w(); }

    constexpr auto vec4() const { return static_cast<const Vector<T, 4>&>(*this); }

    constexpr auto data() const { return Vector::data(); }
};

using Quatf = Quaternion<float>;

template <typename T>
constexpr auto identityQuaternion() {
    return Quaternion<T>{T(0), T(0), T(0), T(1)};
}

// Assumes normalized axis
template <typename U, typename V>
inline auto fromAxisAngle(const Vector<U, 3>& axis, V angle) {
    const auto t = angle * V(0.5);
    return Quaternion<U>{axis * std::sin(t), std::cos(t)};
}

template <typename T>
constexpr auto norm(const Quaternion<T>& x) {
    return magnitude(x.vec4());
}

template<typename T, typename U>
constexpr auto operator==(const Quaternion<T>& x, const Quaternion<U>& y) {
    return x.vec4() == y.vec4();
}

template <typename T, typename U>
constexpr auto operator+(const Quaternion<T>& x, const Quaternion<U>& y) {
    return Quaternion<std::common_type_t<T, U>>{x.vec4() + y.vec4()};
}

template <typename T, typename U>
constexpr auto operator-(const Quaternion<T>& x, const Quaternion<U>& y) {
    return Quaternion<std::common_type_t<T, U>>{x.vec4() - y.vec4()};
}

// operator~() is used for Quaternion conjugate, seems less potential for confusion than op*()
template<typename T>
constexpr auto operator~(const Quaternion<T>& x) {
    return Quaternion<T>{-x.xyz(), x.w()};
}

template <typename T, typename U>
constexpr auto operator*(const Quaternion<T>& x, const Quaternion<U>& y) {
    return Quaternion<std::common_type_t<T, U>>{x.v() * y.s() + y.v() * x.s() + cross(x.v(), y.v()),
                                                x.s() * y.s() - (x.v() | y.v())};
}

template<typename T, typename U>
inline auto rotate(const Vector<T, 3>& v, const Quaternion<U>& q) {
    const auto t = times2(cross(q.v(), v));
    return v + q.s() * t + cross(q.v(), t);
}

} // namespace mathlib
