#pragma once

#include "vector.h"

namespace mathlib {

template <typename T>
class Quaternion {
public:
    Quaternion() = default;
    Quaternion(const Quaternion&) = default;
    Quaternion(const Vector<T, 3>& qv, T qs) : qv_{ qv }, qs_{ qs } {}

    const Vector<T, 3>& qv() const { return qv_; }
    T qs() const { return qs_; }
    auto x() const { return qv_.x(); }
    auto y() const { return qv_.y(); }
    auto z() const { return qv_.z(); }
    auto w() const { return qs_; }

private:
    Vector<T, 3> qv_;
    T qs_;
};

using Quatf = Quaternion<float>;

}
