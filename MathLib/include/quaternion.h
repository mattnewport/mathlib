#pragma once

#include "vector.h"

namespace mathlib {

template <typename T>
class Quaternion : private Vector<T, 4> {
public:
    using Vector::Vector;

    const Vector<T, 3>& qv() const { return vec().xyz(); }
    T qs() const { return vec().w(); }
    auto x() const { return vec().x(); }
    auto y() const { return vec().y(); }
    auto z() const { return vec().z(); }
    auto w() const { return vec().w(); }

private:
    constexpr auto vec() const { return static_cast<const Vector<T, 4>&>(*this); }
};

using Quatf = Quaternion<float>;

}
