#pragma once

#include "unittestwrapper.h"

#include "vector.h"
#include "vectorio.h"

namespace Microsoft {
namespace VisualStudio {
namespace CppUnitTestFramework {
template <typename T, size_t N>
auto ToStringHelper(const mathlib::Vector<T, N>& x) {
    RETURN_WIDE_STRING(x);
}

template <>
std::wstring ToString<mathlib::Vec2d>(const mathlib::Vec2d& x) {
    return ToStringHelper(x);
}

template <>
std::wstring ToString<mathlib::Vec3d>(const mathlib::Vec3d& x) {
    return ToStringHelper(x);
}

template <>
std::wstring ToString<mathlib::Vec4d>(const mathlib::Vec4d& x) {
    return ToStringHelper(x);
}

template <>
std::wstring ToString<mathlib::Vec2f>(const mathlib::Vec2f& x) {
    return ToStringHelper(x);
}

template <>
std::wstring ToString<mathlib::Vec3f>(const mathlib::Vec3f& x) {
    return ToStringHelper(x);
}

template <>
std::wstring ToString<mathlib::Vec4f>(const mathlib::Vec4f& x) {
    return ToStringHelper(x);
}

template <>
std::wstring ToString<mathlib::Vec2i>(const mathlib::Vec2i& x) {
    return ToStringHelper(x);
}

template <>
std::wstring ToString<mathlib::Vec3i>(const mathlib::Vec3i& x) {
    return ToStringHelper(x);
}

template <>
std::wstring ToString<mathlib::Vec4i>(const mathlib::Vec4i& x) {
    return ToStringHelper(x);
}

template <>
std::wstring ToString<mathlib::Vector<mathlib::Vec2f, 2>>(const mathlib::Vector<mathlib::Vec2f, 2>& x) {
    return ToStringHelper(x);
}

}  // namespace CppUnitTestFramework
}  // namespace VisualStudio
}  // namespace Microsoft
