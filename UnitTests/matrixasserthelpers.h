#pragma once

#include "unittestwrapper.h"

#include "matrix.h"
#include "mathio.h"

namespace Microsoft {
namespace VisualStudio {
namespace CppUnitTestFramework {
template <typename T, size_t M, size_t N>
auto ToStringHelper(const mathlib::Matrix<T, M, N>& x) {
    RETURN_WIDE_STRING(x);
}

template <>
std::wstring ToString<mathlib::Mat4f>(const mathlib::Mat4f& x) {
    return ToStringHelper(x);
}

template <>
std::wstring ToString<mathlib::Matrix<float, 3, 4>>(const mathlib::Matrix<float, 3, 4>& x) {
    return ToStringHelper(x);
}

template <>
std::wstring ToString<mathlib::Matrix<float, 2, 3>>(const mathlib::Matrix<float, 2, 3>& x) {
    return ToStringHelper(x);
}

}  // namespace CppUnitTestFramework
}  // namespace VisualStudio
}  // namespace Microsoft
