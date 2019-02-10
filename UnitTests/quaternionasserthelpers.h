#pragma once

#include "unittestwrapper.h"

#include "quaternion.h"
#include "quaternionio.h"

namespace Microsoft {
namespace VisualStudio {
namespace CppUnitTestFramework {
template <typename T>
auto ToStringHelper(const mathlib::Quaternion<T>& x) {
    RETURN_WIDE_STRING(x);
}

template <>
std::wstring ToString<mathlib::Quatf>(const mathlib::Quatf& x) {
    return ToStringHelper(x);
}

}  // namespace CppUnitTestFramework
}  // namespace VisualStudio
}  // namespace Microsoft
