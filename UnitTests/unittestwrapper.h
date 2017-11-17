#pragma once

#if defined(__clang__)

#define TEST_CLASS(X) class X
#define TEST_METHOD(X) void X()

namespace Assert {
template <typename T, typename U>
void AreEqual(T&&, U&&) {}

inline void IsTrue(bool) {}
inline void IsFalse(bool) {}
}

#elif defined(_MSC_VER)

#include <CppUnitTest.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#endif

