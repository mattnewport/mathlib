#include "stdafx.h"

#include <algorithm>
#include <iterator>
#include <type_traits>

#include "unittestwrapper.h"

#include "mathconstants.h"
#include "mathfuncs.h"
#include "vector.h"
#include "vectorio.h"

using namespace mathlib;

using namespace std::literals;

namespace Microsoft {
    namespace VisualStudio {
        namespace CppUnitTestFramework {
            template <typename T, size_t N>
            auto ToString(const mathlib::Vector<T, N>& x) { RETURN_WIDE_STRING(x); }
        }
    }
}


namespace UnitTests {

    template<typename T>
    constexpr auto areNearlyEqual(const T& a, const T& b, const T& eps) {
        return std::abs(a - b) < eps;
    }

TEST_CLASS(FuncsUnitTests) {
public:
    TEST_METHOD(TestLerp) {
        auto x0 = lerp(0.0f, 1.0f, 0.3f);
        Assert::AreEqual(x0, 0.3f);
        auto x1 = lerp(1.0f, 3.0f, 0.6f);
        Assert::AreEqual(x1, 2.2f);
    }

    TEST_METHOD(TestAbs) {
        constexpr auto v0 = Vec3f{ -0.5f, 2.0f, 1.1f };
        const auto v2 = abs(v0);
        Assert::AreEqual(v2, Vec3f{ 0.5f, 2.0f, 1.1f });
    }

    TEST_METHOD(TestMax) {
        constexpr auto v0 = Vec3f{ -0.5f, 2.0f, 1.1f };
        constexpr auto v1 = Vec3f{ 0.5f, 0.6f, 0.7f };
        const auto v3 = max(v0, v1);
        Assert::AreEqual(v3, Vec3f{ 0.5f, 2.0f, 1.1f });
    }

    TEST_METHOD(TestSaturate) {
        constexpr auto v0 = Vec3f{ -0.5f, 2.0f, 0.9f };
        const auto v2 = saturate(v0);
        Assert::AreEqual(v2, Vec3f{ 0.0f, 1.0f, 0.9f });
    }

    TEST_METHOD(TestClamp) {
        constexpr auto v0 = Vec3f{ -0.5f, 2.0f, 1.1f };
        const auto v1 = clamp(v0, 1.0f, 2.0f);
        Assert::AreEqual(v1, Vec3f{ 1.0f, 2.0f, 1.1f });
    }
};

}
