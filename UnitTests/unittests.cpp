#include "stdafx.h"

#include <type_traits>

#include "CppUnitTest.h"

#include "mathio.h"
#include "mathconstants.h"
#include "matrix.h"
#include "vector.h"

#include <DirectXMath.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DirectX;

using namespace mathlib;

namespace Microsoft { namespace VisualStudio { namespace CppUnitTestFramework { 
template<typename T, size_t N>
std::wstring ToString(const Vector<T, N>& x) { RETURN_WIDE_STRING(x); }
}}}

namespace UnitTests {
TEST_CLASS(VectorUnitTests){
public :

    TEST_METHOD(TestBasics) {
        const Vec4f v0{1.0f};
        Assert::IsTrue(v0.x() == 1.0f && v0.y() == 1.0f && v0.z() == 1.0f && v0.w() == 1.0f);
        constexpr Vec4f v1{1.0f, 2.0f, 3.0f, 4.0f};
        Assert::IsTrue(v1.x() == 1.0f && v1.y() == 2.0f && v1.z() == 3.0f && v1.w() == 4.0f);
        Assert::IsTrue(v1.e(0) == 1.0f && v1.e(1) == 2.0f && v1.e(2) == 3.0f && v1.e(3) == 4.0f);
        constexpr auto v2 = v1;
        Assert::IsTrue(v2 == v1);
        Assert::IsTrue(v0 != v1);
        Assert::IsFalse(v2 != v1);
        Assert::IsFalse(v0 == v1);
        Assert::IsTrue(v0 < v1);
        Assert::IsFalse(v1 < v0);
        Assert::IsFalse(v0 < v0);
        constexpr Vector<double, 4> v3{ 1.0, 2.0, 3.0, 4.0 };
        Assert::IsTrue(v1 == v3);
        const Vector<double, 4> v4{v1};
        Assert::IsTrue(v4 == v1);
        const Vector<double, 4> v5{1.0f};
        Assert::IsTrue(v5 == v0);
        const Vec2f v6{1.0f, 2.0f};
        Assert::IsTrue(v6.x() == 1.0f && v6.y() == 2.0f);
    }

    TEST_METHOD(TestAdd) {
        constexpr Vec4f v0{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Vec4f v1{ 2.0f, 4.0f, 6.0f, 8.0f };
        constexpr auto v2 = v0 + v1;
        Assert::AreEqual(Vec4f{ 3.0f, 6.0f, 9.0f, 12.0f }, v2);
        constexpr Vector<double, 4> v3{ 2.0, 4.0, 6.0, 8.0 };
        constexpr auto v4 = v0 + v3;
        Assert::AreEqual(v4, Vector<double, 4>{3.0, 6.0, 9.0, 12.0});
    }

    TEST_METHOD(TestSubtract) {
        constexpr Vec4f v0{2.0f, 4.0f, 6.0f, 8.0f};
        constexpr Vec4f v1{1.0f, 2.0f, 3.0f, 4.0f};
        constexpr auto v2 = v0 - v1;
        Assert::AreEqual(v2, Vec4f{1.0f, 2.0f, 3.0f, 4.0f});
        const Vector<double, 4> v3{v0};
        Assert::IsTrue(v3 == v0);
        constexpr Vector<double, 4> v4{ 2.0, 4.0, 6.0, 8.0 };
        constexpr auto v5 = v4 - v1;
        static_assert(std::is_same<decltype(v5), const Vector<double, 4>>::value, "");
        static_assert(v5 == v2, "");
        Assert::IsTrue(v5 == v2);
        constexpr auto v6{v1};
        constexpr auto v7 = v6 - v1;
        static_assert(v7 == Vec4f{0.0f, 0.0f, 0.0f, 0.0f}, "");
    }

    TEST_METHOD(TestScalarMultiply) {
        constexpr Vec4f v0{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr auto v1 = v0 * 2.0f;
        constexpr auto v2 = 2.0f * v0;
        constexpr auto v3 = v0 * 2.0;
        static_assert(std::is_same<decltype(v3), const Vector<double, 4>>::value, "");
        Assert::AreEqual(v1, v2);
        Assert::AreEqual(v1, Vec4f{ 2.0f, 4.0f, 6.0f, 8.0f });
        Assert::AreEqual(v3, Vector<double, 4>{2.0, 4.0, 6.0, 8.0});
        Assert::IsTrue(v1 == v3);
    }

    TEST_METHOD(TestDot) {
        constexpr Vec4f v0{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Vec4f v1{ 2.0f, 4.0f, 6.0f, 8.0f };
        constexpr auto s0 = dot(v0, v1);
        Assert::AreEqual(s0, 1.0f * 2.0f + 2.0f * 4.0f + 3.0f * 6.0f + 4.0f * 8.0f);
        constexpr Vector<double, 4> v3{ 2.0, 4.0, 6.0, 8.0 };
        constexpr auto s1 = dot(v0, v3);
        Assert::AreEqual(s1, 1.0f * 2.0 + 2.0f * 4.0 + 3.0f * 6.0 + 4.0f * 8.0);
        static_assert(s1 == 1.0f * 2.0 + 2.0f * 4.0 + 3.0f * 6.0 + 4.0f * 8.0, "");
    }

    TEST_METHOD(TestDivide) {
        constexpr Vec3f v0{2.0f, 4.0f, 6.0f};
        constexpr auto v1 = v0 / 2.0f;
        static_assert(v1 == Vec3f{1.0f, 2.0f, 3.0f}, "");
        Assert::AreEqual(v1, Vec3f{1.0f, 2.0f, 3.0f});

        constexpr Vec3i v2{2, 4, 6};
        constexpr auto v3 = v2 / 2;
        static_assert(std::is_same<decltype(v3), const Vec3i>::value && v3 == Vec3i{1, 2, 3}, "");
        Assert::AreEqual(v3, Vec3i{1, 2, 3});
    }

    TEST_METHOD(TestMagnitude) {
        constexpr Vec2f v0{ 3.0f, 4.0f };
        const auto s0 = magnitude(v0);
        Assert::AreEqual(s0, 5.0f);
        constexpr Vec3f v1{ 1.0f, 1.0f, 1.0f };
        const auto s1 = magnitude(v1);
        Assert::AreEqual(s1, sqrt(3.0f));
    }

    TEST_METHOD(TestNormalize) {
        constexpr Vec3f v0{1.0f, 2.0f, 3.0f};
        const auto s0 = magnitude(v0);
        const auto v1 = normalize(v0);
        const auto s1 = magnitude(v1);
        auto closeEnough = [](auto a, auto b) { return abs(a - b) < 1e-6f; };
        Assert::IsTrue(closeEnough(1.0f, s1));
        const auto v2 = v1 * s0;
        Assert::IsTrue(closeEnough(v0.x(), v2.x()) && closeEnough(v0.y(), v2.y()) &&
                       closeEnough(v0.z(), v2.z()));
    }

    TEST_METHOD(TestOpAssignment) {
        Vec3f v0{1.0f, 2.0f, 3.0f};
        constexpr Vec3f v1{1.0f, 2.0f, 3.0f};
        v0 += v1;
        Assert::AreEqual(v0, Vec3f{2.0f, 4.0f, 6.0f});
        v0 -= v1;
        Assert::AreEqual(v0, Vec3f{1.0f, 2.0f, 3.0f});
        v0 *= 2.0f;
        Assert::AreEqual(v0, Vec3f{2.0f, 4.0f, 6.0f});
        v0 /= 2.0f;
        Assert::AreEqual(v0, Vec3f{ 1.0f, 2.0f, 3.0f });
        Vec3i v2{2, 3, 4};
        v2 /= 2;
        Assert::IsTrue(v2.x() == 2 / 2 && v2.y() == 3 / 2 && v2.z() == 4 / 2);
    }
};

TEST_CLASS(MatrixUnitTests){
    public : TEST_METHOD(TestMatrixColumnAccess){
        const auto m = Mat4f{Vec4f{1.0f, 2.0f, 3.0f, 4.0f}, Vec4f{5.0f, 6.0f, 7.0f, 8.0f},
                       Vec4f{9.0f, 10.0f, 11.0f, 12.0f}, Vec4f{13.0f, 14.0f, 15.0f, 16.0f}};
        const auto c0 = m.column(0);
        Assert::AreEqual(c0, Vec4f{1.0f, 5.0f, 9.0f, 13.0f});
        const auto c1 = m.column(1);
        Assert::AreEqual(c1, Vec4f{ 2.0f, 6.0f, 10.0f, 14.0f });
        const auto c2 = m.column(2);
        Assert::AreEqual(c2, Vec4f{ 3.0f, 7.0f, 11.0f, 15.0f });
        const auto c3 = m.column(3);
        Assert::AreEqual(c3, Vec4f{ 4.0f, 8.0f, 12.0f, 16.0f });
    }

    TEST_METHOD(TestVectorMatrixMultiply) {
        const auto m0 = Mat4f{ Vec4f{ 1.0f, 2.0f, 3.0f, 4.0f }, Vec4f{ 5.0f, 6.0f, 7.0f, 8.0f },
            Vec4f{ 9.0f, 10.0f, 11.0f, 12.0f }, Vec4f{ 13.0f, 14.0f, 15.0f, 16.0f } };
        const auto v0 = Vec4f{ 1.0f, 1.0f, 1.0f, 1.0f };
        const auto v1 = v0 * m0;

        auto xmv0 = XMVECTOR{};
        memcpy(&xmv0, &v0, sizeof(xmv0));
        auto xmm = XMMATRIX{};
        memcpy(&xmm, &m0, sizeof(xmm));
        auto xmv1 = XMVector4Transform(xmv0, xmm);
        Assert::IsTrue(memcmp(&v1, &xmv1, sizeof(v1)) == 0);
    }

    TEST_METHOD(TestMatrixMatrixMultiply) {
        const auto m0 = Mat4f{ Vec4f{ 1.0f, 2.0f, 3.0f, 4.0f }, Vec4f{ 5.0f, 6.0f, 7.0f, 8.0f },
            Vec4f{ 9.0f, 10.0f, 11.0f, 12.0f }, Vec4f{ 13.0f, 14.0f, 15.0f, 16.0f } };
        const auto m1 = Mat4f{ Vec4f{ 21.0f, 22.0f, 23.0f, 24.0f }, Vec4f{ 25.0f, 26.0f, 27.0f, 28.0f },
            Vec4f{ 29.0f, 30.0f, 31.0f, 32.0f }, Vec4f{ 33.0f, 34.0f, 35.0f, 36.0f } };
        const auto m2 = m0 * m1;
        auto xmm0 = XMMATRIX{};
        memcpy(&xmm0, &m0, sizeof(xmm0));
        auto xmm1 = XMMATRIX{};
        memcpy(&xmm1, &m1, sizeof(xmm1));
        auto xmm2 = XMMatrixMultiply(xmm0, xmm1);
        Assert::IsTrue(memcmp(&m2, &xmm2, sizeof(m2)) == 0);
    }

    TEST_METHOD(TestMatrix4fRotationY) {
        const auto angle = pif / 4.0f;
        auto m = Mat4fRotationY(angle);
        auto xmm = XMMatrixRotationY(angle);
        Assert::IsTrue(memcmp(&m, &xmm, sizeof(m)) == 0);
    }
};

}