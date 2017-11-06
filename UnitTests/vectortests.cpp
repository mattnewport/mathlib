#include "stdafx.h"

#include <algorithm>
#include <iterator>
#include <type_traits>

#include "CppUnitTest.h"

#include "mathconstants.h"
#include "vector.h"
#include "vectorio.h"

#include <DirectXMath.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DirectX;

using namespace mathlib;

using namespace std::literals;

namespace Microsoft {
namespace VisualStudio {
namespace CppUnitTestFramework {
template <typename T, size_t N>
auto ToString(const Vector<T, N>& x) {
    RETURN_WIDE_STRING(x);
}
}  // namespace CppUnitTestFramework
}  // namespace VisualStudio
}  // namespace Microsoft

namespace UnitTests {

inline auto toXmVector(const Vec3f& v) { return XMLoadFloat3(std::data({XMFLOAT3{v.data()}})); }
inline auto toXmVector(const Vec4f& v) { return XMLoadFloat4(std::data({XMFLOAT4{v.data()}})); }

template <typename T>
constexpr auto areNearlyEqual(const T& a, const T& b, const T& eps) {
    return std::abs(a - b) < eps;
}

inline auto areNearlyEqual(const Vec3f& v, const XMVECTOR& xmv, float eps) {
    const auto diff = toXmVector(v) - xmv;
    return XMVectorGetX(XMVector3Length(diff)) < eps;
}

TEST_CLASS(VectorUnitTests) {
public: 
    
TEST_METHOD(TestBasics){const Vec4f v0{1.0f};
    Assert::IsTrue(v0.x() == 1.0f && v0.y() == 1.0f && v0.z() == 1.0f && v0.w() == 1.0f);
    constexpr Vec4f v1{1.0f, 2.0f, 3.0f, 4.0f};
    Assert::IsTrue(v1.x() == 1.0f && v1.y() == 2.0f && v1.z() == 3.0f && v1.w() == 4.0f);
    Assert::IsTrue(v1[0] == 1.0f && v1[1] == 2.0f && v1[2] == 3.0f && v1[3] == 4.0f);
    constexpr auto v2 = v1;
    Assert::IsTrue(v2 == v1);
    Assert::IsTrue(v0 != v1);
    Assert::IsFalse(v2 != v1);
    Assert::IsFalse(v0 == v1);
    constexpr Vector<double, 4> v3{1.0, 2.0, 3.0, 4.0};
    const Vector<double, 4> v4{v1};
    Assert::IsTrue(v4 == v3);
    const Vector<double, 4> v5{1.0f};
    Assert::IsTrue(v5 == Vector<double, 4>{v0});
    const Vec2f v6{1.0f, 2.0f};
    Assert::IsTrue(v6.x() == 1.0f && v6.y() == 2.0f);

    const Vector<Vec2f, 2> v7{Vec2f{1.0f, 2.0f}, Vec2f{3.0f, 4.0f}};
    Assert::IsTrue(std::equal(std::begin(v7[0]), std::end(v7[1]),
                              stdext::make_unchecked_array_iterator(std::begin({1.0f, 2.0f, 3.0f,
                                                                                4.0f}))));

    const auto v8 = Vec3f{v6, 3.0f};
    Assert::AreEqual(v8, Vec3f{1.0f, 2.0f, 3.0f});

    static_assert(IsVector<Vec4f>{}, "");
    static_assert(!IsVector<std::tuple<int, int>>{}, "");
    static_assert(VectorDimension<Vec4f>{} == 4, "");
    static_assert(std::is_same<float, VectorElementType_t<Vec4f>>{}, "");

    constexpr auto& v9 = v1;
    constexpr auto v10{v9};

    const auto v11 = Vec3f{1};
    const auto v12 = Vec3f{1.0f, 2.0f, 3.0f};
    const auto v13 = Vec3i{1, 2, 3};
    Vec3f v14 = Vec3f{v13};

    Assert::IsTrue(v12[0] == v12.x() && v12[1] == v12.y() && v12[2] == v12.z());
    Assert::IsTrue(-v14 + +v14 == zeroVector<Vec3f>());

    auto pv = new (&v14) Vec3f;
    Assert::AreEqual(*pv, v12);
    pv = new (&v14) Vec3f{};  // check value initialization syntax works
    Assert::AreEqual(*pv, Vec3f{0.0f, 0.0f, 0.0f});

    auto v15 = basisVector<float, 3>(Y);
    Assert::AreEqual(v15, Vec3f{0.0f, 1.0f, 0.0f});
    auto v16 = basisVector<Vec3f>(Y);
    Assert::AreEqual(v16, v15);

    const auto v17 = Vec3f{v1.data()};
    Assert::AreEqual(v17, v1.xyz());

    auto v18 = Vec4f{};
    v18.x() = 1.0f;
    v18.y() = 2.0f;
    v18.z() = 3.0f;
    v18.w() = 4.0f;
    Assert::AreEqual(v18, Vec4f{1.0f, 2.0f, 3.0f, 4.0f});

    constexpr auto v19 = zeroVector<Vec4f>();
    static_assert(v19 == Vec4f{0.0f, 0.0f, 0.0f, 0.0f}, "");

    const auto v20 = basisVector<Vec4f>(Z);
    Assert::IsTrue(v20 == Vec4f{0.0f, 0.0f, 1.0f, 0.0f});

    const auto v21 = Vec3f{v18};
    Assert::AreEqual(v21, Vec3f{1.0f, 2.0f, 3.0f});
}

TEST_METHOD(TestAdd) {
    constexpr Vec4f v0{1.0f, 2.0f, 3.0f, 4.0f};
    constexpr Vec4f v1{2.0f, 4.0f, 6.0f, 8.0f};
    constexpr auto v2 = v0 + v1;
    Assert::AreEqual(Vec4f{3.0f, 6.0f, 9.0f, 12.0f}, v2);
    constexpr Vector<double, 4> v3{2.0, 4.0, 6.0, 8.0};
    static_assert(std::is_convertible<float, double>{}, "");
    static_assert(std::is_convertible<double, float>{}, "");

    auto v5 = v0;
    v5 += v1;
    Assert::AreEqual(v5, v2);

    const auto v6 = Vector<Vec2f, 2>{Vec2f{1.0f, 2.0f}, Vec2f{3.0f, 4.0f}};
    auto v7 = v6;
    v7 += v6;
    Assert::AreEqual(v7, v6 + v6);
}

TEST_METHOD(TestSubtract) {
    constexpr Vec4f v0{2.0f, 4.0f, 6.0f, 8.0f};
    constexpr Vec4f v1{1.0f, 2.0f, 3.0f, 4.0f};
    constexpr auto v2 = v0 - v1;
    Assert::AreEqual(v2, Vec4f{1.0f, 2.0f, 3.0f, 4.0f});
    constexpr Vector<double, 4> v3{v1};
    constexpr Vector<double, 4> v4{2.0, 4.0, 6.0, 8.0};
    constexpr auto v5 = v4 - v3;
    Assert::IsTrue(v5 == Vector<double, 4>{v2});
    constexpr auto v6{v1};
    constexpr auto v7 = v6 - v1;
    static_assert(v7 == Vec4f{0.0f, 0.0f, 0.0f, 0.0f}, "");
}

TEST_METHOD(TestScalarMultiply) {
    constexpr Vec4f v0{1.0f, 2.0f, 3.0f, 4.0f};
    auto v3{ v0 };
    v3 *= 2.0f;
    Assert::AreEqual(v3, Vec4f{ 2.0f, 4.0f, 6.0f, 8.0f });
    const auto v1 = v0 * 2.0f;
    constexpr auto v2 = 2.0f * v0;
    Assert::AreEqual(v1, v2);
    Assert::AreEqual(v1, Vec4f{2.0f, 4.0f, 6.0f, 8.0f});
    Assert::AreEqual(v1, v0 + v0);
}

TEST_METHOD(TestDot) {
    constexpr Vec4f v0{1.0f, 2.0f, 3.0f, 4.0f};
    constexpr Vec4f v1{2.0f, 4.0f, 6.0f, 8.0f};
    constexpr auto s0 = dot(v0, v1);
    Assert::AreEqual(s0, 1.0f * 2.0f + 2.0f * 4.0f + 3.0f * 6.0f + 4.0f * 8.0f);
    static_assert(s0 == 1.0f * 2.0f + 2.0f * 4.0f + 3.0f * 6.0f + 4.0f * 8.0f);
}

TEST_METHOD(TestDivide) {
    constexpr Vec3f v0{2.0f, 4.0f, 6.0f};
    const auto v1 = v0 / 2.0f;
    // static_assert(v1 == Vec3f{1.0f, 2.0f, 3.0f}, "");
    Assert::AreEqual(v1, Vec3f{1.0f, 2.0f, 3.0f});

    constexpr Vec3i v2{2, 4, 6};
    constexpr auto v3 = v2 / 2;
    static_assert(std::is_same<decltype(v3), const Vec3i>{} && v3 == Vec3i{1, 2, 3}, "");
    Assert::AreEqual(v3, Vec3i{1, 2, 3});
}

TEST_METHOD(TestMagnitude) {
    const Vec2f v0{3.0f, 4.0f};
    const auto s0 = magnitude(v0);
    Assert::AreEqual(s0, 5.0f);
    const Vec3f v1{1.0f, 1.0f, 1.0f};
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
    Assert::AreEqual(v0, Vec3f{1.0f, 2.0f, 3.0f});
    Vec3i v2{2, 3, 4};
    v2 /= 2;
    Assert::IsTrue(v2.x() == 2 / 2 && v2.y() == 3 / 2 && v2.z() == 4 / 2);
}

TEST_METHOD(TestSwizzle) {
    static_assert(detail::Max<5, 3, 7, 1, 3>{} == 7, "");
    constexpr auto v0 = Vec4f{1.0f, 2.0f, 3.0f, 4.0f};
    constexpr auto v1 = swizzle<X, Y, Z, W>(v0);
    Assert::AreEqual(v0, v1);
    const auto v2 = swizzle<X>(v0);
    Assert::AreEqual(v2, 1.0f);
    const auto v3 = swizzle<X, X>(v0);
    Assert::AreEqual(v3, Vec2f{1.0f});
    const auto v4 = swizzle<Y, Z>(v0);
    Assert::AreEqual(v4, Vec2f{v0.y(), v0.z()});
    constexpr auto v5 = swizzle<Z, Z, X, Y>(v0);
    Assert::AreEqual(v5, Vec4f{v0.z(), v0.z(), v0.x(), v0.y()});
    constexpr auto v6 = v0.xyz();
    Assert::AreEqual(v6, Vec3f{v0.x(), v0.y(), v0.z()});
    constexpr auto v7 = v0.xy();
    Assert::AreEqual(v7, Vec2f{v0.x(), v0.y()});
    Assert::AreEqual(v0.xz(), Vec2f{v0.x(), v0.z()});
    constexpr auto v8 = swizzle<X, X, Y>(Vec2f{1.0f, 2.0f});
    Assert::AreEqual(v8, Vec3f{1.0f, 1.0f, 2.0f});
}

TEST_METHOD(TestCross) {
    constexpr auto v0 = Vec3f{1.0f, 2.0f, 3.0f};
    constexpr auto v1 = Vec3f{4.0f, 5.0f, 6.0f};
    const auto v2 = cross(v0, v1);
    const auto xv0 = toXmVector(v0);
    const auto xv1 = toXmVector(v1);
    auto xv2 = XMVector3Cross(xv0, xv1);
    Assert::IsTrue(memcmp(&xv2, &v2, sizeof(v2)) == 0);
}
};

}
