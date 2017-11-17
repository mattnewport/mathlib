#include "stdafx.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <type_traits>

#include "unittestwrapper.h"

#include "mathconstants.h"
#include "vector.h"
#include "vectorio.h"

#if defined(__clang__)
#elif defined(_MSC_VER)
#include <DirectXMath.h>
#define USE_DIRECTXMATH
using namespace DirectX;
#endif

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

template <typename T>
constexpr auto areNearlyEqual(const T& a, const T& b, const T& eps) {
    return std::abs(a - b) < eps;
}

#ifdef USE_DIRECTXMATH
inline auto toXmVector(const Vec3f& v) { return XMLoadFloat3(std::data({XMFLOAT3{v.data()}})); }
inline auto toXmVector(const Vec4f& v) { return XMLoadFloat4(std::data({XMFLOAT4{v.data()}})); }

inline auto areNearlyEqual(const Vec3f& v, const XMVECTOR& xmv, float eps) {
    const auto diff = toXmVector(v) - xmv;
    return XMVectorGetX(XMVector3Length(diff)) < eps;
}
#endif

TEST_CLASS(VectorUnitTests){public :

TEST_METHOD(TestVectorConstructors){
    using namespace std;
    // Vec4f cannot be constructed from a single float
    static_assert(!is_constructible_v<Vec4f, float>);
    // Vec4f cannot be constructed from a char*
    static_assert(!is_constructible_v<Vec4f, char*>);
    // Vec4f cannot be constructed from a double
    static_assert(!is_constructible_v<Vec4f, double>);
    // Vec4f can be constructed from 4 floats
    static_assert(is_constructible_v<Vec4f, float, float, float, float>);
    // Vec4f cannot be constructed from 4 doubles
    static_assert(!is_constructible_v<Vec4f, double, double, double, double>);
    // Vec4f can be constructed from Vec4i
    static_assert(is_constructible_v<Vec4f, Vec4i>);
    // Vec4f cannot be constructed from Vec3i
    static_assert(!is_constructible_v<Vec4f, Vec3i>);
    // Vec4f can be constructed from Vec3f and float
    static_assert(is_constructible_v<Vec4f, Vec3f, float>);
    // Vec4f cannot be constructed from Vec3i and int
    static_assert(!is_constructible_v<Vec4f, Vec3i, int>);

    // Construct from N Ts
    constexpr Vec4f v1{1.0f, 2.0f, 3.0f, 4.0f};
    static_assert(v1[0] == 1.0f && v1[1] == 2.0f && v1[2] == 3.0f && v1[3] == 4.0f);
    Assert::IsTrue(v1.x() == 1.0f && v1.y() == 2.0f && v1.z() == 3.0f && v1.w() == 4.0f);
    Assert::IsTrue(v1[0] == 1.0f && v1[1] == 2.0f && v1[2] == 3.0f && v1[3] == 4.0f);

    // Construct Vector<T, N> from Vector<T, N-1> and a T
    constexpr Vector<float, 1> v2{ 1.0f };
    static_assert(v1[0] == 1.0f);
    constexpr Vec2f v3{ v2, 2.0f };
    static_assert(v3[0] == 1.0f && v3[1] == 2.0f);
    constexpr Vec3f v4{ v3, 3.0f };
    static_assert(v4[0] == 1.0f && v4[1] == 2.0f && v4[2] == 3.0f);
    Assert::IsTrue(v4[0] == 1.0f && v4[1] == 2.0f && v4[2] == 3.0f);

    // Construct Vector<T, N> from Vector<T, M> where M > N (take first N elements)
    constexpr Vec4f v5{ 1.0f, 2.0f, 3.0f, 4.0f };
    constexpr Vec3f v6{ v5 };
    static_assert(v6[0] == 1.0f && v6[1] == 2.0f && v6[2] == 3.0f);

    // Construct Vector<U, N> from Vector<T, N>
    constexpr Vector<double, 4> vd1{ v5 };
    static_assert(vd1[0] == 1.0 && vd1[1] == 2.0 && vd1[2] == 3.0 && vd1[3] == 4.0);

    // Construct Vector<T, N> from array of N Ts
    constexpr float fs[]{ 5.0f, 6.0f, 7.0f, 8.0f };
    constexpr Vec4f v7{ fs };
    static_assert(v7[0] == 5.0f && v7[1] == 6.0f && v7[2] == 7.0f && v7[3] == 8.0f);
}

TEST_METHOD(TestVectorValueInitialization) {
    using namespace std;
    union {
        Vec4f v;
        char data[sizeof(Vec4f)];
    } x;
    memset(x.data, 0xbd, sizeof(x.data));
    new (x.data) Vec4f{};
    Assert::IsTrue(x.v[0] == 0.0f && x.v[1] == 0.0f && x.v[2] == 0.0f && x.v[3] == 0.0f);
}

TEST_METHOD(TestVectorEquality) {
    constexpr Vec4f v0{ 1.0f, 1.0f, 1.0f, 1.0f };
    constexpr Vec4f v1{ 1.0f, 2.0f, 3.0f, 4.0f };
    constexpr auto v2 = v1;
    static_assert(v2 == v1);
    Assert::IsTrue(v2 == v1);
    static_assert(v0 != v1);
    Assert::IsTrue(v0 != v1);
    static_assert(!(v2 != v1));
    Assert::IsFalse(v2 != v1);
    static_assert(!(v0 == v1));
    Assert::IsFalse(v0 == v1);
    constexpr Vec3f v7{ v1 };
    static_assert(v7 == Vec3f{ 1.0f, 2.0f, 3.0f });

    constexpr Vector<double, 4> v3{ 1.0, 2.0, 3.0, 4.0 };
    const Vector<double, 4> v4{ v1 };
    Assert::IsTrue(v4 == v3);
    const Vector<double, 4> v5{ 1.0, 1.0, 1.0, 1.0 };
    Assert::IsTrue(v5 == Vector<double, 4>{v0});
}

TEST_METHOD(TestVectorMemberAccess) {
    constexpr Vec3f v0{ 1.0f, 2.0f, 3.0f };
    static_assert(v0.x() == 1.0f && v0.y() == 2.0f && v0.z() == 3.0f);
    static_assert(v0[0] == v0.x() && v0[1] == v0.y() && v0[2] == v0.z());
    static_assert(v0.e(0) == v0.x() && v0.e(1) == v0.y() && v0.e(2) == v0.z());
    static_assert(v0.e(X) == v0.x() && v0.e(Y) == v0.y() && v0.e(Z) == v0.z());
    // This should not compile
    // static_assert(v0.w() == 0.0f);

    Vec4f v1{};
    Assert::AreEqual(v1.x(), 0.0f);
    v1[0] = 1.0f;
    Assert::AreEqual(v1.x(), 1.0f);
    Assert::AreEqual(v1[0], 1.0f);

    // Tuple style / structured bindings
    static_assert(v0.get<0>() == 1.0f && v0.get<1>() == 2.0f && v0.get<2>() == 3.0f);
    static_assert(get<0>(v0) == 1.0f && get<1>(v0) == 2.0f && get<2>(v0) == 3.0f);
    const auto [x, y, z] = v0;
    Assert::IsTrue(x == 1.0f && y == 2.0f && z == 3.0f);
    Vec3f v2{ 1.0f, 2.0f, 3.0f };
    auto& [xr, yr, zr] = v2;
    xr = 4.0f;
    Assert::IsTrue(v2.x() == 4.0f && yr == 2.0f && zr == 3.0f);
}
    
TEST_METHOD(TestVectorBasics){
    const Vec2f v6{ 1.0f, 2.0f };
    Assert::IsTrue(v6.x() == 1.0f && v6.y() == 2.0f);

    const Vector<Vec2f, 2> v7{Vec2f{1.0f, 2.0f}, Vec2f{3.0f, 4.0f}};
    Assert::IsTrue(std::equal(v7[0].data(), v7[0].data() + 4,
                              stdext::make_unchecked_array_iterator(std::begin({1.0f, 2.0f, 3.0f,
                                                                                4.0f}))));

    const auto v8 = Vec3f{v6, 3.0f};
    Assert::AreEqual(v8, Vec3f{1.0f, 2.0f, 3.0f});

    static_assert(IsVector<Vec4f>{});
    static_assert(!IsVector<std::tuple<int, int>>{});
    static_assert(VectorDimension<Vec4f>{} == 4);
    static_assert(std::is_same<float, VectorElementType_t<Vec4f>>{});

    constexpr Vec4f v1{ 1.0f, 2.0f, 3.0f, 4.0f };

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

    const auto vd1 = basisVector<Vector<double, 3>>(Y);
    Assert::IsTrue(vd1[0] == 0.0 && vd1[1] == 1.0 && vd1[2] == 0.0);

    const auto v17 = Vec3f{v1.data()};
    Assert::AreEqual(v17, v1.xyz());

    constexpr auto v19 = zeroVector<Vec4f>();
    static_assert(v19 == Vec4f{0.0f, 0.0f, 0.0f, 0.0f}, "");
    static_assert(v19 == Vec4f::zero());

    const auto v20 = basisVector<Vec4f>(Z);
    Assert::IsTrue(v20 == Vec4f{0.0f, 0.0f, 1.0f, 0.0f});
}

TEST_METHOD(TestVectorAdd) {
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

TEST_METHOD(TestVectorSubtract) {
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

TEST_METHOD(TestVectorScalarMultiply) {
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

TEST_METHOD(TestVectorDot) {
    constexpr Vec4f v0{1.0f, 2.0f, 3.0f, 4.0f};
    constexpr Vec4f v1{2.0f, 4.0f, 6.0f, 8.0f};
    constexpr auto s0 = dot(v0, v1);
    Assert::AreEqual(s0, 1.0f * 2.0f + 2.0f * 4.0f + 3.0f * 6.0f + 4.0f * 8.0f);
    static_assert(s0 == 1.0f * 2.0f + 2.0f * 4.0f + 3.0f * 6.0f + 4.0f * 8.0f);
}

TEST_METHOD(TestVectorDivide) {
    constexpr Vec3f v0{2.0f, 4.0f, 6.0f};
    const auto v1 = v0 / 2.0f;
    // static_assert(v1 == Vec3f{1.0f, 2.0f, 3.0f}, "");
    Assert::AreEqual(v1, Vec3f{1.0f, 2.0f, 3.0f});

    constexpr Vec3i v2{2, 4, 6};
    constexpr auto v3 = v2 / 2;
    static_assert(std::is_same<decltype(v3), const Vec3i>{} && v3 == Vec3i{1, 2, 3}, "");
    Assert::AreEqual(v3, Vec3i{1, 2, 3});
}

TEST_METHOD(TestVectorMagnitude) {
    const Vec2f v0{3.0f, 4.0f};
    const auto s0 = magnitude(v0);
    Assert::AreEqual(s0, 5.0f);
    const Vec3f v1{1.0f, 1.0f, 1.0f};
    const auto s1 = magnitude(v1);
    Assert::AreEqual(s1, sqrt(3.0f));
}

TEST_METHOD(TestVectorNormalize) {
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

TEST_METHOD(TestVectorOpAssignment) {
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

TEST_METHOD(TestVectorSwizzle) {
    static_assert(detail::Max<5, 3, 7, 1, 3>{} == 7, "");
    constexpr auto v0 = Vec4f{1.0f, 2.0f, 3.0f, 4.0f};
    constexpr auto v1 = v0.swizzled<X, Y, Z, W>();
    Assert::AreEqual(v0, v1);
    const auto v2 = v0.swizzled<X>();
    Assert::AreEqual(v2, 1.0f);
    const auto v3 = v0.swizzled<X, X>();
    Assert::AreEqual(v3, Vec2f{1.0f, 1.0f});
    const auto v4 = v0.swizzled<Y, Z>();
    Assert::AreEqual(v4, Vec2f{v0.y(), v0.z()});
    constexpr auto v5 = v0.swizzled<Z, Z, X, Y>();
    Assert::AreEqual(v5, Vec4f{v0.z(), v0.z(), v0.x(), v0.y()});
    constexpr auto v6 = v0.xyz();
    Assert::AreEqual(v6, Vec3f{v0.x(), v0.y(), v0.z()});
    constexpr auto v7 = v0.xy();
    Assert::AreEqual(v7, Vec2f{v0.x(), v0.y()});
    Assert::AreEqual(v0.xz(), Vec2f{v0.x(), v0.z()});
    constexpr auto v8 = Vec2f{1.0f, 2.0f}.swizzled<X, X, Y>();
    Assert::AreEqual(v8, Vec3f{1.0f, 1.0f, 2.0f});

    const Vector<double, 3> vd1{ 1.0, 2.0, 3.0 };
    const auto vd2 = vd1.swizzled<Z, X, Y>();
    Assert::IsTrue(vd2[0] == 3.0 && vd2[1] == 1.0 && vd2[2] == 2.0);
}

TEST_METHOD(TestVectorCross) {
    constexpr auto v0 = Vec3f{1.0f, 2.0f, 3.0f};
    constexpr auto v1 = Vec3f{4.0f, 5.0f, 6.0f};
    [[maybe_unused]] const auto v2 = cross(v0, v1);
#ifdef USE_DIRECTXMATH
    const auto xv0 = toXmVector(v0);
    const auto xv1 = toXmVector(v1);
    auto xv2 = XMVector3Cross(xv0, xv1);
    Assert::IsTrue(memcmp(&xv2, &v2, sizeof(v2)) == 0);
#endif
}
};

}
