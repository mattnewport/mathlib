#include "stdafx.h"

#include <algorithm>
#include <iterator>
#include <type_traits>

#include "CppUnitTest.h"

#include "mathconstants.h"
#include "mathio.h"
#include "matrix.h"
#include "quaternion.h"
#include "vector.h"

#include <DirectXMath.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DirectX;

using namespace mathlib;

using namespace std::literals;

namespace Microsoft { namespace VisualStudio { namespace CppUnitTestFramework {
template <typename T, size_t N>
auto ToString(const Vector<T, N>& x) { RETURN_WIDE_STRING(x); }
template <typename T, size_t M, size_t N>
auto ToString(const Matrix<T, M, N>& x) { RETURN_WIDE_STRING(x); }
template <typename T>
auto ToString(const Quaternion<T>& x) { RETURN_WIDE_STRING(x); }
}}}

namespace UnitTests {

inline auto toXmVector(const Vec3f& v) { return XMLoadFloat3(std::data({XMFLOAT3{v.data()}})); }
inline auto toXmVector(const Vec4f& v) { return XMLoadFloat4(std::data({XMFLOAT4{v.data()}})); }
inline auto toXmVector(const Quatf& q) { return XMLoadFloat4(std::data({XMFLOAT4{q.data()}})); }

inline auto toMat4f(const XMMATRIX& xmm) {
    return MatrixFromDataPointer<Mat4f>(&xmm.r[0].m128_f32[0]);
}

template<typename T>
constexpr auto areNearlyEqual(const T& a, const T& b, const T& eps) {
    return std::abs(a - b) < eps;
}

inline auto areNearlyEqual(const Quatf& q, const XMVECTOR& xmq, float eps) {
    const auto diff = toXmVector(q) - xmq;
    return XMVectorGetX(XMVector4Length(diff)) < eps;
}

inline auto areNearlyEqual(const Vec3f& v, const XMVECTOR& xmv, float eps) {
    const auto diff = toXmVector(v) - xmv;
    return XMVectorGetX(XMVector3Length(diff)) < eps;
}

template <typename T, size_t M, size_t N>
inline auto areNearlyEqual(const Matrix<T, M, N>& x, const Matrix<T, M, N>& y, const T& eps) {
    using namespace std;
    T diffs[M * N] = {};
    transform(begin(x), end(x), begin(y), stdext::make_unchecked_array_iterator(begin(diffs)),
              [](const T& a, const T& b) { return abs(a - b); });
    return all_of(begin(diffs), end(diffs), [eps](const T& d) { return d < eps; });
}

TEST_CLASS(FuncsUnitTests) {
public:
    TEST_METHOD(TestLerp) {
        auto x0 = lerp(0.0f, 1.0f, 0.3f);
        Assert::AreEqual(x0, 0.3f);
        auto x1 = lerp(1.0f, 3.0f, 0.6f);
        Assert::AreEqual(x1, 2.2f);
    }

    TEST_METHOD(TestMax) {
        constexpr auto v0 = Vec3f{ -0.5f, 2.0f, 1.1f };
        constexpr auto v1 = Vec3f{ 0.5f, 0.6f, 0.7f };
        const auto v3 = mathlib::max(v0, v1);
        Assert::AreEqual(v3, Vec3f{ 0.5f, 2.0f, 1.1f });
    }

    TEST_METHOD(TestAbs) {
        constexpr auto v0 = Vec3f{ -0.5f, 2.0f, 1.1f };
        const auto v2 = abs(v0);
        Assert::AreEqual(v2, Vec3f{ 0.5f, 2.0f, 1.1f });
    }

    TEST_METHOD(TestSaturate) {
        constexpr auto v0 = Vec3f{ -0.5f, 2.0f, 0.9f };
        const auto v2 = saturate(v0);
        Assert::AreEqual(v2, Vec3f{ 0.0f, 1.0f, 0.9f });
    }

    TEST_METHOD(TestMinMaxElement) {
        constexpr auto v0 = Vec3f{ 0.9f, 2.0f, -0.5f };
        const auto s0 = minElement(v0);
        const auto s1 = maxElement(v0);
        Assert::AreEqual(s0, -0.5f);
        Assert::AreEqual(s1, 2.0f);
    }

    TEST_METHOD(TestClamp) {
        constexpr auto v0 = Vec3f{-0.5f, 2.0f, 1.1f};
        constexpr auto v1 = Vec3f{0.5f, 0.6f, 0.7f};
        constexpr auto v2 = Vec3f{1.5f, 1.6f, 1.7f};
        const auto v3 = clamp(v0, v1, v2);
        Assert::AreEqual(v3, Vec3f{0.5f, 1.6f, 1.1f});
        const auto v4 = clamp(v0, 1.0f, 2.0f);
        Assert::AreEqual(v4, Vec3f{1.0f, 2.0f, 1.1f});
    }
};

TEST_CLASS(VectorUnitTests){
public:
    TEST_METHOD(TestBasics) {
        const Vec4f v0{1.0f};
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
        Assert::IsTrue(v1 == v3);
        const Vector<double, 4> v4{v1};
        Assert::IsTrue(v4 == v1);
        const Vector<double, 4> v5{1.0f};
        Assert::IsTrue(v5 == v0);
        const Vec2f v6{1.0f, 2.0f};
        Assert::IsTrue(v6.x() == 1.0f && v6.y() == 2.0f);

        const Vector<Vec2f, 2> v7{Vec2f{1.0f, 2.0f}, Vec2f{3.0f, 4.0f}};
        Assert::IsTrue(std::equal(
            std::begin(v7[0]), std::end(v7[1]),
            stdext::make_unchecked_array_iterator(std::begin({1.0f, 2.0f, 3.0f, 4.0f}))));

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
        Vec3f v14 = Vec3f{ v13 };

        Assert::IsTrue(v12[0] == v12.x() && v12[1] == v12.y() && v12[2] == v12.z());
        Assert::IsTrue(-v14 + +v14 == zeroVector<Vec3f>());

        auto pv = new(&v14) Vec3f;
        Assert::AreEqual(*pv, v12);
        pv = new(&v14) Vec3f{}; // check value initialization syntax works
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
        constexpr auto v4 = v0 + v3;
        constexpr auto vvv = v3 + v0;
        Assert::IsTrue(v4 == Vector<double, 4>{3.0, 6.0, 9.0, 12.0});

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
        const Vector<double, 4> v3{v0};
        Assert::IsTrue(v3 == v0);
        constexpr Vector<double, 4> v4{2.0, 4.0, 6.0, 8.0};
        constexpr auto v5 = v4 - v1;
        static_assert(std::is_same<decltype(v5), const Vector<double, 4>>{}, "");
        static_assert(v5 == v2, "");
        Assert::IsTrue(v5 == v2);
        constexpr auto v6{v1};
        constexpr auto v7 = v6 - v1;
        static_assert(v7 == Vec4f{0.0f, 0.0f, 0.0f, 0.0f}, "");
    }

    TEST_METHOD(TestScalarMultiply) {
        constexpr Vec4f v0{1.0f, 2.0f, 3.0f, 4.0f};
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
        constexpr Vector<double, 4> v3{2.0, 4.0, 6.0, 8.0};
        constexpr auto s1 = dot(v0, v3);
        Assert::AreEqual(s1, 1.0f * 2.0 + 2.0f * 4.0 + 3.0f * 6.0 + 4.0f * 8.0);
        static_assert(s1 == 1.0f * 2.0 + 2.0f * 4.0 + 3.0f * 6.0 + 4.0f * 8.0, "");
        Assert::AreEqual(dot(v0, v1), v0 | v1);
        constexpr Vector<float, 5> v4{v0, 5.0f};
        constexpr Vector<float, 5> v5{v1, 10.0f};
        constexpr auto s2 = dot(v4, v5);
        Assert::AreEqual(s2, 1.0f * 2.0f + 2.0f * 4.0f + 3.0f * 6.0f + 4.0f * 8.0f + 5.0f * 10.0f);
    }

    TEST_METHOD(TestDivide) {
        constexpr Vec3f v0{2.0f, 4.0f, 6.0f};
        const auto v1 = v0 / 2.0f;
        //static_assert(v1 == Vec3f{1.0f, 2.0f, 3.0f}, "");
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

TEST_CLASS(MatrixUnitTests){
public:
    TEST_METHOD(TestMatrixBasics) {
        const auto m0 = MatrixFromRows(Vec4f{1.0f, 2.0f, 3.0f, 4.0f}, Vec4f{5.0f, 6.0f, 7.0f, 8.0f},
                                       Vec4f{9.0f, 10.0f, 11.0f, 12.0f});
        const auto m1 = MatrixFromColumns(Vec3f{1.0f, 5.0f, 9.0f}, Vec3f{2.0f, 6.0f, 10.0f},
                                          Vec3f{3.0f, 7.0f, 11.0f}, Vec3f{4.0f, 8.0f, 12.0f});
        Assert::AreEqual(m0, m1);
        Assert::IsTrue(m0.e(0, 0) == m0[0][0] && m0[0][0] == 1.0f);
        Assert::IsTrue(m0.e(1, 2) == m0[1][2] && m0[1][2] == 7.0f);
        Assert::AreEqual(-m0 + +m1, zeroMatrix<float, 3, 4>());

        const auto m2 = identityMatrix<float, 3, 4>();
        const auto m3 = MatrixFromRows(Vec4f{1.0f, 0.0f, 0.0f, 0.0f}, Vec4f{0.0f, 1.0f, 0.0f, 0.0f},
                                       Vec4f{0.0f, 0.0f, 1.0f, 0.0f});
        Assert::AreEqual(m2, m3);
        Assert::AreEqual(m2, identityMatrix<Matrix<float, 3, 4>>());

        const auto scale = scaleMat4f(3.0f);
        Assert::AreEqual(scale,
                         Mat4f{Vec4f{3.0f, 0.0f, 0.0f, 0.0f}, Vec4f{0.0f, 3.0f, 0.0f, 0.0f},
                               Vec4f{0.0f, 0.0f, 3.0f, 0.0f}, Vec4f{0.0f, 0.0f, 0.0f, 1.0f}});

        const auto m4 = Mat4fFromRows({1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f},
                                      {9.0f, 10.0f, 11.0f, 12.0f}, {13.0f, 14.0f, 15.0f, 16.0f});
        Assert::IsTrue(std::equal(std::begin(m4), std::end(m4),
                                  stdext::make_unchecked_array_iterator(std::begin(
                                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                       11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}))));

        const auto m5 = zeroMatrix<float, 3, 4>();
        Assert::IsTrue(std::all_of(std::begin(m5), std::end(m5), [](auto x) { return x == 0.0f; }));

        Assert::AreEqual(ToString(m0), L"{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}"s);

        const auto m6 = m4 + Mat4f{Vec4f{5e-7f}};
        Assert::IsTrue(areNearlyEqual(m6, m4, 1e-6f));
    }

    TEST_METHOD(TestMatrixAdd) {
        const auto m1 = identityMatrix<float, 4, 4>();
        const auto m2 = scaleMat4f(3.0f);
        const auto m3 = m1 + m2;
        Assert::AreEqual(m3, Mat4f{Vec4f{4.0f, 0.0f, 0.0f, 0.0f}, Vec4f{0.0f, 4.0f, 0.0f, 0.0f},
                                   Vec4f{0.0f, 0.0f, 4.0f, 0.0f}, Vec4f{0.0f, 0.0f, 0.0f, 2.0f}});

        auto m4 = m1;
        m4 += m2;
        Assert::AreEqual(m4, m3);
    }

    TEST_METHOD(TestMatrixSubtract) {
        const auto m0 = MatrixFromRows(Vec4f{1.0f, 2.0f, 3.0f, 4.0f}, Vec4f{5.0f, 6.0f, 7.0f, 8.0f},
                                       Vec4f{9.0f, 10.0f, 11.0f, 12.0f});
        const auto m1 = 2.0f * m0;
        Assert::AreEqual(m1 - m0, m0);
    }

    TEST_METHOD(TestMatrixScalarMultiply) {
        const auto m1 = Mat4fFromRows({1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f},
                                      {9.0f, 10.0f, 11.0f, 12.0f}, {13.0f, 14.0f, 15.0f, 16.0f});
        const auto m2 = m1 * 0.5f;
        Assert::AreEqual(m2, Mat4fFromRows({0.5f, 1.0f, 1.5f, 2.0f}, {2.5f, 3.0f, 3.5f, 4.0f},
                                           {4.5f, 5.0f, 5.5f, 6.0f}, {6.5f, 7.0f, 7.5f, 8.0f}));

        auto m3 = m1;
        m3 *= 0.5f;
        Assert::AreEqual(m3, m2);

        auto m4 = m1;
        m4 /= 2.0f;
        Assert::AreEqual(m4, m2);
    }

    TEST_METHOD(TestMatrixColumnAccess){
        const auto m = Mat4f{Vec4f{1.0f, 2.0f, 3.0f, 4.0f}, Vec4f{5.0f, 6.0f, 7.0f, 8.0f},
                             Vec4f{9.0f, 10.0f, 11.0f, 12.0f}, Vec4f{13.0f, 14.0f, 15.0f, 16.0f}};
        const auto c0 = m.column(0);
        Assert::AreEqual(c0, Vec4f{1.0f, 5.0f, 9.0f, 13.0f});
        const auto c1 = m.column(1);
        Assert::AreEqual(c1, Vec4f{2.0f, 6.0f, 10.0f, 14.0f});
        const auto c2 = m.column(2);
        Assert::AreEqual(c2, Vec4f{3.0f, 7.0f, 11.0f, 15.0f});
        const auto c3 = m.column(3);
        Assert::AreEqual(c3, Vec4f{4.0f, 8.0f, 12.0f, 16.0f});
    }

    TEST_METHOD(TestVectorMatrixMultiply) {
        const auto m0 = Mat4f{Vec4f{1.0f, 2.0f, 3.0f, 4.0f}, Vec4f{5.0f, 6.0f, 7.0f, 8.0f},
                              Vec4f{9.0f, 10.0f, 11.0f, 12.0f}, Vec4f{13.0f, 14.0f, 15.0f, 16.0f}};
        const auto v0 = Vec4f{1.0f, 1.0f, 1.0f, 1.0f};
        const auto v1 = v0 * m0;

        const auto xmv0 = toXmVector(v0);
        auto xmm = XMMATRIX{};
        memcpy(&xmm, &m0, sizeof(xmm));
        auto xmv1 = XMVector4Transform(xmv0, xmm);
        Assert::IsTrue(memcmp(&v1, &xmv1, sizeof(v1)) == 0);
        Assert::AreEqual(v1, Vec4f{&xmv1.m128_f32[0]});
    }

    TEST_METHOD(TestMatrixMatrixMultiply) {
        const auto m0 = Mat4f{Vec4f{1.0f, 2.0f, 3.0f, 4.0f}, Vec4f{5.0f, 6.0f, 7.0f, 8.0f},
                              Vec4f{9.0f, 10.0f, 11.0f, 12.0f}, Vec4f{13.0f, 14.0f, 15.0f, 16.0f}};
        const auto m1 = Mat4f{Vec4f{21.0f, 22.0f, 23.0f, 24.0f}, Vec4f{25.0f, 26.0f, 27.0f, 28.0f},
                              Vec4f{29.0f, 30.0f, 31.0f, 32.0f}, Vec4f{33.0f, 34.0f, 35.0f, 36.0f}};
        const auto m2 = m0 * m1;
        auto xmm0 = XMMATRIX{};
        memcpy(&xmm0, &m0, sizeof(xmm0));
        auto xmm1 = XMMATRIX{};
        memcpy(&xmm1, &m1, sizeof(xmm1));
        auto xmm2 = XMMatrixMultiply(xmm0, xmm1);
        Assert::IsTrue(memcmp(&m2, &xmm2, sizeof(m2)) == 0);
        Assert::AreEqual(m2, toMat4f(xmm2));
    }

    TEST_METHOD(TestMatrix4fRotationY) {
        const auto angle = pif / 4.0f;
        auto m = rotationYMat4f(angle);
        auto xmm = XMMatrixRotationY(angle);
        Assert::IsTrue(areNearlyEqual(m, toMat4f(xmm), 1e-6f));
    }

    TEST_METHOD(TestMat4FromQuat) {
        const auto axis = normalize(Vec3f{1.0f, 2.0f, 3.0f});
        const auto angle = pif / 6.0f;
        const auto q0 = QuaternionFromAxisAngle(axis, angle);
        const auto m0 = Mat4FromQuat(q0);
        const auto xmq0 = XMQuaternionRotationAxis(toXmVector(axis), angle);
        const auto xmm0 = XMMatrixRotationQuaternion(xmq0);
        XMMATRIX xmm;
        const auto m1 = toMat4f(xmm0);
        Assert::IsTrue(areNearlyEqual(m0, m1, 1e-6f));
    }

    TEST_METHOD(TestTranspose) {
        constexpr auto v0 = Vec4f{1.0f, 2.0f, 3.0f, 4.0f};
        constexpr auto v1 = Vec4f{5.0f, 6.0f, 7.0f, 8.0f};
        constexpr auto v2 = Vec4f{9.0f, 10.0f, 11.0f, 12.0f};
        constexpr auto v3 = Vec4f{13.0f, 14.0f, 15.0f, 16.0f};
        const auto m0 = Mat4f{v0, v1, v2, v3};
        const auto m1 = transpose(m0);
        const auto m2 = Mat4f{Vec4f{1.0f, 5.0f, 9.0f, 13.0f}, Vec4f{2.0f, 6.0f, 10.0f, 14.0f},
                              Vec4f{3.0f, 7.0f, 11.0f, 15.0f}, Vec4f{4.0f, 8.0f, 12.0f, 16.0f}};
        Assert::IsTrue(m1 == m2);
    }

    TEST_METHOD(TestRotationYMat4f) {
        const auto angle = pif / 6.0f;
        const auto m0 = rotationYMat4f(angle);
        const auto xmm0 = XMMatrixRotationY(angle);
        Assert::IsTrue(areNearlyEqual(m0, toMat4f(xmm0), 1e-6f));
    }

    TEST_METHOD(TestLookAtRhMat4f) {
        const auto eyePos = Vec3f{1.0f, 2.0f, 3.0f};
        const auto at = Vec3f{4.0f, 5.0f, 6.0f};
        const auto up = basisVector<Vec3f>(Y);
        const auto m0 = lookAtRhMat4f(eyePos, at, up);
        const auto xmm0 = DirectX::XMMatrixLookAtRH(
            toXmVector(Vec4f{eyePos, 0}), toXmVector(Vec4f{at, 0}), toXmVector(Vec4f{up, 0}));
        Assert::IsTrue(areNearlyEqual(m0, toMat4f(xmm0), 1e-6f));
    }
};

TEST_CLASS(QuaternionUnitTests) {
    TEST_METHOD(TestQuaternionValueInit) {
        const auto q0 = Quatf{};
        Assert::AreEqual(q0.v(), zeroVector<Vec3f>());
    }

    TEST_METHOD(TestQuaternionBasics) {
        const auto q0 = Quatf{Vec3f{1.0f, 2.0f, 3.0f}, 4.0f};
        const auto q1 = Quatf{q0};
        Assert::IsTrue(q0.v() == Vec3f{1.0f, 2.0f, 3.0f} && q0.s() == 4.0f);
        Assert::IsTrue(q0.x() == q0.v().x() && q0.y() == q0.v().y() && q0.z() == q0.v().z() &&
                       q0.w() == q0.s());
        auto q2 = Quatf{};
        q2.x() = 1.0f;
        q2.y() = 2.0f;
        q2.z() = 3.0f;
        q2.w() = 4.0f;
        Assert::AreEqual(q2, q0);
    }

    TEST_METHOD(TestQuaternionFromAxisAngle) {
        const auto axis = normalize(Vec3f{1.0f, 2.0f, 3.0f});
        const auto angle = pif / 6.0f;
        const auto q0 = QuaternionFromAxisAngle(axis, angle);
        const auto xmAxis = toXmVector(axis);
        const auto xmq0 = XMQuaternionRotationAxis(xmAxis, angle);
        Assert::IsTrue(areNearlyEqual(q0, xmq0, 1e-6f));
    }

    TEST_METHOD(TestQuaternionVectorRotate) {
        const auto axis = normalize(Vec3f{1.0f, 2.0f, 3.0f});
        const auto angle = pif / 6.0f;
        const auto q0 = QuaternionFromAxisAngle(axis, angle);
        const auto xmAxis = toXmVector(axis);
        const auto xmq0 = XMQuaternionRotationAxis(xmAxis, angle);
        const auto v0 = Vec3f{4.0f, 5.0f, 6.0f};
        const auto xmv0 = toXmVector(v0);
        const auto v1 = rotate(v0, q0);
        const auto xmv1 = XMVector3Rotate(xmv0, xmq0);
        Assert::IsTrue(areNearlyEqual(v1, xmv1, 1e-6f));
    }

    TEST_METHOD(TestQuaternionNorm) {
        const auto q0 = Quatf{1.0f, 2.0f, 3.0f, 4.0f};
        const auto s0 = norm(q0);
        const auto xq0 = toXmVector(q0);
        const auto xs0 = XMQuaternionLength(xq0);
        Assert::IsTrue(memcmp(&xq0, &q0, sizeof(q0)) == 0);
    }

    TEST_METHOD(TestQuaternionAdd) {
        const auto q0 = Quatf{1.0f, 2.0f, 3.0f, 4.0f};
        const auto q1 = Quatf{5.0f, 6.0f, 7.0f, 8.0f};
        const auto q2 = q0 + q1;
        const auto xq0 = toXmVector(q0);
        const auto xq1 = toXmVector(q1);
        const auto xq2 = xq0 + xq1;
        Assert::IsTrue(memcmp(&xq2, &q2, sizeof(q2)) == 0);
    }

    TEST_METHOD(TestQuaternionSubtract) {
        const auto q0 = Quatf{1.0f, 2.0f, 3.0f, 4.0f};
        const auto q1 = Quatf{5.0f, 6.0f, 7.0f, 8.0f};
        const auto q2 = q1 - q0;
        const auto xq0 = toXmVector(q0);
        const auto xq1 = toXmVector(q1);
        const auto xq2 = xq1 - xq0;
        Assert::IsTrue(memcmp(&xq2, &q2, sizeof(q2)) == 0);
    }

    TEST_METHOD(TestQuaternionConjugate) {
        const auto q0 = Quatf{1.0f, 2.0f, 3.0f, 4.0f};
        const auto q1 = ~q0;
        const auto xq1 = XMQuaternionConjugate(toXmVector(q0));
        Assert::IsTrue(memcmp(&xq1, &q1, sizeof(q1)) == 0);
    }

    TEST_METHOD(TestQuaternionMultiply) {
        const auto q0 = Quatf{1.0f, 2.0f, 3.0f, 4.0f};
        const auto q1 = Quatf{5.0f, 6.0f, 7.0f, 8.0f};
        const auto q2 = q0 * q1;
        const auto xq0 = toXmVector(q0);
        const auto xq1 = toXmVector(q1);
        const auto xq2 = XMQuaternionMultiply(xq1, xq0);
        Assert::IsTrue(memcmp(&xq2, &q2, sizeof(q2)) == 0);
    }
};

}