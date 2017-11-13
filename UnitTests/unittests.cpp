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
}