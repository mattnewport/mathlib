#include "stdafx.h"

#include <algorithm>
#include <iterator>
#include <type_traits>

#include "CppUnitTest.h"

#include "mathconstants.h"
#include "quaternion.h"
#include "quaternionio.h"

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
template <typename T>
auto ToString(const Quaternion<T>& x) {
    RETURN_WIDE_STRING(x);
}
}  // namespace CppUnitTestFramework
}  // namespace VisualStudio
}  // namespace Microsoft

namespace UnitTests {

inline auto toXmVector(const Vec3f& v) { return XMLoadFloat3(std::data({XMFLOAT3{v.data()}})); }
inline auto toXmVector(const Vec4f& v) { return XMLoadFloat4(std::data({XMFLOAT4{v.data()}})); }
inline auto toXmVector(const Quatf& q) { return XMLoadFloat4(std::data({XMFLOAT4{q.data()}})); }

template <typename T>
constexpr auto areNearlyEqual(const T& a, const T& b, const T& eps) {
    return std::abs(a - b) < eps;
}

inline auto areNearlyEqual(const Vec3f& v, const XMVECTOR& xmv, float eps) {
    const auto diff = toXmVector(v) - xmv;
    return XMVectorGetX(XMVector3Length(diff)) < eps;
}

inline auto areNearlyEqual(const Quatf& q, const XMVECTOR& xmq, float eps) {
    const auto diff = toXmVector(q) - xmq;
    return XMVectorGetX(XMVector4Length(diff)) < eps;
}

TEST_CLASS(QuaternionUnitTests) {
    TEST_METHOD(TestQuaternionValueInit) {
        const auto q0 = Quatf{};
        Assert::AreEqual(q0.v(), zeroVector<Vec3f>());
    }

    TEST_METHOD(TestQuaternionEquality) {
        constexpr auto q0 = Quatf{ Vec3f{ 1.0f, 2.0f, 3.0f }, 4.0f };
        constexpr auto q1 = Quatf{ q0 };
        constexpr auto q2 = Quatf{ 1.0f, 2.0f, 3.0f, 4.0f };
        static_assert(q2 == q0);
        static_assert(q0 != Quatf::identity());
    }

    TEST_METHOD(TestQuaternionBasics) {
        constexpr auto q0 = Quatf{Vec3f{1.0f, 2.0f, 3.0f}, 4.0f};
        constexpr auto q1 = Quatf{q0};
        constexpr auto q2 = Quatf{ 1.0f, 2.0f, 3.0f, 4.0f };
        static_assert(q2 == q0);
        constexpr auto q3 = Quatf{ Vec4f{5.0f, 6.0f, 7.0f, 8.0f} };
        static_assert(q3.x() == 5.0f && q3.y() == 6.0f && q3.z() == 7.0f && q3.w() == 8.0f);
        Assert::IsTrue(q0.v() == Vec3f{1.0f, 2.0f, 3.0f} && q0.s() == 4.0f);
        Assert::IsTrue(q0.x() == q0.v().x() && q0.y() == q0.v().y() && q0.z() == q0.v().z() &&
                       q0.w() == q0.s());
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

        const auto init = basisVector<Vector<double, 3>>(Y);
        const auto rotZ = QuaternionFromAxisAngle(basisVector<Vector<double, 3>>(X), 30.0);
        const auto elev = rotate(init, rotZ);
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

}  // namespace UnitTests
