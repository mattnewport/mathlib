#include "stdafx.h"
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
template<> inline std::wstring ToString<Vec4f>(const Vec4f& t) { RETURN_WIDE_STRING(t); }
}}}

namespace UnitTests {
TEST_CLASS(VectorUnitTests){
public :

    TEST_METHOD(TestAdd) {
        Vec4f v0{ 1.0f, 2.0f, 3.0f, 4.0f };
        Vec4f v1{ 2.0f, 4.0f, 6.0f, 8.0f };
        auto v2 = v0 + v1;
        Assert::AreEqual(Vec4f{ 3.0f, 6.0f, 9.0f, 12.0f }, v2);
    }

    TEST_METHOD(TestScalarMultiply) {
        Vec4f v0{ 1.0f, 2.0f, 3.0f, 4.0f };
        Vec4f v1 = v0 * 2.0f;
        Vec4f v2 = 2.0f * v0;
        Assert::AreEqual(v1, v2);
        Assert::AreEqual(v1, Vec4f{ 2.0f, 4.0f, 6.0f, 8.0f });
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