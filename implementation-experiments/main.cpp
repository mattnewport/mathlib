#include <iostream>

template<typename Vector, typename T, size_t N, size_t... Is>
struct VectorBase {
    T e[N];

    constexpr const T& operator[](size_t i) const noexcept { return e[i]; }

    constexpr Vector& operator+=(const Vector& x) {
        ((e[Is] += x.e[Is]), ...);
        return static_cast<Vector&>(*this);
    }
};

template <typename T>
struct VectorDimension;

template <typename T>
constexpr size_t VectorDimension_v = VectorDimension<T>::value;

template <typename T>
struct VectorElementType;

template <typename T>
using VectorElementType_t = typename VectorElementType<T>::type;

template <typename Vector>
struct MakeVectorBase {
private:
    template <size_t... Is>
    static constexpr auto make(std::index_sequence<Is...>) {
        return VectorBase<Vector, VectorElementType_t<Vector>, sizeof...(Is), Is...>{};
    }

public:
    using type = decltype(make(std::make_index_sequence<VectorDimension_v<Vector>>{}));
};

template <typename Vector>
using MakeVectorBase_t = typename MakeVectorBase<Vector>::type;


template<typename T, size_t N>
class Vector : private MakeVectorBase_t<Vector<T, N>> {
private:
    using base = MakeVectorBase_t<Vector<T, N>>;
    friend base;
public:
    template <typename... Ts>
    constexpr Vector(const Ts&... ts) noexcept : base{ ts... } {}

    using base::operator[];
    using base::operator+=;
};

template <typename T, size_t N>
struct VectorDimension<Vector<T, N>> : std::integral_constant<size_t, N> {};

template <typename T, size_t N>
struct VectorElementType<Vector<T, N>> {
    using type = T;
};

std::ostream& operator<<(std::ostream& os, const Vector<float, 3>& x) {
    return os << "(" << x[0] << ", " << x[1] << ", " << x[2] << ")";
}

int main() {
    constexpr Vector<float, 3> v1{ 1.0f, 2.0f, 3.0f }, v2{ 4.0f, 5.0f, 6.0f };
    auto v3{ v1 };
    std::cout << v1 << '\n' << v2 << '\n' << v3 << '\n';
    v3 += v2;
    std::cout << v3 << '\n';
}
