#pragma once

#include <cmath>
#include <functional>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

// Vectors are treated as row vectors for the purposes of matrix multiplication (so to transform a
// Vector v by a Matrix M use v * M rather than M * v)

// This code is very generic and uses multiple layers of function helpers, it compiles down to
// pretty efficient code in release builds but in debug builds without any inlining it will be
// pretty inefficient. Using /Ob1
// http://msdn.microsoft.com/en-us/library/47238hez.aspx for debug builds in Visual Studio will help
// debug performance a lot.

namespace mathlib {

template<typename... Ts>
inline void eval(Ts&&...) {}

struct integral_tag {};
struct floating_point_tag {};
template <typename T>
using tag = std::conditional_t<std::is_floating_point<T>::value, floating_point_tag, integral_tag>;

template <typename T, size_t N>
class Vector {
public:
    constexpr Vector() = default;

    // Standard constructor taking a sequence of exactly N objects convertible to T (no narrowing
    // conversions).
    template <typename U, typename V, typename... Us,
              typename = std::enable_if_t<(sizeof...(Us) == N - 2)>>
    constexpr Vector(U u, V v, Us... us) : e_{u, v, us...} {}
    // For convenience we have an explicit constructor taking a single argument that sets all
    // members of the vector to the value of that argument.
    template <typename U, size_t... Is>
    explicit constexpr Vector(U u, std::index_sequence<Is...>) : e_{((void)Is, u)...} {}
    // MNTODO: can't make this constexpr due to a bug in VS2015:
    // http://stackoverflow.com/questions/32489702/constexpr-with-delegating-constructors
    template <typename U>
    explicit Vector(U u) : Vector{u, std::make_index_sequence<N>{}} {}

    // Templated conversion constructor from a Vector<U, N>
    template <typename U, size_t... Is>
    constexpr Vector(const Vector<U, N>& x, std::index_sequence<Is...>) : e_{x.e_[Is]...} {}
    // MNTODO: can't make this constexpr due to a bug in VS2015:
    // http://stackoverflow.com/questions/32489702/constexpr-with-delegating-constructors
    template <typename U>
    Vector(const Vector<U, N>& x) : Vector{x, std::make_index_sequence<N>{}} {}

    T& e(size_t i) { return e_[i]; }
    constexpr const T& e(size_t i) const { return e_[i]; }

    T& x() {
        return e_[0];
    }
    constexpr T x() const {
        return e_[0];
    }
    template<size_t M = N>
    std::enable_if_t<(M > 1), T&> y() {
        return e_[1];
    }
    template<size_t M = N>
    constexpr std::enable_if_t<(M > 1), T> y() const {
        return e_[1];
    }
    template<size_t M = N>
    std::enable_if_t<(M > 2), T&> z() {
        return e_[2];
    }
    template<size_t M = N>
    constexpr std::enable_if_t<(M > 2), T> z() const {
        return e_[2];
    }
    template<size_t M = N>
    std::enable_if_t<(M > 3), T&> w() {
        return e_[3];
    }
    template<size_t M = N>
    constexpr std::enable_if_t<(M > 3), T> w() const {
        return e_[3];
    }

    auto begin() { return std::begin(e_); }
    auto end() { return std::end(e_); }
    auto begin() const { return std::begin(e_); }
    auto end() const { return std::end(e_); }

    const T* data() const { return e_; }

    template<typename U>
    Vector& operator+=(const Vector<U, N>& x) {
        return plusEquals(x, std::make_index_sequence<N>{});
    }

    template<typename U>
    Vector& operator-=(const Vector<U, N>& x) {
        return minusEquals(x, std::make_index_sequence<N>{});
    }

    template<typename U>
    Vector& operator*=(U x) {
        return multiplyEquals(x, std::make_index_sequence<N>{});
    }

    template<typename U>
    Vector& operator/=(U x) {
        return divideEquals(x, std::make_index_sequence<N>{}, tag<U>{});
    }

private:
    T e_[N];

    template<typename U, size_t M> friend class Vector;

    template <typename U, size_t... Is>
    Vector& plusEquals(const Vector<U, N>& x, std::index_sequence<Is...>) {
        eval(e_[Is] += x.e_[Is]...);
        return *this;
    }

    template <typename U, size_t... Is>
    Vector& minusEquals(const Vector<U, N>& x, std::index_sequence<Is...>) {
        eval(e_[Is] -= x.e_[Is]...);
        return *this;
    }

    template <typename U, size_t... Is>
    Vector& multiplyEquals(U x, std::index_sequence<Is...>) {
        eval(e_[Is] *= x...);
        return *this;
    }

    template <typename U, size_t... Is>
    Vector& divideEquals(U x, std::index_sequence<Is...>, integral_tag) {
        eval(e_[Is] /= x...);
        return *this;
    }

    template <typename U, size_t... Is>
    Vector& divideEquals(U x, std::index_sequence<Is...>, floating_point_tag) {
        const auto s = U{1} / x;
        eval(e_[Is] *= s...);
        return *this;
    }
};

template<typename T>
struct IsVector : std::false_type {};

template<typename T, size_t N>
struct IsVector<Vector<T, N>> : std::true_type {};

template<typename T>
struct VectorDimension;

template<typename T, size_t N>
struct VectorDimension<Vector<T, N>> : std::integral_constant<size_t, N> {};

template<typename T>
struct VectorElementType;

template<typename T, size_t N>
struct VectorElementType<Vector<T, N>> {
    using type = T;
};

template<typename T>
using VectorElementType_t = typename VectorElementType<T>::type;

using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;

using Vec2i = Vector<int, 2>;
using Vec3i = Vector<int, 3>;
using Vec4i = Vector<int, 4>;

template<typename T, size_t N>
constexpr auto zeroVector() {
    return Vector<T, N>{T(0)};
}

template <typename T, size_t N, size_t... Js>
constexpr auto basisVector(size_t i, std::index_sequence<Js...>) {
    return Vector<T, N>{T(i == Js)...};
}

template<typename T, size_t N>
constexpr auto basisVector(size_t i) {
    return basisVector<T, N>(i, std::make_index_sequence<N>{});
}

// Apply a function F(T) to each element of Vector x and return a new Vector of the results
template <typename F, size_t... Is, typename T, size_t N>
constexpr auto memberwise(F f, std::index_sequence<Is...>, const Vector<T, N>& x) {
    using resvec = Vector<decltype(f(x.e(0))), N>;
    return resvec{f(x.e(Is))...};
}

template <typename F, typename T, size_t N>
constexpr auto memberwise(F f, const Vector<T, N>& x) {
    return memberwise(f, std::make_index_sequence<N>{}, x);
}

// Apply a function F(T, U) pairwise to elements of x and y and return a new Vector of the results
template <typename F, size_t... Is, typename T, typename U, size_t N>
constexpr auto memberwise(F f, std::index_sequence<Is...>, const Vector<T, N>& x, const Vector<U, N>& y) {
    using resvec = Vector<decltype(f(x.e(0), y.e(0))), N>;
    return resvec{f(x.e(Is), y.e(Is))...};
}

template <typename F, typename T, typename U, size_t N>
constexpr auto memberwise(F f, const Vector<T, N>& x, const Vector<U, N>& y) {
    return memberwise(f, std::make_index_sequence<N>{}, x, y);
}

// Apply a function F(T, U) to elements of x with scalar y and return a new Vector of the results
template <typename F, size_t... Is, typename T, typename U, size_t N>
constexpr auto memberwise(F f, std::index_sequence<Is...>, const Vector<T, N>& x, U y) {
    using resvec = Vector<decltype(f(x.e(0), y)), N>;
    return resvec{ f(x.e(Is), y)... };
}

template <typename F, typename T, typename U, size_t N>
constexpr auto memberwise(F f, const Vector<T, N>& x, U y) {
    return memberwise(f, std::make_index_sequence<N>{}, x, y);
}

// MNTODO: replace this fold machinery with C++17 fold expressions once available
template <typename F, typename T>
constexpr auto fold_impl(F, T t) {
    return t;
}

template <typename F, typename T>
constexpr auto fold_impl(F f, T x, T y) {
    return f(x, y);
}

template <typename F, typename T, typename... Ts>
constexpr auto fold_impl(F f, T x, T y, Ts... ts) {
    return f(x, fold_impl(f, y, ts...));
}

template <typename F, typename T, size_t N, size_t... Is>
constexpr auto fold(F f, const Vector<T, N>& v, std::index_sequence<Is...>) {
    return fold_impl(f, v.e(Is)...);
}

template <typename F, typename T, size_t N>
constexpr auto fold(F f, const Vector<T, N>& v) {
    return fold(f, v, std::make_index_sequence<N>{});
}

// Make a tuple of the elements of Vector a, a helper for implementing operators but also
// potentially useful on its own.
template <typename T, size_t N, size_t... Is>
constexpr auto make_tuple(const Vector<T, N>& a, std::index_sequence<Is...>) {
    return std::make_tuple(a.e(Is)...);
}

template <typename T, typename U, size_t N>
constexpr bool operator==(const Vector<T, N>& a, const Vector<U, N>& b) {
    return make_tuple(a, std::make_index_sequence<N>{}) ==
           make_tuple(b, std::make_index_sequence<N>{});
}

template <typename T, typename U, size_t N>
constexpr bool operator!=(const Vector<T, N>& a, const Vector<U, N>& b) {
    return !(a == b);
}

template <typename T, typename U, size_t N>
constexpr bool operator<(const Vector<T, N>& a, const Vector<U, N>& b) {
    return make_tuple(a, std::make_index_sequence<N>{}) <
           make_tuple(b, std::make_index_sequence<N>{});
}

template <typename T, typename U, size_t N>
constexpr auto operator+(const Vector<T, N>& a, const Vector<U, N>& b) {
    return memberwise(std::plus<>{}, a, b);
}

template <typename T, typename U, size_t N>
constexpr auto operator-(const Vector<T, N>& a, const Vector<U, N>& b) {
    return memberwise(std::minus<>{}, a, b);
}

template <typename T, typename U, size_t N>
constexpr auto operator*(const Vector<T, N>& a, U s) {
    return memberwise(std::multiplies<>{}, a, s);
}

template <typename T, typename U, size_t N>
constexpr auto operator*(T s, const Vector<U, N>& a) {
    return a * s;
}

template <typename T, typename U, size_t N>
constexpr auto divide(const Vector<T, N>& a, U s, integral_tag) {
    return memberwise(std::divides<>{}, a, s);
}
template <typename T, typename U, size_t N>
constexpr auto divide(const Vector<T, N>& a, U s, floating_point_tag) {
    return a * (T{1} / s);
}

template <typename T, typename U, size_t N>
constexpr auto operator/(const Vector<T, N>& a, U s) {
    return divide(a, s, tag<U>{});
}

template <typename T, typename U, size_t N>
constexpr auto dot(const Vector<T, N>& a, const Vector<U, N>& b) {
    return fold(std::plus<>{}, memberwise(std::multiplies<>{}, a, b));
}

template <typename T, size_t N>
constexpr T magnitude(const Vector<T, N>& a) {
    return sqrt(dot(a, a));
}

template <typename T, size_t N>
constexpr auto normalize(const Vector<T, N>& a) {
    return a * (T{1} / magnitude(a));
}

template<typename T>
constexpr Vector<T, 3> cross(const Vector<T, 3>& a, const Vector<T, 3>& b) {
    return {a.y()*b.z() - a.z()*b.y(), a.z()*b.x() - a.x()*b.z(), a.x()*b.y() - a.y()*b.x()};
}

}  // namespace mathlib
