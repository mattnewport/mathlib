#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

// Vectors are treated as row vectors for the purposes of matrix multiplication (so to transform a
// Vector v by a Matrix M use v * M rather than M * v)

// This code is very generic and uses multiple layers of function helpers, it compiles down to
// pretty efficient code in release builds but in debug builds without any inlining it will be
// pretty inefficient. Using /Ob1 http://msdn.microsoft.com/en-us/library/47238hez.aspx for debug
// builds in Visual Studio will help debug performance a lot.

// See this great blog post by Nathan Reed for some discussion of design decisions around vector
// math libraries. He independently reaches many of the same design decisions that have been made
// for this library: http://www.reedbeta.com/blog/2013/12/28/on-vector-math-libraries/

namespace mathlib {

namespace detail {}

struct integral_tag {};
struct floating_point_tag {};
template <typename T>
using tag = std::conditional_t<std::is_floating_point<T>::value, floating_point_tag, integral_tag>;

enum VectorComponents { X = 0, Y = 1, Z = 2, W = 3 };

// IS defaulted template argument is an implementation trick to make it possible to write operators
// that can directly deduce an index sequence and therefore be implemented without calling a helper
// function. This both reduces the amount of boilerplate code that needs to be written and reduces
// inlining depth which is particulary helpful for performance in debug builds if inlining is not
// enabled.
template <typename T, size_t N, typename IS = std::make_index_sequence<N>>
class Vector {
    // This helper should not be necessary once we have C++17 fold expressions
    template <typename... Ts>
    static constexpr void eval(Ts&&...) {}

    // Make all Vectors friends of eachother
    template<typename, size_t, typename> friend class Vector;
public:
    constexpr Vector() = default;
    constexpr Vector(const Vector&) = default;

    // Standard constructor taking a sequence of exactly N objects convertible to T.
    // Question: do we want to allow narrowing conversions? Currently allowed but maybe a bad idea?
    template <typename... Us, typename = std::enable_if_t<(sizeof...(Us) == N - 2)>>
    constexpr Vector(const T& x, const T& y, Us&&... us) : e_{x, y, us...} {}

    // For convenience we have an explicit constructor taking a single argument that sets all
    // members of the vector to the value of that argument.
    // MNTODO: can't make this constructors constexpr due to a bug in VS2015:
    // http://stackoverflow.com/questions/32489702/constexpr-with-delegating-constructors
    explicit Vector(const T& x) : Vector{x, IS{}} {}
    // Templated conversion constructor from a Vector<U, N>
    // Question: do we want to allow narrowing conversions? Currently allowed but maybe a bad idea?
    template <typename U, size_t... Is>
    constexpr explicit Vector(const Vector<U, N, std::index_sequence<Is...>>& x) : e_{T(x.e_[Is])...} {}
    // Templated conversion constructor from a Vector<T, M> with M > N where we take the first N
    // elements
    template <size_t M, size_t... Is, typename = std::enable_if_t<(M > N)>>
    explicit Vector(const Vector<T, M, std::index_sequence<Is...>>& x) : Vector{x.e_, IS{}} {}
    // Templated conversion constructor from a Vector<U, N-1> and a scalar V
    // Question: do we want to allow narrowing conversions? Currently allowed but maybe a bad idea?
    template <size_t... Is>
    constexpr explicit Vector(const Vector<T, N - 1, std::index_sequence<Is...>>& x, const T& s)
        : e_{T(x.e_[Is])..., T(s)} {}

    // Construct from a pointer to N Ts
    explicit Vector(const T* p) : Vector{p, IS{}} {}

    // Generic element access
    T& operator[](size_t i) { return e_[i]; }
    constexpr const T& operator[](size_t i) const { return e_[i]; }
    T& e(size_t i) { return e_[i]; }
    constexpr const T& e(size_t i) const { return e_[i]; }

    // Named element access through x(), y(), z(), w() functions, enable_if is used to disable these
    // accessor functions for vectors with too few elements.
    T& x() { return e_[0]; }
    constexpr T x() const { return e_[0]; }
    template <size_t M = N, typename = std::enable_if_t<(M > 1)>>
    T& y() {
        return e_[1];
    }
    template <size_t M = N, typename = std::enable_if_t<(M > 1)>>
    constexpr T y() const {
        return e_[1];
    }
    template <size_t M = N, typename = std::enable_if_t<(M > 2)>>
    T& z() {
        return e_[2];
    }
    template <size_t M = N, typename = std::enable_if_t<(M > 2)>>
    constexpr T z() const {
        return e_[2];
    }
    template <size_t M = N, typename = std::enable_if_t<(M > 3)>>
    T& w() {
        return e_[3];
    }
    template <size_t M = N, typename = std::enable_if_t<(M > 3)>>
    constexpr T w() const {
        return e_[3];
    }

    // Common swizzles
    constexpr const auto xy() const { return swizzle<X, Y>(*this); }
    constexpr const auto xz() const { return swizzle<X, Z>(*this); }
    constexpr const auto xyz() const { return swizzle<X, Y, Z>(*this); }
    constexpr const auto yzx() const { return swizzle<Y, Z, X>(*this); }
    constexpr const auto zxy() const { return swizzle<Z, X, Y>(*this); }

    // These begin() and end() functions allow a Vector to be used like a container for element
    // access. Not generally recommended but sometimes useful.
    auto begin() { return std::begin(e_); }
    auto end() { return std::end(e_); }
    auto begin() const { return std::begin(e_); }
    auto end() const { return std::end(e_); }

    // Return a pointer to the raw underlying contiguous element data.
    T* data() { return e_; }
    constexpr const T* data() const { return e_; }

    template <typename U, size_t... Is>
    Vector& operator+=(const Vector<U, N, std::index_sequence<Is...>>& x) {
        eval(e_[Is] += x.e_[Is]...);
        return *this;
    }

    template <typename U, size_t... Is>
    Vector& operator-=(const Vector<U, N, std::index_sequence<Is...>>& x) {
        eval(e_[Is] -= x.e_[Is]...);
        return *this;
    }

    template <typename U>
    Vector& operator*=(U x) {
        return multiplyEquals(x, IS{});
    }

    template <typename U>
    Vector& operator/=(U x) {
        return divideEquals(x, IS{}, tag<U>{});
    }

    // Vector operators
    // It would be a bit cleaner to make all operators free functions but to minimize inline depth
    // in debug builds we want to directly access elements via e_[N] and declaring binary operators
    // in such a way that they are friends of both the left and right hand side vectors gets a bit
    // tricky so the most common binary operators are made members.
    template <typename U, size_t... Is>
    constexpr bool operator==(const Vector<U, N, std::index_sequence<Is...>>& x) const {
        return fold(std::logical_and<>{}, true, Vector<bool, N>{e_[Is] == x.e_[Is]...});
    }

    template <typename U, size_t N>
    constexpr bool operator!=(const Vector<U, N>& x) const {
        return !(*this == x);
    }

    constexpr auto operator+() const { return *this; }

    template <typename U, size_t... Is>
    constexpr auto operator+(const Vector<U, N, std::index_sequence<Is...>>& x) const {
        return Vector<decltype(e_[0] + x.e_[0]), N>{e_[Is] + x.e_[Is]...};
    }

    template<size_t... Is>
    friend constexpr auto operator-(const Vector<T, N, std::index_sequence<Is...>>& x) {
        return Vector<T, N>{-x.e_[Is]...};
    }

    template <typename U, size_t... Is>
    constexpr auto operator-(const Vector<U, N, std::index_sequence<Is...>>& x) const {
        return Vector<decltype(e_[0] - x.e_[0]), N>{e_[Is] - x.e_[Is]...};
    }

    // Multiply elements of Vectors x and y memberwise
    // e.g. memberwiseMultiply(Vec3f x, Vec3f y) == Vec3f{x.x * y.x, x.y * y.y, x.z * y.z}
    template <typename U, size_t... Is>
    constexpr auto memberwiseMultiply(const Vector<U, N, std::index_sequence<Is...>>& x) const {
        return Vector<decltype(e_[0] * x.e_[0]), N>{e_[Is] * x.e_[Is]...};
    }

    // Apply a function F(T, U) memberwise to elements of Vector x with fixed bound arg U y and
    // return a Vector of the results.
    // e.g. memberwiseBoundArg(op*, Vex3f x, float y) == Vec3f{x.x*y, x.y*y, x.z*y}
    template <typename F, typename U, size_t... Is>
    friend constexpr auto memberwiseBoundArg(F&& f,
                                             const Vector<T, N, std::index_sequence<Is...>>& x,
                                             U&& y) {
        return Vector<decltype(f(x.e_[0], y)), N>{f(x.e_[Is], y)...};
    }

    template<typename U>
    friend constexpr auto dot(const Vector& x, const Vector<U, N>& y) {
        return x.dotImpl(y);
    }

    // Swizzle members of Vector: call with swizzle<X, Y, Z, W>(v) where the order of X, Y, Z, W
    // determines the swizzle pattern. Output Vector dimension is determined by the number of swizzle
    // constants,
    // e.g. result of swizzle<X, Y>(Vec4f) is a Vec2f
    // Special case for a single swizzle constant: return value is scalar T rather than Vector<T, 1>,
    // e.g. result of swizzle<X>(Vec4f) is a float not a Vector<float, 1>
    template <size_t... Is>
    friend constexpr auto swizzle(const Vector& x) {
        static_assert(N > detail::Max<Is...>::value,
                      "All swizzle args must be <= Vector dimension.");
        using ReturnType = std::conditional_t<sizeof...(Is) == 1, T, Vector<T, sizeof...(Is)>>;
        return ReturnType{x.e_[Is]...};
    }

protected:
    T e_[N];

private:
    // Helper constructors
    // From a single value of type T
    template <size_t... Is>
    explicit constexpr Vector(const T& x, std::index_sequence<Is...>) : e_{((void)Is, x)...} {}
    // From a const T* of N contiguous elements
    template <size_t... Is>
    explicit constexpr Vector(const T* ts, std::index_sequence<Is...>) : e_{ts[Is]...} {}

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

    // manually handle dot for 1 <= N <= 4 to reduce inlining depth for this common operation
    template<typename U, size_t M>
    constexpr auto dotImpl(const Vector<U, M>& x) const {
        return fold(std::plus<>{}, decltype(e_[0] * x.e_[0])(0), memberwiseMultiply(x));
    }

    template<typename U>
    constexpr auto dotImpl(const Vector<U, 1>& x) const {
        return e_[0] * x.e_[0]
    }

    template<typename U>
    constexpr auto dotImpl(const Vector<U, 2>& x) const {
        return e_[0] * x.e_[0] + e_[1] * x.e_[1];
    }

    template<typename U>
    constexpr auto dotImpl(const Vector<U, 3>& x) const {
        return e_[0] * x.e_[0] + e_[1] * x.e_[1] + e_[2] * x.e_[2];
    }

    template<typename U>
    constexpr auto dotImpl(const Vector<U, 4>& x) const {
        return e_[0] * x.e_[0] + e_[1] * x.e_[1] + e_[2] * x.e_[2] + e_[3] * x.e_[3];
    }
};

using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;

using Vec2i = Vector<int, 2>;
using Vec3i = Vector<int, 3>;
using Vec4i = Vector<int, 4>;

// Useful type traits for working with Vectors
template<typename T>
struct IsVector : std::false_type {};

template<typename T, size_t N>
struct IsVector<Vector<T, N, std::make_index_sequence<N>>> : std::true_type {};

template<typename T>
struct VectorDimension;

template <typename T, size_t N>
struct VectorDimension<Vector<T, N, std::make_index_sequence<N>>>
    : std::integral_constant<size_t, N> {};

template<typename T>
struct VectorElementType;

template <typename T, size_t N>
struct VectorElementType<Vector<T, N>> {
    using type = T;
};

template<typename T>
using VectorElementType_t = typename VectorElementType<T>::type;

// Implementation helpers for operators and free functions, not part of public API
namespace detail {

// MNTODO: replace this fold machinery with C++17 fold expressions once available

template <typename F, typename T>
constexpr auto foldImpl(F&& f, T&& t) {
    return t;
}

template <typename F, typename T, typename... Ts>
constexpr auto foldImpl(F&& f, T&& t, Ts&&... ts) {
    return f(std::forward<T>(t), foldImpl(std::forward<F>(f), std::forward<Ts>(ts)...));
}

template <typename T, size_t N, size_t... Js>
constexpr auto basisVectorImpl(size_t i, std::index_sequence<Js...>) {
    return Vector<T, N>{T(i == Js)...};
}

template <typename T, typename U, size_t N, size_t... Is>
constexpr auto divide(const Vector<T, N, std::index_sequence<Is...>>& a, U s, integral_tag) {
    return memberwiseBoundArg(std::divides<>{}, a, s);
}

template <typename T, typename U, size_t N, size_t... Is>
constexpr auto divide(const Vector<T, N, std::index_sequence<Is...>>& a, U s, floating_point_tag) {
    return a * (T{1} / s);
}

template <size_t I, size_t J>
struct MaxImpl : public std::integral_constant<size_t, (J > I ? J : I)> {};

template <size_t... Is>
struct Max;

template <size_t I>
struct Max<I> : public std::integral_constant<size_t, I> {};

template <size_t I, size_t... Is>
struct Max<I, Is...> : public detail::MaxImpl<I, Max<Is...>::value> {};

}  // namespace detail

// Returns a vector of all zeros
// e.g. zeroVector<float, 3>() == Vec3f{0.0f, 0.0f, 0.0f}
// e.g. zeroVector<Vec3f>() = Vec3f{0.0f, 0.0f, 0.0f};
template <typename T, size_t N>
constexpr auto zeroVector() {
    return Vector<T, N>{};
}

template <typename V>
constexpr auto zeroVector() {
    return Vector<VectorElementType_t<V>, VectorDimension<V>::value>{};
}

// Returns a basis vector with 1 in the specified position and 0 elsewhere,
// e.g. basisVector<float, 3>(2) == Vec3f{0.0f, 1.0f, 0.0f}
// e.g. basisVector<Vec3f>(Y) == Vec3f{0.0f, 1.0f, 0.0f}
template <typename T, size_t N>
constexpr auto basisVector(size_t i) {
    return detail::basisVectorImpl<T, N>(i, std::make_index_sequence<N>{});
}

template <typename V>
constexpr auto basisVector(size_t i) {
    return basisVector<VectorElementType_t<V>, VectorDimension<V>::value>(i);
}

// Fold a function F(T, U) over elements of Vector<U, N> v.
// e.g. fold(op+, 0.0f, Vec3f x) == float{x.x + x.y + x.z}
template <typename F, typename T, typename U, size_t N, size_t... Is>
constexpr auto fold(F f, T t, const Vector<U, N, std::index_sequence<Is...>>& v) {
    return detail::foldImpl(f, t, v[Is]...);
}

// free function operators and vector specific functions (dot() etc.)
template <typename T, typename U, size_t N>
constexpr auto operator*(const Vector<T, N>& a, U s) {
    return memberwiseBoundArg(std::multiplies<>{}, a, s);
}

template <typename T, typename U, size_t N>
constexpr auto operator*(T s, const Vector<U, N>& a) {
    return a * s;
}

template <typename T, typename U, size_t N>
constexpr auto operator/(const Vector<T, N>& a, U s) {
    return detail::divide(a, s, tag<U>{});
}

template <typename T, typename U, size_t N>
constexpr auto operator|(const Vector<T, N>& a, const Vector<U, N>& b) {
    return dot(a, b);
}

template <typename T, size_t N>
constexpr T magnitude(const Vector<T, N>& a) {
    return sqrt(dot(a, a));
}

template <typename T, size_t N>
constexpr auto normalize(const Vector<T, N>& a) {
    return a * (T{1} / magnitude(a));
}

template <typename T, size_t N, size_t... Is>
constexpr auto min(const Vector<T, N, std::index_sequence<Is...>>& x, const Vector<T, N>& y) {
    return Vector<T, N>{std::min(x[Is], y[Is])...};
}

template <typename T, size_t N, size_t... Is>
constexpr auto max(const Vector<T, N, std::index_sequence<Is...>>& x, const Vector<T, N>& y) {
    return Vector<T, N>{std::max(x[Is], y[Is])...};
}

template <typename T, size_t N, size_t... Is>
constexpr auto minElement(const Vector<T, N, std::index_sequence<Is...>>& x) {
    return std::min({x[Is]...});
}

template <typename T, size_t N, size_t... Is>
constexpr auto maxElement(const Vector<T, N, std::index_sequence<Is...>>& x) {
    return std::max({x[Is]...});
}

template <typename T, size_t N, size_t... Is>
constexpr auto abs(const Vector<T, N, std::index_sequence<Is...>>& x) {
    return Vector<T, N>{std::abs(x[Is])...};
}

template <typename T, size_t N, size_t... Is>
constexpr auto saturate(const Vector<T, N, std::index_sequence<Is...>>& x) {
    return Vector<T, N>{saturate(x[Is])...};
}

template <typename T, size_t N, size_t... Is>
constexpr auto clamp(const Vector<T, N, std::index_sequence<Is...>>& x, const Vector<T, N>& a,
                     const Vector<T, N>& b) {
    return Vector<T, N>{clamp(x[Is], a[Is], b[Is])...};
}

template <typename T, size_t N, size_t... Is>
constexpr auto clamp(const Vector<T, N, std::index_sequence<Is...>>& x, const T& a, const T& b) {
    return Vector<T, N>{clamp(x[Is], a, b)...};
}

template <typename T, typename U>
constexpr auto cross(const Vector<T, 3>& a, const Vector<U, 3>& b) {
    return a.yzx().memberwiseMultiply(b.zxy()) - a.zxy().memberwiseMultiply(b.yzx());
}

}  // namespace mathlib
