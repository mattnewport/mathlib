#pragma once

#include <algorithm>
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

// See this great blog post by Nathan Reed for some discussion of design decisions around vector
// math libraries. He independently reaches many of the same design decisions that have been made
// for this library: http://www.reedbeta.com/blog/2013/12/28/on-vector-math-libraries/

namespace mathlib {

struct integral_tag {};
struct floating_point_tag {};
template <typename T>
using tag = std::conditional_t<std::is_floating_point<T>::value, floating_point_tag, integral_tag>;

template <typename T, size_t N>
class Vector {
public:
    using is = decltype(std::make_index_sequence<N>{});

    constexpr Vector() = default;
    constexpr Vector(const Vector&) = default;

    // Standard constructor taking a sequence of exactly N objects convertible to T.
    // Question: do we want to allow narrowing conversions? Currently allowed but maybe a bad idea?
    template <typename... Us, typename = std::enable_if_t<(sizeof...(Us) == N - 2)>>
    constexpr Vector(const T& x, const T& y, Us&&... us) : e_{x, y, T(us)...} {}

    // MNTODO: can't make these constructors constexpr due to a bug in VS2015:
    // http://stackoverflow.com/questions/32489702/constexpr-with-delegating-constructors
    
    // For convenience we have an explicit constructor taking a single argument that sets all
    // members of the vector to the value of that argument.
    explicit Vector(const T& x) : Vector{x, is{}} {}
    // Templated conversion constructor from a Vector<U, N>
    // Question: do we want to allow narrowing conversions? Currently allowed but maybe a bad idea?
    template <typename U>
    Vector(const Vector<U, N>& x) : Vector{x, is{}} {}
    // Templated conversion constructor from a Vector<U, N-1> and a scalar V
    // Question: do we want to allow narrowing conversions? Currently allowed but maybe a bad idea?
    template <typename U, typename V>
    explicit Vector(const Vector<U, N - 1>& x, V&& s)
        : Vector{x, std::forward<V>(s), std::make_index_sequence<N - 1>{}} {}

    // Construct from a pointer to N Ts
    explicit Vector(const T* p) : Vector{p, is{}} {}

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
    const T* data() const { return e_; }

    template <typename U>
    Vector& operator+=(const Vector<U, N>& x) {
        return plusEquals(x, is{});
    }

    template <typename U>
    Vector& operator-=(const Vector<U, N>& x) {
        return minusEquals(x, is{});
    }

    template <typename U>
    Vector& operator*=(U x) {
        return multiplyEquals(x, is{});
    }

    template <typename U>
    Vector& operator/=(U x) {
        return divideEquals(x, is{}, tag<U>{});
    }

private:
    T e_[N];

    template <typename U, size_t M>
    friend class Vector;

    // Helper constructors
    // From a single value of type T
    template <size_t... Is>
    explicit constexpr Vector(const T& x, std::index_sequence<Is...>) : e_{((void)Is, x)...} {}
    // From a Vector<U, N> where U is convertible to T
    template <typename U, size_t... Is>
    constexpr Vector(const Vector<U, N>& x, std::index_sequence<Is...>) : e_{T(x.e_[Is])...} {}
    // From a Vector<U, N-1> and a scalar V where U and V are convertible to T
    template <typename U, typename V, size_t... Is>
    explicit constexpr Vector(const Vector<U, N - 1>& x, V s, std::index_sequence<Is...>)
        : e_{T(x.e_[Is])..., T(s)} {}
    // From a const T* of N contiguous elements
    template <size_t... Is>
    explicit Vector(const T* ts, std::index_sequence<Is...>) : e_{ts[Is]...} {}

    template <typename... Ts>
    static constexpr void eval(Ts&&...) {}

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

// Implementation helpers for operators and free functions, not part of public API
namespace detail {

template <size_t I, typename F, typename... Us>
constexpr auto memberApply(F&& f, Us&&... us) {
    return f(us[I]...);
}

// Implementation helpers for memberwise and memberwiseBoundArg
template <typename F, size_t... Is, typename... Us>
constexpr auto memberwiseImpl(F f, std::index_sequence<Is...>, Us&&... us) {
    using T = decltype(memberApply<0>(f, us...));
    return Vector<T, sizeof...(Is)>{memberApply<Is>(f, std::forward<Us>(us)...)...};
}

template <typename F, size_t... Is, typename T, typename U>
constexpr auto memberwiseBoundArgImpl(F f, std::index_sequence<Is...>,
                                    const Vector<T, sizeof...(Is)>& x, U&& y) {
    return Vector<decltype(f(x[0], y)), sizeof...(Is)>{f(x[Is], y)...};
}

// MNTODO: replace this fold machinery with C++17 fold expressions once available
// Manually expanded for first few arguments to reduce inlining depth.
template <typename F, typename T>
constexpr auto foldImpl(F, T t) {
    return t;
}

template <typename F, typename T>
constexpr auto foldImpl(F f, T x, T y) {
    return f(x, y);
}

template <typename F, typename T, typename... Ts>
constexpr auto foldImpl(F f, T x, T y, Ts... ts) {
    return f(x, foldImpl(f, y, ts...));
}

template <typename F, typename T, size_t N, size_t... Is>
constexpr auto fold(F f, const Vector<T, N>& v, std::index_sequence<Is...>) {
    return foldImpl(f, v[Is]...);
}

template <typename T, size_t N, size_t... Js>
constexpr auto basisVectorImpl(size_t i, std::index_sequence<Js...>) {
    return Vector<T, N>{T(i == Js)...};
}

template <typename T, typename U, size_t N>
constexpr auto divide(const Vector<T, N>& a, U s, integral_tag) {
    return memberwiseBoundArg(std::divides<>{}, a, s);
}

template <typename T, typename U, size_t N>
constexpr auto divide(const Vector<T, N>& a, U s, floating_point_tag) {
    return a * (T{1} / s);
}

template <typename T, size_t N, size_t... Is>
constexpr auto asTuple(const Vector<T, N>& a, std::index_sequence<Is...>) {
    return std::make_tuple(a[Is]...);
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

// Swizzle members of Vector: call with swizzle<X, Y, Z, W>(v) where the order of X, Y, Z, W
// determines the swizzle pattern. Output Vector dimension is determined by the number of swizzle
// constants,
// e.g. result of swizzle<X, Y>(Vec4f) is a Vec2f
// Special case for a single swizzle constant: return value is scalar T rather than Vector<T, 1>,
// e.g. result of swizzle<X>(Vec4f) is a float not a Vector<float, 1>
enum SwizzleConstants { X = 0, Y = 1, Z = 2, W = 3 };

template <size_t... Is, typename V>
constexpr auto swizzle(const V& x) {
    static_assert(IsVector<V>::value, "Argument to swizzle() must be a Vector.");
    static_assert(VectorDimension<V>::value > detail::Max<Is...>::value,
                  "All swizzle args must be <= Vector dimension.");
    using T = VectorElementType_t<V>;
    using ReturnType = std::conditional_t<sizeof...(Is) == 1, T, Vector<T, sizeof...(Is)>>;
    return ReturnType{x[Is]...};
}

// Returns a vector of all zeros
// e.g. zeroVector<float, 3>() == Vec3f{0.0f, 0.0f, 0.0f}
template <typename T, size_t N>
constexpr auto zeroVector() {
    return Vector<T, N>{T(0)};
}

// Returns a basis vector with 1 in the specified position and 0 elsewhere,
// e.g. basisVector<float, 3>(2) == Vec3f{0.0f, 1.0f, 0.0f}
template <typename T, size_t N>
constexpr auto basisVector(size_t i) {
    return detail::basisVectorImpl<T, N>(i, std::make_index_sequence<N>{});
}

template<typename V>
constexpr auto basisVector(size_t i) {
    return basisVector<VectorElementType_t<V>, VectorDimension<V>::value>(i);
}

// Takes a function f that takes one or more arguments of type Vector<_, N> and calls it memberwise
// on elements of the arguments, returning the memberwise result.
// e.g. memberwise(op+, Vec3f x, Vec3f y) == Vec3f{x.x + y.x, x.y + y.y, x.z + y.z}
template <typename F, typename... Us>
constexpr auto memberwise(F&& f, Us&&... us) {
    using Vector = std::common_type_t<Us...>;
    static_assert(IsVector<Vector>::value, "memberwise(f, us...): us... should all be Vectors.");
    return detail::memberwiseImpl(std::forward<F>(f), Vector::is(), std::forward<Us>(us)...);
}

// Apply a function F(T, U) memberwise to elements of Vector x with fixed bound arg U y and return a
// Vector of the results. This could be implemented in terms of memberwise with a lambda or
// std::bind but can't use std::bind or lambdas in a constexpr function.
// e.g. memberwiseBoundArg(op*, Vex3f x, float y) == Vec3f{x.x*y, x.y*y, x.z*y}
template <typename F, typename T, typename U, size_t N>
constexpr auto memberwiseBoundArg(F&& f, const Vector<T, N>& x, U&& y) {
    return detail::memberwiseBoundArgImpl(std::forward<F>(f), Vector<T, N>::is{}, x,
                                          std::forward<U>(y));
}

// Fold a function F(T, T) over elements of Vector v.
// e.g. fold(op+, Vec3f x) == float{x.x + x.y + x.z}
template <typename F, typename T, size_t N>
constexpr auto fold(F f, const Vector<T, N>& v) {
    return detail::fold(f, v, Vector<T, N>::is{});
}

// Returns a tuple of the elements of Vector a, used to implement operators but also potentially
// useful on its own.
template <typename T, size_t N>
constexpr auto asTuple(const Vector<T, N>& a) {
    return detail::asTuple(a, Vector<T, N>::is{});
}

// standard operators and vector specific functions (dot() etc.)
template <typename T, typename U, size_t N>
constexpr bool operator==(const Vector<T, N>& a, const Vector<U, N>& b) {
    return asTuple(a) == asTuple(b);
}

template <typename T, typename U, size_t N>
constexpr bool operator!=(const Vector<T, N>& a, const Vector<U, N>& b) {
    return !(a == b);
}

template <typename T, typename U, size_t N>
constexpr bool operator<(const Vector<T, N>& a, const Vector<U, N>& b) {
    return asTuple(a) < asTuple(b);
}

template<typename T, size_t N>
constexpr auto operator+(const Vector<T, N>& x) {
    return x;
}

template <typename T, typename U, size_t N>
constexpr auto operator+(const Vector<T, N>& a, const Vector<U, N>& b) {
    return memberwise(std::plus<>{}, a, b);
}

template<typename T, size_t N>
constexpr auto operator-(const Vector<T, N>& x) {
    return memberwise(std::negate<>{}, x);
}

template <typename T, typename U, size_t N>
constexpr auto operator-(const Vector<T, N>& a, const Vector<U, N>& b) {
    return memberwise(std::minus<>{}, a, b);
}

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

template<typename T, typename U, size_t N>
constexpr auto memberwiseMultiply(const Vector<T, N>& x, const Vector<U, N>& y) {
    return memberwise(std::multiplies<>{}, x, y);
}

template <typename T, typename U, size_t N>
constexpr auto dot(const Vector<T, N>& a, const Vector<U, N>& b) {
    return fold(std::plus<>{}, memberwiseMultiply(a, b));
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

template <typename T, typename U, size_t N>
constexpr auto min(const Vector<T, N>& x, const Vector<U, N>& y) {
    return memberwise([](auto x, auto y) { return std::min(x, y); }, x, y);
}

template <typename T, typename U, size_t N>
constexpr auto max(const Vector<T, N>& x, const Vector<U, N>& y) {
    return memberwise([](auto x, auto y) { return std::max(x, y); }, x, y);
}

template <typename T, size_t N>
inline auto& minElement(const Vector<T, N>& x) {
    return *std::min_element(std::begin(x), std::end(x));
}

template <typename T, size_t N>
inline auto& maxElement(const Vector<T, N>& x) {
    return *std::max_element(std::begin(x), std::end(x));
}

template <typename T, size_t N>
constexpr auto abs(const Vector<T, N>& x) {
    return memberwise([](auto x) { return std::abs(x); }, x);
}

template <typename T, size_t N>
constexpr auto saturate(const Vector<T, N>& x) {
    return memberwise([](auto x) { return saturate(x); }, x);
}

template <typename T, typename U, size_t N>
constexpr auto clamp(const Vector<T, N>& x, const Vector<U, N>& a, const Vector<U, N>& b) {
    return memberwise([](auto x, auto a, auto b) { return clamp(x, a, b); }, x, a, b);
}

template <typename T, typename U>
constexpr Vector<T, 3> cross(const Vector<T, 3>& a, const Vector<U, 3>& b) {
    return memberwiseMultiply(a.yzx(), b.zxy()) - memberwiseMultiply(a.zxy(), b.yzx());
}

}  // namespace mathlib
