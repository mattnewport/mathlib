#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "mathfuncs.h"

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

namespace detail {

template <typename T, size_t N, size_t... Is>
struct ValArray {
    T e[N];

    constexpr ValArray& operator+=(const ValArray &x) noexcept {
        ((e[Is] += x.e[Is]), ...);
        return *this;
    }
    constexpr ValArray& operator-=(const ValArray &x) noexcept {
        ((e[Is] -= x.e[Is]), ...);
        return *this;
    }
    constexpr ValArray& operator*=(const T &x) noexcept {
        ((e[Is] *= x), ...);
        return *this;
    }
    constexpr ValArray& operator/=(const T &x) noexcept {
        ((e[Is] /= x), ...);
        return *this;
    }

    constexpr ValArray operator-() const noexcept { return ValArray{-e[Is]...}; }

    constexpr bool operator==(const ValArray& x) const noexcept {
        return (... && (e[Is] == x.e[Is]));
    }

    template <typename F>
    friend constexpr auto map(F f, const ValArray& x) noexcept {
        return ValArray{f(x.e[Is])...};
    }

    template <typename F>
    friend constexpr auto map(F f, const ValArray& x, const ValArray& y) noexcept {
        return ValArray{f(x.e[Is], y.e[Is])...};
    }

    /*
    template <typename F>
    friend constexpr auto map(F f, const ValArray& x, const ValArray& y, const ValArray& z) noexcept
    { return ValArray{ f(x.e[Is], y.e[Is], z.e[Is])... };
    }
    */

    template <typename F>
    constexpr auto fold(F f, T&& t) const noexcept {
        return foldImpl(f, t, e[Is]...);
    }

    constexpr ValArray memberwiseMultiply(const ValArray& x) const noexcept {
        return ValArray{e[Is] * x.e[Is]...};
    }

    constexpr auto dot(const ValArray& x) const noexcept {
        const auto mm = ValArray{ e[Is] * x.e[Is]... };
        return (... + (mm.e[Is]));
    }

private:
    template <typename F>
    static constexpr auto foldImpl(F&& f, T&& t) noexcept {
        return t;
    }

    template <typename F, typename... Ts>
    static constexpr auto foldImpl(F&& f, T&& t, Ts&&... ts) noexcept {
        return f(std::forward<T>(t), foldImpl(std::forward<F>(f), std::forward<Ts>(ts)...));
    }
};

template <typename T, size_t N>
struct MakeValArray {
private:
    template <size_t... Is>
    static constexpr auto make(std::index_sequence<Is...>) {
        return ValArray<T, N, Is...>{};
    }

public:
    using type = decltype(make(std::make_index_sequence<N>{}));
};

template <typename T, size_t N>
using MakeValArray_t = typename MakeValArray<T, N>::type;

}  // namespace detail

enum VectorComponents { X = 0, Y = 1, Z = 2, W = 3 };

template <typename T>
struct ScalarType {
    using type = T;
};

template <typename T>
using ScalarType_t = typename ScalarType<T>::type;

// IS defaulted template argument is an implementation trick to make it possible to write operators
// that can directly deduce an index sequence and therefore be implemented without calling a helper
// function. This both reduces the amount of boilerplate code that needs to be written and reduces
// inlining depth which is particulary helpful for performance in debug builds if inlining is not
// enabled.
template <typename T, size_t N>
class Vector {
    using data_t = detail::MakeValArray_t<T, N>;
    using IS = std::make_index_sequence<N>;

    // This helper should not be necessary once we have C++17 fold expressions
    template <typename... Ts>
    static constexpr void eval(Ts&&...) {}

    data_t es;

    constexpr Vector(const data_t& es_) : es{es_} {}

public:
    constexpr Vector() noexcept = default;
    constexpr Vector(const Vector&) noexcept = default;

    // Standard constructor taking a sequence of exactly N objects convertible to T.
    // Question: do we want to allow narrowing conversions?
    template <typename... Ts, typename = std::enable_if_t<(sizeof...(Ts) == N)>>
    constexpr Vector(Ts&&... ts) noexcept : es{{ts...}} {}

    // For convenience we have an explicit constructor taking a single argument that sets all
    // members of the vector to the value of that argument.
    constexpr explicit Vector(const T& x) noexcept : Vector{x, IS{}} {}
    // Templated explicit conversion constructor from a Vector<U, N>
    template <typename U>
    constexpr explicit Vector(const Vector<U, N>& x) noexcept : Vector{x, IS{}} {}
    // Templated explicit conversion constructor from a Vector<T, M>, M > N: take first N elements
    template <size_t M, typename = std::enable_if_t<(M > N)>>
    explicit Vector(const Vector<T, M>& x) noexcept : Vector{x, IS{}} {}
    // Templated explicit conversion constructor from a Vector<U, N-1> and a scalar V
    template <size_t... Is>
    constexpr explicit Vector(const Vector<T, N - 1>& x, const T& s) noexcept : Vector{x} {
        es.e[N - 1] = s;
    }

    // Explicit constructor from a pointer to N Ts
    explicit Vector(const T* p) noexcept : Vector{p, IS{}} {}

    // Generic element access
    constexpr auto operator[](size_t i) noexcept { return es.e[i]; }
    constexpr auto e(size_t i) noexcept { return es.e[i]; }
    constexpr auto operator[](size_t i) const noexcept { return es.e[i]; }
    constexpr auto e(size_t i) const noexcept { return es.e[i]; }

    // Named element access through x(), y(), z(), w() functions, enable_if is used to disable these
    // accessor functions for vectors with too few elements.
    T& x() noexcept { return es.e[0]; }
    constexpr T x() const noexcept { return es.e[0]; }
    template <size_t M = N, typename = std::enable_if_t<(M > 1)>>
    T& y() noexcept {
        return es.e[1];
    }
    template <size_t M = N, typename = std::enable_if_t<(M > 1)>>
    constexpr T y() const noexcept {
        return es.e[1];
    }
    template <size_t M = N, typename = std::enable_if_t<(M > 2)>>
    T& z() noexcept {
        return es.e[2];
    }
    template <size_t M = N, typename = std::enable_if_t<(M > 2)>>
    constexpr T z() const noexcept {
        return es.e[2];
    }
    template <size_t M = N, typename = std::enable_if_t<(M > 3)>>
    T& w() noexcept {
        return es.e[3];
    }
    template <size_t M = N, typename = std::enable_if_t<(M > 3)>>
    constexpr T w() const noexcept {
        return es.e[3];
    }

    // Common swizzles
    constexpr const auto xy() const noexcept { return swizzle<X, Y>(*this); }
    constexpr const auto xz() const noexcept { return swizzle<X, Z>(*this); }
    constexpr const auto xyz() const noexcept { return swizzle<X, Y, Z>(*this); }
    constexpr const auto yzx() const noexcept { return swizzle<Y, Z, X>(*this); }
    constexpr const auto zxy() const noexcept { return swizzle<Z, X, Y>(*this); }

    // These begin() and end() functions allow a Vector to be used like a container for element
    // access. Not generally recommended but sometimes useful.
    constexpr T* begin() noexcept { return std::begin(es.e); }
    constexpr T* end() noexcept { return std::end(es.e); }
    constexpr const T* begin() const noexcept { return std::cbegin(es.e); }
    constexpr const T* end() const noexcept { return std::cend(es.e); }

    // Return a pointer to the raw underlying contiguous element data.
    constexpr T* data() noexcept { return es.e; }
    constexpr const T* data() const noexcept { return es.e; }

    // @=() op assignment operators
    constexpr Vector& operator+=(const Vector& x) noexcept {
        es += x.es;
        return *this;
    }

    constexpr Vector& operator-=(const Vector& x) noexcept {
        es -= x.es;
        return *this;
    }

    constexpr Vector& operator*=(const ScalarType_t<T>& s) noexcept {
        es *= s;
        return *this;
    }

    constexpr Vector& operator/=(const ScalarType_t<T>& s) noexcept {
        if constexpr (std::is_floating_point<ScalarType_t<T>>::value) {
            auto m = T{1} / s;
            es *= m;
        } else {
            es /= s;
        }
        return *this;
    }

    constexpr auto operator+() const noexcept { return *this; }

    constexpr auto operator-() const noexcept { return Vector{-es}; }

    constexpr auto operator+(const Vector& x) const noexcept {
        auto res{*this};
        return res += x;
    }

    constexpr auto operator-(const Vector& x) const noexcept {
        auto res{*this};
        return res -= x;
    }

    friend constexpr auto operator*(Vector x, const ScalarType_t<T>& s) noexcept {
        return x *= s;
    }

    friend constexpr auto operator/(Vector x, const ScalarType_t<T>& s) noexcept {
        return x /= s;
    }

    friend constexpr auto operator*(const T& s, const Vector& x) noexcept { return x * s; }

    // Multiply elements of Vectors x and y memberwise
    // e.g. memberwiseMultiply(Vec3f x, Vec3f y) == Vec3f{x.x * y.x, x.y * y.y, x.z * y.z}
    constexpr auto memberwiseMultiply(const Vector& x) const noexcept {
        return Vector{es.memberwiseMultiply(x.es)};
    }

    friend constexpr auto dot(const Vector& x, const Vector& y) noexcept {
        return x.es.dot(y.es);
    }

    // Swizzle members of Vector: call with swizzle<X, Y, Z, W>(v) where the order of X, Y, Z, W
    // determines the swizzle pattern. Output Vector dimension is determined by the number of
    // swizzle constants, e.g. result of swizzle<X, Y>(Vec4f) is a Vec2f Special case for a single
    // swizzle constant: return value is scalar T rather than Vector<T, 1>, e.g. result of
    // swizzle<X>(Vec4f) is a float not a Vector<float, 1>
    template <size_t... Is>
    friend constexpr auto swizzle(const Vector& x) noexcept {
        static_assert(N > detail::Max<Is...>{}, "All swizzle args must be <= Vector dimension.");
        using ReturnType = std::conditional_t<sizeof...(Is) == 1, T, Vector<T, sizeof...(Is)>>;
        return ReturnType{x.es.e[Is]...};
    }

    template <typename F>
    friend constexpr auto vector_map(F f, const Vector& x) noexcept {
        return Vector{mathlib::detail::map(f, x.es)};
    }

    friend constexpr auto abs(const Vector& x) noexcept {
        return Vector{map(mathlib::abs<T>, x.es)};
    }

    friend constexpr auto min(const Vector& x, const Vector& y) noexcept {
        return Vector{ map([](const T& x, const T& y) { return std::min<T>(x, y); }, x.es, y.es) };
    }

    friend constexpr auto max(const Vector& x, const Vector& y) noexcept {
        return Vector{map([](const T& x, const T& y) { return std::max<T>(x, y); }, x.es, y.es)};
    }

    friend constexpr auto saturate(const Vector& x) noexcept {
        return Vector{ map(saturate<T>, x.es) };
    }

    friend constexpr auto clamp(const Vector<T, N>& x, const T& a, const T& b) noexcept {
        return Vector{ map([=](const T& y) { return clamp(y, a, b); }, x.es) };
    }

    friend constexpr bool operator==(const Vector& x, const Vector& y) noexcept {
        return x.es == y.es;
    }

    friend constexpr bool operator!=(const Vector& x, const Vector& y) noexcept { return !(x.es == y.es); }

private:
    // Helper constructors
    // From a single value of type T
    template <size_t... Is>
    explicit constexpr Vector(const T& x, std::index_sequence<Is...>) noexcept
        : es{((void)Is, x)...} {}
    // From a const T* of N contiguous elements
    template <size_t... Is>
    explicit constexpr Vector(const T* ts, std::index_sequence<Is...>) noexcept : es{{ts[Is]...}} {}
    // From a Vector<U, N>
    template <typename U, size_t... Is>
    explicit constexpr Vector(const Vector<U, N>& x, std::index_sequence<Is...>) noexcept
        : es{ {T(x[Is])...} } {}
};

using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;

using Vec2i = Vector<int, 2>;
using Vec3i = Vector<int, 3>;
using Vec4i = Vector<int, 4>;

// Useful type traits for working with Vectors
template <typename T>
struct IsVector : std::false_type {};

template <typename T, size_t N>
struct IsVector<Vector<T, N>> : std::true_type {};

template <typename T>
struct VectorDimension;

template <typename T, size_t N>
struct VectorDimension<Vector<T, N>> : std::integral_constant<size_t, N> {};

template <typename T>
struct VectorElementType;

template <typename T, size_t N>
struct VectorElementType<Vector<T, N>> {
    using type = T;
};

template <typename T>
using VectorElementType_t = typename VectorElementType<T>::type;

template <typename T, size_t N>
struct ScalarType<Vector<T, N>> {
    using type = typename ScalarType_t<T>;
};

// Implementation helpers for operators and free functions, not part of public API
namespace detail {
// MNTODO: replace this fold machinery with C++17 fold expressions once available

template <typename F, typename T>
constexpr auto foldImpl(F&& f, T&& t) noexcept {
    return t;
}

template <typename F, typename T, typename... Ts>
constexpr auto foldImpl(F&& f, T&& t, Ts&&... ts) noexcept {
    return f(std::forward<T>(t), foldImpl(std::forward<F>(f), std::forward<Ts>(ts)...));
}

template <typename T, size_t N, size_t... Js>
constexpr auto basisVectorImpl(size_t i, std::index_sequence<Js...>) noexcept {
    return Vector<T, N>{T(i == Js)...};
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
constexpr auto zeroVector() noexcept {
    return Vector<T, N>{};
}

template <typename V>
constexpr auto zeroVector() noexcept {
    return Vector<VectorElementType_t<V>, VectorDimension<V>::value>{};
}

// Returns a basis vector with 1 in the specified position and 0 elsewhere,
// e.g. basisVector<float, 3>(2) == Vec3f{0.0f, 1.0f, 0.0f}
// e.g. basisVector<Vec3f>(Y) == Vec3f{0.0f, 1.0f, 0.0f}
template <typename T, size_t N>
constexpr auto basisVector(size_t i) noexcept {
    return detail::basisVectorImpl<T, N>(i, std::make_index_sequence<N>{});
}

template <typename V>
constexpr auto basisVector(size_t i) noexcept {
    return basisVector<VectorElementType_t<V>, VectorDimension<V>::value>(i);
}

// Fold a function F(T, U) over elements of Vector<U, N> v.
// e.g. fold(op+, 0.0f, Vec3f x) == float{x.x + x.y + x.z}
template <typename F, typename T, size_t N>
constexpr auto fold(F f, T t, const Vector<T, N>& v) noexcept {
    return v.fold(f, t);
}

// free function operators and vector specific functions (dot() etc.)
template <typename T, typename U, size_t N>
constexpr auto memberwiseMultiply(const Vector<T, N>& x, const Vector<U, N>& y) noexcept {
    return x.memberwiseMultiply(y);
}

template <typename T, size_t N>
constexpr T magnitude(const Vector<T, N>& a) noexcept {
    return sqrt(dot(a, a));
}

template <typename T, size_t N>
constexpr auto normalize(const Vector<T, N>& a) noexcept {
    return a * (T{1} / magnitude(a));
}

template <typename T, typename U>
constexpr auto cross(const Vector<T, 3>& a, const Vector<U, 3>& b) noexcept {
    return a.yzx().memberwiseMultiply(b.zxy()) - a.zxy().memberwiseMultiply(b.yzx());
}

}  // namespace mathlib
