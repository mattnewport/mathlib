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
// fairly inefficient. Using /Ob1 http://msdn.microsoft.com/en-us/library/47238hez.aspx for debug
// builds in Visual Studio will help debug performance a lot.

// See this great blog post by Nathan Reed for some discussion of design decisions around vector
// math libraries. He independently reaches many of the same design decisions that have been made
// for this library: http://www.reedbeta.com/blog/2013/12/28/on-vector-math-libraries/

namespace mathlib {

// Declare some useful helpers for working with Vector<T, N>'s (specialized for Vector<T, N> later)
template <typename T>
struct IsVector : std::false_type {};

template <typename T>
constexpr bool IsVector_v = IsVector<std::decay_t<T>>::value;

template <typename... Ts>
constexpr bool AllVectors_v = std::conjunction_v<IsVector<std::decay_t<Ts>>...>;

template <typename T, typename... Ts>
constexpr bool AllOfType_v = std::conjunction_v<std::is_same<T, std::decay_t<Ts>>...>;

template <typename T>
struct VectorDimension;

template <typename T>
constexpr size_t VectorDimension_v = VectorDimension<T>::value;

template <typename T>
struct VectorElementType;

template <typename T>
using VectorElementType_t = typename VectorElementType<std::decay_t<T>>::type;

template <typename T>
struct ScalarType {
    using type = std::decay_t<T>;
};

template <typename T>
using ScalarType_t = typename ScalarType<std::decay_t<T>>::type;

namespace detail {

// Useful for debugging template errors
template<typename... Ts> struct Debug;

// ValArray exists mainly to take advantage of aggregate initialization without having to define a
// bunch of constructors (ValArray supports lots of ways to initialize, Vector can restrict them to
// ones we want to support via providing mostly simple constructors that forward to ValArray) and to
// provide easy access from any function to the raw variadic indices used in fold expressions and
// pack expansions for direct array indexing. It doesn't try to provide all functions that might be
// useful for an end user, rather it just defines functions that can be used as basic building
// blocks for Vector to provide functions useful to end users.
template <typename Vector, typename T, size_t N, size_t... Is>
struct VectorBase {
    T e[N];

    constexpr const T& operator[](size_t i) const noexcept { return e[i]; }

    // Return a (const) pointer to the raw underlying contiguous element data.
    constexpr const T* data() const noexcept { return e; }

    // Op assignment
    constexpr Vector& operator+=(const Vector& x) noexcept {
        ((e[Is] += x.e[Is]), ...);
        return static_cast<Vector&>(*this);
    }
    constexpr Vector& operator-=(const Vector& x) noexcept {
        ((e[Is] -= x.e[Is]), ...);
        return static_cast<Vector&>(*this);
    }
    constexpr Vector& operator*=(const ScalarType_t<Vector>& x) noexcept {
        ((e[Is] *= x), ...);
        return static_cast<Vector&>(*this);
    }
    constexpr Vector& operator/=(const ScalarType_t<Vector>& x) noexcept {
        ((e[Is] /= x), ...);
        return static_cast<Vector&>(*this);
    }

    // Unary +/-
    constexpr Vector operator-() const noexcept { return Vector{-e[Is]...}; }
    constexpr Vector operator+() const noexcept { return Vector{+e[Is]...}; }

    // Binary +/-
    constexpr Vector operator+(const Vector& x) const noexcept {
        return Vector{(e[Is] + x.e[Is])...};
    }
    constexpr Vector operator-(const Vector& x) const noexcept {
        return Vector{(e[Is] - x.e[Is])...};
    }

    // Binary *//
    constexpr Vector operator*(const ScalarType_t<Vector>& x) const noexcept {
        return Vector{(e[Is] * x)...};
    }
    constexpr Vector operator/(const ScalarType_t<Vector>& x) const noexcept {
        if constexpr (std::is_floating_point_v<ScalarType_t<Vector>>) {
            const auto s = ScalarType_t<Vector>{1} / x;
            return Vector{(e[Is] * s)...};
        }
        return Vector{(e[Is] / x)...};
    }

    // Comparison
    constexpr bool operator==(const Vector& x) const noexcept {
        return (... && (e[Is] == x.e[Is]));
    }
    constexpr bool operator!=(const Vector& x) const noexcept { return !(*this == x); }

    template <typename F>
    constexpr auto mapHelper(F&& f) const noexcept {
        return Vector{f(e[Is])...};
    }

    template <typename F>
    constexpr auto mapHelper(F&& f, const Vector& x) const noexcept {
        return Vector{f(e[Is], x.e[Is])...};
    }

    constexpr Vector memberwiseMultiply(const Vector& x) const noexcept {
        return Vector{e[Is] * x.e[Is]...};
    }

    constexpr auto dot(const Vector& x) const noexcept {
        const auto mm = VectorBase{{e[Is] * x.e[Is]...}};
        return (... + (mm.e[Is]));
    }

    constexpr auto magnitude() const noexcept {
        const auto mm = VectorBase{{e[Is] * e[Is]...}};
        return std::sqrt((... + (mm.e[Is])));
    }

    static constexpr Vector basis(size_t i) noexcept { return Vector{T(i == Is)...}; }
    static constexpr Vector ones() noexcept {
        if constexpr (std::is_same_v<T, ScalarType_t<T>>)
            return Vector{T(((void)Is, 1))...};
        else
            return Vector{((void)Is, T::ones())...};
    }
};

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

}  // namespace detail

enum VectorComponents { X = 0, Y = 1, Z = 2, W = 3 };

template <typename T, size_t N>
class Vector : private detail::MakeVectorBase_t<Vector<T, N>> {
    using base = detail::MakeVectorBase_t<Vector<T, N>>;
    friend base;
    using IS = std::make_index_sequence<N>;

    // Helper constructors
    // From a const T* of N contiguous elements
    template <size_t... Is>
    explicit constexpr Vector(const T* ts, std::index_sequence<Is...>) noexcept
        : base{{ts[Is]...}} {}
    // From a Vector<U, M>
    template <typename U, size_t M, size_t... Is>
    explicit constexpr Vector(const Vector<U, M>& x, std::index_sequence<Is...>) noexcept
        : base{{T(x[Is])...}} {}

    explicit Vector(const base& x) : base{x} {}

public:
    constexpr Vector() noexcept = default;
    constexpr Vector(const Vector&) noexcept = default;
    // Standard constructor taking a sequence of exactly N Ts.
    template <typename... Ts,
              typename = std::enable_if_t<(sizeof...(Ts) == N) && AllOfType_v<T, Ts...>>>
    constexpr Vector(const Ts&... ts) noexcept : base{{ts...}} {}
    // Templated explicit conversion constructor from a Vector<U, N>
    template <typename U>
    constexpr explicit Vector(const Vector<U, N>& x) noexcept : Vector{x, IS{}} {}
    // Templated explicit conversion constructor from a Vector<T, M>, M > N: take first N elements
    template <size_t M, typename = std::enable_if_t<(M > N)>>
    constexpr explicit Vector(const Vector<T, M>& x) noexcept : Vector{x, IS{}} {}
    // Templated explicit conversion constructor from a Vector<T, N-1> and a scalar T s
    constexpr explicit Vector(const Vector<T, N - 1>& x, const T& s) noexcept
        : Vector{x, std::make_index_sequence<N - 1>{}} {
        this->e[N - 1] = s;
    }
    // Explicit constructor from a pointer to N or more Ts
    constexpr explicit Vector(const T* p) noexcept : Vector{p, IS{}} {}

    // Generic element access
    using base::operator[];

    // Data pointer access
    using base::data;

    // Tuple style element access
    template <size_t I>
    constexpr const T& get() const noexcept {
        return this->e[I];
    }

    // Named element access through x(), y(), z(), w() functions.
    constexpr const T& x() const noexcept { return this->e[0]; }
    constexpr const T& y() const noexcept {
        static_assert(N > 1);
        return this->e[1];
    }
    constexpr const T& z() const noexcept {
        static_assert(N > 2);
        return this->e[2];
    }
    constexpr const T& w() const noexcept {
        static_assert(N > 3);
        return this->e[3];
    }

    // Swizzle members of Vector: call with v.swizzled<X, Y, Z, W> where the order of X, Y, Z, W
    // determines the swizzle pattern. Output Vector dimension is determined by the number of
    // swizzle constants, e.g. result of swizzle<X, Y>(Vec4f) is a Vec2f. Special case for a single
    // swizzle constant: return value is scalar T rather than Vector<T, 1>, e.g. result of
    // swizzle<X>(Vec4f) is a float not a Vector<float, 1>
    template <size_t... Is>
    constexpr auto swizzled() const noexcept {
        using ReturnType = std::conditional_t<sizeof...(Is) == 1, T, Vector<T, sizeof...(Is)>>;
        return ReturnType{this->e[Is]...};
    }

    // Common swizzles
    constexpr auto xy() const noexcept { return this->swizzled<X, Y>(); }
    constexpr auto xz() const noexcept { return this->swizzled<X, Z>(); }
    constexpr auto xyz() const noexcept { return this->swizzled<X, Y, Z>(); }
    constexpr auto yzx() const noexcept { return this->swizzled<Y, Z, X>(); }
    constexpr auto zxy() const noexcept { return this->swizzled<Z, X, Y>(); }

    // These begin() and end() functions allow a Vector to be used like a container for element
    // access. Not generally recommended but sometimes useful. Note no non-const versions provided.
    constexpr auto begin() const noexcept { return std::cbegin(this->e); }
    constexpr auto end() const noexcept { return std::cend(this->e); }

    // Equality
    using base::operator==;
    using base::operator!=;

    // @=() op assignment operators
    using base::operator+=;
    using base::operator-=;
    using base::operator*=;
    using base::operator/=;

    // Unary and binary operator +/-
    using base::operator+;
    using base::operator-;

    // Binary operator *//
    using base::operator*;
    using base::operator/;
    friend constexpr auto operator*(const ScalarType_t<Vector>& s, const Vector& x) noexcept {
        return x * s;
    }

    // Multiply elements of Vectors x and y memberwise
    // e.g. Vec3f x, y; x.memberwiseMultiply(y) == Vec3f{x.x * y.x, x.y * y.y, x.z * y.z}
    using base::memberwiseMultiply;
    friend constexpr auto memberwiseMultiply(const Vector& x, const Vector& y) noexcept {
        return x.memberwiseMultiply(y);
    }

    // Dot product
    using base::dot;
    friend constexpr auto dot(const Vector& x, const Vector& y) noexcept { return x.dot(y); }

    // magnitude and normalize
    using base::magnitude;
    friend constexpr auto magnitude(const Vector& x) noexcept { return x.magnitude(); }

    friend constexpr auto normalize(const Vector& x) noexcept { return x * (T{1} / x.magnitude()); }

    template <typename F>
    friend constexpr auto map(F f, const Vector& x) noexcept {
        return Vector{x.mapHelper(f)};
    }

    friend constexpr auto abs(const Vector& x) noexcept {
        return Vector{x.mapHelper([](const T& x) { return std::abs(x); })};
    }

    friend constexpr auto min(const Vector& x, const Vector& y) noexcept {
        return Vector{x.mapHelper([](const T& x, const T& y) { return std::min<T>(x, y); }, y)};
    }

    friend constexpr auto max(const Vector& x, const Vector& y) noexcept {
        return Vector{x.mapHelper([](const T& x, const T& y) { return std::max<T>(x, y); }, y)};
    }

    friend constexpr auto saturate(const Vector& x) noexcept {
        return Vector{x.mapHelper(saturate<T>)};
    }

    friend constexpr auto clamp(const Vector<T, N>& x, const ScalarType_t<Vector>& a,
                                const ScalarType_t<Vector>& b) noexcept {
        return Vector{x.mapHelper([=](const T& y) { return std::clamp(y, a, b); })};
    }

    static constexpr auto zero() { return Vector{}; }
    using base::basis;
    using base::ones;
};

using Vec2d = Vector<double, 2>;
using Vec3d = Vector<double, 3>;
using Vec4d = Vector<double, 4>;

using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;

using Vec2i = Vector<int, 2>;
using Vec3i = Vector<int, 3>;
using Vec4i = Vector<int, 4>;

// Specializations of type traits for working with Vectors
template <typename T, size_t N>
struct IsVector<Vector<T, N>> : std::true_type {};

template <typename T, size_t N>
struct VectorDimension<Vector<T, N>> : std::integral_constant<size_t, N> {};

template <typename T, size_t N>
struct VectorElementType<Vector<T, N>> {
    using type = T;
};

template <typename T, size_t N>
struct ScalarType<Vector<T, N>> {
    using type = ScalarType_t<T>;
};

// Free function tuple style get<>()
template <std::size_t I, typename T, std::size_t N>
constexpr auto get(const mathlib::Vector<T, N>& x) noexcept -> const T& {
    return x[I];
}

template <std::size_t I, typename T, std::size_t N>
constexpr auto get(mathlib::Vector<T, N>& x) noexcept -> T& {
    return x[I];
}

// Returns a basis vector with 1 in the specified position and 0 elsewhere,
// e.g. basisVector<float, 3>(2) == Vec3f{0.0f, 1.0f, 0.0f}
template <typename T, size_t N>
constexpr auto basisVector(size_t i) noexcept {
    return Vector<T, N>::basis(i);
}

// free function operators and vector specific functions (dot() etc.)
template <typename T>
constexpr auto cross(const Vector<T, 3>& a, const Vector<T, 3>& b) noexcept {
    return a.yzx().memberwiseMultiply(b.zxy()) - a.zxy().memberwiseMultiply(b.yzx());
}

}  // namespace mathlib

// Specialize std::tuple_size and std::tuple_element for Vector to support structured bindings
namespace std {
template <typename T, size_t N>
struct tuple_size<mathlib::Vector<T, N>> : integral_constant<size_t, N> {};

template <size_t I, typename T, size_t N>
struct tuple_element<I, mathlib::Vector<T, N>> {
    using type = T;
};
}  // namespace std
