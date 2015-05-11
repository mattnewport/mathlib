#pragma once

#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

// Vectors are treated as row vectors for the purposes of matrix multiplication (so to transform a
// Vector v by a Matrix M use v * M rather than M * v)

// Observations on efficient usage of Vector types.
//
// This code is very generic and uses multiple layers of function helpers, it compiles down to
// pretty efficient code in release builds but in debug builds without any inlining it will be
// pretty inefficient. Using /Ob1
// http: //msdn.microsoft.com/en-us/library/47238hez.aspx for debug builds in Visual Studio will help
// debug performance a lot.
//
// You probably want to work with vec4fs as local variables and do all / most of your calculations
// with them, keeping vec3fs as mostly a storage format. This is because vec4fs are implemented as
// native SIMD vector types and most simple operations map to a single intrinsic. It is inefficient
// to work with vec3fs as native SIMD types since the compiler can not make intelligent decisions
// about when to keep values in registers. The exception to this is when doing bulk operations on
// arrays of vec3fs in which case for optimum efficiency you would write custom vector code in SoA
// rather than AoS style operating on multiple elements of the array at once.

namespace mathlib {

template <typename T, size_t N>
class Vector {
public:
    static const size_t dimension = N;

    template <typename... Us>
    Vector(Us&&... us)
        : aw({std::forward<Us>(us)...}) {}

    T& e(size_t i) { return aw.e_[i]; }
    constexpr const T& e(size_t i) const { return aw.e_[i]; }

    T& x() {
        static_assert(N > 0, "accessing x element of empty vector");
        return aw.e_[0];
    }
    constexpr const T& x() const {
        static_assert(N > 0, "accessing x element of empty vector");
        return aw.e_[0];
    }
    T& y() {
        static_assert(N > 1, "accessing y element of vector of dimension less than 2");
        return aw.e_[1];
    }
    constexpr const T& y() const {
        static_assert(N > 1, "accessing y element of vector of dimension less than 2");
        return aw.e_[1];
    }
    T& z() {
        static_assert(N > 2, "accessing z element of vector of dimension less than 3");
        return aw.e_[2];
    }
    constexpr const T& z() const {
        static_assert(N > 2, "accessing z element of vector of dimension less than 3");
        return aw.e_[2];
    }
    T& w() {
        static_assert(N > 3, "accessing w element of vector of dimension less than 4");
        return aw.e_[3];
    }
    constexpr const T& w() const {
        static_assert(N > 3, "accessing w element of vector of dimension less than 4");
        return aw.e_[3];
    }

    auto begin() { return std::begin(aw.e_); }
    auto end() { return std::end(aw.e_); }
    auto begin() const { return std::cbegin(aw.e_); }
    auto end() const { return std::cend(aw.e_); }
    auto cbegin() const { return std::cbegin(aw.e_); }
    auto cend() const { return std::cend(aw.e_); }

private:
    struct ArrayWrapper {
        T e_[N];
    } aw;  // ArrayWrapper lets us initialize in constructor initializer
};

using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;

template <size_t I, typename F, typename... Args>
auto apply(F f, Args&&... args) {
    return f(args.e(I)...);
}

template <typename F, size_t... Is, typename... Args>
auto apply(F f, std::index_sequence<Is...>, Args&&... args) {
    using vec = std::common_type_t<Args...>;
    using resvec = Vector<decltype(apply<0>(f, args...)), vec::dimension>;
    return resvec{apply<Is>(f, args...)...};
}

template <typename F, typename... Args>
inline auto memberwise(F f, const Args&... args) {
    using vec = std::common_type_t<Args...>;
    return apply(f, std::make_index_sequence<vec::dimension>{}, args...);
}

template <typename F, typename Arg>
auto reduce_impl(F, Arg arg) {
    return arg;
}

template <typename F, typename Arg, typename... Args>
auto reduce_impl(F f, Arg arg, Args... args) {
    return f(arg, reduce_impl(f, args...));
}

template <typename F, typename T, size_t N, size_t... Is>
inline T reduce(F f, const Vector<T, N>& v, std::index_sequence<Is...>) {
    return reduce_impl(f, v.e(Is)...);
}

template <typename F, typename T, size_t N>
inline T reduce(F f, const Vector<T, N>& v) {
    return reduce(f, v, std::make_index_sequence<N>{});
}

template <typename T, size_t N>
inline bool operator==(const Vector<T, N>& a, const Vector<T, N>& b) {
    return reduce(std::logical_and<>{}, memberwise(std::equal_to<>{}, a, b));
}

template <typename T, size_t N>
inline bool operator<(const Vector<T, N>& a, const Vector<T, N>& b) {
    return std::lexicographical_compare(begin(a), end(a), begin(b), end(b));
}

template <typename T, size_t N>
inline Vector<T, N> operator+(const Vector<T, N>& a, const Vector<T, N>& b) {
    return memberwise(std::plus<>{}, a, b);
}

template <typename T, size_t N>
inline Vector<T, N> operator-(const Vector<T, N>& a, const Vector<T, N>& b) {
    return memberwise(std::minus<>{}, a, b);
}

template <typename T, size_t N>
inline Vector<T, N> operator*(const Vector<T, N>& a, T s) {
    return memberwise([s](T e) { return e * s; }, a);
}

template <typename T, size_t N>
inline Vector<T, N> operator*(T s, const Vector<T, N>& a) {
    return a * s;
}

template <typename T, size_t N>
inline Vector<T, N> operator/(const Vector<T, N>& a, T s) {
    return a * (T{1} / s);
}

template <typename T, size_t N>
inline T dot(const Vector<T, N>& a, const Vector<T, N>& b) {
    return reduce(std::plus<>{}, memberwise(std::multiplies<>{}, a, b));
}

}  // namespace mathlib
