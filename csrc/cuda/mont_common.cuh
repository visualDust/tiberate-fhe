#pragma once

#include <torch/torch.h>

// ------------------------------------------------------------------
// macros and type definitions
// ------------------------------------------------------------------

#define BLOCK_SIZE 256

#define makeAcc32Restrict(tensor, scalar_t, dim) \
  tensor.packed_accessor32<scalar_t, dim, torch::RestrictPtrTraits>()

#define makeAcc32(tensor, scalar_t, dim) \
  tensor.packed_accessor32<scalar_t, dim>()

template <typename scalar_t, int dim>
using TensorAcc32Restrict =
    torch::PackedTensorAccessor32<scalar_t, dim, torch::RestrictPtrTraits>;

template <typename scalar_t, int dim>
using TensorAcc32 = torch::PackedTensorAccessor32<scalar_t, dim>;

// ------------------------------------------------------------------
// mont scalar cuda kernels
// ------------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ scalar_t
mont_mult_scalar_cuda_kernel(const scalar_t a,
                             const scalar_t b,
                             const scalar_t ql,
                             const scalar_t qh,
                             const scalar_t kl,
                             const scalar_t kh) {
  // Masks.
  constexpr scalar_t one = 1;
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  constexpr scalar_t fb_mask = ((one << nbits) - one);
  constexpr scalar_t lb_mask = (one << half_nbits) - one;

  const scalar_t al = a & lb_mask;
  const scalar_t ah = a >> half_nbits;
  const scalar_t bl = b & lb_mask;
  const scalar_t bh = b >> half_nbits;

  const scalar_t alpha = ah * bh;
  const scalar_t beta = ah * bl + al * bh;
  const scalar_t gamma = al * bl;

  // s = xk mod R
  const scalar_t gammal = gamma & lb_mask;
  const scalar_t gammah = gamma >> half_nbits;
  const scalar_t betal = beta & lb_mask;
  const scalar_t betah = beta >> half_nbits;

  scalar_t upper = gammal * kh;
  upper = upper + (gammah + betal) * kl;
  upper = upper << half_nbits;
  scalar_t s = upper + gammal * kl;
  s = upper + gammal * kl;
  s = s & fb_mask;

  // t = x + sq
  // u = t/R
  const scalar_t sl = s & lb_mask;
  const scalar_t sh = s >> half_nbits;
  const scalar_t sqb = sh * ql + sl * qh;
  const scalar_t sqbl = sqb & lb_mask;
  const scalar_t sqbh = sqb >> half_nbits;

  scalar_t carry = (gamma + sl * ql) >> half_nbits;
  carry = (carry + betal + sqbl) >> half_nbits;

  return alpha + betah + sqbh + carry + sh * qh;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
reduce_2q_scalar_cuda_kernel(const scalar_t x, const scalar_t _2q) {
  constexpr scalar_t one = 1;
  const scalar_t q = _2q >> one;
  // Reduce 2q, bound 2q → q
  return (x < q) ? x : x - q;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t mont_add_scalar_cuda_kernel(
    const scalar_t a, const scalar_t b, const scalar_t _2q) {
  // Add.
  const scalar_t aplusb = a + b;
  // Reduce 2q, bound 2q → q
  return (aplusb < _2q) ? aplusb : aplusb - _2q;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t mont_sub_scalar_cuda_kernel(
    const scalar_t a, const scalar_t b, const scalar_t _2q) {
  // Subtract.
  const scalar_t aminusb = a - b;
  // Reduce 2q, bound 2q → q
  return (aminusb < _2q) ? aminusb : aminusb - _2q;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
mont_reduce_scalar_cuda_kernel(const scalar_t a,
                               const scalar_t ql,
                               const scalar_t qh,
                               const scalar_t kl,
                               const scalar_t kh) {
  // Masks.
  constexpr scalar_t one = 1;
  // nbits is set to 2 bits less than the full width for overflow safety
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  // lb_mask: mask for the lower half (like 0xFFFF)
  constexpr scalar_t lb_mask = (one << half_nbits) - one;
  // fb_mask: mask to chop the result below 2^nbits to avoid overflow
  constexpr scalar_t fb_mask = ((one << nbits) - one);

  // s = (a * k) mod R, with k = -q^{-1} mod R
  const scalar_t xl = a & lb_mask;
  const scalar_t xh = a >> half_nbits;
  const scalar_t xkb = xh * kl + xl * kh;
  scalar_t s = (xkb << half_nbits) + xl * kl;
  s = s & fb_mask;

  // t = a + s * q, then u = t / R
  // Note that x gets erased in t/R operation if x < R.
  const scalar_t sl = s & lb_mask;
  const scalar_t sh = s >> half_nbits;
  const scalar_t sqb = sh * ql + sl * qh;
  const scalar_t sqbl = sqb & lb_mask;
  const scalar_t sqbh = sqb >> half_nbits;
  scalar_t carry = (a + sl * ql) >> half_nbits;
  carry = (carry + sqbl) >> half_nbits;

  // The final result is approximated by skipping an actual full a + s * q and
  // >> r division Assume we have satisfied the condition 4*q < R. Return the
  // calculated value directly without conditional subtraction.
  return sqbh + carry + sh * qh;
  // result within [0,2q)
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
make_signed_scalar_cuda_kernel(const scalar_t a, const scalar_t _2q) {
  // Masks.
  constexpr scalar_t one = 1;
  const scalar_t q = _2q >> one;
  const scalar_t q_half = q >> one;
  return (a <= q_half) ? a : a - q;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
make_unsigned_scalar_cuda_kernel(const scalar_t a, const scalar_t _2q) {
  // Masks.
  constexpr scalar_t one = 1;
  const scalar_t q = _2q >> one;
  return a + q;
}
