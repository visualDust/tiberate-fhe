#pragma once

#include "../extensions.h"
#include "mont_common.cuh"

#define BLOCK_SIZE 256

//------------------------------------------------------------------
// mont enter
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> Rs_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t Rs = Rs_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];

  // Store the result.
  a_acc[i][j] = mont_mult_scalar_cuda_kernel(a, Rs, ql, qh, kl, kh);
}

//------------------------------------------------------------------
// mont reduce
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_reduce_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Masks.
  constexpr scalar_t one = 1;
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  constexpr scalar_t fb_mask = ((one << nbits) - one);
  constexpr scalar_t lb_mask = (one << half_nbits) - one;

  // Inputs.
  const scalar_t x = a_acc[i][j];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];

  // Implementation.
  // s= xk mod R
  const scalar_t xl = x & lb_mask;
  const scalar_t xh = x >> half_nbits;
  const scalar_t xkb = xh * kl + xl * kh;
  scalar_t s = (xkb << half_nbits) + xl * kl;
  s = s & fb_mask;

  // t = x + sq
  // u = t/R
  // Note that x gets erased in t/R operation if x < R.
  const scalar_t sl = s & lb_mask;
  const scalar_t sh = s >> half_nbits;
  const scalar_t sqb = sh * ql + sl * qh;
  const scalar_t sqbl = sqb & lb_mask;
  const scalar_t sqbh = sqb >> half_nbits;
  scalar_t carry = (x + sl * ql) >> half_nbits;
  carry = (carry + sqbl) >> half_nbits;

  // Assume we have satisfied the condition 4*q < R.
  // Return the calculated value directly without conditional subtraction.
  a_acc[i][j] = sqbh + carry + sh * qh;
}

//------------------------------------------------------------------
// reduce 2q
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void reduce_2q_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
  // This kernel reduces each element a_acc[i][j] modulo q = _2q_acc[i] / 2,
  // assuming that a < 2q. It's a fast, branchless way to compute a % q under
  // certain assumptions.

  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t a = a_acc[i][j];
  const scalar_t q = _2q_acc[i] >> one;

  // Reduce. bound 2q → q
  a_acc[i][j] = (a < q) ? a : a - q;
}

//------------------------------------------------------------------
// Misc
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void make_signed_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t a = a_acc[i][j];
  const scalar_t q = _2q_acc[i] >> one;
  const scalar_t q_half = q >> one;

  // Make signed.
  a_acc[i][j] = (a <= q_half) ? a : a - q;
}
