#pragma once

#include "../../extensions.cuh"
#include "constant_mem.cuh"
#include "mont_scalar_kernel.cuh"

//------------------------------------------------------------------
// mont enter
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_Rs_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc, const int sp_prime_len) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];

  // Montgomery inputs.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t Rs = const_mem_2q[-RS_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t _2q = const_mem_2q[-_2Q_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];

  // Store the result.
  a_acc[i][j] = mont_mult_scalar_cuda_kernel(a, Rs, ql, qh, kl, kh);
}

template <typename scalar_t>
__global__ void mont_enter_Ninv_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc, const int sp_prime_len) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];

  // Montgomery inputs.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t Ninv = const_mem_2q[-NINV_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t _2q = const_mem_2q[-_2Q_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];

  // Store the result.
  a_acc[i][j] = mont_mult_scalar_cuda_kernel(a, Ninv, ql, qh, kl, kh);
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

  // Inputs.
  const scalar_t x = a_acc[i][j];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];

  a_acc[i][j] = mont_reduce_scalar_cuda_kernel(x, ql, qh, kl, kh);
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
  const scalar_t a = a_acc[i][j];
  const scalar_t _2q = _2q_acc[i];
  // Reduce. bound 2q → q
  a_acc[i][j] = reduce_2q_scalar_cuda_kernel(a, _2q);
}

//------------------------------------------------------------------
// Make signed
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
  const scalar_t _2q = _2q_acc[i];

  // Make signed.
  a_acc[i][j] = make_signed_scalar_cuda_kernel(a, _2q);
}

// -------------------------------------------------------------------
// mont enter + mont reduce
// -------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_Ninv_mont_reduce_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc, const int sp_prime_len) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Masks.
  constexpr scalar_t one = 1;
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  constexpr scalar_t fb_mask = ((one << nbits) - one);
  constexpr scalar_t lb_mask = (one << half_nbits) - one;

  const scalar_t a = a_acc[i][j];
  // Montgomery inputs.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t Ninv = const_mem_2q[-NINV_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t _2q = const_mem_2q[-_2Q_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];

  scalar_t x =
      mont_mult_scalar_cuda_kernel(a, Ninv, ql, qh, kl, kh);  // mont enter
  x = mont_reduce_scalar_cuda_kernel(x, ql, qh, kl, kh);      // mont reduce

  // write the result
  a_acc[i][j] = x;
}

// -------------------------------------------------------------------
// mont enter + mont reduce + reduce 2q
// -------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_Ninv_mont_reduce_reduce_2q_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc, const int sp_prime_len) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Masks.
  constexpr scalar_t one = 1;
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  constexpr scalar_t fb_mask = ((one << nbits) - one);
  constexpr scalar_t lb_mask = (one << half_nbits) - one;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  // Montgomery inputs.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t Ninv = const_mem_2q[-NINV_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t _2q = const_mem_2q[-_2Q_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t q = _2q >> 1;

  scalar_t x =
      mont_mult_scalar_cuda_kernel(a, Ninv, ql, qh, kl, kh);  // mont enter
  x = mont_reduce_scalar_cuda_kernel(x, ql, qh, kl, kh);      // mont reduce
  x = reduce_2q_scalar_cuda_kernel(x, _2q);                   // reduce 2q

  // write the result
  a_acc[i][j] = x;
}

// -------------------------------------------------------------------
// mont enter + mont reduce + reduce 2q + make signed
// -------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_Ninv_mont_reduce_reduce_2q_make_signed_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc, const int sp_prime_len) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Masks.
  constexpr scalar_t one = 1;
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  constexpr scalar_t fb_mask = ((one << nbits) - one);
  constexpr scalar_t lb_mask = (one << half_nbits) - one;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  // Montgomery inputs.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t Ninv = const_mem_2q[-NINV_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t _2q = const_mem_2q[-_2Q_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];

  // mont enter
  scalar_t x = mont_mult_scalar_cuda_kernel(a, Ninv, ql, qh, kl, kh);
  x = mont_reduce_scalar_cuda_kernel(x, ql, qh, kl, kh);  // mont reduce
  x = reduce_2q_scalar_cuda_kernel(x, _2q);               // reduce 2q
  x = make_signed_scalar_cuda_kernel(x, _2q);             // make signed

  // write the result
  a_acc[i][j] = x;
}
