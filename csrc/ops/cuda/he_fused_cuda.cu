#include "he_fused_cuda.h"
#include <cstdint>
#include <cstdio>
#include "../../extensions.cuh"
#include "mont_scalar_kernel.cuh"

// ------------------------------------------------------------------
// pc_add_fused_cuda_kernel: mont enter + mont add + mont reduce + reduce 2q
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void pc_add_fused_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,
    const TensorAcc32Restrict<scalar_t, 2> ct_acc,
    const TensorAcc32Restrict<scalar_t, 2> pt_acc,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc,
    const TensorAcc32Restrict<scalar_t, 1> Rs_acc,
    const TensorAcc32Restrict<scalar_t, 1> ql_acc,
    const TensorAcc32Restrict<scalar_t, 1> qh_acc,
    const TensorAcc32Restrict<scalar_t, 1> kl_acc,
    const TensorAcc32Restrict<scalar_t, 1> kh_acc) {
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Masks.
  constexpr scalar_t one = 1;
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  constexpr scalar_t fb_mask = ((one << nbits) - one);
  constexpr scalar_t lb_mask = (one << half_nbits) - one;

  // Inputs.
  const scalar_t ct_in = ct_acc[i][j];
  const scalar_t pt_in = pt_acc[i][j];

  const scalar_t Rs = Rs_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];
  const scalar_t _2q = _2q_acc[i];

  scalar_t x =
      mont_mult_scalar_cuda_kernel(ct_in, Rs, ql, qh, kl, kh);  // mont mult
  x = mont_add_scalar_cuda_kernel(x, pt_in, _2q);               // mont add
  x = mont_reduce_scalar_cuda_kernel(x, ql, qh, kl, kh);        // mont reduce
  x = reduce_2q_scalar_cuda_kernel(x, _2q);  // reduce 2q, bound 2q → q

  // write the result
  out_acc[i][j] = x;
}

template <typename scalar_t>
void pc_add_fused_cuda_typed(const torch::Tensor ct_data,
                             const torch::Tensor pt_data,
                             torch::Tensor out,
                             const torch::Tensor _2q,
                             const torch::Tensor Rs,
                             const torch::Tensor ql,
                             const torch::Tensor qh,
                             const torch::Tensor kl,
                             const torch::Tensor kh) {
  // Retrieve the device index, then set the corresponding device and stream.
  auto device_id = ct_data.device().index();
  cudaSetDevice(device_id);

  // Use a preallocated pytorch stream.
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  // The problem dimension.
  auto C = ct_data.size(0);
  auto N = ct_data.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  // Run the cuda kernel.
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto ct_acc = makeAcc32Restrict(ct_data, scalar_t, 2);
  const auto pt_acc = makeAcc32Restrict(pt_data, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  const auto Rs_acc = makeAcc32Restrict(Rs, scalar_t, 1);
  const auto ql_acc = makeAcc32Restrict(ql, scalar_t, 1);
  const auto qh_acc = makeAcc32Restrict(qh, scalar_t, 1);
  const auto kl_acc = makeAcc32Restrict(kl, scalar_t, 1);
  const auto kh_acc = makeAcc32Restrict(kh, scalar_t, 1);

  pc_add_fused_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
      out_acc, ct_acc, pt_acc, _2q_acc, Rs_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

torch::Tensor pc_add_fused_cuda(const torch::Tensor a,  // ct_data
                                const torch::Tensor b,  // pt_data
                                const torch::Tensor _2q,
                                const torch::Tensor Rs,
                                const torch::Tensor ql,
                                const torch::Tensor qh,
                                const torch::Tensor kl,
                                const torch::Tensor kh) {
  // Dispatch to the correct data type.
  torch::Tensor out = torch::empty_like(a);
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_pc_add_fused_cuda", ([&] {
                               pc_add_fused_cuda_typed<scalar_t>(
                                   a, b, out, _2q, Rs, ql, qh, kl, kh);
                             }));
  return out;
}

// ------------------------------------------------------------------
// rescale + exact rounding
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void rescale_exact_rounding_fused_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> Rs_acc,
    const TensorAcc32Restrict<scalar_t, 1> rescaler,  // rescaler0
    const int64_t round_at,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc,
    const TensorAcc32Restrict<scalar_t, 1> ql_acc,
    const TensorAcc32Restrict<scalar_t, 1> qh_acc,
    const TensorAcc32Restrict<scalar_t, 1> kl_acc,
    const TensorAcc32Restrict<scalar_t, 1> kh_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Masks.
  constexpr scalar_t one = 1;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t b = Rs_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];
  const scalar_t _2q = _2q_acc[i];

  // in python, its rounder = torch.where(rescaler > round_at, 1, 0)
  const scalar_t resclr = rescaler[j];
  const scalar_t rounder = (resclr > round_at) ? 1 : 0;

  // data0 = [(d - s) for d, s in zip(data0, rescaler0)]
  scalar_t x = a - resclr;
  // mont_enter
  x = mont_mult_scalar_cuda_kernel(x, b, ql, qh, kl, kh);
  // data0 = [(d + r) for d, r in zip(data0, rounder0)]
  x = x + rounder;
  // reduce 2q
  x = reduce_2q_scalar_cuda_kernel(x, _2q);
  // write the result
  a_acc[i][j] = x;
}

template <typename scalar_t>
void rescale_exact_rounding_fused_cuda_typed(
    torch::Tensor a,
    const torch::Tensor Rs,
    const torch::Tensor rescaler,  // rescaler0
    const int64_t round_at,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  auto C = a.size(0);
  auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  // Run the cuda kernel.
  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto Rs_acc = makeAcc32Restrict(Rs, scalar_t, 1);
  const auto rescaler_acc = makeAcc32Restrict(rescaler, scalar_t, 1);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  const auto ql_acc = makeAcc32Restrict(ql, scalar_t, 1);
  const auto qh_acc = makeAcc32Restrict(qh, scalar_t, 1);
  const auto kl_acc = makeAcc32Restrict(kl, scalar_t, 1);
  const auto kh_acc = makeAcc32Restrict(kh, scalar_t, 1);

  rescale_exact_rounding_fused_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc,
                                           Rs_acc,
                                           rescaler_acc,
                                           round_at,
                                           _2q_acc,
                                           ql_acc,
                                           qh_acc,
                                           kl_acc,
                                           kh_acc);
}

void rescale_exact_rounding_fused_cuda(
    torch::Tensor a,  // inplace of a
    const torch::Tensor Rs,
    const torch::Tensor rescaler,  // rescaler0
    const int64_t round_at,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_rescale_exact_rounding_fused_cuda", ([&] {
        rescale_exact_rounding_fused_cuda_typed<scalar_t>(
            a, Rs, rescaler, round_at, _2q, ql, qh, kl, kh);
      }));
}

// ------------------------------------------------------------------
// rescale without exact rounding
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void rescale_non_exact_rounding_fused_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> Rs_acc,
    const TensorAcc32Restrict<scalar_t, 1> rescaler,  // rescaler0
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc,
    const TensorAcc32Restrict<scalar_t, 1> ql_acc,
    const TensorAcc32Restrict<scalar_t, 1> qh_acc,
    const TensorAcc32Restrict<scalar_t, 1> kl_acc,
    const TensorAcc32Restrict<scalar_t, 1> kh_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Masks.
  constexpr scalar_t one = 1;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t b = Rs_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];
  const scalar_t _2q = _2q_acc[i];
  // in python, its rounder = torch.where(rescaler > round_at, 1, 0)
  const scalar_t resclr = rescaler[j];

  // data0 = [(d - s) for d, s in zip(data0, rescaler0)]
  scalar_t x = a - resclr;
  // mont_enter
  x = mont_mult_scalar_cuda_kernel(x, b, ql, qh, kl, kh);
  // data0 = [(d + r) for d, r in zip(data0, rounder0)]
  // reduce 2q, bound 2q → q
  x = reduce_2q_scalar_cuda_kernel(x, _2q);
  // write the result
  a_acc[i][j] = x;
}

template <typename scalar_t>
void rescale_non_exact_rounding_fused_cuda_typed(
    torch::Tensor a,
    const torch::Tensor Rs,
    const torch::Tensor rescaler,  // rescaler0
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  auto C = a.size(0);
  auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  // Run the cuda kernel.
  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto Rs_acc = makeAcc32Restrict(Rs, scalar_t, 1);
  const auto rescaler_acc = makeAcc32Restrict(rescaler, scalar_t, 1);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  const auto ql_acc = makeAcc32Restrict(ql, scalar_t, 1);
  const auto qh_acc = makeAcc32Restrict(qh, scalar_t, 1);
  const auto kl_acc = makeAcc32Restrict(kl, scalar_t, 1);
  const auto kh_acc = makeAcc32Restrict(kh, scalar_t, 1);

  rescale_non_exact_rounding_fused_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(
          a_acc, Rs_acc, rescaler_acc, _2q_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

void rescale_non_exact_rounding_fused_cuda(
    torch::Tensor a,  // inplace of a
    const torch::Tensor Rs,
    const torch::Tensor rescaler,  // rescaler0
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_rescale_non_exact_rounding_fused_cuda", ([&] {
        rescale_non_exact_rounding_fused_cuda_typed<scalar_t>(
            a, Rs, rescaler, _2q, ql, qh, kl, kh);
      }));
}

// ------------------------------------------------------------------
// key switching - switch layer part - extend
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void switch_key_switch_later_part_extend(
    TensorAcc32Restrict<scalar_t, 2> out_acc,
    const TensorAcc32Restrict<scalar_t, 2> state_acc,
    const TensorAcc32Restrict<scalar_t, 2> l_enter_acc,
    const int64_t l_enter_start_offset,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc,
    const TensorAcc32Restrict<scalar_t, 1> Rs_acc,  // Rs_prepack
    const TensorAcc32Restrict<scalar_t, 1> ql_acc,  // *mont_prepack
    const TensorAcc32Restrict<scalar_t, 1> qh_acc,
    const TensorAcc32Restrict<scalar_t, 1> kl_acc,
    const TensorAcc32Restrict<scalar_t, 1> kh_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t _2q = _2q_acc[i];
  const scalar_t Rs = Rs_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];

  // mont enter
  const scalar_t state_0 = state_acc[0][j];
  scalar_t x = mont_mult_scalar_cuda_kernel(state_0, Rs, ql, qh, kl, kh);

  for (int k = 0; k < state_acc.size(0) - 1; ++k) {
    const scalar_t state_k = state_acc[k + 1][j];
    const scalar_t l_enter_k = l_enter_acc[k][l_enter_start_offset + i];
    const scalar_t y =
        mont_mult_scalar_cuda_kernel(state_k, l_enter_k, ql, qh, kl, kh);
    x = mont_add_scalar_cuda_kernel(x, y, _2q);
  }

  // Store the result back
  out_acc[i][j] = x;
}

template <typename scalar_t>
void switch_key_switch_later_part_extend_cuda_typed(
    torch::Tensor out,
    const torch::Tensor state,
    const torch::Tensor l_enter,
    const int64_t l_enter_start_offset,
    const torch::Tensor _2q,
    const torch::Tensor Rs,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
  auto device_id = state.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);
  auto C = out.size(0);
  auto N = state.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto state_acc = makeAcc32Restrict(state, scalar_t, 2);
  const auto l_enter_acc = makeAcc32Restrict(l_enter, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  const auto Rs_acc = makeAcc32Restrict(Rs, scalar_t, 1);
  const auto ql_acc = makeAcc32Restrict(ql, scalar_t, 1);
  const auto qh_acc = makeAcc32Restrict(qh, scalar_t, 1);
  const auto kl_acc = makeAcc32Restrict(kl, scalar_t, 1);
  const auto kh_acc = makeAcc32Restrict(kh, scalar_t, 1);

  switch_key_switch_later_part_extend<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc,
                                           state_acc,
                                           l_enter_acc,
                                           l_enter_start_offset,
                                           _2q_acc,
                                           Rs_acc,
                                           ql_acc,
                                           qh_acc,
                                           kl_acc,
                                           kh_acc);
}

torch::Tensor switch_key_switch_later_part_extend_cuda(
    const int64_t rns_len,
    const torch::Tensor state,
    const torch::Tensor l_enter,
    const int64_t l_enter_start_offset,
    const torch::Tensor _2q,
    const torch::Tensor Rs,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
  torch::Tensor out = torch::empty({rns_len, state.size(1)}, state.options());

  AT_DISPATCH_INTEGRAL_TYPES(
      state.scalar_type(), "switch_key_switch_later_part_extend_cuda", [&] {
        switch_key_switch_later_part_extend_cuda_typed<scalar_t>(
            out, state, l_enter, l_enter_start_offset, _2q, Rs, ql, qh, kl, kh);
      });

  return out;
}

// ------------------------------------------------------------------
// rotate_single - codec_rotate
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void codec_rotate_make_unsigned_reduce_2q_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,
    const TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> perm_acc,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  const int i = blockIdx.x;                             // batch index
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;  // position index

  const int N = a_acc.size(1);
  if (j >= N) {
    printf("debug: j >= N, j: %d, N: %d\n", j, N);
    return;
  }

  const scalar_t perm = perm_acc[j];
  const scalar_t perm_folded = perm % N;

  // Compute sign = (-1)^(perm // N)
  const scalar_t perm_sign = ((perm / N) & 1) ? -1 : 1;

  // Read input
  scalar_t x = a_acc[i][j];
  x *= perm_sign;

  // Load 2q
  const scalar_t _2q = _2q_acc[i];

  // Apply unsigned conversion and reduction
  x = make_unsigned_scalar_cuda_kernel(x, _2q);
  x = reduce_2q_scalar_cuda_kernel(x, _2q);

  // Write output
  out_acc[i][perm_folded] = x;
}

template <typename scalar_t>
void codec_rotate_make_unsigned_reduce_2q_cuda_typed(torch::Tensor out,
                                                     const torch::Tensor a,
                                                     const torch::Tensor perm,
                                                     const torch::Tensor _2q) {
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);
  auto C = out.size(0);
  auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto perm_acc = makeAcc32Restrict(perm, scalar_t, 1);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);

  codec_rotate_make_unsigned_reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, a_acc, perm_acc, _2q_acc);
}

torch::Tensor codec_rotate_make_unsigned_reduce_2q_cuda(
    const torch::Tensor a, const torch::Tensor perm, const torch::Tensor _2q) {
  torch::Tensor out = torch::empty_like(a);

  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "codec_rotate_make_unsigned_reduce_2q_cuda", [&] {
        codec_rotate_make_unsigned_reduce_2q_cuda_typed<scalar_t>(
            out, a, perm, _2q);
      });

  return out;
}

// ----------------------------------------------------------------------
// create_switcher Divide by P
// ----------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_chain_backward_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> p_acc,
    const int prime_row_offset,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc,
    const scalar_t** PiRi_ptr,
    const TensorAcc32Restrict<scalar_t, 1> ql_acc,
    const TensorAcc32Restrict<scalar_t, 1> qh_acc,
    const TensorAcc32Restrict<scalar_t, 1> kl_acc,
    const TensorAcc32Restrict<scalar_t, 1> kh_acc) {
  // const int i = blockIdx.x; // i is useless here because only need 1d kernel
  // launch
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
  const int num_primes = p_acc.size(0);

  for (int row = num_primes - 2; row >= 0; --row) {
    scalar_t x = p_acc[row][j];
    const scalar_t _2q = _2q_acc[row + prime_row_offset];
    const scalar_t ql = ql_acc[row + prime_row_offset];
    const scalar_t qh = qh_acc[row + prime_row_offset];
    const scalar_t kl = kl_acc[row + prime_row_offset];
    const scalar_t kh = kh_acc[row + prime_row_offset];

    // Loop over all rows below
    for (int k = num_primes - 1; k > row; --k) {
      const scalar_t PiRi =
          PiRi_ptr[num_primes - k - 1][row + prime_row_offset];
      const scalar_t after_sub =
          mont_sub_scalar_cuda_kernel(x, p_acc[k][j], _2q);  // mont sub
      const scalar_t after_mult = mont_mult_scalar_cuda_kernel(
          after_sub, PiRi, ql, qh, kl, kh);  // mont enter scalar PiRi
      x = after_mult;
    }

    p_acc[row][j] = x;
    // __syncthreads();  // previous row depends on the next row
  }
}

template <typename scalar_t>
__global__ void create_switcher_d_divide_by_p_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,  // same shape as c_acc
    const TensorAcc32Restrict<scalar_t, 2>
        c_acc,  // d[: -self.ckksCfg.num_special_primes]
    const TensorAcc32Restrict<scalar_t, 2>
        p_acc,  // d[-self.ckksCfg.num_special_primes:], assume p_acc is already
                // processed with create_switcher_p_self_divide_iter_cuda_kernel
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc,
    const TensorAcc32Restrict<scalar_t, 1> Rs_acc,
    const scalar_t** PiRi_ptr,
    const TensorAcc32Restrict<scalar_t, 1> ql_acc,
    const TensorAcc32Restrict<scalar_t, 1> qh_acc,
    const TensorAcc32Restrict<scalar_t, 1> kl_acc,
    const TensorAcc32Restrict<scalar_t, 1> kh_acc) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  scalar_t x = c_acc[i][j];
  const scalar_t _2q = _2q_acc[i];
  const scalar_t Rs = Rs_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];

  // mont_enter
  x = mont_mult_scalar_cuda_kernel(x, Rs, ql, qh, kl, kh);

  // iterate from last p_acc to first
  const int num_primes = p_acc.size(0);
  for (int k = p_acc.size(0) - 1; k >= 0; --k) {
    const scalar_t PiRi = PiRi_ptr[num_primes - k - 1][i];
    const scalar_t p = p_acc[k][j];
    const scalar_t p_enter =
        mont_mult_scalar_cuda_kernel(p, Rs, ql, qh, kl, kh);  // mont enter
    x = mont_sub_scalar_cuda_kernel(x, p_enter, _2q);         // mont sub
    x = mont_mult_scalar_cuda_kernel(
        x, PiRi, ql, qh, kl, kh);  // mont enter scalar PiRi
  }

  x = mont_reduce_scalar_cuda_kernel(x, ql, qh, kl, kh);  // mont reduce
  x = reduce_2q_scalar_cuda_kernel(x, _2q);               // reduce 2q

  // Store the result.
  out_acc[i][j] = x;
}

template <typename scalar_t>
void create_switcher_divide_by_p_cuda_typed(
    torch::Tensor out,
    const torch::Tensor c,
    const torch::Tensor p,
    const torch::Tensor _2q,
    const torch::Tensor Rs,
    const std::vector<torch::Tensor> PiRi,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
  auto device_id = c.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  auto C = out.size(0);
  auto N = c.size(1);  // c.size(1) == p.size(1)
  int dim_block = BLOCK_SIZE;
  dim3 dim_grid_p_back(1, N / BLOCK_SIZE);
  dim3 dim_grid_divide_p(C, N / BLOCK_SIZE);

  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto c_acc = makeAcc32Restrict(c, scalar_t, 2);
  const auto p_acc = makeAcc32Restrict(p, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  const auto Rs_acc = makeAcc32Restrict(Rs, scalar_t, 1);

  // need to create pointer for PiRi
  std::vector<scalar_t*> PiRi_ptr_h(PiRi.size());
  for (size_t i = 0; i < PiRi.size(); ++i)
    PiRi_ptr_h[i] = PiRi[i].data_ptr<scalar_t>();
  scalar_t** PiRi_ptr_d = nullptr;
  cudaMalloc(&PiRi_ptr_d, PiRi.size() * sizeof(scalar_t*));
  cudaMemcpyAsync(PiRi_ptr_d,
                  PiRi_ptr_h.data(),
                  PiRi.size() * sizeof(scalar_t*),
                  cudaMemcpyHostToDevice,
                  stream);

  // const auto PiRi_acc = makeAcc32Restrict(PiRi, scalar_t, 1);
  const auto ql_acc = makeAcc32Restrict(ql, scalar_t, 1);
  const auto qh_acc = makeAcc32Restrict(qh, scalar_t, 1);
  const auto kl_acc = makeAcc32Restrict(kl, scalar_t, 1);
  const auto kh_acc = makeAcc32Restrict(kh, scalar_t, 1);

  // This is actually done in successive order.
  // Rescale from the most outer prime channel.
  // Start from the special len and drop channels one by one.

  // process primes
  mont_chain_backward_cuda_kernel<scalar_t>
      <<<dim_grid_p_back, dim_block, 0, stream>>>(
          p_acc,
          c.size(0),  // others are for both c and p
          _2q_acc,
          (const scalar_t**)PiRi_ptr_d,
          ql_acc,
          qh_acc,
          kl_acc,
          kh_acc);

  // run the main kernel
  create_switcher_d_divide_by_p_cuda_kernel<scalar_t>
      <<<dim_grid_divide_p, dim_block, 0, stream>>>(
          out_acc,
          c_acc,
          p_acc,
          _2q_acc,
          Rs_acc,
          (const scalar_t**)PiRi_ptr_d,
          ql_acc,
          qh_acc,
          kl_acc,
          kh_acc);
}

torch::Tensor create_switcher_divide_by_p_cuda(
    const torch::Tensor c,  // d[: -self.ckksCfg.num_special_primes]
    const torch::Tensor p,  // d[-self.ckksCfg.num_special_primes:]
    const torch::Tensor _2q,
    const torch::Tensor Rs,
    const std::vector<torch::Tensor> PiRi,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
  // Create output tensor.
  torch::Tensor out = torch::empty_like(c);

  AT_DISPATCH_INTEGRAL_TYPES(
      c.scalar_type(), "create_switcher_divide_by_p_cuda", [&] {
        create_switcher_divide_by_p_cuda_typed<scalar_t>(
            out, c, p, _2q, Rs, PiRi, ql, qh, kl, kh);
      });

  return out;
}

// ------------------------------------------------------------------
// create_switcher - pre_extend
// ------------------------------------------------------------------
