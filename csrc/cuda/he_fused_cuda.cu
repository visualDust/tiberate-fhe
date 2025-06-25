#include "he_fused_cuda.h"
#include <cstdint>
#include <cstdio>
#include "extensions.cuh"
#include "mont_scalar_kernel.cuh"

// ------------------------------------------------------------------
// pc_add_fused_cuda_kernel
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

// ------------------------------------------------------------------
// create_switcher - pre_extend
// ------------------------------------------------------------------

// template <typename scalar_t>
// __global__ void create_switcher_pre_extend_cuda_kernel(
//     TensorAcc32Restrict<scalar_t, 2> out_acc,
//     const TensorAcc32Restrict<scalar_t, 2> a_part_acc,
//     const TensorAcc32Restrict<scalar_t, 1> perm_acc,
//     const TensorAcc32Restrict<scalar_t, 1> Rs_acc,  // Rs_prepack
//     const TensorAcc32Restrict<scalar_t, 1> ql_acc,  //
//     *mont_prepack const TensorAcc32Restrict<scalar_t, 1> qh_acc,
//     const TensorAcc32Restrict<scalar_t, 1> kl_acc,
//     const TensorAcc32Restrict<scalar_t, 1> kh_acc) {

// ----------------------------------------------------------------------
// create_switcher Divide by P
// ----------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_chain_backward_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> PiRi_acc) {
  const int tid = threadIdx.x;
  const int N = a_acc.size(1);
  const int C = a_acc.size(0);

  if (tid >= N) return;

  for (int row = C - 2; row >= 0; --row) {
    scalar_t val = a_acc[row][tid];
    scalar_t PiRi = PiRi_acc[row];

    // Loop over all rows below
    for (int k = C - 1; k > row; --k) {
      val = mont_mult_scalar_cuda_kernel(val - a_acc[k][tid], PiRi);
    }

    a_acc[row][tid] = val;
    __syncthreads();  // ensure that row is finished before the next uses it
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
    const TensorAcc32Restrict<scalar_t, 1>
        PiRi_acc,  // same shape as c_acc.size(0)
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
  const scalar_t PiRi = PiRi_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];

  // mont_enter
  x = mont_mult_scalar_cuda_kernel(x, Rs, ql, qh, kl, kh);

  // iterate from last p_acc to first
  for (int k = p_acc.size(0) - 1; k >= 0; --k) {
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
void create_switcher_divide_by_p_cuda_typed(torch::Tensor out,
                                            const torch::Tensor c,
                                            const torch::Tensor p,
                                            const torch::Tensor _2q,
                                            const torch::Tensor Rs,
                                            const torch::Tensor PiRi,
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
  dim3 dim_grid(C, N / BLOCK_SIZE);

  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto c_acc = makeAcc32Restrict(c, scalar_t, 2);
  const auto p_acc = makeAcc32Restrict(p, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  const auto Rs_acc = makeAcc32Restrict(Rs, scalar_t, 1);
  const auto PiRi_acc = makeAcc32Restrict(PiRi, scalar_t, 1);
  const auto ql_acc = makeAcc32Restrict(ql, scalar_t, 1);
  const auto qh_acc = makeAcc32Restrict(qh, scalar_t, 1);
  const auto kl_acc = makeAcc32Restrict(kl, scalar_t, 1);
  const auto kh_acc = makeAcc32Restrict(kh, scalar_t, 1);

  create_switcher_d_divide_by_p_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc,
                                           c_acc,
                                           p_acc,
                                           _2q_acc,
                                           Rs_acc,
                                           PiRi_acc,
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
    const torch::Tensor PiRi,  // same shape as c.size(0)
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
