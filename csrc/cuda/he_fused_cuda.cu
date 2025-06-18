#include "he_fused_cuda.h"
#include <ATen/core/TensorAccessor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <cstdint>
#include "../extensions.h"
#include "mont_common.cuh"

// ------------------------------------------------------------------
// key switching - switch layer part - extend
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void switch_key_switch_later_part_extend(
    torch::PackedTensorAccessor32<scalar_t, 2> out_acc,
    const torch::PackedTensorAccessor32<scalar_t, 2> state_acc,
    const torch::PackedTensorAccessor32<scalar_t, 2> l_enter_acc,
    const int64_t l_enter_start_offset,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> Rs_acc,  // Rs_prepack
    const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,  // *mont_prepack
    const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc) {
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

  auto out_acc = out.packed_accessor32<scalar_t, 2>();
  const auto state_acc = state.packed_accessor32<scalar_t, 2>();
  const auto l_enter_acc = l_enter.packed_accessor32<scalar_t, 2>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  const auto Rs_acc = Rs.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();

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
