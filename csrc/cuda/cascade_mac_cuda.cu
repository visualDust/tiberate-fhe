#include "cascade_mac_cuda.h"
#include <cstdio>
#include "extensions.cuh"
#include "mont_scalar_kernel.cuh"

template <typename scalar_t, int K_STEP>
__global__ void mont_mult_sum_many_3d_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,           // [C, N]
    const TensorAcc32Restrict<scalar_t, 3> a_many_acc,  // [K, C, N]
    const TensorAcc32Restrict<scalar_t, 3> b_many_acc,  // [K, C, N]
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc,
    const TensorAcc32Restrict<scalar_t, 1> ql_acc,
    const TensorAcc32Restrict<scalar_t, 1> qh_acc,
    const TensorAcc32Restrict<scalar_t, 1> kl_acc,
    const TensorAcc32Restrict<scalar_t, 1> kh_acc) {
  const int c = blockIdx.x;
  const int n = blockIdx.y * BLOCK_SIZE + threadIdx.x;
  if (n >= a_many_acc.size(2)) return;

  const int K = a_many_acc.size(0);
  const scalar_t _2q = _2q_acc[c];
  const scalar_t ql = ql_acc[c];
  const scalar_t qh = qh_acc[c];
  const scalar_t kl = kl_acc[c];
  const scalar_t kh = kh_acc[c];

  scalar_t acc = 0;
  int k = 0;

  // Unrolled main loop
  for (; k + K_STEP - 1 < K; k += K_STEP) {
#pragma unroll
    for (int offset = 0; offset < K_STEP; ++offset) {
      scalar_t a_val = a_many_acc[k + offset][c][n];
      scalar_t b_val = b_many_acc[k + offset][c][n];
      scalar_t prod =
          mont_mult_scalar_cuda_kernel(a_val, b_val, ql, qh, kl, kh);
      acc = mont_add_scalar_cuda_kernel(acc, prod, _2q);
    }
  }

  // Tail loop for remaining K
  for (; k < K; ++k) {
    scalar_t a_val = a_many_acc[k][c][n];
    scalar_t b_val = b_many_acc[k][c][n];
    scalar_t prod = mont_mult_scalar_cuda_kernel(a_val, b_val, ql, qh, kl, kh);
    acc = mont_add_scalar_cuda_kernel(acc, prod, _2q);
  }

  out_acc[c][n] = acc;
}

template <typename scalar_t>
void mont_mult_sum_many_3d_cuda_typed(torch::Tensor out,
                                      const torch::Tensor a,
                                      const torch::Tensor b,
                                      const torch::Tensor _2q,
                                      const torch::Tensor ql,
                                      const torch::Tensor qh,
                                      const torch::Tensor kl,
                                      const torch::Tensor kh) {
  const int device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const int C = a.size(1);  // out dim 0
  const int N = a.size(2);  // out dim 1

  dim3 dim_block(BLOCK_SIZE);
  dim3 dim_grid(C, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  const auto a_acc = makeAcc32Restrict(a, scalar_t, 3);
  const auto b_acc = makeAcc32Restrict(b, scalar_t, 3);
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  const auto ql_acc = makeAcc32Restrict(ql, scalar_t, 1);
  const auto qh_acc = makeAcc32Restrict(qh, scalar_t, 1);
  const auto kl_acc = makeAcc32Restrict(kl, scalar_t, 1);
  const auto kh_acc = makeAcc32Restrict(kh, scalar_t, 1);

  mont_mult_sum_many_3d_cuda_kernel<scalar_t, 8>  // Unroll factor of 8
      <<<dim_grid, dim_block, 0, stream>>>(
          out_acc, a_acc, b_acc, _2q_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

torch::Tensor mont_mult_sum_many_3d_cuda(const torch::Tensor a,
                                         const torch::Tensor b,
                                         const torch::Tensor _2q,
                                         const torch::Tensor ql,
                                         const torch::Tensor qh,
                                         const torch::Tensor kl,
                                         const torch::Tensor kh) {
  TORCH_CHECK(a.dim() == 3, "Input must be 3D (K, C, N)");
  TORCH_CHECK(_2q.dim() == 1 && _2q.size(0) == a.size(1),
              "_2q must be 1D and match C dimension");

  torch::Tensor out = torch::empty({a.size(1), a.size(2)}, a.options());

  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "mont_add_many_3d_cuda_typed", [&] {
        mont_mult_sum_many_3d_cuda_typed<scalar_t>(
            out, a, b, _2q, ql, qh, kl, kh);
      });

  return out;
}
