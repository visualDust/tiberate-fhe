#include "ntt_radix2_cuda.h"
#include "../../extensions.cuh"
#include "mont_used_in_ntt.cuh"

//------------------------------------------------------------------
// ntt
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void ntt_radix2_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<int, 2> even_acc,
    const torch::PackedTensorAccessor32<int, 2> odd_acc,
    const torch::PackedTensorAccessor32<scalar_t, 3> psi_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc,
    const int level) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Montgomery inputs.
  const scalar_t _2q = _2q_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];

  // Butterfly.
  const int even_j = even_acc[level][j];
  const int odd_j = odd_acc[level][j];

  const scalar_t U = a_acc[i][even_j];
  const scalar_t S = psi_acc[i][level][j];
  const scalar_t O = a_acc[i][odd_j];
  const scalar_t V = mont_mult_scalar_cuda_kernel(S, O, ql, qh, kl, kh);

  // Store back.
  const scalar_t UplusV = U + V;
  const scalar_t UminusV = U + _2q - V;

  a_acc[i][even_j] = (UplusV < _2q) ? UplusV : UplusV - _2q;
  a_acc[i][odd_j] = (UminusV < _2q) ? UminusV : UminusV - _2q;
}

template <typename scalar_t>
void ntt_radix2_cuda_typed(torch::Tensor a,
                           const torch::Tensor even,
                           const torch::Tensor odd,
                           const torch::Tensor psi,
                           const torch::Tensor _2q,
                           const torch::Tensor ql,
                           const torch::Tensor qh,
                           const torch::Tensor kl,
                           const torch::Tensor kh) {
  // Retrieve the device index, then set the corresponding device and stream.
  auto device_id = a.device().index();
  cudaSetDevice(device_id);

  // Use a preallocated pytorch stream.
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  // The problem dimension.
  const auto C = ql.size(0);
  const auto logN = even.size(0);
  const auto N = even.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  // Run the cuda kernel.
  auto a_acc = a.packed_accessor32<scalar_t, 2>();

  const auto even_acc = even.packed_accessor32<int, 2>();
  const auto odd_acc = odd.packed_accessor32<int, 2>();
  const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();

  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();

  for (int i = 0; i < logN; ++i) {
    ntt_radix2_cuda_kernel<scalar_t>
        <<<dim_grid, dim_block, 0, stream>>>(a_acc,
                                             even_acc,
                                             odd_acc,
                                             psi_acc,
                                             _2q_acc,
                                             ql_acc,
                                             qh_acc,
                                             kl_acc,
                                             kh_acc,
                                             i);
  }
}

void ntt_radix2_cuda(torch::Tensor a,
                     const torch::Tensor even,
                     const torch::Tensor odd,
                     const torch::Tensor psi,
                     const torch::Tensor _2q,
                     const torch::Tensor ql,
                     const torch::Tensor qh,
                     const torch::Tensor kl,
                     const torch::Tensor kh) {
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_ntt_cuda", ([&] {
                               ntt_radix2_cuda_typed<scalar_t>(
                                   a, even, odd, psi, _2q, ql, qh, kl, kh);
                             }));
}

//------------------------------------------------------------------
// enter_ntt
//------------------------------------------------------------------

template <typename scalar_t>
void enter_ntt_radix2_cuda_typed(torch::Tensor a,
                                 const torch::Tensor Rs,
                                 const torch::Tensor even,
                                 const torch::Tensor odd,
                                 const torch::Tensor psi,
                                 const torch::Tensor _2q,
                                 const torch::Tensor ql,
                                 const torch::Tensor qh,
                                 const torch::Tensor kl,
                                 const torch::Tensor kh) {
  // Retrieve the device index, then set the corresponding device and stream.
  auto device_id = a.device().index();
  cudaSetDevice(device_id);

  // Use a preallocated pytorch stream.
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  // The problem dimension.
  // Be careful. even and odd has half the length of the a.
  const auto C = ql.size(0);
  const auto logN = even.size(0);
  const auto N_half = even.size(1);
  const auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid_ntt(C, N_half / BLOCK_SIZE);
  dim3 dim_grid_enter(C, N / BLOCK_SIZE);

  // Run the cuda kernel.
  auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto Rs_acc = Rs.packed_accessor32<scalar_t, 1>();

  const auto even_acc = even.packed_accessor32<int, 2>();
  const auto odd_acc = odd.packed_accessor32<int, 2>();
  const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();

  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();

  // enter.
  mont_enter_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(
      a_acc, Rs_acc, ql_acc, qh_acc, kl_acc, kh_acc);

  // ntt.
  for (int i = 0; i < logN; ++i) {
    ntt_radix2_cuda_kernel<scalar_t>
        <<<dim_grid_ntt, dim_block, 0, stream>>>(a_acc,
                                                 even_acc,
                                                 odd_acc,
                                                 psi_acc,
                                                 _2q_acc,
                                                 ql_acc,
                                                 qh_acc,
                                                 kl_acc,
                                                 kh_acc,
                                                 i);
  }
}

void enter_ntt_radix2_cuda(torch::Tensor a,
                           const torch::Tensor Rs,
                           const torch::Tensor even,
                           const torch::Tensor odd,
                           const torch::Tensor psi,
                           const torch::Tensor _2q,
                           const torch::Tensor ql,
                           const torch::Tensor qh,
                           const torch::Tensor kl,
                           const torch::Tensor kh) {
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_enter_ntt_cuda", ([&] {
                               enter_ntt_radix2_cuda_typed<scalar_t>(
                                   a, Rs, even, odd, psi, _2q, ql, qh, kl, kh);
                             }));
}
