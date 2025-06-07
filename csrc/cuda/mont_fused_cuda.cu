#include "mont_fused_cuda.h"
#include <c10/cuda/CUDAStream.h>

#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void mont_add_reduce_2q_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 2> b_acc,
    torch::PackedTensorAccessor32<scalar_t, 2> c_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t a = a_acc[i][j];
  const scalar_t b = b_acc[i][j];
  const scalar_t _2q = _2q_acc[i];
  const scalar_t q = _2q >> one;

  // Add.
  const scalar_t aplusb = a + b;
  const scalar_t c = (aplusb < _2q) ? aplusb : aplusb - _2q;

  // Reduce. bound 2q → q
  c_acc[i][j] = (c < q) ? c : c - q;
}

template <typename scalar_t>
void mont_add_reduce_2q_cuda_typed(const torch::Tensor a,
                                   const torch::Tensor b,
                                   torch::Tensor c,
                                   const torch::Tensor _2q) {
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  auto C = a.size(0);
  auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  // Run the cuda kernel.
  const auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto b_acc = b.packed_accessor32<scalar_t, 2>();
  auto c_acc = c.packed_accessor32<scalar_t, 2>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  mont_add_reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, b_acc, c_acc, _2q_acc);
}

torch::Tensor mont_add_reduce_2q_cuda(const torch::Tensor a,
                                      const torch::Tensor b,
                                      const torch::Tensor _2q) {
  torch::Tensor c = torch::empty_like(a);
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_mont_add_reduce_2q_cuda", ([&] {
        mont_add_reduce_2q_cuda_typed<scalar_t>(a, b, c, _2q);
      }));
  return c;
}
