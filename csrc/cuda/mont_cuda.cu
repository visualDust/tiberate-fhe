#include "mont_cuda.h"
#include <c10/cuda/CUDAStream.h>
#include "mont_cuda_kernel.cuh"

//------------------------------------------------------------------
// mont_mult
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_mult_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 2> b_acc,
    torch::PackedTensorAccessor32<scalar_t, 2> out_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t b = b_acc[i][j];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];

  // Store the result.
  out_acc[i][j] = mont_mult_scalar_cuda_kernel(a, b, ql, qh, kl, kh);
}

template <typename scalar_t>
void mont_mult_cuda_typed(const torch::Tensor a,
                          const torch::Tensor b,
                          torch::Tensor out,
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
  auto C = a.size(0);
  auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  // Run the cuda kernel.
  const auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto b_acc = b.packed_accessor32<scalar_t, 2>();
  auto out_acc = out.packed_accessor32<scalar_t, 2>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
  mont_mult_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
      a_acc, b_acc, out_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

torch::Tensor mont_mult_cuda(const torch::Tensor a,
                             const torch::Tensor b,
                             const torch::Tensor ql,
                             const torch::Tensor qh,
                             const torch::Tensor kl,
                             const torch::Tensor kh) {
  // Prepare the output.
  torch::Tensor out = torch::empty_like(a);

  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_mont_mult_cuda", ([&] {
                               mont_mult_cuda_typed<scalar_t>(
                                   a, b, out, ql, qh, kl, kh);
                             }));

  return out;
}

//------------------------------------------------------------------
// mont_enter
//------------------------------------------------------------------

template <typename scalar_t>
void mont_enter_cuda_typed(torch::Tensor a,
                           const torch::Tensor Rs,
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
  auto C = a.size(0);
  auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  // Run the cuda kernel.
  auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto Rs_acc = Rs.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
  mont_enter_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
      a_acc, Rs_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

void mont_enter_cuda(torch::Tensor a,
                     const torch::Tensor Rs,
                     const torch::Tensor ql,
                     const torch::Tensor qh,
                     const torch::Tensor kl,
                     const torch::Tensor kh) {
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_mont_enter_cuda", ([&] {
                               mont_enter_cuda_typed<scalar_t>(
                                   a, Rs, ql, qh, kl, kh);
                             }));
}

//------------------------------------------------------------------
// reduce 2q
//------------------------------------------------------------------

template <typename scalar_t>
void reduce_2q_cuda_typed(torch::Tensor a, const torch::Tensor _2q) {
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const auto C = a.size(0);
  const auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();

  reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, _2q_acc);
}

void reduce_2q_cuda(torch::Tensor a, const torch::Tensor _2q) {
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_reduce_2q_cuda", ([&] {
                               reduce_2q_cuda_typed<scalar_t>(a, _2q);
                             }));
}

//------------------------------------------------------------------
// mont reduce
//------------------------------------------------------------------

template <typename scalar_t>
void mont_reduce_cuda_typed(torch::Tensor a,
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
  auto C = a.size(0);
  auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  // Run the cuda kernel.
  auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
  mont_reduce_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
      a_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

void mont_reduce_cuda(torch::Tensor a,
                      const torch::Tensor ql,
                      const torch::Tensor qh,
                      const torch::Tensor kl,
                      const torch::Tensor kh) {
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_mont_reduce_cuda", ([&] {
                               mont_reduce_cuda_typed<scalar_t>(
                                   a, ql, qh, kl, kh);
                             }));
}

template <typename scalar_t>
__global__ void mont_add_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 2> b_acc,
    torch::PackedTensorAccessor32<scalar_t, 2> out_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t a = a_acc[i][j];
  const scalar_t b = b_acc[i][j];
  const scalar_t _2q = _2q_acc[i];

  // Add.
  const scalar_t aplusb = a + b;
  out_acc[i][j] = (aplusb < _2q) ? aplusb : aplusb - _2q;
}

template <typename scalar_t>
__global__ void mont_sub_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 2> b_acc,
    torch::PackedTensorAccessor32<scalar_t, 2> out_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t a = a_acc[i][j];
  const scalar_t b = b_acc[i][j];
  const scalar_t _2q = _2q_acc[i];

  // Sub.
  const scalar_t aminusb = a + _2q - b;
  out_acc[i][j] = (aminusb < _2q) ? aminusb : aminusb - _2q;
}

template <typename scalar_t>
void mont_add_cuda_typed(const torch::Tensor a,
                         const torch::Tensor b,
                         torch::Tensor out,
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
  auto out_acc = out.packed_accessor32<scalar_t, 2>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  mont_add_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, b_acc, out_acc, _2q_acc);
}

template <typename scalar_t>
void mont_sub_cuda_typed(const torch::Tensor a,
                         const torch::Tensor b,
                         torch::Tensor out,
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
  auto out_acc = out.packed_accessor32<scalar_t, 2>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  mont_sub_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, b_acc, out_acc, _2q_acc);
}

torch::Tensor mont_add_cuda(const torch::Tensor a,
                            const torch::Tensor b,
                            const torch::Tensor _2q) {
  torch::Tensor out = torch::empty_like(a);
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_mont_add_cuda", ([&] {
                               mont_add_cuda_typed<scalar_t>(a, b, out, _2q);
                             }));
  return out;
}

torch::Tensor mont_sub_cuda(const torch::Tensor a,
                            const torch::Tensor b,
                            const torch::Tensor _2q) {
  torch::Tensor out = torch::empty_like(a);
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_mont_sub_cuda", ([&] {
                               mont_sub_cuda_typed<scalar_t>(a, b, out, _2q);
                             }));
  return out;
}

//------------------------------------------------------------------
// Misc
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void make_unsigned_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t q = _2q_acc[i] >> one;

  // Make unsigned.
  a_acc[i][j] += q;
}

template <typename scalar_t>
__global__ void tile_unsigned_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1> a_acc,
    torch::PackedTensorAccessor32<scalar_t, 2> dst_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t q = _2q_acc[i] >> one;
  const scalar_t a = a_acc[j];

  // Make unsigned.
  dst_acc[i][j] = a + q;
}

template <typename scalar_t>
void make_signed_cuda_typed(torch::Tensor a, const torch::Tensor _2q) {
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const auto C = a.size(0);
  const auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();

  make_signed_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, _2q_acc);
}

template <typename scalar_t>
void tile_unsigned_cuda_typed(const torch::Tensor a,
                              torch::Tensor dst,
                              const torch::Tensor _2q) {
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const auto C = _2q.size(0);
  const auto N = a.size(0);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  const auto a_acc = a.packed_accessor32<scalar_t, 1>();
  auto dst_acc = dst.packed_accessor32<scalar_t, 2>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();

  tile_unsigned_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, dst_acc, _2q_acc);
}

template <typename scalar_t>
void make_unsigned_cuda_typed(torch::Tensor a, const torch::Tensor _2q) {
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const auto C = a.size(0);
  const auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();

  make_unsigned_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, _2q_acc);
}

void make_signed_cuda(torch::Tensor a, const torch::Tensor _2q) {
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_make_signed_cuda", ([&] {
                               make_signed_cuda_typed<scalar_t>(a, _2q);
                             }));
}

void make_unsigned_cuda(torch::Tensor a, const torch::Tensor _2q) {
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_make_unsigned_cuda", ([&] {
                               make_unsigned_cuda_typed<scalar_t>(a, _2q);
                             }));
}

torch::Tensor tile_unsigned_cuda(const torch::Tensor a,
                                 const torch::Tensor _2q) {
  a.squeeze_();
  const auto C = _2q.size(0);
  const auto N = a.size(0);
  auto c = a.new_empty({C, N});
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_tile_unsigned_cuda", ([&] {
                               tile_unsigned_cuda_typed<scalar_t>(a, c, _2q);
                             }));
  return c;
}
