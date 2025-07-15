#include "mont_extra_cuda.h"
#include "../../extensions.cuh"
#include "constant_mem.cuh"
#include "mont_scalar_kernel.cuh"

// ------------------------------------------------------------------
// mont_add_many_cuda_kernel, it adds up on 1st dimension of a 3D tensor
// Input: [K, C, N] → Output: [C, N]
// this is for K is small that no need to reduce
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_add_many_3d_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,          // [C, N]
    const TensorAcc32Restrict<scalar_t, 3> input_acc,  // [K, C, N]
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  const int K = input_acc.size(0);
  scalar_t acc = 0;
  scalar_t _2q = _2q_acc[i];

  for (int k = 0; k < K / 2; ++k) {
    acc = mont_add_scalar_cuda_kernel(
        acc,
        mont_add_scalar_cuda_kernel(
            input_acc[k * 2][i][j], input_acc[k * 2 + 1][i][j], _2q),
        _2q);
  }

  if (K % 2 == 1) {
    acc = mont_add_scalar_cuda_kernel(acc, input_acc[K - 1][i][j], _2q);
  }

  out_acc[i][j] = acc;
}

template <typename scalar_t>
void mont_add_many_3d_cuda_typed(const torch::Tensor input,
                                 torch::Tensor out,
                                 const torch::Tensor _2q) {
  auto device_id = input.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  auto C = input.size(1);
  auto N = input.size(2);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  const auto input_acc = makeAcc32Restrict(input, scalar_t, 3);
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);

  mont_add_many_3d_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, input_acc, _2q_acc);
}

torch::Tensor mont_add_many_3d_cuda(const torch::Tensor input,
                                    const torch::Tensor _2q) {
  torch::Tensor out =
      torch::empty({input.size(1), input.size(2)}, input.options());
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      input.scalar_type(), "typed_mont_add_many_3d_cuda_typed", ([&] {
        mont_add_many_3d_cuda_typed<scalar_t>(input, out, _2q);
      }));
  return out;
}

// ------------------------------------------------------------------
// mont_add_many_3d_reduce_cuda_kernel
// ------------------------------------------------------------------

template <typename scalar_t, int K_STEP>
__global__ void mont_reduce_add_many_3d_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,          // [C, N]
    const TensorAcc32Restrict<scalar_t, 3> input_acc,  // [K, C, N]
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  const int c = blockIdx.x;                             // channel index (C)
  const int n = blockIdx.y * BLOCK_SIZE + threadIdx.x;  // data index (N)

  const int K = input_acc.size(0);
  const scalar_t _2q = _2q_acc[c];

  scalar_t acc = 0;
  int k = 0;

  // Unrolled main loop
  for (; k + K_STEP - 1 < K; k += K_STEP) {
#pragma unroll
    for (int offset = 0; offset < K_STEP; ++offset) {
      scalar_t val = input_acc[k + offset][c][n];
      acc = mont_add_scalar_cuda_kernel(acc, val, _2q);
    }
  }

  // Tail loop
  for (; k < K; ++k) {
    scalar_t val = input_acc[k][c][n];
    acc = mont_add_scalar_cuda_kernel(acc, val, _2q);
  }

  out_acc[c][n] = acc;
}

template <typename scalar_t>
void mont_reduce_add_many_3d_cuda_typed(const torch::Tensor input,
                                        torch::Tensor out,
                                        const torch::Tensor _2q) {
  const int device_id = input.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const int C = input.size(1);  // out dim 0
  const int N = input.size(2);  // out dim 1

  dim3 dim_block(BLOCK_SIZE);
  dim3 dim_grid(C, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  const auto input_acc = makeAcc32Restrict(input, scalar_t, 3);
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);

  mont_reduce_add_many_3d_cuda_kernel<scalar_t, 8>  // dispatch with unroll 8
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, input_acc, _2q_acc);
}

torch::Tensor mont_reduce_add_many_3d_cuda(const torch::Tensor input,
                                           const torch::Tensor _2q) {
  TORCH_CHECK(input.dim() == 3, "Input must be 3D (K, C, N)");
  TORCH_CHECK(_2q.dim() == 1 && _2q.size(0) == input.size(1),
              "_2q must be 1D and match C dimension");

  torch::Tensor out =
      torch::empty({input.size(1), input.size(2)}, input.options());

  AT_DISPATCH_INTEGRAL_TYPES(
      input.scalar_type(), "mont_add_many_3d_cuda_typed", [&] {
        mont_reduce_add_many_3d_cuda_typed<scalar_t>(input, out, _2q);
      });

  return out;
}

// ------------------------------------------------------------------
// mont_add_reduce_2q_cuda_kernel
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_add_reduce_2q_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,
    const TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 2> b_acc,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t b = b_acc[i][j];
  const scalar_t _2q = _2q_acc[i];

  // Add.
  scalar_t x = mont_add_scalar_cuda_kernel(a, b, _2q);

  // Reduce. bound 2q → q
  x = reduce_2q_scalar_cuda_kernel(x, _2q);

  // Write the result.
  out_acc[i][j] = x;
}

template <typename scalar_t>
void mont_add_reduce_2q_cuda_typed(const torch::Tensor a,
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
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto b_acc = makeAcc32Restrict(b, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  mont_add_reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, a_acc, b_acc, _2q_acc);
}

torch::Tensor mont_add_reduce_2q_cuda(const torch::Tensor a,
                                      const torch::Tensor b,
                                      const torch::Tensor _2q) {
  torch::Tensor out = torch::empty_like(a);
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_mont_add_reduce_2q_cuda", ([&] {
        mont_add_reduce_2q_cuda_typed<scalar_t>(a, b, out, _2q);
      }));
  return out;
}

// ------------------------------------------------------------------
// mont_sub_reduce_2q_cuda_kernel
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_sub_reduce_2q_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,
    const TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 2> b_acc,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t b = b_acc[i][j];
  const scalar_t _2q = _2q_acc[i];

  // Subtract.
  scalar_t x = mont_sub_scalar_cuda_kernel(a, b, _2q);

  // Reduce. bound 2q → q
  x = reduce_2q_scalar_cuda_kernel(x, _2q);

  // Write the result.
  out_acc[i][j] = x;
}

template <typename scalar_t>
void mont_sub_reduce_2q_cuda_typed(const torch::Tensor a,
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
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto b_acc = makeAcc32Restrict(b, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  mont_sub_reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, a_acc, b_acc, _2q_acc);
}

torch::Tensor mont_sub_reduce_2q_cuda(const torch::Tensor a,
                                      const torch::Tensor b,
                                      const torch::Tensor _2q) {
  torch::Tensor out = torch::empty_like(a);
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_mont_sub_reduce_2q_cuda", ([&] {
        mont_sub_reduce_2q_cuda_typed<scalar_t>(a, b, out, _2q);
      }));
  return out;
}

// ------------------------------------------------------------------
// mont_enter + reduce 2q
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_reduce_2q_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,
    const TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> Rs_acc,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc,
    const TensorAcc32Restrict<scalar_t, 1> ql_acc,
    const TensorAcc32Restrict<scalar_t, 1> qh_acc,
    const TensorAcc32Restrict<scalar_t, 1> kl_acc,
    const TensorAcc32Restrict<scalar_t, 1> kh_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t Rs = Rs_acc[i];
  const scalar_t _2q = _2q_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];

  // mont_enter
  scalar_t x = mont_mult_scalar_cuda_kernel(a, Rs, ql, qh, kl, kh);

  // Reduce. bound 2q → q
  x = reduce_2q_scalar_cuda_kernel(x, _2q);

  // Write the result.
  out_acc[i][j] = x;
}

template <typename scalar_t>
void mont_enter_reduce_2q_cuda_typed(torch::Tensor out,
                                     const torch::Tensor a,
                                     const torch::Tensor Rs,
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
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto Rs_acc = makeAcc32Restrict(Rs, scalar_t, 1);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  const auto ql_acc = makeAcc32Restrict(ql, scalar_t, 1);
  const auto qh_acc = makeAcc32Restrict(qh, scalar_t, 1);
  const auto kl_acc = makeAcc32Restrict(kl, scalar_t, 1);
  const auto kh_acc = makeAcc32Restrict(kh, scalar_t, 1);

  mont_enter_reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(
          out_acc, a_acc, Rs_acc, _2q_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

torch::Tensor mont_enter_reduce_2q_cuda(torch::Tensor a,
                                        const torch::Tensor Rs,
                                        const torch::Tensor _2q,
                                        const torch::Tensor ql,
                                        const torch::Tensor qh,
                                        const torch::Tensor kl,
                                        const torch::Tensor kh) {
  auto out = torch::empty_like(a);
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_mont_enter_reduce_2q_cuda", ([&] {
        mont_enter_reduce_2q_cuda_typed<scalar_t>(
            out, a, Rs, _2q, ql, qh, kl, kh);
      }));
  return a;
}
