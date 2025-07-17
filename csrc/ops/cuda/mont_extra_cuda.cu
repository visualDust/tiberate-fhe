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
    const int sp_prime_len) {
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  const int K = input_acc.size(0);
  scalar_t acc = 0;

  // Montgomery constants.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t _2q = const_mem_2q[-_2Q_CONST_IDX * CONST_MEM_REGION_LEN];

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
                                 const int sp_prime_len) {
  auto device_id = input.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  auto C = input.size(1);
  auto N = input.size(2);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  const auto input_acc = makeAcc32Restrict(input, scalar_t, 3);
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);

  mont_add_many_3d_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, input_acc, sp_prime_len);
}

torch::Tensor mont_add_many_3d_cuda(const torch::Tensor input,
                                    const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);

  torch::Tensor out =
      torch::empty({input.size(1), input.size(2)}, input.options());
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      input.scalar_type(), "typed_mont_add_many_3d_cuda_typed", ([&] {
        mont_add_many_3d_cuda_typed<scalar_t>(input, out, prime_len_int);
      }));
  return out;
}

// ------------------------------------------------------------------
// mont_reduce_add_many_3d_cuda_kernel
// ------------------------------------------------------------------

template <typename scalar_t, int K_STEP>
__global__ void mont_reduce_add_many_3d_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,          // [C, N]
    const TensorAcc32Restrict<scalar_t, 3> input_acc,  // [K, C, N]
    const int sp_prime_len) {
  const int c = blockIdx.x;                             // channel index (C)
  const int n = blockIdx.y * BLOCK_SIZE + threadIdx.x;  // data index (N)

  const int K = input_acc.size(0);

  // Montgomery constants.
  const int prime_offset = -gridDim.x - sp_prime_len + c;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t _2q = const_mem_2q[-_2Q_CONST_IDX * CONST_MEM_REGION_LEN];

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
                                        const int sp_prime_len) {
  const int device_id = input.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const int C = input.size(1);  // out dim 0
  const int N = input.size(2);  // out dim 1

  dim3 dim_block(BLOCK_SIZE);
  dim3 dim_grid(C, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  const auto input_acc = makeAcc32Restrict(input, scalar_t, 3);
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);

  mont_reduce_add_many_3d_cuda_kernel<scalar_t, 8>  // dispatch with unroll 8
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, input_acc, sp_prime_len);
}

torch::Tensor mont_reduce_add_many_3d_cuda(const torch::Tensor input,
                                           const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);
  TORCH_CHECK(input.dim() == 3, "Input must be 3D (K, C, N)");

  torch::Tensor out =
      torch::empty({input.size(1), input.size(2)}, input.options());

  AT_DISPATCH_INTEGRAL_TYPES(
      input.scalar_type(), "mont_add_many_3d_cuda_typed", [&] {
        mont_reduce_add_many_3d_cuda_typed<scalar_t>(input, out, prime_len_int);
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
    const int sp_prime_len) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t b = b_acc[i][j];

  // Montgomery constants.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t _2q = const_mem_2q[-_2Q_CONST_IDX * CONST_MEM_REGION_LEN];

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
                                   const int sp_prime_len) {
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
  mont_add_reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, a_acc, b_acc, sp_prime_len);
}

torch::Tensor mont_add_reduce_2q_cuda(const torch::Tensor a,
                                      const torch::Tensor b,
                                      const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);
  torch::Tensor out = torch::empty_like(a);
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_mont_add_reduce_2q_cuda", ([&] {
        mont_add_reduce_2q_cuda_typed<scalar_t>(a, b, out, prime_len_int);
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
    const int sp_prime_len) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t b = b_acc[i][j];

  // Montgomery constants.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t _2q = const_mem_2q[-_2Q_CONST_IDX * CONST_MEM_REGION_LEN];

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
                                   const int sp_prime_len) {
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
  mont_sub_reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, a_acc, b_acc, sp_prime_len);
}

torch::Tensor mont_sub_reduce_2q_cuda(const torch::Tensor a,
                                      const torch::Tensor b,
                                      const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);
  torch::Tensor out = torch::empty_like(a);
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_mont_sub_reduce_2q_cuda", ([&] {
        mont_sub_reduce_2q_cuda_typed<scalar_t>(a, b, out, prime_len_int);
      }));
  return out;
}

// ------------------------------------------------------------------
// mont_enter + reduce 2q
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_scalar_reduce_2q_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,
    const TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> b_acc,
    const int sp_prime_len) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t Rs = b_acc[i];
  // Montgomery constants.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t _2q = const_mem_2q[-_2Q_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];

  // mont_enter
  scalar_t x = mont_mult_scalar_cuda_kernel(a, Rs, ql, qh, kl, kh);

  // Reduce. bound 2q → q
  x = reduce_2q_scalar_cuda_kernel(x, _2q);

  // Write the result.
  out_acc[i][j] = x;
}

template <typename scalar_t>
void mont_enter_scalar_reduce_2q_cuda_typed(torch::Tensor out,
                                            const torch::Tensor a,
                                            const torch::Tensor b,
                                            const int sp_prime_len) {
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
  const auto b_acc = makeAcc32Restrict(b, scalar_t, 1);

  mont_enter_scalar_reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, a_acc, b_acc, sp_prime_len);
}

torch::Tensor mont_enter_scalar_reduce_2q_cuda(torch::Tensor a,
                                               const torch::Tensor Rs,
                                               const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);
  auto out = torch::empty_like(a);
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_mont_enter_scalar_reduce_2q_cuda", ([&] {
        mont_enter_scalar_reduce_2q_cuda_typed<scalar_t>(
            out, a, Rs, prime_len_int);
      }));
  return a;
}
