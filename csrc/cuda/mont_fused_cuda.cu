#include "mont_fused_cuda.h"
#include <cstdint>
#include "extensions.cuh"
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

template <typename scalar_t>
__global__ void mont_add_many_3d_reduce_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,          // 输出张量 [C, N]
    const TensorAcc32Restrict<scalar_t, 3> input_acc,  // 输入张量 [K, C, N]
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  // shared memory for block-level reduction
  extern __shared__ scalar_t sdata[];

  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y;

  const int tid = threadIdx.x;
  const int K = input_acc.size(0);
  const scalar_t _2q = _2q_acc[i];

  // 1. Load the first element from global memory to shared memory
  // Looping grid-stride over K dimension
  scalar_t acc = 0;  // 蒙哥马利加法的单位元是 0
  for (int k = tid; k < K; k += blockDim.x) {
    acc = input_acc[k][i][j];
  }
  sdata[tid] = acc;

  // 2. Do the reduction in shared memory
  // Halfing the number of active threads in each iteration
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s) {
      sdata[tid] = mont_add_scalar_cuda_kernel(sdata[tid], sdata[tid + s], _2q);
    }
  }

  // 3. Write the result for this block to global memory
  if (tid == 0) {
    out_acc[i][j] = sdata[0];
  }
}

template <typename scalar_t>
void mont_add_many_3d_reduce_cuda_typed(const torch::Tensor input,
                                        torch::Tensor out,
                                        const torch::Tensor _2q) {
  auto device_id = input.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  auto C = input.size(1);
  auto N = input.size(2);

  // 1. 定义块大小
  // 归约操作通常使用 128, 256 或 512。必须是 2 的幂次。

  // 2. 定义网格大小
  // 我们需要 C * N 个线程块，每个块负责一个输出元素
  dim3 dim_grid(C, N);
  dim3 dim_block(BLOCK_SIZE);

  // 3. 计算所需的共享内存大小
  // 每个线程在归约中需要一个位置
  size_t shared_mem_size = BLOCK_SIZE * sizeof(scalar_t);

  const auto input_acc = makeAcc32Restrict(input, scalar_t, 3);
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);

  // 4. 启动新的归约核函数
  // 注意：我们传入了4个参数到 <<<...>>>
  // 网格维度, 块维度, 动态共享内存大小, CUDA流
  mont_add_many_3d_reduce_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, shared_mem_size, stream>>>(
          out_acc, input_acc, _2q_acc);
}

torch::Tensor mont_add_many_3d_reduce_cuda(const torch::Tensor input,
                                           const torch::Tensor _2q) {
  torch::Tensor out =
      torch::empty({input.size(1), input.size(2)}, input.options());
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      input.scalar_type(), "typed_mont_add_many_3d_reduce_cuda", ([&] {
        mont_add_many_3d_cuda_typed<scalar_t>(input, out, _2q);
      }));
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
