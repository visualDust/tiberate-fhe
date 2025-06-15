#include "mont_fused_cuda.h"
#include <ATen/core/TensorAccessor.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "mont_common.cuh"

#define BLOCK_SIZE 256

// ------------------------------------------------------------------
// mont_add_reduce_2q_cuda_kernel
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_add_reduce_2q_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 2> b_acc,
    torch::PackedTensorAccessor32<scalar_t, 2> out_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
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
  const auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto b_acc = b.packed_accessor32<scalar_t, 2>();
  auto out_acc = out.packed_accessor32<scalar_t, 2>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  mont_add_reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, b_acc, out_acc, _2q_acc);
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
    const torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 2> b_acc,
    torch::PackedTensorAccessor32<scalar_t, 2> out_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
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
  const auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto b_acc = b.packed_accessor32<scalar_t, 2>();
  auto out_acc = out.packed_accessor32<scalar_t, 2>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  mont_sub_reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, b_acc, out_acc, _2q_acc);
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
// pc_add_fused_cuda_kernel
// ------------------------------------------------------------------

template <typename scalar_t>
__global__ void pc_add_fused_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> ct_acc,
    torch::PackedTensorAccessor32<scalar_t, 2> pt_acc,
    torch::PackedTensorAccessor32<scalar_t, 2> out_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> Rs_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc) {
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
  auto ct_acc = ct_data.packed_accessor32<scalar_t, 2>();
  auto pt_acc = pt_data.packed_accessor32<scalar_t, 2>();
  auto out_acc = out.packed_accessor32<scalar_t, 2>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  const auto Rs_acc = Rs.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();

  pc_add_fused_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
      ct_acc, pt_acc, out_acc, _2q_acc, Rs_acc, ql_acc, qh_acc, kl_acc, kh_acc);
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
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> Rs_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> rescaler,  // rescaler0
    const int64_t round_at,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc) {
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
  auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto Rs_acc = Rs.packed_accessor32<scalar_t, 1>();
  const auto rescaler_acc = rescaler.packed_accessor32<scalar_t, 1>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();

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
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> Rs_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> rescaler,  // rescaler0
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc) {
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
  auto a_acc = a.packed_accessor32<scalar_t, 2>();
  const auto Rs_acc = Rs.packed_accessor32<scalar_t, 1>();
  const auto rescaler_acc = rescaler.packed_accessor32<scalar_t, 1>();
  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();

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
// codec.rotate
// ------------------------------------------------------------------

// template <typename scalar_t>
// __global__ void codec_rotate_make_unsigned_reduce_2q_kernel(
//     const torch::PackedTensorAccessor32<scalar_t, 2> in_acc,
//     torch::PackedTensorAccessor32<scalar_t, 2> out_acc,
//     const torch::PackedTensorAccessor32<scalar_t, 1> perm_acc,
//     const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
//   const int i = blockIdx.x;                             // batch index
//   const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;  // within-row index

//   // if (j >= in_acc.size(1)) {
//   //   printf("j >= in_acc.size(1) %d >= %d\n", j, in_acc.size(1));  // debug
//   //   return;
//   // }

//   const scalar_t in_val = in_acc[i][j];
//   const scalar_t p = perm_acc[j];  // permutation index
//   const scalar_t folded_j = p % in_acc.size(1);
//   const scalar_t sign = (p / in_acc.size(1)) % 2 == 0 ? 1 : -1;
//   const scalar_t q = _2q_acc[i] >> 1;

//   scalar_t rotated = sign * in_val;

//   // Make unsigned
//   rotated += q;

//   // Reduce to q
//   rotated = (rotated < q) ? rotated : rotated - q;

//   // Store result
//   out_acc[i][folded_j] = rotated;
// }

// template <typename scalar_t>
// void codec_rotate_make_unsigned_reduce_2q_cuda_typed(const torch::Tensor in,
//                                                      torch::Tensor out,
//                                                      const torch::Tensor
//                                                      perm, const
//                                                      torch::Tensor _2q) {
//   auto device_id = in.device().index();
//   cudaSetDevice(device_id);
//   auto stream = at::cuda::getCurrentCUDAStream(device_id);

//   const int C = in.size(0);
//   const int N = in.size(1);

//   int dim_block = BLOCK_SIZE;
//   dim3 dim_grid(C, N / BLOCK_SIZE);

//   auto out_acc = out.packed_accessor32<scalar_t, 2>();
//   auto in_acc = in.packed_accessor32<scalar_t, 2>();
//   auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
//   auto perm_acc = perm.packed_accessor32<scalar_t, 1>();

//   codec_rotate_make_unsigned_reduce_2q_kernel<scalar_t>
//       <<<dim_grid, dim_block, 0, stream>>>(in_acc, out_acc, perm_acc,
//       _2q_acc);
// }

// torch::Tensor codec_rotate_make_unsigned_reduce_2q_cuda(
//     const torch::Tensor in, const torch::Tensor perm, const torch::Tensor
//     _2q) {
//   torch::Tensor out = torch::empty_like(in);

//   AT_DISPATCH_INTEGRAL_TYPES(
//       in.scalar_type(), "typed_codec_rotate_unsigned_reduce_2q", ([&] {
//         codec_rotate_make_unsigned_reduce_2q_cuda_typed<scalar_t>(
//             in, out, perm, _2q);
//       }));

//   return out;
// }
