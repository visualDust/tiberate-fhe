#include "mont_fused_cuda.h"
#include <ATen/core/TensorAccessor.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

#define BLOCK_SIZE 256

// ------------------------------------------------------------------
// mont_mult_scalar_cuda_kernel
// ------------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ scalar_t
mont_mult_scalar_cuda_kernel(const scalar_t a,
                             const scalar_t b,
                             const scalar_t ql,
                             const scalar_t qh,
                             const scalar_t kl,
                             const scalar_t kh) {
  // Masks.
  constexpr scalar_t one = 1;
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  constexpr scalar_t fb_mask = ((one << nbits) - one);
  constexpr scalar_t lb_mask = (one << half_nbits) - one;

  const scalar_t al = a & lb_mask;
  const scalar_t ah = a >> half_nbits;
  const scalar_t bl = b & lb_mask;
  const scalar_t bh = b >> half_nbits;

  const scalar_t alpha = ah * bh;
  const scalar_t beta = ah * bl + al * bh;
  const scalar_t gamma = al * bl;

  // s = xk mod R
  const scalar_t gammal = gamma & lb_mask;
  const scalar_t gammah = gamma >> half_nbits;
  const scalar_t betal = beta & lb_mask;
  const scalar_t betah = beta >> half_nbits;

  scalar_t upper = gammal * kh;
  upper = upper + (gammah + betal) * kl;
  upper = upper << half_nbits;
  scalar_t s = upper + gammal * kl;
  s = upper + gammal * kl;
  s = s & fb_mask;

  // t = x + sq
  // u = t/R
  const scalar_t sl = s & lb_mask;
  const scalar_t sh = s >> half_nbits;
  const scalar_t sqb = sh * ql + sl * qh;
  const scalar_t sqbl = sqb & lb_mask;
  const scalar_t sqbh = sqb >> half_nbits;

  scalar_t carry = (gamma + sl * ql) >> half_nbits;
  carry = (carry + betal + sqbl) >> half_nbits;

  return alpha + betah + sqbh + carry + sh * qh;
}

// ------------------------------------------------------------------
// mont_add_reduce_2q_cuda_kernel
// ------------------------------------------------------------------

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
  const scalar_t q = _2q >> one;

  // mont mult
  scalar_t x = mont_mult_scalar_cuda_kernel(ct_in, Rs, ql, qh, kl, kh);

  // mont add
  x = x + pt_in;
  x = (x < _2q) ? x : x - _2q;

  // mont reduce
  // s= xk mod R
  const scalar_t xl = x & lb_mask;
  const scalar_t xh = x >> half_nbits;
  const scalar_t xkb = xh * kl + xl * kh;
  scalar_t s = (xkb << half_nbits) + xl * kl;
  s = s & fb_mask;

  // t = x + sq
  // u = t/R
  // Note that x gets erased in t/R operation if x < R.
  const scalar_t sl = s & lb_mask;
  const scalar_t sh = s >> half_nbits;
  const scalar_t sqb = sh * ql + sl * qh;
  const scalar_t sqbl = sqb & lb_mask;
  const scalar_t sqbh = sqb >> half_nbits;
  scalar_t carry = (x + sl * ql) >> half_nbits;
  carry = (carry + sqbl) >> half_nbits;

  x = sqbh + carry + sh * qh;

  // reduce 2q, bound 2q → q
  x = (x < q) ? x : x - q;

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
  const scalar_t q = _2q >> one;
  // in python, its rounder = torch.where(rescaler > round_at, 1, 0)
  const scalar_t resclr = rescaler[j];
  const scalar_t rounder = (resclr > round_at) ? 1 : 0;

  // data0 = [(d - s) for d, s in zip(data0, rescaler0)]
  scalar_t x = a - resclr;
  // mont_enter
  x = mont_mult_scalar_cuda_kernel(x, b, ql, qh, kl, kh);
  // data0 = [(d + r) for d, r in zip(data0, rounder0)]
  x = x + rounder;
  // reduce 2q, bound 2q → q
  x = (x < q) ? x : x - q;
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
  const scalar_t q = _2q >> one;
  // in python, its rounder = torch.where(rescaler > round_at, 1, 0)
  const scalar_t resclr = rescaler[j];

  // data0 = [(d - s) for d, s in zip(data0, rescaler0)]
  scalar_t x = a - resclr;
  // mont_enter
  x = mont_mult_scalar_cuda_kernel(x, b, ql, qh, kl, kh);
  // data0 = [(d + r) for d, r in zip(data0, rounder0)]
  // reduce 2q, bound 2q → q
  x = (x < q) ? x : x - q;
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
