
#include "ntt_fused_cuda.h"
#include <c10/cuda/CUDAStream.h>

// -------------------------------------------------------------------
// montgomery common kernels
// -------------------------------------------------------------------

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

// -------------------------------------------------------------------
// mont enter + mont reduce
// -------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_reduce_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> Rs_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Masks.
  constexpr scalar_t one = 1;
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  constexpr scalar_t fb_mask = ((one << nbits) - one);
  constexpr scalar_t lb_mask = (one << half_nbits) - one;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t Rs = Rs_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];

  // mont enter
  scalar_t x = mont_mult_scalar_cuda_kernel(a, Rs, ql, qh, kl, kh);

  // mont reduce
  const scalar_t xl = x & lb_mask;
  const scalar_t xh = x >> half_nbits;
  const scalar_t xkb = xh * kl + xl * kh;
  scalar_t s = (xkb << half_nbits) + xl * kl;
  s = s & fb_mask;

  const scalar_t sl = s & lb_mask;
  const scalar_t sh = s >> half_nbits;
  const scalar_t sqb = sh * ql + sl * qh;
  const scalar_t sqbl = sqb & lb_mask;
  const scalar_t sqbh = sqb >> half_nbits;
  scalar_t carry = (x + sl * ql) >> half_nbits;
  carry = (carry + sqbl) >> half_nbits;

  // write back the result
  a_acc[i][j] = sqbh + carry + sh * qh;
}

// -------------------------------------------------------------------
// mont enter + mont reduce + reduce 2q
// -------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_reduce_2q_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> Rs_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Masks.
  constexpr scalar_t one = 1;
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  constexpr scalar_t fb_mask = ((one << nbits) - one);
  constexpr scalar_t lb_mask = (one << half_nbits) - one;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t Rs = Rs_acc[i];
  const scalar_t _2q = _2q_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];
  const scalar_t q = _2q >> 1;

  // mont enter
  scalar_t x = mont_mult_scalar_cuda_kernel(a, Rs, ql, qh, kl, kh);

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
  a_acc[i][j] = x;
}

// -------------------------------------------------------------------
// mont enter + mont reduce + reduce 2q + make signed
// -------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_reduce_2q_make_signed_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> Rs_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Masks.
  constexpr scalar_t one = 1;
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  constexpr scalar_t fb_mask = ((one << nbits) - one);
  constexpr scalar_t lb_mask = (one << half_nbits) - one;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t Rs = Rs_acc[i];
  const scalar_t _2q = _2q_acc[i];
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];
  const scalar_t q = _2q >> one;
  const scalar_t q_half = q >> one;

  // mont enter
  scalar_t x = mont_mult_scalar_cuda_kernel(a, Rs, ql, qh, kl, kh);

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

  // make signed
  x = (x <= q_half) ? x : x - q;

  // write the result
  a_acc[i][j] = x;
}

// -------------------------------------------------------------------
// intt cuda kernel
// -------------------------------------------------------------------

template <typename scalar_t>
__global__ void intt_cuda_kernel(
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
  const scalar_t V = a_acc[i][odd_j];

  const scalar_t UminusV = U + _2q - V;
  const scalar_t O = (UminusV < _2q) ? UminusV : UminusV - _2q;

  const scalar_t W = mont_mult_scalar_cuda_kernel(S, O, ql, qh, kl, kh);
  a_acc[i][odd_j] = W;

  const scalar_t UplusV = U + V;
  a_acc[i][even_j] = (UplusV < _2q) ? UplusV : UplusV - _2q;
}

// -------------------------------------------------------------------
// intt exit
// -------------------------------------------------------------------

template <typename scalar_t>
void intt_exit_cuda_typed(torch::Tensor a,
                          const torch::Tensor even,
                          const torch::Tensor odd,
                          const torch::Tensor psi,
                          const torch::Tensor Ninv,
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

  const auto even_acc = even.packed_accessor32<int, 2>();
  const auto odd_acc = odd.packed_accessor32<int, 2>();
  const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();
  const auto Ninv_acc = Ninv.packed_accessor32<scalar_t, 1>();

  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();

  for (int i = 0; i < logN; ++i) {
    intt_cuda_kernel<scalar_t><<<dim_grid_ntt, dim_block, 0, stream>>>(a_acc,
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

  // Normalize and Exit
  mont_enter_reduce_cuda_kernel<scalar_t>
      <<<dim_grid_enter, dim_block, 0, stream>>>(
          a_acc, Ninv_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

void intt_exit_cuda(torch::Tensor a,
                    const torch::Tensor even,
                    const torch::Tensor odd,
                    const torch::Tensor psi,
                    const torch::Tensor Ninv,
                    const torch::Tensor _2q,
                    const torch::Tensor ql,
                    const torch::Tensor qh,
                    const torch::Tensor kl,
                    const torch::Tensor kh) {
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_intt_exit_cuda", ([&] {
        intt_exit_cuda_typed<scalar_t>(
            a, even, odd, psi, Ninv, _2q, ql, qh, kl, kh);
      }));
}

// ----------------------------------------------------------------------
// intt exit reduce
// -------------------------------------------------------------------

template <typename scalar_t>
void intt_exit_reduce_cuda_typed(torch::Tensor a,
                                 const torch::Tensor even,
                                 const torch::Tensor odd,
                                 const torch::Tensor psi,
                                 const torch::Tensor Ninv,
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

  const auto even_acc = even.packed_accessor32<int, 2>();
  const auto odd_acc = odd.packed_accessor32<int, 2>();
  const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();
  const auto Ninv_acc = Ninv.packed_accessor32<scalar_t, 1>();

  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();

  for (int i = 0; i < logN; ++i) {
    intt_cuda_kernel<scalar_t><<<dim_grid_ntt, dim_block, 0, stream>>>(a_acc,
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

  // Normalize, Exit and Reduce.
  mont_enter_reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid_enter, dim_block, 0, stream>>>(
          a_acc, Ninv_acc, ql_acc, qh_acc, kl_acc, kh_acc, _2q_acc);
}

void intt_exit_reduce_cuda(torch::Tensor a,
                           const torch::Tensor even,
                           const torch::Tensor odd,
                           const torch::Tensor psi,
                           const torch::Tensor Ninv,
                           const torch::Tensor _2q,
                           const torch::Tensor ql,
                           const torch::Tensor qh,
                           const torch::Tensor kl,
                           const torch::Tensor kh) {
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_intt_exit_reduce_cuda", ([&] {
        intt_exit_reduce_cuda_typed<scalar_t>(
            a, even, odd, psi, Ninv, _2q, ql, qh, kl, kh);
      }));
}

// ----------------------------------------------------------------------
// intt exit reduce signed
// ----------------------------------------------------------------------

template <typename scalar_t>
void intt_exit_reduce_signed_cuda_typed(torch::Tensor a,
                                        const torch::Tensor even,
                                        const torch::Tensor odd,
                                        const torch::Tensor psi,
                                        const torch::Tensor Ninv,
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

  const auto even_acc = even.packed_accessor32<int, 2>();
  const auto odd_acc = odd.packed_accessor32<int, 2>();
  const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();
  const auto Ninv_acc = Ninv.packed_accessor32<scalar_t, 1>();

  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();

  for (int i = 0; i < logN; ++i) {
    intt_cuda_kernel<scalar_t><<<dim_grid_ntt, dim_block, 0, stream>>>(a_acc,
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

  // Normalize.
  mont_enter_reduce_2q_make_signed_cuda_kernel<scalar_t>
      <<<dim_grid_enter, dim_block, 0, stream>>>(
          a_acc, Ninv_acc, ql_acc, qh_acc, kl_acc, kh_acc, _2q_acc);
}

void intt_exit_reduce_signed_cuda(torch::Tensor a,
                                  const torch::Tensor even,
                                  const torch::Tensor odd,
                                  const torch::Tensor psi,
                                  const torch::Tensor Ninv,
                                  const torch::Tensor _2q,
                                  const torch::Tensor ql,
                                  const torch::Tensor qh,
                                  const torch::Tensor kl,
                                  const torch::Tensor kh) {
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_intt_exit_reduce_signed_cuda", ([&] {
        intt_exit_reduce_signed_cuda_typed<scalar_t>(
            a, even, odd, psi, Ninv, _2q, ql, qh, kl, kh);
      }));
}
