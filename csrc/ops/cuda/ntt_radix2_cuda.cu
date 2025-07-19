#include "ntt_radix2_cuda.h"
#include "../../extensions.cuh"
#include "constant_mem.cuh"
#include "mont_used_in_ntt.cuh"

//------------------------------------------------------------------
// ntt
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void ntt_radix2_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<int, 2> even_acc,
    const TensorAcc32Restrict<int, 2> odd_acc,
    const TensorAcc32Restrict<scalar_t, 3> psi_acc,
    const int sp_prime_len,
    const int level) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Montgomery constants.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t _2q = const_mem_2q[-_2Q_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];

  // Butterfly.
  const int even_j = even_acc[level][j];
  const int odd_j = odd_acc[level][j];

  const scalar_t U = a_acc[i][even_j];
  const scalar_t S = psi_acc[psi_acc.size(0) + prime_offset][level][j];
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
                           const int sp_prime_len) {
  // Retrieve the device index, then set the corresponding device and stream.
  auto device_id = a.device().index();
  cudaSetDevice(device_id);

  // Use a preallocated pytorch stream.
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  // The problem dimension.
  const auto C = a.size(0) - sp_prime_len;
  const auto logN = even.size(0);
  const auto N = even.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto even_acc = makeAcc32Restrict(even, int, 2);
  const auto odd_acc = makeAcc32Restrict(odd, int, 2);
  const auto psi_acc = makeAcc32Restrict(psi, scalar_t, 3);

  for (int i = 0; i < logN; ++i) {
    ntt_radix2_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
        a_acc, even_acc, odd_acc, psi_acc, sp_prime_len, i);
  }
}

void ntt_radix2_cuda(torch::Tensor a,
                     const torch::Tensor even,
                     const torch::Tensor odd,
                     const torch::Tensor psi,
                     const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_ntt_cuda", ([&] {
                               ntt_radix2_cuda_typed<scalar_t>(
                                   a, even, odd, psi, prime_len_int);
                             }));
}

//------------------------------------------------------------------
// enter_ntt
//------------------------------------------------------------------

template <typename scalar_t>
void enter_ntt_radix2_cuda_typed(torch::Tensor a,
                                 const torch::Tensor even,
                                 const torch::Tensor odd,
                                 const torch::Tensor psi,
                                 const int sp_prime_len) {
  // Retrieve device idx and stream.
  auto device_id = a.device().index();
  cudaSetDevice(device_id);

  // Use a preallocated pytorch stream.
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  // The problem dimension.
  // Be careful. even and odd has half the length of the a.
  const auto C = a.size(0);
  const auto logN = even.size(0);
  const auto N_half = even.size(1);
  const auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid_ntt(C, N_half / BLOCK_SIZE);
  dim3 dim_grid_enter(C, N / BLOCK_SIZE);

  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto even_acc = makeAcc32Restrict(even, int, 2);
  const auto odd_acc = makeAcc32Restrict(odd, int, 2);
  const auto psi_acc = makeAcc32Restrict(psi, scalar_t, 3);

  // enter.
  mont_enter_Rs_cuda_kernel<scalar_t>
      <<<dim_grid_enter, dim_block, 0, stream>>>(a_acc, sp_prime_len);

  // ntt.
  for (int i = 0; i < logN; ++i) {
    ntt_radix2_cuda_kernel<scalar_t><<<dim_grid_ntt, dim_block, 0, stream>>>(
        a_acc, even_acc, odd_acc, psi_acc, sp_prime_len, i);
  }
}

void enter_ntt_radix2_cuda(torch::Tensor a,
                           const torch::Tensor even,
                           const torch::Tensor odd,
                           const torch::Tensor psi,
                           const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_enter_ntt_cuda", ([&] {
                               enter_ntt_radix2_cuda_typed<scalar_t>(
                                   a, even, odd, psi, prime_len_int);
                             }));
}
