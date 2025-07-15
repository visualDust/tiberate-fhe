#include "mont_cuda.h"
#include <c10/cuda/CUDAStream.h>
#include "../../extensions.cuh"
#include "constant_mem.cuh"
#include "mont_scalar_kernel.cuh"

//------------------------------------------------------------------
// mont_mult
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_mult_cuda_kernel(
    const TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 2> b_acc,
    TensorAcc32Restrict<scalar_t, 2> out_acc,
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
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];

  // Store the result.
  out_acc[i][j] = mont_mult_scalar_cuda_kernel(a, b, ql, qh, kl, kh);
}

template <typename scalar_t>
void mont_mult_cuda_typed(const torch::Tensor a,
                          const torch::Tensor b,
                          torch::Tensor out,
                          const int sp_prime_len) {
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
  const auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto b_acc = makeAcc32Restrict(b, scalar_t, 2);
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  mont_mult_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, b_acc, out_acc, sp_prime_len);
}

torch::Tensor mont_mult_cuda(const torch::Tensor a,
                             const torch::Tensor b,
                             const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);
  // Prepare the output.
  torch::Tensor out = torch::empty_like(a);

  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_mont_mult_cuda", ([&] {
                               mont_mult_cuda_typed<scalar_t>(
                                   a, b, out, prime_len_int);
                             }));

  return out;
}

//------------------------------------------------------------------
// mont_enter_scalar
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_scalar_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> b_acc,
    const int sp_prime_len) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  scalar_t& a = a_acc[i][j];
  const scalar_t b = b_acc[i];

  // Montgomery constants.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];

  // Store the result.
  a = mont_mult_scalar_cuda_kernel(a, b, ql, qh, kl, kh);
}

template <typename scalar_t>
void mont_enter_scalar_cuda_typed(torch::Tensor a,
                                  const torch::Tensor b,
                                  const int sp_prime_len) {
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
  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto b_acc = makeAcc32Restrict(b, scalar_t, 1);
  mont_enter_scalar_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, b_acc, sp_prime_len);
}

void mont_enter_scalar_cuda(torch::Tensor a,
                            const torch::Tensor b,
                            const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_mont_mult_inplace_cuda", ([&] {
        mont_enter_scalar_cuda_typed<scalar_t>(a, b, prime_len_int);
      }));
}

//------------------------------------------------------------------
// mont enter Rs
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_Rs_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> a_acc, const int sp_prime_len) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];

  // Montgomery constants.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t Rs = const_mem_2q[-RS_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];

  // Store the result.
  a_acc[i][j] = mont_mult_scalar_cuda_kernel(a, Rs, ql, qh, kl, kh);
}

template <typename scalar_t>
void mont_enter_Rs_cuda_typed(torch::Tensor a, int sp_prime_len) {
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
  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);

  mont_enter_Rs_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, sp_prime_len);
}

void mont_enter_Rs_cuda(torch::Tensor a, const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_mont_enter_Rs_cuda", ([&] {
                               mont_enter_Rs_cuda_typed<scalar_t>(
                                   a, prime_len_int);
                             }));
}

//------------------------------------------------------------------
// mont enter Rs scale
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_Rs_scale_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> a_acc, const int sp_prime_len) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];

  // Montgomery constants.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t Rs_scale =
      const_mem_2q[-RS_SCALE_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];

  // Store the result.
  a_acc[i][j] = mont_mult_scalar_cuda_kernel(a, Rs_scale, ql, qh, kl, kh);
}

template <typename scalar_t>
void mont_enter_Rs_scale_cuda_typed(torch::Tensor a, int sp_prime_len) {
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
  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);

  mont_enter_Rs_scale_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, sp_prime_len);
}

void mont_enter_Rs_scale_cuda(torch::Tensor a, const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_mont_enter_Rs_scale_cuda", ([&] {
        mont_enter_Rs_scale_cuda_typed<scalar_t>(a, prime_len_int);
      }));
}

//------------------------------------------------------------------
// mont enter legacy support
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_enter_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> Rs_acc,
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
  const scalar_t ql = ql_acc[i];
  const scalar_t qh = qh_acc[i];
  const scalar_t kl = kl_acc[i];
  const scalar_t kh = kh_acc[i];

  // Store the result.
  a_acc[i][j] = mont_mult_scalar_cuda_kernel(a, Rs, ql, qh, kl, kh);
}

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
  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto Rs_acc = makeAcc32Restrict(Rs, scalar_t, 1);
  const auto ql_acc = makeAcc32Restrict(ql, scalar_t, 1);
  const auto qh_acc = makeAcc32Restrict(qh, scalar_t, 1);
  const auto kl_acc = makeAcc32Restrict(kl, scalar_t, 1);
  const auto kh_acc = makeAcc32Restrict(kh, scalar_t, 1);
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
// mont reduce
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_reduce_cuda_kernel(TensorAcc32Restrict<scalar_t, 2> a_acc,
                                        const int sp_prime_len) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t x = a_acc[i][j];
  // Montgomery constants.
  const int prime_offset = -gridDim.x - sp_prime_len + i;
  const scalar_t* const_mem_2q =
      get_const_ptr_gright<scalar_t>(0, prime_offset);
  const scalar_t ql = const_mem_2q[-QL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t qh = const_mem_2q[-QH_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kl = const_mem_2q[-KL_CONST_IDX * CONST_MEM_REGION_LEN];
  const scalar_t kh = const_mem_2q[-KH_CONST_IDX * CONST_MEM_REGION_LEN];

  a_acc[i][j] = mont_reduce_scalar_cuda_kernel(x, ql, qh, kl, kh);
}

template <typename scalar_t>
void mont_reduce_cuda_typed(torch::Tensor a, const int sp_prime_len) {
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
  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);

  mont_reduce_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, sp_prime_len);
}

void mont_reduce_cuda(torch::Tensor a, const int64_t sp_prime_len) {
  const int prime_len_int = static_cast<int>(sp_prime_len);
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_mont_reduce_cuda", ([&] {
                               mont_reduce_cuda_typed<scalar_t>(a,
                                                                prime_len_int);
                             }));
}

//------------------------------------------------------------------
// reduce 2q
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void reduce_2q_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  // This kernel reduces each element a_acc[i][j] modulo q = _2q_acc[i] / 2,
  // assuming that a < 2q. It's a fast, branchless way to compute a % q under
  // certain assumptions.

  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  const scalar_t a = a_acc[i][j];
  const scalar_t _2q = _2q_acc[i];
  // Reduce. bound 2q → q
  a_acc[i][j] = reduce_2q_scalar_cuda_kernel(a, _2q);
}

template <typename scalar_t>
void reduce_2q_cuda_typed(torch::Tensor a, const torch::Tensor _2q) {
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const auto C = a.size(0);
  const auto N = a.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);

  reduce_2q_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, _2q_acc);
}

void reduce_2q_cuda(torch::Tensor a, const torch::Tensor _2q) {
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_reduce_2q_cuda", ([&] {
                               reduce_2q_cuda_typed<scalar_t>(a, _2q);
                             }));
}

//------------------------------------------------------------------
// mont_add
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_add_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,
    const TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 2> b_acc,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t a = a_acc[i][j];
  const scalar_t b = b_acc[i][j];
  const scalar_t _2q = _2q_acc[i];

  // Add.
  out_acc[i][j] = mont_add_scalar_cuda_kernel(a, b, _2q);
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
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto b_acc = makeAcc32Restrict(b, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  mont_add_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, a_acc, b_acc, _2q_acc);
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

// ------------------------------------------------------------------
// mont_sub
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void mont_sub_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,
    const TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 2> b_acc,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t a = a_acc[i][j];
  const scalar_t b = b_acc[i][j];
  const scalar_t _2q = _2q_acc[i];

  // Sub.
  out_acc[i][j] = mont_sub_scalar_cuda_kernel(a, b, _2q);
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
  auto out_acc = makeAcc32Restrict(out, scalar_t, 2);
  const auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto b_acc = makeAcc32Restrict(b, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);
  mont_sub_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, a_acc, b_acc, _2q_acc);
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
// Make signed
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void make_signed_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t a = a_acc[i][j];
  const scalar_t _2q = _2q_acc[i];

  // Make signed.
  a_acc[i][j] = make_signed_scalar_cuda_kernel(a, _2q);
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

  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);

  make_signed_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, _2q_acc);
}

void make_signed_cuda(torch::Tensor a, const torch::Tensor _2q) {
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_make_signed_cuda", ([&] {
                               make_signed_cuda_typed<scalar_t>(a, _2q);
                             }));
}

//------------------------------------------------------------------
// Make unsigned
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void make_unsigned_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t q = _2q_acc[i] >> one;

  // Make unsigned.
  a_acc[i][j] = make_unsigned_scalar_cuda_kernel(a_acc[i][j], q);
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

  auto a_acc = makeAcc32Restrict(a, scalar_t, 2);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);

  make_unsigned_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(a_acc, _2q_acc);
}

void make_unsigned_cuda(torch::Tensor a, const torch::Tensor _2q) {
  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "typed_make_unsigned_cuda", ([&] {
                               make_unsigned_cuda_typed<scalar_t>(a, _2q);
                             }));
}

//------------------------------------------------------------------
// Tile unsigned
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void tile_unsigned_cuda_kernel(
    TensorAcc32Restrict<scalar_t, 2> out_acc,
    const TensorAcc32Restrict<scalar_t, 1> a_acc,
    const TensorAcc32Restrict<scalar_t, 1> _2q_acc) {
  // Indexing
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Inputs.
  constexpr scalar_t one = 1;
  const scalar_t q = _2q_acc[i] >> one;
  const scalar_t a = a_acc[j];

  // Make unsigned.
  out_acc[i][j] = a + q;
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

  auto out_acc = makeAcc32Restrict(dst, scalar_t, 2);
  const auto a_acc = makeAcc32Restrict(a, scalar_t, 1);
  const auto _2q_acc = makeAcc32Restrict(_2q, scalar_t, 1);

  tile_unsigned_cuda_kernel<scalar_t>
      <<<dim_grid, dim_block, 0, stream>>>(out_acc, a_acc, _2q_acc);
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
