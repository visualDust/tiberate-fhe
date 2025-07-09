#include "ntt_radix2_cuda.h"
#include <cstdint>
#include "../../extensions.cuh"
#include "mont_used_in_ntt.cuh"

// ------------------------------------------------------------------
// constant memory pool
// ------------------------------------------------------------------

__device__ __constant__ uint8_t constant_mem_pool[MAX_CONST_BYTES];

int upload_tensor_list_cuda(const std::vector<torch::Tensor> tensors,
                            const std::vector<int64_t> offsets,
                            const int64_t layout_int64_t,
                            const int64_t device_id_int64_t) {
  auto layout = static_cast<ConstantMemoryGravity>(layout_int64_t);
  auto device_id = static_cast<int>(device_id_int64_t);
  cudaSetDevice(device_id);
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& t = tensors[i];
    TORCH_CHECK(t.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(tensors.size() == offsets.size(),
                "Mismatch: tensor list and offset list must have same length");
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    const size_t size_bytes = t.nbytes();
    size_t effective_offset = offsets[i];

    if (layout == Right) {
      effective_offset = MAX_CONST_BYTES - offsets[i] - size_bytes;
    }

    TORCH_CHECK(effective_offset + size_bytes <= MAX_CONST_BYTES,
                "Upload exceeds constant memory");

    cudaMemcpyToSymbolAsync(constant_mem_pool,
                            t.data_ptr(),
                            size_bytes,
                            effective_offset,
                            cudaMemcpyHostToDevice,
                            stream);
  }
  return 0;
}

template <typename scalar_t>
__global__ void read_chunk_kernel(scalar_t* out_ptr,
                                  const size_t offset_bytes,
                                  const size_t count,
                                  const ConstantMemoryGravity layout) {
  int idx = threadIdx.x;
  if (idx >= count) return;

  size_t effective_offset = offset_bytes;
  if (layout == Right) {
    effective_offset =
        MAX_CONST_BYTES - offset_bytes - count * sizeof(scalar_t);
  }

  const scalar_t* const_mem =
      reinterpret_cast<const scalar_t*>(&constant_mem_pool[effective_offset]);

  // if (idx == 0) {
  //   // Debugging output.
  //   printf("read: constant_mem_pool address: %p : %ld\n",
  //          constant_mem_pool,
  //          const_mem[idx]);
  // }
  out_ptr[idx] = const_mem[idx];
}

torch::Tensor read_constant_chunk_cuda(const int64_t device_id_int64_t,
                                       const int64_t offset_bytes_int64_t,
                                       const int64_t count_int64_t,
                                       const torch::Dtype dtype,
                                       const int64_t layout_int64_t) {
  auto device_id = static_cast<int>(device_id_int64_t);
  auto offset_bytes = static_cast<size_t>(offset_bytes_int64_t);
  auto count = static_cast<size_t>(count_int64_t);
  auto layout = static_cast<ConstantMemoryGravity>(layout_int64_t);
  at::cuda::set_device(device_id);

  auto options =
      torch::TensorOptions().device(torch::kCUDA, device_id).dtype(dtype);
  torch::Tensor out = torch::empty({static_cast<int64_t>(count)}, options);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  AT_DISPATCH_INTEGRAL_TYPES(
      dtype, "read_constant_chunk_cuda", ([&] {
        read_chunk_kernel<scalar_t><<<1, count, 0, stream.stream()>>>(
            out.data_ptr<scalar_t>(), offset_bytes, count, layout);
      }));

  return out;
}

//------------------------------------------------------------------
// ntt
//------------------------------------------------------------------

template <typename scalar_t>
__global__ void ntt_radix2_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
    const torch::PackedTensorAccessor32<int, 2> even_acc,
    const torch::PackedTensorAccessor32<int, 2> odd_acc,
    const torch::PackedTensorAccessor32<scalar_t, 3> psi_acc,
    //     const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc,
    //     const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
    //     const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
    //     const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
    //     const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc,
    const int prime_len,
    const int level) {
  // Where am I?
  const int i = blockIdx.x;
  const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Montgomery inputs.
  const scalar_t* const_mem_2q = reinterpret_cast<const scalar_t*>(
      &constant_mem_pool[MAX_CONST_BYTES -
                         (_2Q_CONST_IDX * CONST_MEM_REGION_LEN) *
                             sizeof(scalar_t)]);
  const int prime_offset = -gridDim.x - prime_len + i;
  const scalar_t _2q = const_mem_2q[prime_offset];
  const scalar_t ql = const_mem_2q[-2 * CONST_MEM_REGION_LEN + prime_offset];
  const scalar_t qh = const_mem_2q[-3 * CONST_MEM_REGION_LEN + prime_offset];
  const scalar_t kl = const_mem_2q[-4 * CONST_MEM_REGION_LEN + prime_offset];
  const scalar_t kh = const_mem_2q[-5 * CONST_MEM_REGION_LEN + prime_offset];

  // Butterfly.
  const int even_j = even_acc[level][j];
  const int odd_j = odd_acc[level][j];

  const scalar_t U = a_acc[i][even_j];
  const scalar_t S = psi_acc[i][level][j];
  const scalar_t O = a_acc[i][odd_j];
  const scalar_t V = mont_mult_scalar_cuda_kernel(S, O, ql, qh, kl, kh);

  // Store back.
  const scalar_t UplusV = U + V;
  const scalar_t UminusV = U + _2q - V;

  a_acc[i][even_j] = (UplusV < _2q) ? UplusV : UplusV - _2q;
  a_acc[i][odd_j] = (UminusV < _2q) ? UminusV : UminusV - _2q;
}

// template <typename scalar_t>
// __global__ void ntt_radix2_cuda_kernel(
//     torch::PackedTensorAccessor32<scalar_t, 2> a_acc,
//     const torch::PackedTensorAccessor32<int, 2> even_acc,
//     const torch::PackedTensorAccessor32<int, 2> odd_acc,
//     const torch::PackedTensorAccessor32<scalar_t, 3> psi_acc,
//     const torch::PackedTensorAccessor32<scalar_t, 1> _2q_acc,
//     const torch::PackedTensorAccessor32<scalar_t, 1> ql_acc,
//     const torch::PackedTensorAccessor32<scalar_t, 1> qh_acc,
//     const torch::PackedTensorAccessor32<scalar_t, 1> kl_acc,
//     const torch::PackedTensorAccessor32<scalar_t, 1> kh_acc,
//     const int prime_len,
//     const int level) {
//   // Where am I?
//   const int i = blockIdx.x;
//   const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

//   // Montgomery inputs.
//   const scalar_t _2q = _2q_acc[i];
//   const scalar_t ql = ql_acc[i];
//   const scalar_t qh = qh_acc[i];
//   const scalar_t kl = kl_acc[i];
//   const scalar_t kh = kh_acc[i];

//   // Butterfly.
//   const int even_j = even_acc[level][j];
//   const int odd_j = odd_acc[level][j];

//   const scalar_t U = a_acc[i][even_j];
//   const scalar_t S = psi_acc[i][level][j];
//   const scalar_t O = a_acc[i][odd_j];
//   const scalar_t V = mont_mult_scalar_cuda_kernel(S, O, ql, qh, kl, kh);

//   // Store back.
//   const scalar_t UplusV = U + V;
//   const scalar_t UminusV = U + _2q - V;

//   a_acc[i][even_j] = (UplusV < _2q) ? UplusV : UplusV - _2q;
//   a_acc[i][odd_j] = (UminusV < _2q) ? UminusV : UminusV - _2q;
// }

template <typename scalar_t>
void ntt_radix2_cuda_typed(torch::Tensor a,
                           const torch::Tensor even,
                           const torch::Tensor odd,
                           const torch::Tensor psi,
                           const torch::Tensor _2q,
                           const torch::Tensor ql,
                           const torch::Tensor qh,
                           const torch::Tensor kl,
                           const torch::Tensor kh,
                           const int prime_len) {
  // Retrieve the device index, then set the corresponding device and stream.
  auto device_id = a.device().index();
  cudaSetDevice(device_id);

  // Use a preallocated pytorch stream.
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  // The problem dimension.
  const auto C = ql.size(0);
  const auto logN = even.size(0);
  const auto N = even.size(1);

  int dim_block = BLOCK_SIZE;
  dim3 dim_grid(C, N / BLOCK_SIZE);

  // Run the cuda kernel.
  auto a_acc = a.packed_accessor32<scalar_t, 2>();

  const auto even_acc = even.packed_accessor32<int, 2>();
  const auto odd_acc = odd.packed_accessor32<int, 2>();
  const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();

  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();

  for (int i = 0; i < logN; ++i) {
    ntt_radix2_cuda_kernel<scalar_t>
        <<<dim_grid, dim_block, 0, stream>>>(a_acc,
                                             even_acc,
                                             odd_acc,
                                             psi_acc,
                                             //  _2q_acc,
                                             //  ql_acc,
                                             //  qh_acc,
                                             //  kl_acc,
                                             //  kh_acc,
                                             prime_len,
                                             i);
  }
}

void ntt_radix2_cuda(torch::Tensor a,
                     const torch::Tensor even,
                     const torch::Tensor odd,
                     const torch::Tensor psi,
                     const torch::Tensor _2q,
                     const torch::Tensor ql,
                     const torch::Tensor qh,
                     const torch::Tensor kl,
                     const torch::Tensor kh,
                     const int64_t prime_len) {
  const int prime_len_int = static_cast<int>(prime_len);
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_ntt_cuda", ([&] {
        ntt_radix2_cuda_typed<scalar_t>(
            a, even, odd, psi, _2q, ql, qh, kl, kh, prime_len_int);
      }));
}

//------------------------------------------------------------------
// enter_ntt
//------------------------------------------------------------------

template <typename scalar_t>
void enter_ntt_radix2_cuda_typed(torch::Tensor a,
                                 const torch::Tensor Rs,
                                 const torch::Tensor even,
                                 const torch::Tensor odd,
                                 const torch::Tensor psi,
                                 const torch::Tensor _2q,
                                 const torch::Tensor ql,
                                 const torch::Tensor qh,
                                 const torch::Tensor kl,
                                 const torch::Tensor kh,
                                 const int prime_len) {
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
  const auto Rs_acc = Rs.packed_accessor32<scalar_t, 1>();

  const auto even_acc = even.packed_accessor32<int, 2>();
  const auto odd_acc = odd.packed_accessor32<int, 2>();
  const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();

  const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
  const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
  const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
  const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
  const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();

  // enter.
  mont_enter_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(
      a_acc, Rs_acc, ql_acc, qh_acc, kl_acc, kh_acc);

  // ntt.
  for (int i = 0; i < logN; ++i) {
    ntt_radix2_cuda_kernel<scalar_t>
        <<<dim_grid_ntt, dim_block, 0, stream>>>(a_acc,
                                                 even_acc,
                                                 odd_acc,
                                                 psi_acc,
                                                 //  _2q_acc,
                                                 //  ql_acc,
                                                 //  qh_acc,
                                                 //  kl_acc,
                                                 //  kh_acc,
                                                 prime_len,
                                                 i);
  }
}

void enter_ntt_radix2_cuda(torch::Tensor a,
                           const torch::Tensor Rs,
                           const torch::Tensor even,
                           const torch::Tensor odd,
                           const torch::Tensor psi,
                           const torch::Tensor _2q,
                           const torch::Tensor ql,
                           const torch::Tensor qh,
                           const torch::Tensor kl,
                           const torch::Tensor kh,
                           const int64_t prime_len) {
  const int prime_len_int = static_cast<int>(prime_len);
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "typed_enter_ntt_cuda", ([&] {
        enter_ntt_radix2_cuda_typed<scalar_t>(
            a, Rs, even, odd, psi, _2q, ql, qh, kl, kh, prime_len_int);
      }));
}
