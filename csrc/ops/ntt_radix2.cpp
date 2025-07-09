#include "cuda/ntt_radix2_cuda.h"

//------------------------------------------------------------------
// Wrap functions for ntt transformation
//------------------------------------------------------------------

void upload_tensor_list(const std::vector<torch::Tensor> tensor_list,
                        const std::vector<int64_t> offset_list,
                        const int64_t layout,
                        const int64_t device_id) {
  upload_tensor_list_cuda(tensor_list, offset_list, layout, device_id);
}

torch::Tensor read_constant_chunk(const torch::Tensor& dummy,
                                  const int64_t offset_bytes,
                                  const int64_t count,
                                  const torch::Dtype dtype,
                                  const int64_t layout) {
  const int device_id =
      dummy.device().index();  // Extract from dummy tensor, since pytorch needs
                               // a tensor to dispatch op
  return read_constant_chunk_cuda(
      device_id, offset_bytes, count, dtype, layout);
}

void ntt_radix2(std::vector<torch::Tensor> a,
                const std::vector<torch::Tensor> even,
                const std::vector<torch::Tensor> odd,
                const std::vector<torch::Tensor> psi,
                const std::vector<torch::Tensor> _2q,
                const std::vector<torch::Tensor> ql,
                const std::vector<torch::Tensor> qh,
                const std::vector<torch::Tensor> kl,
                const std::vector<torch::Tensor> kh,
                const int64_t prime_len) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    ntt_radix2_cuda(a[i],
                    even[i],
                    odd[i],
                    psi[i],
                    _2q[i],
                    ql[i],
                    qh[i],
                    kl[i],
                    kh[i],
                    prime_len);
  }
}

void enter_ntt_radix2(std::vector<torch::Tensor> a,
                      const std::vector<torch::Tensor> Rs,
                      const std::vector<torch::Tensor> even,
                      const std::vector<torch::Tensor> odd,
                      const std::vector<torch::Tensor> psi,
                      const std::vector<torch::Tensor> _2q,
                      const std::vector<torch::Tensor> ql,
                      const std::vector<torch::Tensor> qh,
                      const std::vector<torch::Tensor> kl,
                      const std::vector<torch::Tensor> kh,
                      const int64_t prime_len) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    enter_ntt_radix2_cuda(a[i],
                          Rs[i],
                          even[i],
                          odd[i],
                          psi[i],
                          _2q[i],
                          ql[i],
                          qh[i],
                          kl[i],
                          kh[i],
                          prime_len);
  }
}

TORCH_LIBRARY_FRAGMENT(tiberate_ntt2_ops, m) {
  m.def(
      "upload_tensor_list(Tensor[] tensor_list, int[] offset_list, "
      "int layout, int device_id) -> ()");

  m.def(
      "read_constant_chunk(Tensor dummy, int offset_bytes, "
      "int count, ScalarType dtype, int layout) -> Tensor");
  m.def(
      "ntt_radix2(Tensor[](a!) a, Tensor[] even, Tensor[] odd, Tensor[] psi, "
      "Tensor[] _2q, Tensor[] ql, Tensor[] qh, Tensor[] kl, "
      "Tensor[] kh, int prime_len) -> ()");
  m.def(
      "enter_ntt_radix2(Tensor[](a!) a, Tensor[] Rs, Tensor[] even, Tensor[] "
      "odd, "
      "Tensor[] psi, Tensor[] _2q, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh, int prime_len) -> ()");
}

TORCH_LIBRARY_IMPL(tiberate_ntt2_ops, CUDA, m) {
  m.impl("upload_tensor_list", upload_tensor_list);
  m.impl("read_constant_chunk", read_constant_chunk);
  m.impl("ntt_radix2", &ntt_radix2);
  m.impl("enter_ntt_radix2", &enter_ntt_radix2);
}
