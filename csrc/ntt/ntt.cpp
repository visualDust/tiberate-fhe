#include "ntt.h" // inside is forward declarations for cuda kernels
#include <torch/extension.h>
#include <vector>

//------------------------------------------------------------------
// Wrap functions for Montgomery space operations
//------------------------------------------------------------------

std::vector<torch::Tensor> mont_mult(const std::vector<torch::Tensor> a,
                                     const std::vector<torch::Tensor> b,
                                     const std::vector<torch::Tensor> ql,
                                     const std::vector<torch::Tensor> qh,
                                     const std::vector<torch::Tensor> kl,
                                     const std::vector<torch::Tensor> kh) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto c = mont_mult_cuda(a[i], b[i], ql[i], qh[i], kl[i], kh[i]);

    outputs.push_back(c);
  }

  return outputs;
}

void mont_enter(std::vector<torch::Tensor> a,
                const std::vector<torch::Tensor> Rs,
                const std::vector<torch::Tensor> ql,
                const std::vector<torch::Tensor> qh,
                const std::vector<torch::Tensor> kl,
                const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    mont_enter_cuda(a[i], Rs[i], ql[i], qh[i], kl[i], kh[i]);
  }
}

void reduce_2q(std::vector<torch::Tensor> a,
               const std::vector<torch::Tensor> _2q) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    reduce_2q_cuda(a[i], _2q[i]);
  }
}

void mont_reduce(std::vector<torch::Tensor> a,
               const std::vector<torch::Tensor> ql,
               const std::vector<torch::Tensor> qh,
               const std::vector<torch::Tensor> kl,
               const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    mont_reduce_cuda(a[i], ql[i], qh[i], kl[i], kh[i]);
  }
}

std::vector<torch::Tensor> mont_add(const std::vector<torch::Tensor> a,
                                    const std::vector<torch::Tensor> b,
                                    const std::vector<torch::Tensor> _2q) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto c = mont_add_cuda(a[i], b[i], _2q[i]);
    outputs.push_back(c);
  }
  return outputs;
}

std::vector<torch::Tensor> mont_sub(const std::vector<torch::Tensor> a,
                                    const std::vector<torch::Tensor> b,
                                    const std::vector<torch::Tensor> _2q) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto c = mont_sub_cuda(a[i], b[i], _2q[i]);
    outputs.push_back(c);
  }
  return outputs;
}

void make_signed(std::vector<torch::Tensor> a,
                 const std::vector<torch::Tensor> _2q) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    make_signed_cuda(a[i], _2q[i]);
  }
}

void make_unsigned(std::vector<torch::Tensor> a,
                   const std::vector<torch::Tensor> _2q) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    make_unsigned_cuda(a[i], _2q[i]);
  }
}

std::vector<torch::Tensor> tile_unsigned(std::vector<torch::Tensor> a,
                                         const std::vector<torch::Tensor> _2q) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = _2q.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto result = tile_unsigned_cuda(a[i], _2q[i]);
    outputs.push_back(result);
  }
  return outputs;
}

//------------------------------------------------------------------
// Wrap functions for ntt transformation
//------------------------------------------------------------------

void ntt(std::vector<torch::Tensor> a,
         const std::vector<torch::Tensor> even,
         const std::vector<torch::Tensor> odd,
         const std::vector<torch::Tensor> psi,
         const std::vector<torch::Tensor> _2q,
         const std::vector<torch::Tensor> ql,
         const std::vector<torch::Tensor> qh,
         const std::vector<torch::Tensor> kl,
         const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    ntt_cuda(a[i], even[i], odd[i], psi[i], _2q[i], ql[i], qh[i], kl[i], kh[i]);
  }
}

void enter_ntt(std::vector<torch::Tensor> a,
               const std::vector<torch::Tensor> Rs,
               const std::vector<torch::Tensor> even,
               const std::vector<torch::Tensor> odd,
               const std::vector<torch::Tensor> psi,
               const std::vector<torch::Tensor> _2q,
               const std::vector<torch::Tensor> ql,
               const std::vector<torch::Tensor> qh,
               const std::vector<torch::Tensor> kl,
               const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    enter_ntt_cuda(a[i],
                   Rs[i],
                   even[i],
                   odd[i],
                   psi[i],
                   _2q[i],
                   ql[i],
                   qh[i],
                   kl[i],
                   kh[i]);
  }
}

void intt(std::vector<torch::Tensor> a,
          const std::vector<torch::Tensor> even,
          const std::vector<torch::Tensor> odd,
          const std::vector<torch::Tensor> psi,
          const std::vector<torch::Tensor> Ninv,
          const std::vector<torch::Tensor> _2q,
          const std::vector<torch::Tensor> ql,
          const std::vector<torch::Tensor> qh,
          const std::vector<torch::Tensor> kl,
          const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    intt_cuda(a[i],
              even[i],
              odd[i],
              psi[i],
              Ninv[i],
              _2q[i],
              ql[i],
              qh[i],
              kl[i],
              kh[i]);
  }
}

void intt_exit(std::vector<torch::Tensor> a,
               const std::vector<torch::Tensor> even,
               const std::vector<torch::Tensor> odd,
               const std::vector<torch::Tensor> psi,
               const std::vector<torch::Tensor> Ninv,
               const std::vector<torch::Tensor> _2q,
               const std::vector<torch::Tensor> ql,
               const std::vector<torch::Tensor> qh,
               const std::vector<torch::Tensor> kl,
               const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    intt_exit_cuda(a[i],
                   even[i],
                   odd[i],
                   psi[i],
                   Ninv[i],
                   _2q[i],
                   ql[i],
                   qh[i],
                   kl[i],
                   kh[i]);
  }
}

void intt_exit_reduce(std::vector<torch::Tensor> a,
                      const std::vector<torch::Tensor> even,
                      const std::vector<torch::Tensor> odd,
                      const std::vector<torch::Tensor> psi,
                      const std::vector<torch::Tensor> Ninv,
                      const std::vector<torch::Tensor> _2q,
                      const std::vector<torch::Tensor> ql,
                      const std::vector<torch::Tensor> qh,
                      const std::vector<torch::Tensor> kl,
                      const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    intt_exit_reduce_cuda(a[i],
                          even[i],
                          odd[i],
                          psi[i],
                          Ninv[i],
                          _2q[i],
                          ql[i],
                          qh[i],
                          kl[i],
                          kh[i]);
  }
}

void intt_exit_reduce_signed(std::vector<torch::Tensor> a,
                             const std::vector<torch::Tensor> even,
                             const std::vector<torch::Tensor> odd,
                             const std::vector<torch::Tensor> psi,
                             const std::vector<torch::Tensor> Ninv,
                             const std::vector<torch::Tensor> _2q,
                             const std::vector<torch::Tensor> ql,
                             const std::vector<torch::Tensor> qh,
                             const std::vector<torch::Tensor> kl,
                             const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    intt_exit_reduce_signed_cuda(a[i],
                                 even[i],
                                 odd[i],
                                 psi[i],
                                 Ninv[i],
                                 _2q[i],
                                 ql[i],
                                 qh[i],
                                 kl[i],
                                 kh[i]);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mont_mult", &mont_mult, "MONTGOMERY MULTIPLICATION");
  m.def("mont_enter", &mont_enter, "ENTER MONTGOMERY");
  m.def("mont_reduce", &mont_reduce, "MONTGOMERY REDUCTION");
  m.def("reduce_2q", &reduce_2q, "REDUCE RANGE TO 2q");
  m.def("mont_add", &mont_add, "MONTGOMERY ADDITION");
  m.def("mont_sub", &mont_sub, "MONTGOMERY SUBTRACTION");
  m.def("make_signed", &make_signed, "MAKE SIGNED");
  m.def("make_unsigned", &make_unsigned, "MAKE UNSIGNED");
  m.def("tile_unsigned", &tile_unsigned, "TILE -> MAKE UNSIGNED");
  //------------------------------------------------------------------
  m.def("ntt", &ntt, "FORWARD NTT");
  m.def("enter_ntt", &enter_ntt, "ENTER -> FORWARD NTT");
  m.def("intt", &intt, "INVERSE NTT");
  m.def("intt_exit", &intt_exit, "INVERSE NTT -> EXIT");
  m.def("intt_exit_reduce", &intt_exit_reduce, "INVERSE NTT -> EXIT -> REDUCE");
  m.def("intt_exit_reduce_signed",
        &intt_exit_reduce_signed,
        "INVERSE NTT -> EXIT -> REDUCE -> MAKE SIGNED");
}
