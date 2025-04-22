#include <vector>
#include "cuda/mont_cuda.h"
#include "extensions.h"
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

void reduce_2q(std::vector<torch::Tensor> a,
               const std::vector<torch::Tensor> _2q) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    reduce_2q_cuda(a[i], _2q[i]);
  }
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

static auto registry =
    torch::RegisterOperators()
        .op("torch_tiberate::mont_mult",
            &mont_mult,
            torch::RegisterOperators::options().aliasAnalysis(
                torch::AliasAnalysisKind::FROM_SCHEMA))
        .op("torch_tiberate::mont_enter", &mont_enter)
        .op("torch_tiberate::mont_reduce", &mont_reduce)
        .op("torch_tiberate::mont_add", &mont_add)
        .op("torch_tiberate::mont_sub", &mont_sub)
        .op("torch_tiberate::reduce_2q", &reduce_2q)
        .op("torch_tiberate::make_signed", &make_signed)
        .op("torch_tiberate::make_unsigned", &make_unsigned)
        .op("torch_tiberate::tile_unsigned", &tile_unsigned);
