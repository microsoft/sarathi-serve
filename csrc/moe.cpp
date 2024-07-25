#include <torch/extension.h>

void topk_softmax(
    torch::Tensor& topk_weights, 
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    torch::Tensor& gating_output
);

void moe_align_block_size(
    torch::Tensor topk_ids, 
    int64_t num_experts,
    int64_t block_size, 
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "topk_softmax",
    &topk_softmax,
    "Apply topk softmax to the gating outputs.");
  m.def(
    "moe_align_block_size",
    &moe_align_block_size,
    "Aligning the number of tokens to be processed by each expert such that it is divisible by the block size.");
}
