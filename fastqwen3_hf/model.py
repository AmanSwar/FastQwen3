import torch

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from fastqwen3_hf.arch.qwen_fast_cuda import FastQwen3
from fastqwen3_hf.arch.config import QwenConfig_float16


class FastQwenConfigHF(PretrainedConfig):
    model_type = "qwen_cuda"

    def __init__(
        self,
        vocab_size=32000,
        embed_dim=1024,
        n_heads=16,
        n_layers=28,
        head_dim=128,
        n_kv_heads=8,
        rope_base=1e6,
        context_length=32768,
        hidden_dim=3072,
        dtype="float16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.rope_base = rope_base
        self.context_length = context_length
        self.hidden_dim = hidden_dim
        self.dtype = dtype


class FastQwenForCausalLM(PreTrainedModel):
    config_class = FastQwenConfigHF
    base_model_prefix = "fastqwen"

    def __init__(self, config: FastQwenConfigHF):
        super().__init__(config)

        # map HF config -> your internal QwenConfig_float16
        qwen_cfg = QwenConfig_float16()
        qwen_cfg.vocab_size = config.vocab_size
        qwen_cfg.embed_dim = config.embed_dim
        qwen_cfg.n_heads = config.n_heads
        qwen_cfg.n_layers = config.n_layers
        qwen_cfg.head_dim = config.head_dim
        qwen_cfg.n_kv_heads = config.n_kv_heads
        qwen_cfg.rope_base = config.rope_base
        qwen_cfg.context_length = config.context_length
        qwen_cfg.hidden_dim = config.hidden_dim
        qwen_cfg.dtype = torch.float16

        # instantiate your model
        self.model = FastQwen3(qwen_cfg)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        logits = self.model(input_ids)  # (batch, seq_len, vocab)
        return CausalLMOutputWithCrossAttentions(logits=logits)
