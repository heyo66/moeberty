
import torch
if not hasattr(torch.library, 'wrap_triton'):
    def wrap_triton(fn):
        return fn
    torch.library.wrap_triton = wrap_triton

# Fix graph breaks from scalar outputs
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True

import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutputWithPast, SequenceClassifierOutput

import bert_padding
from attention import FlexBertUnpadRopeAttention

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

try:
    from liger_kernel.transformers import LigerLayerNorm
    LayerNormClass = LigerLayerNorm
except ImportError:
    LayerNormClass = nn.LayerNorm



# ==============================================================================
# HuggingFace-compatible Configuration
# ==============================================================================

class CustomTransformerConfig(PretrainedConfig):
    """
    Configuration class for CustomTransformer model.

    This class stores the configuration of a CustomTransformer model and is compatible
    with HuggingFace's transformers library. It replaces the old ModelConfig dataclass.
    """
    model_type = "custom_transformer"

    # auto_map tells HF which classes to use when loading with AutoModel/AutoConfig
    auto_map = {
        "AutoConfig": "model.CustomTransformerConfig",
        "AutoModel": "model.CustomTransformerModel",
        "AutoModelForMaskedLM": "model.CustomTransformerForMaskedLM",
        "AutoModelForSequenceClassification": "model.CustomTransformerForSequenceClassification",
    }

    def __init__(
        self,
        vocab_size: int = 50368,
        num_dims: int = 768,
        num_heads: int = 12,
        num_kv_heads: int = 12,
        num_layers: int = 12,
        ffn_hidden_dims: int = 1536,
        layernorm_eps: float = 1e-6,
        attention_probs_dropout_prob: float = 0.1,
        attn_qkv_bias: bool = False,
        attn_out_bias: bool = False,
        attn_out_dropout_prob: float = 0.0,
        global_attn_every_n_layers: int = 3,
        sliding_window: int = 128,
        rotary_emb_base: int = 10000,
        context_len: int = 128,
        use_cache: bool = False,
        use_flash: bool = True,
        use_moe: bool = True,
        moe_num_experts: int = 15,
        moe_routed_experts: int = 1,
        moe_eps: float = 1e-6,
        moe_aux_loss_coef: float = 0.01,
        moe_shared_experts: int = 1,
        use_lossfreebalance: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        mask_token_id: int = 3,
        rope_theta: float = 1e5,
        ffn_dim_multiplier: Optional[int] = None,
        rotary_emb_dim: Optional[int] = None,
        local_attn_rotary_emb_base: int = -1,
        local_attn_rotary_emb_dim: Optional[int] = None,
        rotary_emb_scale_base: Optional[float] = None,
        rotary_emb_interleaved: bool = False,
        use_fa2: Optional[bool] = None,
        deterministic_fa2: bool = False,
        use_sdpa_attn_mask: bool = False,
        num_labels: int = 2,
        classifier_dropout: Optional[float] = None,
        **kwargs
    ):
        """Initialize CustomTransformerConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.ffn_hidden_dims = ffn_hidden_dims
        self.layernorm_eps = layernorm_eps
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.attn_qkv_bias = attn_qkv_bias
        self.attn_out_bias = attn_out_bias
        self.attn_out_dropout_prob = attn_out_dropout_prob
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.sliding_window = sliding_window
        self.rotary_emb_base = rotary_emb_base
        self.context_len = context_len
        self.use_cache = use_cache
        self.use_flash = use_flash
        self.use_moe = use_moe
        self.moe_num_experts = moe_num_experts
        self.moe_routed_experts = moe_routed_experts
        self.moe_eps = moe_eps
        self.moe_aux_loss_coef = moe_aux_loss_coef
        self.moe_shared_experts = moe_shared_experts
        self.use_lossfreebalance = use_lossfreebalance
        self.mask_token_id = mask_token_id
        self.rope_theta = rope_theta
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rotary_emb_dim = rotary_emb_dim
        self.local_attn_rotary_emb_base = local_attn_rotary_emb_base
        self.local_attn_rotary_emb_dim = local_attn_rotary_emb_dim
        self.rotary_emb_scale_base = rotary_emb_scale_base
        self.rotary_emb_interleaved = rotary_emb_interleaved
        self.use_fa2 = use_fa2
        self.deterministic_fa2 = deterministic_fa2
        self.use_sdpa_attn_mask = use_sdpa_attn_mask
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout

        # Derived attributes for compatibility with attention module
        self.hidden_size = num_dims
        self.num_attention_heads = num_heads
        self.embedding_size = num_dims

        # Mirror old ModelConfig.__post_init__
        if self.use_fa2 is None:
            self.use_fa2 = self.use_flash


# Keep ModelConfig as a thin alias for backward compatibility with existing training scripts
@dataclass
class ModelConfig:
    vocab_size: int

    num_dims: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    ffn_hidden_dims: int

    context_len: int
    use_cache: bool
    use_flash: bool
    use_moe: bool

    moe_num_experts: int
    moe_routed_experts: int
    moe_eps: float = 1e-6
    moe_aux_loss_coef: float = 0.01
    moe_shared_experts: int = 0
    use_lossfreebalance: bool = False

    layernorm_eps: float = 1e-6
    rope_theta: float = 1e5

    attention_probs_dropout_prob: float = 0.0
    attn_qkv_bias: bool = False
    attn_out_bias: bool = False
    attn_out_dropout_prob: float = 0.0
    global_attn_every_n_layers: int = 0
    sliding_window: int = -1
    rotary_emb_dim: Optional[int] = None
    rotary_emb_base: Optional[float] = None
    local_attn_rotary_emb_base: int = -1
    local_attn_rotary_emb_dim: Optional[int] = None
    rotary_emb_scale_base: Optional[float] = None
    rotary_emb_interleaved: bool = False
    use_fa2: Optional[bool] = None
    deterministic_fa2: bool = False
    use_sdpa_attn_mask: bool = False
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    embedding_size: Optional[int] = None

    ffn_dim_multiplier: Optional[int] = None

    def __post_init__(self):
        if self.hidden_size is None:
            self.hidden_size = self.num_dims
        if self.num_attention_heads is None:
            self.num_attention_heads = self.num_heads
        if self.rotary_emb_base is None:
            self.rotary_emb_base = self.rope_theta
        if self.use_fa2 is None:
            self.use_fa2 = self.use_flash


# ==============================================================================
# Model Layers
# ==============================================================================

class FlexBertUnpadAttention(nn.Module):
    """Thin wrapper that preserves the state_dict key path: block.attention.attn.*

    In ModernBERT-style global unpadding the data is already (total_nnz, dim) so
    this wrapper just forwards directly to FlexBertUnpadRopeAttention without
    any pad/unpad work.  cu_seqlens, max_seqlen, indices, and attn_mask are
    passed through from the Transformer level.
    """
    def __init__(self, config, layer_id: Optional[int] = None):
        super().__init__()
        self.attn = FlexBertUnpadRopeAttention(config=config, layer_id=layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward on already-unpadded data.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: (total_nnz,)
            attn_mask: (batch, seq_len)

        Returns:
            (total_nnz, dim)
        """
        return self.attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            indices=indices,
            attn_mask=attn_mask,
        )


class FeedForward(nn.Module):
    """Default Feed Forward Layer.  Works on both 2D (total_nnz, dim) and 3D inputs."""
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.ffn_hidden_dims

        self.w1 = nn.Linear(config.num_dims, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, config.num_dims, bias=False)
        self.w3 = nn.Linear(config.num_dims, self.hidden_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return self.w2(self.act(self.w1(x)) * self.w3(x)), None


class FFNwMoE(nn.Module):
    """
    Feed Forward with MoE with optional shared experts.
    Works on 2D (total_nnz, dim) unpadded inputs.

    Uses batched_mm (torch.bmm) for expert dispatch. Expert weights are stored
    as stacked nn.Parameters: (num_experts, out_dim, in_dim). Old checkpoints
    with per-expert nn.Linear weights are automatically converted at load time
    via _load_from_state_dict.

    Returns after forward:
        output: Combined outputs from experts
        aux_loss: Auxiliary loss tensor or routing metadata
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.ffn_hidden_dims
        self.num_dims = config.num_dims

        self.moe_routed_experts = config.moe_routed_experts
        self.moe_aux_loss_coef = config.moe_aux_loss_coef
        self.moe_eps = config.moe_eps
        self.moe_shared_experts = config.moe_shared_experts
        self.num_experts = config.moe_num_experts

        self.use_lossfreebalance = config.use_lossfreebalance

        self.router = nn.Linear(config.num_dims, self.num_experts, bias=False)

        # Stacked expert weights — the actual trainable parameters
        # w1: projects dim -> hidden (gate)
        # w2: projects hidden -> dim (down)
        # w3: projects dim -> hidden (up)
        self.w1_stacked = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, config.num_dims))
        self.w2_stacked = nn.Parameter(torch.empty(self.num_experts, config.num_dims, self.hidden_dim))
        self.w3_stacked = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, config.num_dims))

        # Initialize
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.w1_stacked.data[i])
            nn.init.kaiming_uniform_(self.w2_stacked.data[i])
            nn.init.kaiming_uniform_(self.w3_stacked.data[i])

        # shared experts (for DeepSeekMoE)
        self.shared_experts = nn.ModuleList()
        for _ in range(self.moe_shared_experts):
            self.shared_experts.append(
                nn.ModuleList([
                    nn.Linear(config.num_dims, self.hidden_dim, bias=False),
                    nn.Linear(self.hidden_dim, config.num_dims, bias=False),
                    nn.Linear(config.num_dims, self.hidden_dim, bias=False)
                ]))

        # Auxiliary-loss-free load balancing strategy for MoE (DeepSeek)
        if self.use_lossfreebalance:
            self.expert_biases = nn.Parameter(torch.zeros(self.num_experts))

    def forward(self, x: torch.Tensor):
        # x can be (total_nnz, dim) or (batch, seq_len, dim)
        input_shape = x.shape
        if x.ndim == 3:
            c_batch_size, c_context_len, c_dim = input_shape
            x_flat = x.view(-1, c_dim)
        else:
            x_flat = x
            c_dim = x.shape[-1]

        router_out = self.router(x_flat)
        router_probs = F.softmax(router_out, dim=-1)

        _, topk_indices = router_out.topk(self.moe_routed_experts, dim=-1)
        self.last_topk_indices = topk_indices.detach()

        aux_loss, topk_probs = self._compute_aux_loss(router_out, router_probs, topk_indices)

        output = self._compute_expert_outputs(x_flat, topk_indices, topk_probs, router_probs)

        if x.ndim == 3:
            output = output.view(c_batch_size, c_context_len, c_dim)

        return output, aux_loss

    def _compute_aux_loss(self, router_out, router_probs, topk_indices):
        """Computes the auxiliary loss based on whether loss-free balancing is used or not."""
        if not self.use_lossfreebalance:
            topk_probs, _ = router_probs.topk(self.moe_routed_experts, dim=-1)
            expert_mask = F.one_hot(topk_indices[:, 0], self.num_experts).float()
            density = expert_mask.mean(dim=0)
            router_prob_mean = router_probs.mean(dim=0)
            aux_loss = self.moe_aux_loss_coef * torch.sum(density * router_prob_mean) * self.num_experts

        else:
            router_out = router_out + self.expert_biases
            router_probs = torch.sigmoid(router_out)
            topk_probs = router_probs.gather(-1, topk_indices)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

            aux_loss = (router_probs, topk_indices)
        return aux_loss, topk_probs

    def _compute_expert_outputs(self, x_flat, topk_indices, topk_probs, router_probs):
        """Compute expert outputs using sort-based dispatch with stacked weights.

        Sort tokens by expert, slice contiguous chunks, run each expert via
        matmul on the stacked weight tensors. No weight duplication, minimal
        memory overhead.
        """
        num_tokens, dim = x_flat.shape

        # Flatten top-k: (num_tokens * top_k,)
        flat_expert_ids = topk_indices.view(-1)
        flat_probs = topk_probs.view(-1)
        flat_token_ids = torch.arange(num_tokens, device=x_flat.device).unsqueeze(1).expand(-1, self.moe_routed_experts).reshape(-1)

        # Sort by expert id for contiguous batching
        sorted_expert_ids, sort_indices = flat_expert_ids.sort(stable=True)
        sorted_token_ids = flat_token_ids[sort_indices]
        sorted_probs = flat_probs[sort_indices]

        # Gather sorted input tokens
        sorted_x = x_flat[sorted_token_ids]  # (num_tokens * top_k, dim)

        # Find expert boundaries
        expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=x_flat.device)
        torch.cumsum(expert_counts, dim=0, out=expert_offsets[1:])

        # Run each expert on its contiguous slice using stacked weights
        sorted_output = torch.zeros_like(sorted_x)
        for expert_id in range(self.num_experts):
            start = expert_offsets[expert_id].item()
            end = expert_offsets[expert_id + 1].item()
            if start == end:
                continue
            expert_input = sorted_x[start:end]  # (n_tokens, dim)
            # Use stacked weights directly: w1[expert_id] is (hidden, dim)
            h1 = F.linear(expert_input, self.w1_stacked[expert_id])  # (n, hidden)
            h3 = F.linear(expert_input, self.w3_stacked[expert_id])  # (n, hidden)
            h = F.gelu(h1) * h3
            sorted_output[start:end] = F.linear(h, self.w2_stacked[expert_id])  # (n, dim)

        # Weight by router probabilities
        sorted_output = sorted_output * sorted_probs.unsqueeze(-1)

        # Scatter back to original token positions
        output = torch.zeros_like(x_flat)
        output.scatter_add_(0, sorted_token_ids.unsqueeze(-1).expand_as(sorted_output), sorted_output)

        # Shared experts (for DeepSeekMoE) — unchanged
        for shared_expert_id in range(self.moe_shared_experts):
            w1, w2, w3 = self.shared_experts[shared_expert_id]
            expert_output = w2(F.gelu(w1(x_flat)) * w3(x_flat))
            output = output + expert_output

        return output


class Block(nn.Module):
    """Transformer block operating on unpadded (total_nnz, dim) tensors.

    Receives unpadding metadata (cu_seqlens, max_seqlen, indices, attn_mask)
    from the Transformer level and passes them to attention.  Norms and FFN
    operate directly on the 2D unpadded tensor, avoiding wasted compute on
    padding tokens.
    """
    def __init__(self, config, layer_id: Optional[int] = None):
        super().__init__()
        self.is_first_block = (layer_id == 0)

        self.attention = FlexBertUnpadAttention(config, layer_id=layer_id)
        if config.use_moe:
            self.ffn = FFNwMoE(config)
        else:
            self.ffn = FeedForward(config)

        self.norm_attention = LayerNormClass(config.num_dims, eps=config.layernorm_eps)
        self.norm_ffn = LayerNormClass(config.num_dims, eps=config.layernorm_eps)

    def forward(self, x, cu_seqlens, max_seqlen, indices, attn_mask):
        """
        Args:
            x: (total_nnz, dim) - unpadded hidden states
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: (total_nnz,)
            attn_mask: (batch, seq_len)

        Returns:
            x: (total_nnz, dim)
            aux_loss: auxiliary loss from MoE or None
        """
        if self.is_first_block:
            attn_in = x
        else:
            attn_in = self.norm_attention(x)

        x = x + self.attention(
            attn_in,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            indices=indices,
            attn_mask=attn_mask,
        )

        ffn_out, aux_loss = self.ffn(
            self.norm_ffn(x)
        )
        x = x + ffn_out
        return x, aux_loss


# ==============================================================================
# Core Transformer (nn.Module backbone used inside HF wrappers)
# ==============================================================================

class Transformer(nn.Module):
    """ModernBERT-style Transformer: unpad once before embeddings, repad once at
    the end.  All blocks, norms, and FFNs operate on (total_nnz, dim) tensors,
    avoiding wasted compute on padding tokens.
    """
    def __init__(self, config):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.num_dims = config.num_dims
        self.num_heads = config.num_heads
        self.context_len = config.context_len
        self.use_moe = config.use_moe
        self.use_lossfreebalance = config.use_lossfreebalance and self.use_moe

        self.num_layers = config.num_layers

        hidden_dim = 4 * config.num_dims

        self.tokens_embedding = nn.Embedding(config.vocab_size, config.num_dims)
        self.norm_embeddings = LayerNormClass(config.num_dims, eps=config.layernorm_eps)

        self.blocks = nn.ModuleList()
        for layer_id in range(self.num_layers):
            self.blocks.append(Block(config, layer_id=layer_id))

        self.norm = LayerNormClass(config.num_dims, eps=config.layernorm_eps)
        self.ll_head = nn.Linear(config.num_dims, config.vocab_size, bias=False)

        self.tokens_embedding.weight = self.ll_head.weight

    def _unpad(self, input_ids, attention_mask):
        """Compute unpadding metadata and unpad input_ids before embedding.

        Unpads input_ids (cheap 1D integer indexing) so that embedding and
        all subsequent layers only process real tokens.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) or None

        Returns:
            input_ids_unpadded: (total_nnz,)
            indices: (total_nnz,)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            attn_mask: (batch, seq_len)
            batch_size: int
            seq_len: int
        """
        batch_size, seq_len = input_ids.shape

        if attention_mask is None:
            attn_mask = torch.ones((batch_size, seq_len), device=input_ids.device, dtype=torch.int32)
        else:
            attn_mask = attention_mask.to(dtype=torch.int32)

        # Unpad input_ids using the same bert_padding logic but on (batch, seq_len, 1)
        # so we can reuse unpad_input which expects 3D
        input_ids_3d = input_ids.unsqueeze(-1).float()  # (batch, seq_len, 1)
        input_ids_unpadded, indices, cu_seqlens, max_seqlen = bert_padding.unpad_input(input_ids_3d, attn_mask)
        input_ids_unpadded = input_ids_unpadded.squeeze(-1).long()  # (total_nnz,)

        return input_ids_unpadded, indices, cu_seqlens, max_seqlen, attn_mask, batch_size, seq_len

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = x.shape

        # Unpad input_ids before embedding — only embed real tokens
        x_unpadded, indices, cu_seqlens, max_seqlen, attn_mask, batch_size, seq_len = self._unpad(x, attention_mask)

        # Embed only real tokens (total_nnz, dim)
        x = self.tokens_embedding(x_unpadded)
        x = self.norm_embeddings(x)

        total_aux_loss = 0

        for block in self.blocks:
            x, aux_loss = block(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                indices=indices,
                attn_mask=attn_mask,
            )
            if self.use_moe and not self.use_lossfreebalance:
                total_aux_loss += aux_loss

        x = self.norm(x)

        # Repad once — back to (batch, seq_len, dim) for the LM head / loss
        x = bert_padding.pad_input(x, indices, batch_size, seq_len)

        logits = self.ll_head(x)

        if targets is None:
            loss = None
            ce_loss = None
        else:
            c_batch_size, c_context_len, c_dim = logits.shape
            logits = logits.view(c_batch_size * c_context_len, c_dim)
            targets = targets.view(c_batch_size * c_context_len)
            ce_loss = F.cross_entropy(logits, targets)

            if self.use_moe and not self.use_lossfreebalance:
                loss = ce_loss + total_aux_loss
            else:
                loss = ce_loss
                ce_loss = aux_loss

        return logits, loss, ce_loss

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_tokens: int, temperature: float = 1.0, top_k: int = 50,
                 use_cache: bool = False):
        """Generate text from x up to max_tokens."""
        for c_tkn_pos in range(max_tokens):
            if use_cache:
                if c_tkn_pos == 0:
                    logits, _, ce_loss = self.forward(x, start_pos=c_tkn_pos)
                else:
                    logits, _, ce_loss = self.forward(x[:, -1:], start_pos=c_tkn_pos)
            else:
                logits, _, ce_loss = self.forward(x)

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                tkl, idx = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < tkl[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
        return x


# ==============================================================================
# HuggingFace PreTrainedModel Wrappers
# ==============================================================================

class CustomTransformerPreTrainedModel(PreTrainedModel):
    """Base class for CustomTransformer models."""
    config_class = CustomTransformerConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Block"]

    def _init_weights(self, module):
        """Initialize weights - handled by model itself."""
        pass


class CustomTransformerModel(CustomTransformerPreTrainedModel):
    """The bare CustomTransformer Model outputting raw hidden-states."""

    def __init__(self, config: CustomTransformerConfig):
        super().__init__(config)
        self.config = config

        self.transformer = Transformer(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.tokens_embedding

    def set_input_embeddings(self, value):
        self.transformer.tokens_embedding = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Forward pass returning raw hidden states."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Unpad input_ids before embedding
        x_unpadded, indices, cu_seqlens, max_seqlen, attn_mask, batch_size, seq_len = self.transformer._unpad(input_ids, attention_mask)

        # Embed only real tokens
        x = self.transformer.tokens_embedding(x_unpadded)
        x = self.transformer.norm_embeddings(x)

        for block in self.transformer.blocks:
            x, _ = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, indices=indices, attn_mask=attn_mask)

        x = self.transformer.norm(x)

        # Repad once
        hidden_states = bert_padding.pad_input(x, indices, batch_size, seq_len)

        if not return_dict:
            return (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class CustomTransformerForMaskedLM(CustomTransformerPreTrainedModel):
    """CustomTransformer Model with a masked language modeling head on top."""
    _tied_weights_keys = ["transformer.ll_head.weight", "transformer.tokens_embedding.weight"]

    def __init__(self, config: CustomTransformerConfig):
        super().__init__(config)
        self.config = config

        self.transformer = Transformer(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.tokens_embedding

    def set_input_embeddings(self, value):
        self.transformer.tokens_embedding = value

    def get_output_embeddings(self):
        return self.transformer.ll_head

    def set_output_embeddings(self, new_embeddings):
        self.transformer.ll_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """Forward pass for masked language modeling."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits, model_loss, ce_loss = self.transformer(
            input_ids, targets=labels, start_pos=0, attention_mask=attention_mask
        )

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = model_loss

        if not return_dict:
            output = (logits,)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class CustomTransformerForSequenceClassification(CustomTransformerPreTrainedModel):
    """CustomTransformer Model with a sequence classification head on top."""

    def __init__(self, config: CustomTransformerConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.transformer = Transformer(config)

        # Classification head
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.attention_probs_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.num_dims, config.num_labels)

        self._init_classifier_weights()
        self.post_init()

    def _init_classifier_weights(self):
        std = 0.02
        if isinstance(self.classifier, nn.Linear):
            self.classifier.weight.data.normal_(mean=0.0, std=std)
            if self.classifier.bias is not None:
                self.classifier.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """Forward pass for sequence classification."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # Unpad input_ids before embedding
        x_unpadded, indices, cu_seqlens, max_seqlen, attn_mask, batch_size, seq_len = self.transformer._unpad(input_ids, attention_mask)

        # Embed only real tokens
        x = self.transformer.tokens_embedding(x_unpadded)
        x = self.transformer.norm_embeddings(x)

        # Collect hidden states if requested (repad each for the output tuple)
        all_hidden_states = () if output_hidden_states else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (bert_padding.pad_input(x, indices, batch_size, seq_len),)

        for block in self.transformer.blocks:
            x, _ = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, indices=indices, attn_mask=attn_mask)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (bert_padding.pad_input(x, indices, batch_size, seq_len),)

        x = self.transformer.norm(x)

        # Repad once
        hidden_states = bert_padding.pad_input(x, indices, batch_size, seq_len)

        # Use [CLS] token representation (first token) for classification
        pooled_output = hidden_states[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + (all_hidden_states,) + (None,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=None,
        )


# ==============================================================================
# Auto-registration
# ==============================================================================

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification

    AutoConfig.register("custom_transformer", CustomTransformerConfig)
    AutoModel.register(CustomTransformerConfig, CustomTransformerModel)
    AutoModelForMaskedLM.register(CustomTransformerConfig, CustomTransformerForMaskedLM)
    AutoModelForSequenceClassification.register(CustomTransformerConfig, CustomTransformerForSequenceClassification)
except Exception:
    pass


def main():
    pass


if __name__ == "__main__":
    main()
