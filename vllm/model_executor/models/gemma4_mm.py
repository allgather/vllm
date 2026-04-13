# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4 multimodal model (image + audio + video support).

Adds vision tower, audio tower, and multimodal embedders on top of the
text-only Gemma4ForCausalLM.  The vision/audio encoders are loaded via
AutoModel.from_config and run in eager mode while the language model uses
the vLLM-optimized path.

Video support:  Gemma4 does **not** have a native video tower.  Videos are
decomposed into timestamped image frames (up to 32 frames at 70 soft tokens
each) and fed through the same vision tower as regular images.  The
processor inserts ``mm:ss`` timestamps between frames so the model can
reason about temporal order.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Annotated, Any, Literal

import numpy as np
import torch
import torch.nn.functional
from PIL import Image as PILImage
from torch import nn
from transformers import BatchFeature
from transformers.activations import ACT2FN
from transformers.models.gemma4 import (
    Gemma4Config,
    Gemma4Processor,
    Gemma4VisionConfig,
)
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4TextConfig,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions, VideoDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    VideoItem,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)

# Video constants — match transformers Gemma4VideoProcessor defaults.
_VIDEO_MAX_SOFT_TOKENS = 70  # soft tokens per video frame (vs 280 for images)
_VIDEO_MAX_FRAMES = 32  # max sampled frames per video


class Gemma4ClippableLinear(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig | Gemma4AudioConfig,
        in_features: int,
        out_features: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = ReplicatedLinear(
            in_features,
            out_features,
            bias=False,
            prefix=maybe_prefix(prefix, "linear"),
            return_bias=False,
        )

        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def clamp_hidden_states(
        self,
        hidden_states: torch.Tensor,
        min_value: torch.Tensor,
        max_value: torch.Tensor,
    ) -> torch.Tensor:
        return torch.clamp(hidden_states, min_value, max_value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_clipped_linears:
            return self.linear(hidden_states)

        hidden_states = self.clamp_hidden_states(
            hidden_states,
            self.input_min,
            self.input_max,
        )
        hidden_states = self.linear(hidden_states)
        return self.clamp_hidden_states(
            hidden_states,
            self.output_min,
            self.output_max,
        )


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = ReplicatedLinear(
            3 * self.patch_size**2,
            self.hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "input_proj"),
            return_bias=False,
        )
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        clamped_positions = pixel_position_ids.clamp(min=0)
        one_hot = torch.nn.functional.one_hot(
            clamped_positions,
            num_classes=self.position_embedding_size,
        )
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        position_embeddings = one_hot @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(dim=1)
        return torch.where(
            padding_positions.unsqueeze(-1),
            torch.zeros((), device=position_embeddings.device),
            position_embeddings,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        position_embeddings = self.position_embeddings(
            pixel_position_ids,
            padding_positions,
        )
        return hidden_states + position_embeddings


class Gemma4VisionPooler(nn.Module):
    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size**0.5
        self.image_seq_length = getattr(config, "image_seq_length", None)
        self.default_output_length = getattr(config, "default_output_length", None)

    def avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pool patch states into soft-token slots.
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: "
                f"{k=}^2 times {length=} must be {input_seq_len}."
            )

        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = (
            torch.nn.functional.one_hot(kernel_idxs.long(), length).float() / k_squared
        )
        weights = weights.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        output = weights.transpose(1, 2) @ hidden_states.float()
        pooled_padding_positions = (weights == 0).all(dim=1)
        return output.to(hidden_states.dtype), pooled_padding_positions

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_length is None:
            output_length = (
                getattr(self, "image_seq_length", None)
                or getattr(self, "default_output_length", None)
                or hidden_states.shape[1]
            )
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                "Cannot output more soft tokens "
                f"(requested {output_length}) than there are patches "
                f"({hidden_states.shape[1]})."
            )

        hidden_states = hidden_states.masked_fill(
            padding_positions.unsqueeze(-1),
            0.0,
        )
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self.avg_pool_by_positions(
                hidden_states,
                pixel_position_ids,
                padding_positions,
                output_length,
            )

        hidden_states *= self.root_hidden_size
        return hidden_states, padding_positions


class Gemma4VisionMLP(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.gate_proj = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            config.intermediate_size,
            prefix=maybe_prefix(prefix, "gate_proj"),
        )
        self.up_proj = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            config.intermediate_size,
            prefix=maybe_prefix(prefix, "up_proj"),
        )
        self.down_proj = Gemma4ClippableLinear(
            config,
            config.intermediate_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "down_proj"),
        )
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Gemma4VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.attention_scaling = 1.0
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None)
        if dim is None:
            dim = config.hidden_size // config.num_attention_heads
        spatial_dim = dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float()
        inv_freq = inv_freq.expand(position_ids.shape[0], -1, 1)
        inv_freq = inv_freq.to(hidden_states.device)

        all_cos: list[torch.Tensor] = []
        all_sin: list[torch.Tensor] = []
        for dim_idx in range(position_ids.shape[-1]):
            dim_position_ids = position_ids[:, :, dim_idx][:, None, :].float()
            freqs = (inv_freq @ dim_position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos() * self.attention_scaling)
            all_sin.append(emb.sin() * self.attention_scaling)

        cos = torch.cat(all_cos, dim=-1).to(dtype=hidden_states.dtype)
        sin = torch.cat(all_sin, dim=-1).to(dtype=hidden_states.dtype)
        return cos, sin


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first_half = hidden_states[..., : hidden_states.shape[-1] // 2]
    second_half = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_pos_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (hidden_states * cos) + (rotate_half(hidden_states) * sin)


def apply_multidimensional_rope(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    ndim = position_ids.shape[-1]
    num_input_channels = hidden_states.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
    if num_rotated_channels_per_dim <= 0:
        raise ValueError(
            "Invalid configuration: num_rotated_channels_per_dim must be > 0, "
            f"got {num_rotated_channels_per_dim}."
        )

    split_sizes = [num_rotated_channels_per_dim] * ndim
    hidden_parts = torch.split(hidden_states, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    rotated = [
        apply_rotary_pos_emb(
            hidden_parts[idx],
            cos_parts[idx],
            sin_parts[idx],
            unsqueeze_dim=unsqueeze_dim,
        )
        for idx in range(ndim)
    ]
    return torch.cat(rotated, dim=-1)


class Gemma4VisionAttention(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = 1.0
        self.attn = MMEncoderAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            prefix=maybe_prefix(prefix, "attn"),
        )
        self.q_proj = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            self.num_heads * self.head_dim,
            prefix=maybe_prefix(prefix, "q_proj"),
        )
        self.k_proj = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            prefix=maybe_prefix(prefix, "k_proj"),
        )
        self.v_proj = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            prefix=maybe_prefix(prefix, "v_proj"),
        )
        self.o_proj = Gemma4ClippableLinear(
            config,
            self.num_heads * self.head_dim,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "o_proj"),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            has_weight=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        query_states = self.q_norm(query_states)
        query_states = apply_multidimensional_rope(
            query_states,
            cos,
            sin,
            position_ids,
        )
        query_states = query_states.reshape(batch_size, seq_len, -1)

        key_states = self.k_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )
        key_states = self.k_norm(key_states)
        key_states = apply_multidimensional_rope(
            key_states,
            cos,
            sin,
            position_ids,
        )
        key_states = key_states.reshape(batch_size, seq_len, -1)

        value_states = self.v_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )
        value_states = self.v_norm(value_states)
        value_states = value_states.reshape(batch_size, seq_len, -1)

        attn_output = self.attn(
            query_states,
            key_states,
            value_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return self.o_proj(attn_output)


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Gemma4VisionAttention(
            config,
            prefix=maybe_prefix(prefix, "self_attn"),
        )
        self.mlp = Gemma4VisionMLP(
            config,
            prefix=maybe_prefix(prefix, "mlp"),
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(
                    config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx}"),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        padding_positions: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = pixel_position_ids.clamp(min=0)
        hidden_states = inputs_embeds
        cu_seqlens = None
        max_seqlen = None
        if torch.any(padding_positions):
            # Pack valid patches for MMEncoderAttention.
            valid_positions = torch.logical_not(padding_positions)
            valid_lengths = valid_positions.sum(dim=1, dtype=torch.int32)
            hidden_states = hidden_states[valid_positions].unsqueeze(0)
            position_ids = position_ids[valid_positions].unsqueeze(0)
            cu_seqlens = torch.zeros(
                valid_lengths.shape[0] + 1,
                dtype=torch.int32,
                device=inputs_embeds.device,
            )
            cu_seqlens[1:] = valid_lengths.cumsum(dim=0)
            max_seqlen = valid_lengths.max()

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        if cu_seqlens is None:
            return hidden_states

        full_hidden_states = inputs_embeds.new_zeros(inputs_embeds.shape)
        full_hidden_states[torch.logical_not(padding_positions)] = hidden_states[0]
        return full_hidden_states


class Gemma4VisionModel(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.embedder = Gemma4VisionPatchEmbedder(
            config,
            prefix=maybe_prefix(prefix, "embedder"),
        )
        self.encoder = Gemma4VisionEncoder(
            config,
            prefix=maybe_prefix(prefix, "encoder"),
        )
        self.pooler = Gemma4VisionPooler(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        output_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        hidden_states = self.embedder(
            pixel_values,
            pixel_position_ids,
            padding_positions,
        )
        hidden_states = self.encoder(
            hidden_states,
            padding_positions,
            pixel_position_ids,
        )
        hidden_states, padding_positions = self.pooler(
            hidden_states,
            pixel_position_ids,
            padding_positions,
            output_length=output_length,
        )
        return hidden_states, padding_positions


def build_audio_attention_mask(
    attention_mask: torch.Tensor,
    *,
    chunk_size: int,
    max_past_horizon: int,
    max_future_horizon: int,
) -> torch.Tensor:
    # Limit audio attention to each chunk window.
    batch_size, seq_len = attention_mask.shape
    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    pad = num_blocks * chunk_size - seq_len
    query_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=False)
    query_mask = query_mask.view(batch_size, num_blocks, chunk_size)

    context_size = chunk_size + max_past_horizon + max_future_horizon
    key_mask = torch.nn.functional.pad(
        attention_mask,
        (max_past_horizon, max_future_horizon + chunk_size - 1),
        value=False,
    )
    key_mask = key_mask.unfold(1, context_size, chunk_size).contiguous()

    return (query_mask.unsqueeze(-1) & key_mask.unsqueeze(-2)).unsqueeze(1)


class Gemma4AudioRelPositionalEncoding(nn.Module):
    inv_timescales: torch.Tensor

    def __init__(self, config: Gemma4AudioConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(
            num_timescales - 1,
            1,
        )
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales) * -log_timescale_increment
        )
        self.register_buffer(
            "inv_timescales",
            inv_timescales.unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(
            self.max_past_horizon,
            -self.max_future_horizon - 1,
            -1,
            device=hidden_states.device,
        )
        position_ids = position_ids[..., None]
        scaled_time = position_ids * self.inv_timescales.to(device=hidden_states.device)
        pos_embed = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)],
            dim=-1,
        )
        return pos_embed.to(dtype=hidden_states.dtype)


class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False,
        )
        self.norm = nn.LayerNorm(
            out_channels,
            eps=norm_eps,
            elementwise_affine=True,
            bias=False,
        )
        self.act = nn.ReLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mask is not None:
            mask = mask.to(device=hidden_states.device)
            hidden_states = hidden_states * mask[:, None, :, None]

        hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
        hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1))
        hidden_states = self.act(hidden_states).permute(0, 3, 1, 2).contiguous()

        if mask is not None:
            # Downsample the time mask with the conv stride.
            mask = mask[:, ::2]

        return hidden_states, mask


class Gemma4AudioSubSampleConvProjection(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=1,
            out_channels=config.subsampling_conv_channels[0],
            norm_eps=config.rms_norm_eps,
        )
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=config.subsampling_conv_channels[0],
            out_channels=config.subsampling_conv_channels[1],
            norm_eps=config.rms_norm_eps,
        )
        proj_input_dim = (
            config.subsampling_conv_channels[0] // 4
        ) * config.subsampling_conv_channels[1]
        self.input_proj_linear = ReplicatedLinear(
            proj_input_dim,
            config.hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "input_proj_linear"),
            return_bias=False,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = input_features.unsqueeze(1)
        hidden_states, mask = self.layer0(hidden_states, input_features_mask)
        hidden_states, mask = self.layer1(hidden_states, mask)
        batch_size, _, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
        return self.input_proj_linear(hidden_states), mask


class Gemma4AudioFeedForward(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.ffw_layer_1 = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            config.hidden_size * 4,
            prefix=maybe_prefix(prefix, "ffw_layer_1"),
        )
        self.ffw_layer_2 = Gemma4ClippableLinear(
            config,
            config.hidden_size * 4,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "ffw_layer_2"),
        )
        self.pre_layer_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_layer_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gradient_clipping = min(
            self.gradient_clipping,
            torch.finfo(self.ffw_layer_1.linear.weight.dtype).max,
        )
        residual = hidden_states
        hidden_states = torch.clamp(
            hidden_states,
            -gradient_clipping,
            gradient_clipping,
        )
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)
        hidden_states = torch.clamp(
            hidden_states,
            -gradient_clipping,
            gradient_clipping,
        )
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states *= self.post_layer_scale
        hidden_states += residual
        return hidden_states


class Gemma4AudioCausalConv1d(nn.Conv1d):
    @cached_property
    def left_pad(self) -> int:
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.nn.functional.pad(x, (self.left_pad, 0)))


class Gemma4AudioLightConv1d(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.linear_start = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            config.hidden_size * 2,
            prefix=maybe_prefix(prefix, "linear_start"),
        )
        self.linear_end = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "linear_end"),
        )
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
        )
        self.pre_layer_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.conv_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = torch.nn.functional.glu(hidden_states, dim=-1)
        hidden_states = self.depthwise_conv1d(hidden_states.transpose(1, 2))
        hidden_states = hidden_states.transpose(1, 2)
        gradient_clipping = min(
            self.gradient_clipping,
            torch.finfo(self.linear_start.linear.weight.dtype).max,
        )
        hidden_states = torch.clamp(
            hidden_states,
            -gradient_clipping,
            gradient_clipping,
        )
        hidden_states = self.conv_norm(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        return hidden_states + residual


class Gemma4AudioAttention(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.attention_logits_soft_cap = config.attention_logit_cap
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        self.q_proj = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            self.num_heads * self.head_dim,
            prefix=maybe_prefix(prefix, "q_proj"),
        )
        self.k_proj = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            self.num_heads * self.head_dim,
            prefix=maybe_prefix(prefix, "k_proj"),
        )
        self.v_proj = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            self.num_heads * self.head_dim,
            prefix=maybe_prefix(prefix, "v_proj"),
        )
        self.post = Gemma4ClippableLinear(
            config,
            config.hidden_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "post"),
        )
        self.relative_k_proj = ReplicatedLinear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            prefix=maybe_prefix(prefix, "relative_k_proj"),
            return_bias=False,
        )
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))
        self.register_buffer(
            "softcap",
            torch.tensor(self.attention_logits_soft_cap),
            persistent=False,
        )

    def convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, 0, 0, pad))
        return hidden_states.reshape(
            batch_size,
            num_blocks,
            self.chunk_size,
            num_heads,
            head_dim,
        ).contiguous()

    def extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.nn.functional.pad(
            hidden_states,
            (
                0,
                0,
                0,
                0,
                self.max_past_horizon,
                self.max_future_horizon + self.chunk_size - 1,
            ),
        )
        hidden_states = hidden_states.unfold(1, self.context_size, self.chunk_size)
        hidden_states = torch.movedim(hidden_states, -1, 2)
        return hidden_states.contiguous()

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, num_blocks, block_size, _ = x.shape
        x = torch.nn.functional.pad(x, (0, self.context_size + 1 - x.shape[-1]))
        x = x.view(
            batch_size,
            num_heads,
            num_blocks,
            block_size * (self.context_size + 1),
        )
        x = x[..., : block_size * self.context_size]
        return x.view(
            batch_size,
            num_heads,
            num_blocks,
            block_size,
            self.context_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).float().view(hidden_shape)
        key_states = self.k_proj(hidden_states).float().view(hidden_shape)
        value_states = self.v_proj(hidden_states).float().view(hidden_shape)

        query_states = (
            query_states
            * self.q_scale
            * torch.nn.functional.softplus(self.per_dim_scale)
        )
        key_states = key_states * self.k_scale
        query_states = self.convert_to_block(query_states)
        key_states = self.extract_block_context(key_states)
        value_states = self.extract_block_context(value_states)
        num_blocks = query_states.shape[1]

        relative_key_states = self.relative_k_proj(
            position_embeddings.to(dtype=self.relative_k_proj.weight.dtype)
        )
        relative_key_states = relative_key_states.view(
            -1,
            self.num_heads,
            self.head_dim,
        )
        relative_key_states = relative_key_states.to(dtype=query_states.dtype)

        queries = query_states.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)
        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(
            batch_size,
            self.num_heads,
            num_blocks,
            self.chunk_size,
            -1,
        )
        matrix_bd = self.rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = attn_weights / self.softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.softcap
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.logical_not(),
                self.config.attention_invalid_logits_value,
            )

        attn_weights = torch.nn.functional.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(value_states.dtype)
        attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(
            batch_size,
            num_blocks * self.chunk_size,
            -1,
        )
        attn_output = attn_output[:, :seq_length].contiguous()
        attn_output = self.post(attn_output.to(dtype=self.post.linear.weight.dtype))
        return attn_output, attn_weights


class Gemma4AudioLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.feed_forward1 = Gemma4AudioFeedForward(
            config,
            prefix=maybe_prefix(prefix, "feed_forward1"),
        )
        self.self_attn = Gemma4AudioAttention(
            config,
            prefix=maybe_prefix(prefix, "self_attn"),
        )
        self.lconv1d = Gemma4AudioLightConv1d(
            config,
            prefix=maybe_prefix(prefix, "lconv1d"),
        )
        self.feed_forward2 = Gemma4AudioFeedForward(
            config,
            prefix=maybe_prefix(prefix, "feed_forward2"),
        )
        self.norm_pre_attn = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.norm_post_attn = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.norm_out = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.gradient_clipping = config.gradient_clipping

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor | None,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        gradient_clipping = min(
            self.gradient_clipping,
            torch.finfo(self.norm_pre_attn.weight.dtype).max,
        )
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states
        hidden_states = torch.clamp(
            hidden_states,
            -gradient_clipping,
            gradient_clipping,
        )
        hidden_states = self.norm_pre_attn(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = torch.clamp(
            hidden_states,
            -gradient_clipping,
            gradient_clipping,
        )
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states += residual
        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)
        hidden_states = torch.clamp(
            hidden_states,
            -gradient_clipping,
            gradient_clipping,
        )
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


class Gemma4AudioModel(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(
            config,
            prefix=maybe_prefix(prefix, "subsample_conv_projection"),
        )
        self.position_embeddings = Gemma4AudioRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [
                Gemma4AudioLayer(
                    config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx}"),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.output_proj = ReplicatedLinear(
            config.hidden_size,
            config.output_proj_dims,
            bias=False,
            prefix=maybe_prefix(prefix, "output_proj"),
            return_bias=False,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)

        hidden_states, attention_mask = self.subsample_conv_projection(
            input_features,
            attention_mask,
        )
        position_embeddings = self.position_embeddings(hidden_states)
        block_attention_mask = None
        if attention_mask is not None:
            # Reuse one block mask across audio layers.
            block_attention_mask = build_audio_attention_mask(
                attention_mask,
                chunk_size=self.config.attention_chunk_size,
                max_past_horizon=self.config.attention_context_left - 1,
                max_future_horizon=self.config.attention_context_right,
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                block_attention_mask,
                position_embeddings,
            )

        hidden_states = self.output_proj(
            hidden_states.to(dtype=self.output_proj.weight.dtype)
        )
        return hidden_states, attention_mask


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class Gemma4ImagePixelInputs(TensorSchema):
    """
    Pre-patchified image inputs from the Gemma4 image processor.

    Dimensions:
        - bn: Batch size * number of images
        - np: Number of patches (max_patches = max_soft_tokens * pooling_kernel_size²)
        - pp: Patch pixels (patch_size² * 3)

    The HF Gemma4ImageProcessor outputs pixel_values as
    (batch, max_patches, patch_pixels) — already patchified with
    zero-padding for patches beyond the real image content.
    pixel_position_ids provides (x, y) coordinates per patch,
    with (-1, -1) for padding patches.
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("bn", "np", "pp"),
    ]
    pixel_position_ids: Annotated[
        torch.Tensor,
        TensorShape("bn", "np", 2),
    ]


class Gemma4AudioInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of audios
        - s: Sequence length (MEL spectrogram frames)
        - f: Number of features (MEL bins)
    """

    type: Literal["audio"] = "audio"
    input_features_padded: Annotated[torch.Tensor, TensorShape("bn", "s", "f")]
    input_features_mask: Annotated[torch.Tensor, TensorShape("bn", "s")]


Gemma4ImageInputs = Gemma4ImagePixelInputs


class Gemma4VideoInputs(TensorSchema):
    """Video frame inputs — same tensor format as image inputs.

    Gemma4 has no separate video tower; video frames are processed
    through the vision tower at lower resolution (max_soft_tokens=70).
    """

    type: Literal["pixel_values_videos"] = "pixel_values_videos"
    pixel_values_videos: Annotated[
        torch.Tensor,
        TensorShape("bn", "np", "pp"),
    ]
    pixel_position_ids_videos: Annotated[
        torch.Tensor,
        TensorShape("bn", "np", 2),
    ]


# ---------------------------------------------------------------------------
# Processing info
# ---------------------------------------------------------------------------


class Gemma4ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Gemma4Config)

    def get_default_tok_params(self):
        """Gemma4's chat template already embeds a literal ``<bos>`` token in
        the rendered text.  If ``add_special_tokens=True`` (the base-class
        default), the tokenizer prepends *another* BOS, producing a
        ``[2, 2, ...]`` double-BOS sequence that the model was not trained on.

        Setting ``add_special_tokens=False`` here prevents the duplicate and
        ensures both ``llm.generate()`` and the chat/completions API behave
        correctly.
        """
        params = super().get_default_tok_params()
        params = params.with_kwargs(add_special_tokens=False)
        return params

    def get_hf_processor(self, **kwargs: object) -> Gemma4Processor:
        return self.ctx.get_hf_processor(
            Gemma4Processor,
            **kwargs,
        )

    def validate_num_items(self, modality: str, num_items: int) -> None:
        if (
            modality == "audio"
            and num_items > 0
            and self.get_hf_config().audio_config is None
        ):
            model = self.ctx.model_config.model
            raise ValueError(
                f"Audio input was provided but the model "
                f"'{model}' does not have an audio tower. "
                f"Audio inference is only supported for Gemma4 "
                f"models that include an audio_config "
                f"(i.e., models that include an audio_config)."
            )
        super().validate_num_items(modality, num_items)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        limits: dict[str, int | None] = {"image": None}
        if self.get_hf_config().audio_config is not None:
            limits["audio"] = None
        limits["video"] = None
        return limits

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        config = self.get_hf_config()
        # Report per-item encoder-token upper bounds.
        # Upper bound: the pooler outputs default_output_length slots
        # per image (280).  After padding is stripped the actual count
        # is ≤ this value, but vLLM needs the max for memory planning.
        tokens_per_image = config.vision_config.default_output_length
        tokens: dict[str, int] = {"image": tokens_per_image}
        if config.audio_config is not None:
            # Audio max tokens from the processor's audio_seq_length.
            processor = self.get_hf_processor()
            tokens["audio"] = processor.audio_seq_length
        # Video encoder budgeting tracks only embedded soft tokens, not the
        # timestamp or delimiter text that stays on the language-model path.
        tokens["video"] = _VIDEO_MAX_FRAMES * _VIDEO_MAX_SOFT_TOKENS
        return tokens

    def get_data_parser(self) -> MultiModalDataParser:
        config = self.get_hf_config()
        kwargs: dict[str, Any] = {"video_needs_metadata": True}
        if getattr(config, "audio_config", None) is not None:
            processor = self.get_hf_processor()
            kwargs["target_sr"] = processor.feature_extractor.sampling_rate
        return MultiModalDataParser(**kwargs)

    def _compute_num_soft_tokens(
        self,
        image_width: int,
        image_height: int,
        max_soft_tokens: int | None = None,
    ) -> int:
        """Compute the number of soft tokens the vision tower produces
        for an image of the given dimensions, after padding is stripped.

        Args:
            max_soft_tokens: Override for the vision config's
                ``default_output_length``.  When *None*, the value from
                the model config is used.
        """
        vision_cfg = self.get_hf_config().vision_config
        patch_size = vision_cfg.patch_size
        pooling_kernel_size = vision_cfg.pooling_kernel_size

        if max_soft_tokens is None:
            max_soft_tokens = vision_cfg.default_output_length

        unit = patch_size * pooling_kernel_size
        max_patches = max_soft_tokens * pooling_kernel_size**2
        num_patches_orig = (image_height / patch_size) * (image_width / patch_size)
        scale = math.sqrt(max_patches / num_patches_orig)
        target_h = max(unit, int(math.floor(image_height * scale / unit)) * unit)
        target_w = max(unit, int(math.floor(image_width * scale / unit)) * unit)
        num_patches = (target_h // patch_size) * (target_w // patch_size)
        return num_patches // (pooling_kernel_size**2)

    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Gemma4Processor | None,
        max_soft_tokens: int | None = None,
    ) -> PromptUpdateDetails[list[int]]:
        """Return the dynamic image token sequence for this image.

        Computes the exact number of soft tokens the vision tower will
        produce after stripping padding.

        Args:
            max_soft_tokens: Override for the default token budget.
                When *None*, falls back to the model config value.
        """
        if processor is None:
            processor = self.get_hf_processor()

        num_soft = self._compute_num_soft_tokens(
            image_width,
            image_height,
            max_soft_tokens=max_soft_tokens,
        )
        config = self.get_hf_config()
        token_ids = (
            [config.boi_token_id]
            + [processor.image_token_id] * num_soft
            + [config.eoi_token_id]
        )
        return PromptUpdateDetails.select_token_id(token_ids, processor.image_token_id)

    def get_audio_repl(
        self,
        *,
        audio_len: int,
        processor: Gemma4Processor | None,
    ) -> PromptUpdateDetails[list[int]]:
        """Return the dynamic audio token sequence for this audio.

        Computes the number of soft tokens from the audio waveform
        length using ``ceil(duration_ms / audio_ms_per_token)``.
        """
        if processor is None:
            processor = self.get_hf_processor()

        sampling_rate = processor.feature_extractor.sampling_rate
        num_tokens = processor._compute_audio_num_tokens(
            torch.zeros(audio_len), sampling_rate
        )
        config = self.get_hf_config()
        token_ids = (
            [config.boa_token_id]
            + [processor.audio_token_id] * num_tokens
            + [config.eoa_token_id]
        )
        return PromptUpdateDetails.select_token_id(token_ids, processor.audio_token_id)

    def get_video_repl(
        self,
        *,
        timestamps: list[float],
        num_soft_tokens_per_frame: list[int],
        processor: Gemma4Processor,
    ) -> PromptUpdateDetails[list[int]]:
        """Build the full token replacement for one video.

        Produces the same interleaved sequence as the HF Gemma4Processor:
            mm:ss <boi><|video|>*N<eoi> mm:ss <boi><|video|>*N<eoi> ...
        """
        tokenizer = self.ctx.get_tokenizer()
        config = self.get_hf_config()

        boi_token_id = config.boi_token_id
        eoi_token_id = config.eoi_token_id
        video_token_id = processor.video_token_id

        all_token_ids: list[int] = []
        for i, (ts, n_tokens) in enumerate(zip(timestamps, num_soft_tokens_per_frame)):
            # mm:ss timestamp — matches transformers: int-truncated,
            # zero-padded.
            minutes = int(ts // 60)
            seconds = int(ts % 60)
            ts_str = f"{minutes:02d}:{seconds:02d}"

            prefix = f" {ts_str} " if i > 0 else f"{ts_str} "
            ts_token_ids = tokenizer.encode(prefix, add_special_tokens=False)
            all_token_ids.extend(ts_token_ids)

            all_token_ids.append(boi_token_id)
            all_token_ids.extend([video_token_id] * n_tokens)
            all_token_ids.append(eoi_token_id)

        return PromptUpdateDetails.select_token_id(all_token_ids, video_token_id)


# ---------------------------------------------------------------------------
# Dummy inputs builder
# ---------------------------------------------------------------------------


class Gemma4DummyInputsBuilder(BaseDummyInputsBuilder[Gemma4ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        num_videos = mm_counts.get("video", 0)
        processor = self.info.get_hf_processor()
        # Use image_token (<|image|>) with tab prefix — this is what the
        # Gemma4 chat template inserts per image (\t<|image|>).
        # _get_prompt_updates targets image_token and expands it to the
        # full_image_sequence.
        text = ("\t" + processor.image_token) * num_images
        if num_audios > 0 and processor.audio_token:
            text += processor.audio_token * num_audios
        if num_videos > 0:
            text += processor.video_token * num_videos
        return text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        num_videos = mm_counts.get("video", 0)
        processor = self.info.get_hf_processor()
        image_processor = processor.image_processor
        # Use processor's configured image size for dummies.
        # Gemma4ImageProcessor sets size=None (it uses patch_size /
        # max_soft_tokens instead of the standard size dict), so we
        # guard against None with `or {}`.
        size = getattr(image_processor, "size", None) or {}
        img_width = size.get("width", 224)
        img_height = size.get("height", 224)

        image_overrides = mm_options.get("image") if mm_options else None
        audio_overrides = mm_options.get("audio") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None

        data: MultiModalDataDict = {
            "image": self._get_dummy_images(
                width=img_width,
                height=img_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }

        if num_audios > 0:
            audio_len = processor.feature_extractor.fft_length
            data["audio"] = self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )

        if num_videos > 0:
            data["video"] = self._get_dummy_videos(
                width=img_width,
                height=img_height,
                num_frames=_VIDEO_MAX_FRAMES,
                num_videos=num_videos,
                overrides=video_overrides,
            )

        return data

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
        overrides: VideoDummyOptions | None = None,
    ) -> list[VideoItem]:
        num_frames = max(num_frames, 2)
        videos = super()._get_dummy_videos(
            width=width,
            height=height,
            num_frames=num_frames,
            num_videos=num_videos,
            overrides=overrides,
        )
        videos = [v.copy() for v in videos]

        video_items: list[VideoItem] = []
        for video in videos:
            video_num_frames = video.shape[0]
            video_metadata = {
                "fps": 2.0,
                "duration": video_num_frames / 2.0,
                "total_num_frames": video_num_frames,
                "frames_indices": list(range(video_num_frames)),
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            video_items.append((video, video_metadata))

        return video_items


# ---------------------------------------------------------------------------
# Multimodal processor
# ---------------------------------------------------------------------------


class Gemma4MultiModalProcessor(BaseMultiModalProcessor[Gemma4ProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)

        # Expand videos into frame batches for the shared tower.
        # ---- VIDEO HANDLING ----
        # Gemma4 decomposes video into timestamped image frames.
        # Each frame is processed with max_soft_tokens=70 through the
        # same vision tower, matching transformers processing_gemma4.py.
        video_outputs: dict[str, Any] = {}
        if videos := mm_data.pop("videos", []):
            processor = self.info.get_hf_processor()

            all_video_pixel_values: list[torch.Tensor] = []
            all_video_position_ids: list[torch.Tensor] = []
            video_num_soft_tokens_per_video: list[list[int]] = []
            video_timestamps_per_video: list[list[float]] = []
            video_frame_counts: list[int] = []

            for item in videos:
                video_array, metadata = item

                # Convert frames to PIL images
                if isinstance(video_array, np.ndarray):
                    frames = [
                        PILImage.fromarray(video_array[i])
                        for i in range(video_array.shape[0])
                    ]
                else:
                    frames = list(video_array)

                # Compute timestamps from metadata (same as transformers)
                fps = metadata.get("fps") or 24
                frame_indices = metadata.get("frames_indices", list(range(len(frames))))
                timestamps = [idx / fps for idx in frame_indices]

                # Process frames as images with max_soft_tokens=70
                video_mm_kwargs = dict(mm_kwargs)
                video_mm_kwargs["max_soft_tokens"] = _VIDEO_MAX_SOFT_TOKENS

                dummy_prompt = ("\t" + processor.image_token) * len(frames)

                frame_outputs = super()._call_hf_processor(
                    prompt=dummy_prompt,
                    mm_data={"images": frames},
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )

                # Remap HF key name
                if "image_position_ids" in frame_outputs:
                    frame_outputs["pixel_position_ids"] = frame_outputs.pop(
                        "image_position_ids"
                    )

                all_video_pixel_values.append(frame_outputs["pixel_values"])
                all_video_position_ids.append(frame_outputs["pixel_position_ids"])

                # Compute soft tokens per frame
                num_soft_per_frame = []
                for img in frames:
                    w, h = img.size
                    n = self.info._compute_num_soft_tokens(
                        w, h, max_soft_tokens=_VIDEO_MAX_SOFT_TOKENS
                    )
                    num_soft_per_frame.append(n)

                video_num_soft_tokens_per_video.append(num_soft_per_frame)
                video_timestamps_per_video.append(timestamps)
                video_frame_counts.append(len(frames))

                # Build expanded replacement text and replace the
                # <|video|> placeholder in the prompt.
                # Use split(token, 1) to avoid collision — the
                # replacement text itself contains <|video|> tokens.
                ts_strs = [f"{int(s // 60):02d}:{int(s % 60):02d}" for s in timestamps]
                replacement = " ".join(
                    f"{t} {processor.boi_token}"
                    f"{processor.video_token * n}"
                    f"{processor.eoi_token}"
                    for t, n in zip(ts_strs, num_soft_per_frame)
                )
                parts = prompt.split(processor.video_token, 1)
                if len(parts) == 2:
                    prompt = parts[0] + replacement + parts[1]

            video_outputs = {
                "pixel_values_videos": torch.cat(all_video_pixel_values, dim=0),
                "pixel_position_ids_videos": torch.cat(all_video_position_ids, dim=0),
                "video_frame_counts": torch.tensor(video_frame_counts),
                "video_num_soft_tokens": video_num_soft_tokens_per_video,
                "video_timestamps": video_timestamps_per_video,
            }

        # The processor accepts 'audio' not 'audios'.
        if "audios" in mm_data:
            mm_data["audio"] = mm_data.pop("audios")

        # Warn if any audio waveform exceeds the model's max duration.
        if "audio" in mm_data:
            processor = self.info.get_hf_processor()
            sr = processor.feature_extractor.sampling_rate
            max_tokens = processor.audio_seq_length
            ms_per_tok = processor.audio_ms_per_token
            max_duration_s = max_tokens * ms_per_tok / 1000.0
            audios = mm_data["audio"]
            if not isinstance(audios, (list, tuple)):
                audios = [audios]
            for i, waveform in enumerate(audios):
                duration_s = len(waveform) / sr
                if duration_s > max_duration_s:
                    logger.warning(
                        "Audio duration exceeds max: %f > %f seconds",
                        duration_s,
                        max_duration_s,
                    )
        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs,
        )

        # HF uses 'image_position_ids'; vLLM uses 'pixel_position_ids'.
        # Remap here to keep a single translation point.
        if "image_position_ids" in processed_outputs:
            processed_outputs["pixel_position_ids"] = processed_outputs.pop(
                "image_position_ids"
            )

        if "input_features" in processed_outputs:
            # Keep padded features for batched audio tower execution.
            processed_outputs["input_features_padded"] = processed_outputs[
                "input_features"
            ]
            # Unpad per-item so each item's cache entry is self-contained.
            unpadded_features = [
                f[mask]
                for f, mask in zip(
                    processed_outputs["input_features"],
                    processed_outputs["input_features_mask"],
                )
            ]
            processed_outputs["input_features"] = unpadded_features

        # Merge video outputs into the final result
        combined_outputs = dict(processed_outputs, **video_outputs)
        return BatchFeature(combined_outputs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            pixel_position_ids=MultiModalFieldConfig.batched("image"),
            input_features_padded=MultiModalFieldConfig.batched("audio"),
            input_features_mask=MultiModalFieldConfig.batched("audio"),
        )

        # Video fields: frames stored flat, split per video by
        # video_frame_counts.
        video_frame_counts = hf_inputs.get("video_frame_counts")
        if video_frame_counts is not None:
            vfc = video_frame_counts
            if not isinstance(vfc, torch.Tensor):
                vfc = torch.tensor(vfc)
            fields.update(
                pixel_values_videos=(
                    MultiModalFieldConfig.flat_from_sizes("video", vfc)
                ),
                pixel_position_ids_videos=(
                    MultiModalFieldConfig.flat_from_sizes("video", vfc)
                ),
                video_frame_counts=MultiModalFieldConfig.batched(
                    "video",
                ),
                video_num_soft_tokens=MultiModalFieldConfig.batched(
                    "video", keep_on_cpu=True
                ),
                video_timestamps=MultiModalFieldConfig.batched(
                    "video", keep_on_cpu=True
                ),
            )

        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        prompt_updates = []

        if "image" in mm_items:
            # Target image_token (<|image|>) — the single placeholder the
            # Gemma4 chat template inserts once per image in the prompt.
            # vLLM tokenizes the prompt without token expansion, so only
            # one image_token exists per image in the token stream.
            # The replacement expands it to the full image sequence
            # (boi + N×image_token + eoi, where N = max_soft_tokens).
            image_token = hf_processor.image_token

            def get_replacement_image(item_idx: int):
                images = mm_items.get_items("image", ImageProcessorItems)
                image_size = images.get_image_size(item_idx)
                # Resolve the effective max_soft_tokens by merging
                # per-prompt kwargs with the config-level defaults,
                # consistent with how _call_hf_processor resolves it.
                # Without this merge, a missing per-prompt override
                # would fall back to vision_cfg.default_output_length
                # instead of the config's mm_processor_kwargs default.
                merged_kwargs = self.info.ctx.get_merged_mm_kwargs(
                    hf_processor_mm_kwargs,
                )
                max_soft_tokens = merged_kwargs.get("max_soft_tokens")
                return self.info.get_image_repl(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                    max_soft_tokens=max_soft_tokens,
                )

            prompt_updates.append(
                PromptReplacement(
                    modality="image",
                    target=image_token,
                    replacement=get_replacement_image,
                )
            )

        if "video" in mm_items:
            video_token = hf_processor.video_token

            def get_replacement_video(item_idx: int):
                out_item = out_mm_kwargs["video"][item_idx]
                timestamps = out_item["video_timestamps"].data
                num_soft = out_item["video_num_soft_tokens"].data
                return self.info.get_video_repl(
                    timestamps=timestamps,
                    num_soft_tokens_per_frame=num_soft,
                    processor=hf_processor,
                )

            prompt_updates.append(
                PromptReplacement(
                    modality="video",
                    target=video_token,
                    replacement=get_replacement_video,
                )
            )

        if "audio" in mm_items:
            audio_token = hf_processor.audio_token

            def get_replacement_audio(item_idx: int):
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)
                return self.info.get_audio_repl(
                    audio_len=audio_len,
                    processor=hf_processor,
                )

            prompt_updates.append(
                PromptReplacement(
                    modality="audio",
                    target=audio_token,
                    replacement=get_replacement_audio,
                )
            )

        return prompt_updates

    # NOTE: Gemma3/Gemma3n override _apply_token_matches and
    # _find_mm_placeholders to merge adjacent newline tokens that arise
    # when full_image_sequence contains "\n\n" wrappers.  Gemma4's
    # full_image_sequence has NO newlines (just BOI + 280×image_token +
    # EOI), so the base class implementations work correctly as-is.


# ---------------------------------------------------------------------------
# Multimodal embedder
# ---------------------------------------------------------------------------


class Gemma4MultimodalEmbedder(nn.Module):
    """Projects vision/audio soft tokens into LM embedding space.

    Architecture:
        inputs_embeds → embedding_projection → embedding_post_projection_norm

    Unlike Gemma3n which has separate hard/soft embedding paths with
    per-path normalization and a learned embedding table, Gemma4 uses a
    simplified 2-layer design: a linear projection followed by RMSNorm
    (without learnable scale).  The checkpoint confirms this — only
    ``embedding_projection.weight`` exists; there is no embedding table
    or pre-projection norm weights.
    """

    def __init__(
        self,
        multimodal_config: Gemma4VisionConfig | Gemma4AudioConfig,
        text_config: Gemma4TextConfig,
    ):
        super().__init__()

        self.eps = multimodal_config.rms_norm_eps
        self.text_hidden_size = text_config.hidden_size

        # Audio tower uses output_proj_dims (1536) rather than hidden_size
        # (1024); vision uses hidden_size (768) directly.
        embedding_dim = (
            getattr(multimodal_config, "output_proj_dims", None)
            or multimodal_config.hidden_size
        )

        self.embedding_projection = ReplicatedLinear(
            embedding_dim,
            self.text_hidden_size,
            bias=False,
        )

        self.embedding_post_projection_norm = RMSNorm(
            self.text_hidden_size,
            eps=self.eps,
            has_weight=False,
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Project soft tokens from a multimodal tower into LM space."""
        embs_proj, _ = self.embedding_projection(inputs_embeds)
        return self.embedding_post_projection_norm(embs_proj)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class Gemma4ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
    SupportsEagle3,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # Maps checkpoint prefixes to vLLM module paths.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.embed_audio.": "embed_audio.",
            "model.embed_vision.": "embed_vision.",
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.audio_tower.": "audio_tower.",
            "lm_head.": "language_model.lm_head.",
            "model": "language_model.model",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config

        # ---- Vision tower (shared by image and video) ----
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = Gemma4VisionModel(
                config.vision_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.embed_vision = Gemma4MultimodalEmbedder(
                config.vision_config, config.text_config
            )

        # ---- Audio tower (variants with audio_config) ----
        if config.audio_config is not None:
            with self._mark_tower_model(vllm_config, "audio"):
                self.audio_tower = Gemma4AudioModel(
                    config.audio_config,
                    prefix=maybe_prefix(prefix, "audio_tower"),
                )
                self.embed_audio = Gemma4MultimodalEmbedder(
                    config.audio_config, config.text_config
                )
        else:
            self.audio_tower = None
            self.embed_audio = None

        # ---- Language model (vLLM optimised) ----
        with self._mark_language_model(vllm_config):
            self.language_model: Gemma4ForCausalLM = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Gemma4ForCausalLM"],
            )

            # Pre-allocate PLE buffer for CUDA graph compatibility.
            # Some variants have hidden_size_per_layer_input=None (no PLE).
            ple_dim = config.text_config.hidden_size_per_layer_input
            if ple_dim is not None:
                self.per_layer_embeddings = torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.num_hidden_layers,
                    ple_dim,
                    device=(self.language_model.model.embed_tokens.weight.device),
                    dtype=(self.language_model.model.embed_tokens.weight.dtype),
                )
            else:
                self.per_layer_embeddings = None

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # --- MixtureOfExperts delegation to language_model ---
        self.expert_weights = self.language_model.expert_weights
        self.moe_layers = self.language_model.moe_layers
        self.num_moe_layers = self.language_model.num_moe_layers
        self.num_logical_experts = self.language_model.num_logical_experts
        self.num_physical_experts = self.language_model.num_physical_experts
        self.num_local_physical_experts = self.language_model.num_local_physical_experts
        self.num_routed_experts = self.language_model.num_routed_experts
        self.num_expert_groups = self.language_model.num_expert_groups
        self.num_shared_experts = self.language_model.num_shared_experts
        self.num_redundant_experts = self.language_model.num_redundant_experts

    # ------------------------------------------------------------------ #
    # Input parsing
    # ------------------------------------------------------------------ #

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Gemma4ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_position_ids = kwargs.pop("pixel_position_ids", None)
        image_embeds = kwargs.pop("image_embeds", None)
        assert image_embeds is None, "Gemma4 does not support image_embeds."
        if pixel_values is None:
            return None
        return Gemma4ImagePixelInputs(
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Gemma4AudioInputs | None:
        input_features_padded = kwargs.pop("input_features_padded", None)
        if input_features_padded is None:
            return None
        input_features_mask = kwargs.pop("input_features_mask", None)
        if input_features_mask is None:
            return None
        return Gemma4AudioInputs(
            input_features_padded=input_features_padded,
            input_features_mask=input_features_mask,
        )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> dict[str, torch.Tensor] | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        pixel_position_ids_videos = kwargs.pop("pixel_position_ids_videos", None)
        video_frame_counts = kwargs.pop("video_frame_counts", None)
        if pixel_values_videos is None:
            return None
        return {
            "pixel_values_videos": pixel_values_videos,
            "pixel_position_ids_videos": pixel_position_ids_videos,
            "video_frame_counts": video_frame_counts,
        }

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, Gemma4ImageInputs | Gemma4AudioInputs | Gemma4VideoInputs | None]:
        mm_input_by_modality = {}
        for input_key in list(kwargs):
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key == "pixel_values_videos"
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
            if (
                input_key == "input_features_padded"
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        return mm_input_by_modality

    # ------------------------------------------------------------------ #
    # Image processing
    # ------------------------------------------------------------------ #

    def _process_image_input(
        self,
        image_input: Gemma4ImageInputs,
    ) -> list[torch.Tensor]:
        pixel_values = image_input["pixel_values"]
        pixel_position_ids = image_input["pixel_position_ids"]
        vt = self.vision_tower
        pooling_k2 = self.config.vision_config.pooling_kernel_size**2
        output_length = pixel_values.shape[1] // pooling_k2
        hidden_states, padding_positions = vt(
            pixel_values,
            pixel_position_ids,
            output_length=output_length,
        )
        # Project pooled vision states into LM space.
        target_dtype = self.embed_vision.embedding_projection.weight.dtype
        projected_states = self.embed_vision(
            inputs_embeds=hidden_states.to(target_dtype)
        )
        return [
            projected_states[i, torch.logical_not(padding_positions[i])]
            for i in range(projected_states.shape[0])
        ]

    # ------------------------------------------------------------------ #
    # Video processing (frames through vision tower)
    # ------------------------------------------------------------------ #

    def _process_video_input(
        self,
        video_input: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        pixel_values = video_input["pixel_values_videos"]
        pixel_position_ids = video_input["pixel_position_ids_videos"]
        frame_counts = video_input["video_frame_counts"]

        vt = self.vision_tower
        pooling_k2 = self.config.vision_config.pooling_kernel_size**2
        target_dtype = self.embed_vision.embedding_projection.weight.dtype
        output_length = pixel_values.shape[1] // pooling_k2
        hidden_states, padding_positions = vt(
            pixel_values,
            pixel_position_ids,
            output_length=output_length,
        )
        # Project frame states before regrouping per video.
        projected_states = self.embed_vision(
            inputs_embeds=hidden_states.to(target_dtype)
        )
        per_frame_embeddings = [
            projected_states[i, torch.logical_not(padding_positions[i])]
            for i in range(projected_states.shape[0])
        ]

        if isinstance(frame_counts, torch.Tensor):
            fc_list = frame_counts.tolist()
        else:
            fc_list = list(frame_counts)

        per_video_embeddings = []
        frame_idx = 0
        for frame_count in fc_list:
            next_frame_idx = frame_idx + frame_count
            per_video_embeddings.append(
                torch.cat(per_frame_embeddings[frame_idx:next_frame_idx], dim=0)
            )
            frame_idx = next_frame_idx

        return per_video_embeddings

    # ------------------------------------------------------------------ #
    # Audio processing
    # ------------------------------------------------------------------ #

    def _process_audio_input(
        self,
        audio_input: Gemma4AudioInputs,
    ) -> list[torch.Tensor]:
        input_features = audio_input["input_features_padded"].squeeze(1)
        input_features_mask = audio_input["input_features_mask"].squeeze(1)

        # Run audio tower — mask uses standard HF convention
        # (True=valid, False=padding).
        audio_outputs = self.audio_tower(input_features, input_features_mask)
        if isinstance(audio_outputs, tuple):
            audio_encodings, audio_mask = audio_outputs
        else:
            audio_encodings = audio_outputs.last_hidden_state
            audio_mask = audio_outputs.attention_mask

        # Project valid audio states into LM space.
        audio_features = self.embed_audio(inputs_embeds=audio_encodings)

        # Strip padding per-batch element: only keep real (non-padding)
        # tokens. audio_mask is True for valid positions (HF convention).
        per_audio = []
        for enc, mask in zip(audio_features, audio_mask, strict=True):
            per_audio.append(enc[mask])  # [num_real, hidden_size]

        return per_audio

    # ------------------------------------------------------------------ #
    # MultiModalEmbeddings interface
    # ------------------------------------------------------------------ #

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        multimodal_embeddings: list[torch.Tensor] = []

        for modality, multimodal_input in mm_input_by_modality.items():
            if multimodal_input is None:
                continue
            if modality == "image":
                multimodal_embeddings.extend(
                    self._process_image_input(multimodal_input)
                )
            elif modality == "video":
                multimodal_embeddings.extend(
                    self._process_video_input(multimodal_input)
                )
            elif modality == "audio":
                multimodal_embeddings.extend(
                    self._process_audio_input(multimodal_input)
                )

        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Cache per-layer embeddings (PLE) for the language model's
        # forward pass.  During profiling embed_input_ids is not called,
        # so the pre-allocated zeros are used instead.
        if self.per_layer_embeddings is not None:
            # Mask multimodal tokens (image/audio) to 0 for PLE
            # computation (using token_type_ids == 0 as text_mask).
            # Replicate this: map image token positions to token 0.
            if is_multimodal is not None:
                is_multimodal = is_multimodal.to(input_ids.device)
                ple_input_ids = torch.where(
                    is_multimodal, torch.zeros_like(input_ids), input_ids
                )
            else:
                ple_input_ids = input_ids

            per_layer_inputs = self.language_model.model.get_per_layer_inputs(
                ple_input_ids
            )
            if per_layer_inputs is not None:
                per_layer_inputs = per_layer_inputs.reshape(
                    -1,
                    self.config.text_config.num_hidden_layers,
                    self.config.text_config.hidden_size_per_layer_input,
                )
                self.per_layer_embeddings[: per_layer_inputs.shape[0]].copy_(
                    per_layer_inputs
                )

        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # Select the pre-cached PLEs for this batch (None when PLE
        # is disabled for variants without PLE).
        per_layer_inputs = (
            self.per_layer_embeddings[: inputs_embeds.shape[0]]
            if self.per_layer_embeddings is not None and inputs_embeds is not None
            else None
        )

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            per_layer_inputs=per_layer_inputs,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    # ------------------------------------------------------------------ #
    # Weight loading
    # ------------------------------------------------------------------ #

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Some checkpoints have vestigial embed_vision.embedding and
        # embed_audio.embedding weights from the Gemma3n architecture
        # that are not used by Gemma4's MultimodalEmbedder (which only
        # has embedding_projection + embedding_post_projection_norm).
        ignore_prefixes = [
            "embed_vision.embedding.",
            "embed_audio.embedding.",
        ]
        # Models without audio tower should skip
        # audio weights entirely.
        if self.audio_tower is None:
            ignore_prefixes.extend(
                [
                    "audio_tower.",
                    "embed_audio.",
                ]
            )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=ignore_prefixes,
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    # ------------------------------------------------------------------ #
    # LoRA / multimodal mapping
    # ------------------------------------------------------------------ #

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix mapping for multimodal models."""
        # Expose tower and connector prefixes for MM LoRA.
        connectors = ["embed_vision"]
        tower_models = ["vision_tower"]
        if self.audio_tower is not None:
            connectors.append("embed_audio")
            tower_models.append("audio_tower")

        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=connectors,
            tower_model=tower_models,
        )

    def get_num_mm_encoder_tokens(self, num_mm_tokens: int) -> int:
        return num_mm_tokens * self.config.vision_config.pooling_kernel_size**2

    def get_num_mm_connector_tokens(self, num_encoder_tokens: int) -> int:
        return num_encoder_tokens // self.config.vision_config.pooling_kernel_size**2

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<image_soft_token>"
        if modality == "audio":
            return "<audio_soft_token>"
        if modality == "video":
            return "<|video|>"
        raise ValueError(f"Unsupported modality: {modality}")
