import math
import numpy as np
from typing import Optional, Tuple, Union, Callable

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast

import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def cross_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    batch_size = hidden_states.shape[0]
    kv_shape = (batch_size, -1, self.config.num_key_value_heads, self.head_dim)
    # sin and cos are specific to RoPE models; cache_position needed for the static cache
    cos, sin = position_embeddings
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    # get the cache for the previous layer
    k_cache, v_cache = past_key_value.update(
        None, None, self.layer_idx - 1, cache_kwargs
    )

    k_cache = k_cache.transpose(1, 2).reshape(
        -1, self.config.num_key_value_heads * self.head_dim
    )
    v_cache = v_cache.transpose(1, 2).reshape(
        -1, self.config.num_key_value_heads * self.head_dim
    )
    print("k_cache", k_cache.shape)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = (
        self.k_proj(self.up_proj(self.down_proj(k_cache)))
        .view(kv_shape)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(self.up_proj(self.down_proj(v_cache)))
        .view(kv_shape)
        .transpose(1, 2)
    )
    print("query_states", query_states.shape)
    print("key_states", key_states.shape)
    print("value_states", value_states.shape)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


global layer_id
layer_id = 32


def enable_yoco_attention_eval(model, args):
    print("Enabling yoco attention...")
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_yoco_attention_eval(
                module,
                args,
            )

        global layer_id
        if isinstance(module, (LlamaAttention, MistralAttention)):
            # For longchat model
            layer_id -= 1
            if layer_id % 2 == 0:
                continue
            model._modules[name].layer_id = layer_id
            model._modules[name].self_attn_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                cross_attn_forward, model._modules[name]
            )
            print(f"layer {layer_id} is set as cross attention layer")
