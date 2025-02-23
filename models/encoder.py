from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import math

from torch.nn.functional import scaled_dot_product_attention

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )
    
class AliBiRelPositionalEncoding(nn.Module):
    
    # Follow https://github.com/NVIDIA/NeMo/blob/cef98dbaa61971b889bb2484916b90c11a4c2a2d/nemo/collections/nlp/modules/common/megatron/position_embedding/alibi_relative_position_embedding.py#L41
    def get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = (
                get_slopes_power_of_2(closest_power_of_2)
                + self.get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

        return slopes
    
    def build_slopes(self, num_attention_heads, num_attention_heads_alibi):
        """
        Builds a slopes tensor.
        """
        slopes = (
            torch.Tensor(self.get_slopes(num_attention_heads_alibi) + [0] * (num_attention_heads - num_attention_heads_alibi))
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        return slopes
    
    def build_relative_position(self, max_seq_len, full=True):
        """
        full=True:  shape (max_seq_len, max_seq_len)
        full=False: shape (max_seq_len)
        """
        relative_position = torch.arange(1 - max_seq_len, 1)[None, :].mul(-1)  # (1, max_seq_len)

        if full:
            memory_position = torch.arange(1 - max_seq_len, 1)[:, None].mul(-1)
            relative_position = torch.abs(memory_position - relative_position)  # (max_seq_len, max_seq_len)

        return relative_position

    def __init__(
        self, bidirectional, num_attention_heads, num_attention_heads_alibi=None, max_seq_len=512,
    ):
        """
        Args:
            bidirectional: Whether to use bidirectional relative position embedding
            num_attention_heads: Number of attention heads
            num_attention_heads_alibi: Number of attention heads for which alibi bias will be used
            max_seq_len: Maximum sequence length for precomputed relative positions. Larger sizes will result in more memory usage by computing alibi mask on-the-fly.
        """
        super().__init__()

        if (num_attention_heads_alibi is None) or (num_attention_heads_alibi <= 0):
            num_attention_heads_alibi = num_attention_heads

        if num_attention_heads_alibi > num_attention_heads:
            raise ValueError(
                f"num_attention_heads_alibi ({num_attention_heads_alibi}) cannot be larger than num_attention_heads ({num_attention_heads})"
            )
        
        self.bidirectional = bidirectional
        self.num_attention_heads = num_attention_heads
        # define the size of pre-computed relative position slopes.
        # define the number of attention heads for which alibi mask will be pre-computed (the rest are disabled).
        self.num_attention_heads_alibi = num_attention_heads_alibi
        # Larger sizes will result in more memory usage by computing alibi mask on-the-fly.
        self.max_seq_len = max_seq_len

        self.register_buffer("slopes", self.build_slopes(num_attention_heads, num_attention_heads_alibi))
        self.register_buffer("relative_position", self.build_relative_position(max_seq_len, full=bidirectional).unsqueeze(0).expand(num_attention_heads, -1, -1))
        
    def forward(self, query_seq_length, key_seq_length):
        # used cached relative position if possible
        max_seq_len = max(query_seq_length, key_seq_length)
        if max_seq_len > self.max_seq_len:
            relative_position = (
                self.build_relative_position(max_seq_len, full=self.bidirectional)
                .unsqueeze(0)
                .expand(self.num_attention_heads, -1, -1)
            )
        else:
            relative_position = self.relative_position

        relative_position = relative_position[:, -query_seq_length:, -key_seq_length:]

        return -relative_position.unsqueeze(0) * self.slopes
    
    def export_forward(self, query_seq_length, key_seq_length):
        relative_position = self.relative_position
        relative_position = relative_position[:, -query_seq_length:, -key_seq_length:]

        return -relative_position.unsqueeze(0) * self.slopes

class ALiBiMultiHeadAttention(nn.Module):

    def __init__(self, n_state: int, n_head: int, att_context_size: Tuple[int, int]):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        self.att_context_size = att_context_size

        self.alibi = AliBiRelPositionalEncoding(
            bidirectional=False, num_attention_heads=n_head, num_attention_heads_alibi=n_head
        )

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
    ):
        q = self.query(x)
        
        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)

        if k_cache is not None and v_cache is not None:
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            k_cache = k[:, -self.att_context_size[0]:, :]
            v_cache = v[:, -self.att_context_size[0]:, :]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv), k_cache, v_cache

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if self.training:
            # Dynamically create alibi mask based on the current query and key lengths
            alibi_mask = self.alibi(q.size(2), k.size(2))
        else:
            # When streaming, q.size(2) and k.size(2) is always = self.att_context_size[0] + self.att_context_size[1] + 1
            alibi_mask = self.alibi.export_forward(q.size(2), k.size(2))
        mask = mask + alibi_mask
        a = scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False # is_causal=False must be False since we prepare the mask ourselves
        )
        out = a.permute(0, 2, 1, 3).flatten(start_dim=2)

        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, attention_context_size: Tuple[int, int]):
        super().__init__()

        self.attn = ALiBiMultiHeadAttention(n_state, n_head, attention_context_size)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
    ):
        residual = x
        x, k_cache, v_cache = self.attn(self.attn_ln(x), mask=mask, k_cache=k_cache, v_cache=v_cache)
        x = x + residual

        x = x + self.mlp(self.mlp_ln(x))
        return x, k_cache, v_cache


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_state: int, n_head: int, n_layer: int, att_context_size: Tuple[int, int]
    ):
        super().__init__()

        self.att_context_size = att_context_size

        kernel_size = 3
        self.conv1 = Conv1d(n_mels, n_state, kernel_size, padding=0, stride=2)
        self.conv2 = Conv1d(n_state, n_state, kernel_size, padding=0, stride=2)
        self.conv3 = Conv1d(n_state, n_state, kernel_size, padding=0, stride=2)
        # self.register_buffer("positional_embedding", sinusoids(1500, n_state)) # maximum 1500 positions

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, att_context_size) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def form_attention_mask_for_streaming(self, att_context_size, padding_length, offset, device):
        max_audio_length = max(padding_length).detach().item()
        att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)
        if att_context_size[1] == -1:
            if att_context_size[0] >= 0:
                att_mask = att_mask.triu(diagonal=-att_context_size[0])
        else:
            chunk_size = att_context_size[1] + 1
            if att_context_size[0] >= 0:
                left_chunks_num = att_context_size[0] // chunk_size
            else:
                left_chunks_num = 10000

            chunk_idx = torch.arange(0, max_audio_length, dtype=torch.int, device=att_mask.device)
            chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode='trunc')
            diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
            chunked_limited_mask = torch.logical_and(
                torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
            )
            att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))

        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(1)

        # Offset 
        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        # Combine attention mask with padding mask
        # pad_mask_for_att_mask need also be max_audio_length x max_audio_length
        pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))

        att_mask = att_mask[:, :max_audio_length, :max_audio_length]
        att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
        att_mask = ~att_mask
        att_mask = att_mask.to(torch.float) * -10000.0 # For attention_scores + attention_mask.to(attention_scores.dtype) in scaled_dot_product_attention
        att_mask = att_mask.unsqueeze(1) # For head dimension
        return att_mask        

    def get_length_after_conv(self, x_len):
        # Length after three Conv1d
        # Length calculated follow https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        x_len = torch.floor((x_len + 2*0 - 1*(3 - 1) - 1) / 2 + 1) # conv1
        x_len = torch.floor((x_len + 2*0 - 1*(3 - 1) - 1) / 2 + 1) # conv2
        x_len = torch.floor((x_len + 2*0 - 1*(3 - 1) - 1) / 2 + 1) # conv3
        return x_len.int()

    def forward(self, 
                x: Tensor, x_len: Tensor,
                k_cache: Optional[Tensor] = None, v_cache: Optional[Tensor] = None, cache_len: Optional[Tensor] = None):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = x.permute(0, 2, 1)

        # x = (x + self.positional_embedding[:x.shape[1], :]).to(x.dtype)

        x_len = self.get_length_after_conv(x_len)
        
        if k_cache is not None:
            x_len = x_len + self.att_context_size[0]
            offset = torch.neg(cache_len) + self.att_context_size[0]
        else:
            offset = None
        attn_mask = self.form_attention_mask_for_streaming(self.att_context_size, x_len, offset, x.device)

        new_k_cache = []
        new_v_cache = []
        for i, block in enumerate(self.blocks):
            if k_cache is not None:
                x, layer_k_cache, layer_v_cache = block(x, mask=attn_mask, k_cache=k_cache[i], v_cache=v_cache[i])
                new_k_cache.append(layer_k_cache)
                new_v_cache.append(layer_v_cache)
            else:
                x, _, _ = block(x, mask=attn_mask)

        x = self.ln_post(x)

        if k_cache is not None:
            new_k_cache = torch.stack(new_k_cache, dim=0)
            new_v_cache = torch.stack(new_v_cache, dim=0)
            return x, x_len, new_k_cache, new_v_cache, torch.clamp(cache_len + x_len, max=self.att_context_size[0])
        return x, x_len



