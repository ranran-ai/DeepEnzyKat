import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
from typing import Optional, Tuple

class ActivationFunction(Enum):
    silu = nn.SiLU
    sigmoid = nn.Sigmoid
    tanh = nn.Tanh
    softplus = nn.Softplus
    relu = nn.ReLU
    leakyrelu = nn.LeakyReLU
    prelu = nn.PReLU
    selu = nn.SELU
    elu = nn.ELU


class CrossAttention(nn.Module):













    def __init__(self, dim_q, dim_kv, dim_out, num_heads=4, dropout=0.1, ffn_mult=2, 
                 activation_type="leakyrelu", use_bias=True, init_scale=1.0):
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        
        self.activation_type = activation_type
        self.dim = dim_out
        self.h = num_heads
        self.dk = dim_out // num_heads
        self.scale = 1.0 / math.sqrt(self.dk)
        self.init_scale = init_scale


        self.q_proj = nn.Linear(dim_q,  dim_out, bias=use_bias)
        self.k_proj = nn.Linear(dim_kv, dim_out, bias=use_bias)
        self.v_from_kv = nn.Linear(dim_kv, dim_out, bias=use_bias)
        self.v_from_q  = nn.Linear(dim_q,  dim_out, bias=use_bias)


        self.o_proj_q = nn.Linear(dim_out, dim_out, bias=use_bias)
        self.o_proj_k = nn.Linear(dim_out, dim_out, bias=use_bias)


        self.q_short = nn.Identity() if dim_q == dim_out else nn.Linear(dim_q, dim_out, bias=False)
        self.k_short = nn.Identity() if dim_kv == dim_out else nn.Linear(dim_kv, dim_out, bias=False)


        self.ln_q_attn_q = nn.LayerNorm(dim_out, eps=1e-6)
        self.ln_q_attn_k = nn.LayerNorm(dim_out, eps=1e-6)
        self.ln_k_attn_k = nn.LayerNorm(dim_out, eps=1e-6)
        self.ln_k_attn_q = nn.LayerNorm(dim_out, eps=1e-6)


        self.ln_q_ffn = nn.LayerNorm(dim_out, eps=1e-6)
        self.ln_k_ffn = nn.LayerNorm(dim_out, eps=1e-6)


        hid = dim_out * ffn_mult
        self.ffn_q = self._build_ffn(dim_out, hid, dropout)
        self.ffn_k = self._build_ffn(dim_out, hid, dropout)

        self.attn_drop = nn.Dropout(dropout)
        

        self._init_weights()
    
    def _build_ffn(self, dim_in, dim_hidden, dropout):

        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            ActivationFunction[self.activation_type].value(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_in),
            nn.Dropout(dropout),
        )
    
    def _init_weights(self):


        for module in [self.q_proj, self.k_proj, self.v_from_kv, self.v_from_q]:
            nn.init.xavier_uniform_(module.weight, gain=self.init_scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        

        for module in [self.o_proj_q, self.o_proj_k]:
            nn.init.xavier_uniform_(module.weight, gain=self.init_scale * 0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        

        for ffn in [self.ffn_q, self.ffn_k]:
            for module in ffn.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=self.init_scale)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    @staticmethod
    def _masked_softmax(scores, pad_mask):











        scores = scores.masked_fill(pad_mask, float('-inf'))
        


        all_masked = torch.all(pad_mask, dim=-1, keepdim=True)
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        

        attn = F.softmax(scores, dim=-1)
        

        attn = attn.masked_fill(pad_mask, 0.0)
        
        return attn

    def _split_heads(self, x, B, L):











        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def _combine_heads(self, x, B, L):











        return x.transpose(1, 2).contiguous().view(B, L, self.dim)

    def forward(self, q_input, kv_input, mask_x1, mask_x2):













        B, Lq, _ = q_input.shape
        _, Lk, _ = kv_input.shape



        Q = self.q_proj(q_input)
        K = self.k_proj(kv_input)
        V2 = self.v_from_kv(kv_input)
        V1 = self.v_from_q(q_input)



        Qn = self.ln_q_attn_q(Q)
        Kn = self.ln_q_attn_k(K)


        Qh = self._split_heads(Qn, B, Lq)
        Kh = self._split_heads(Kn, B, Lk)
        Vh2 = self._split_heads(V2, B, Lk)


        scores1 = torch.matmul(Qh, Kh.transpose(-1, -2)) * self.scale


        pad_mask1 = ~(mask_x1.unsqueeze(-1) & mask_x2.unsqueeze(1))
        pad_mask1 = pad_mask1.unsqueeze(1)


        attn1 = self._masked_softmax(scores1, pad_mask1)
        attn1 = self.attn_drop(attn1)


        ctx1h = torch.matmul(attn1, Vh2)
        ctx1 = self._combine_heads(ctx1h, B, Lq)


        out1 = self.q_short(q_input) + self.attn_drop(self.o_proj_q(ctx1))
        

        out1 = out1 + self.ffn_q(self.ln_q_ffn(out1))
        

        out1 = out1.masked_fill(~mask_x1.unsqueeze(-1), 0.0)



        Qn2 = self.ln_k_attn_k(K)
        Kn2 = self.ln_k_attn_q(Q)


        Q2h = self._split_heads(Qn2, B, Lk)
        K2h = self._split_heads(Kn2, B, Lq)
        Vh1 = self._split_heads(V1,  B, Lq)


        scores2 = torch.matmul(Q2h, K2h.transpose(-1, -2)) * self.scale


        pad_mask2 = ~(mask_x2.unsqueeze(-1) & mask_x1.unsqueeze(1))
        pad_mask2 = pad_mask2.unsqueeze(1)


        attn2 = self._masked_softmax(scores2, pad_mask2)
        attn2 = self.attn_drop(attn2)


        ctx2h = torch.matmul(attn2, Vh1)
        ctx2 = self._combine_heads(ctx2h, B, Lk)


        out2 = self.k_short(kv_input) + self.attn_drop(self.o_proj_k(ctx2))
        

        out2 = out2 + self.ffn_k(self.ln_k_ffn(out2))
        

        out2 = out2.masked_fill(~mask_x2.unsqueeze(-1), 0.0)

        return out1, out2

class CrossAttentionBlock(nn.Module):






    def __init__(self, dim_q, dim_kv, dim_out, num_heads, attn_dropout, 
                 ffn_mult, activation_type):
        super().__init__()
        self.attn = CrossAttention(
            dim_q=dim_q, 
            dim_kv=dim_kv, 
            dim_out=dim_out,
            num_heads=num_heads, 
            dropout=attn_dropout, 
            ffn_mult=ffn_mult,
            activation_type=activation_type
        )

    def forward(self, x1, x2, mask_x1, mask_x2):













        return self.attn(x1, x2, mask_x1, mask_x2)


class FusionLayer(nn.Module):






    def __init__(self, input_dim, output_dim, dropout=0.1, activation_type="leakyrelu"):
        super().__init__()
        self.activation_type = activation_type
        self.input_dim = input_dim
        self.output_dim = output_dim


        self.fusion = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            ActivationFunction[self.activation_type].value(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            ActivationFunction[self.activation_type].value()
        )
        

        self._init_weights()
    
    def _init_weights(self):

        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x1, x2):














        concat = torch.cat([x1, x2], dim=-1)
        

        return self.fusion(concat)
