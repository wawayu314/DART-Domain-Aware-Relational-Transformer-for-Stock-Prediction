#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOTA-style baseline models for stock return prediction benchmarks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ALSTMModel(nn.Module):
    """LSTM with temporal attention, a common financial forecasting baseline."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(ALSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        hidden, _ = self.lstm(x)
        weights = torch.softmax(self.attn(hidden), dim=1)
        context = torch.sum(weights * hidden, dim=1)
        return self.head(self.dropout(context))


class TemporalTransformerModel(nn.Module):
    """Vanilla Transformer encoder over each stock's historical time window."""

    def __init__(
        self,
        input_dim,
        lookback,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.2,
    ):
        super(TemporalTransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, lookback, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x):
        h = self.input_proj(x) + self.position[:, : x.size(1), :]
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        return self.head(h)


class PatchTSTModel(nn.Module):
    """PatchTST-style patch encoder adapted to daily stock features."""

    def __init__(
        self,
        input_dim,
        lookback,
        patch_len=8,
        stride=4,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.2,
    ):
        super(PatchTSTModel, self).__init__()
        if patch_len > lookback:
            patch_len = lookback
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = 1 + max(0, (lookback - patch_len) // stride)
        self.patch_proj = nn.Linear(input_dim * patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x):
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(x.size(0), patches.size(1), -1)
        h = self.patch_proj(patches) + self.position[:, : patches.size(1), :]
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))
        return self.head(h)


class ITransformerModel(nn.Module):
    """iTransformer-style inverted tokenization across feature variables."""

    def __init__(
        self,
        input_dim,
        lookback,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.2,
    ):
        super(ITransformerModel, self).__init__()
        self.input_dim = input_dim
        self.temporal_proj = nn.Linear(lookback, d_model)
        self.variable_embed = nn.Parameter(torch.zeros(1, input_dim, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.trunc_normal_(self.variable_embed, std=0.02)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.temporal_proj(h) + self.variable_embed[:, : self.input_dim, :]
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))
        return self.head(h)


class TimesNetLiteModel(nn.Module):
    """A compact multi-kernel temporal convolution baseline inspired by TimesNet."""

    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(TimesNetLiteModel, self).__init__()
        kernels = [3, 5, 7]
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                )
                for k in kernels
            ]
        )
        self.gate = nn.Linear(hidden_dim * len(kernels), len(kernels))
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        branch_last = [branch(x_t).transpose(1, 2)[:, -1, :] for branch in self.branches]
        concat = torch.cat(branch_last, dim=-1)
        weights = torch.softmax(self.gate(concat), dim=-1)
        stacked = torch.stack(branch_last, dim=1)
        h = torch.sum(weights.unsqueeze(-1) * stacked, dim=1)
        h = self.norm(h)
        return self.head(h)


class SelectiveStateSpaceBlock(nn.Module):
    """A compact Mamba-style selective state-space block implemented in PyTorch."""

    def __init__(self, d_model, kernel_size=3, dropout=0.2):
        super(SelectiveStateSpaceBlock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 4)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model,
        )
        self.a_proj = nn.Linear(d_model, d_model)
        self.b_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        u, gate, skip, residual_mix = self.in_proj(h).chunk(4, dim=-1)
        u = self.depthwise_conv(u.transpose(1, 2))[:, :, : x.size(1)].transpose(1, 2)
        u = F.silu(u)

        a = torch.sigmoid(self.a_proj(u))
        b = torch.tanh(self.b_proj(u))
        c = torch.sigmoid(self.c_proj(u))

        state = torch.zeros(x.size(0), x.size(2), dtype=x.dtype, device=x.device)
        outputs = []
        for t in range(x.size(1)):
            state = a[:, t, :] * state + (1.0 - a[:, t, :]) * b[:, t, :]
            outputs.append(c[:, t, :] * state + residual_mix[:, t, :])
        y = torch.stack(outputs, dim=1)
        y = self.out_proj(y)
        y = y * torch.sigmoid(gate) + skip
        return residual + self.dropout(y)


class MambaStockModel(nn.Module):
    """A lightweight Mamba-style stock forecasting backbone."""

    def __init__(self, input_dim, d_model=64, n_layers=2, dropout=0.2):
        super(MambaStockModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [SelectiveStateSpaceBlock(d_model=d_model, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.temporal_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        last_h = self.norm(h[:, -1, :])
        mean_h = self.norm(h.mean(dim=1))
        gate = self.temporal_gate(torch.cat([last_h, mean_h], dim=-1))
        fused = gate * last_h + (1.0 - gate) * mean_h
        return self.head(fused)


class SparseCausalGraphLayer(nn.Module):
    """Directed sparse message passing layer for CausalStock-style aggregation."""

    def __init__(self, hidden_dim, edge_index=None, edge_weight=None, dropout=0.2):
        super(SparseCausalGraphLayer, self).__init__()
        if edge_index is None or edge_weight is None:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float32)
        self.register_buffer("edge_src", edge_index[0].long())
        self.register_buffer("edge_dst", edge_index[1].long())
        self.register_buffer("edge_weight", edge_weight.float())
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden):
        if self.edge_weight.numel() == 0:
            return hidden
        messages = self.msg_proj(hidden)[self.edge_src] * self.edge_weight.unsqueeze(-1)
        aggregated = hidden.new_zeros(hidden.size(0), hidden.size(1))
        aggregated.index_add_(0, self.edge_dst, messages)
        updated = torch.cat([self.self_proj(hidden), aggregated], dim=-1)
        return self.norm(hidden + self.update(updated))


class CausalStockModel(nn.Module):
    """A CausalStock-style multimodal model adapted to sentiment + price tensors."""

    def __init__(
        self,
        input_dim,
        lookback,
        price_input_dim,
        news_input_dim,
        edge_index=None,
        edge_weight=None,
        hidden_dim=64,
        news_hidden_dim=32,
        dropout=0.2,
    ):
        super(CausalStockModel, self).__init__()
        if price_input_dim <= 0 or news_input_dim <= 0:
            raise ValueError("CausalStockModel requires both price and news features.")
        self.price_input_dim = price_input_dim
        self.news_input_dim = news_input_dim

        self.price_encoder = nn.GRU(
            input_size=price_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.news_denoise = nn.Sequential(
            nn.Linear(news_input_dim, news_input_dim),
            nn.GELU(),
            nn.Linear(news_input_dim, news_input_dim),
            nn.Sigmoid(),
        )
        self.news_encoder = nn.GRU(
            input_size=news_input_dim,
            hidden_size=news_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.news_proj = nn.Linear(news_hidden_dim, hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.graph_layers = nn.ModuleList(
            [
                SparseCausalGraphLayer(
                    hidden_dim=hidden_dim,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        )
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        price_x = x[:, :, : self.price_input_dim]
        news_x = x[:, :, self.price_input_dim : self.price_input_dim + self.news_input_dim]

        price_seq, _ = self.price_encoder(price_x)
        price_last = price_seq[:, -1, :]
        attn = torch.softmax(self.temporal_attention(price_seq), dim=1)
        price_context = torch.sum(attn * price_seq, dim=1)

        news_gate = self.news_denoise(news_x)
        news_seq, _ = self.news_encoder(news_x * news_gate)
        news_last = self.news_proj(news_seq[:, -1, :])

        fusion_gate = self.fusion_gate(torch.cat([price_last, news_last], dim=-1))
        fused = self.fusion_norm(price_context + fusion_gate * news_last)

        graph_hidden = fused
        for layer in self.graph_layers:
            graph_hidden = layer(graph_hidden)

        combined = torch.cat([price_last, graph_hidden, fusion_gate * news_last], dim=-1)
        return self.head(combined)


def build_sota_model(model_name, input_dim, lookback, **kwargs):
    name = model_name.lower()
    if name == "alstm":
        return ALSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
    if name == "transformer":
        return TemporalTransformerModel(input_dim=input_dim, lookback=lookback)
    if name == "patchtst":
        return PatchTSTModel(input_dim=input_dim, lookback=lookback)
    if name == "itransformer":
        return ITransformerModel(input_dim=input_dim, lookback=lookback)
    if name == "timesnetlite":
        return TimesNetLiteModel(input_dim=input_dim)
    if name == "mambastock":
        return MambaStockModel(input_dim=input_dim, d_model=64, n_layers=2, dropout=0.2)
    if name == "causalstock":
        edge_index = kwargs.get("edge_index")
        edge_weight = kwargs.get("edge_weight")
        price_input_dim = int(kwargs.get("price_input_dim", 0))
        news_input_dim = int(kwargs.get("news_input_dim", 0))
        return CausalStockModel(
            input_dim=input_dim,
            lookback=lookback,
            price_input_dim=price_input_dim,
            news_input_dim=news_input_dim,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
    raise ValueError("Unsupported model: {}".format(model_name))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
