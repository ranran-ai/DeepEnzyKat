








from enum import Enum
import dgl
from typing import List

from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem
from modules_cross_fusion import CrossAttentionBlock, FusionLayer, ActivationFunction
from rdkit import RDLogger



RDLogger.DisableLog('rdApp.*')


class att_gate_layer(nn.Module):
    def __init__(self, input_dim, output_dim, activation_type="leakyrelu"):
        super(att_gate_layer, self).__init__()
        self.activation_type = activation_type
        self.activation = ActivationFunction[activation_type].value()

        self.W_mask = nn.Linear(input_dim, input_dim, bias=False)
        self.W = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        for param in self.W_mask.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_mask = self.W_mask(x)
        x_main = self.W(x)
        x_combined = self.activation(x_mask + x_main)

        e = torch.einsum('ijl,ikl->ijk', (x_combined, x_combined))
        attention = F.softmax(e, dim=2)
        x_attn = torch.einsum('ijk,ikl->ijl', (attention, x_combined))
        out = self.activation(self.fc(x_attn))
        return out


def normalize_with_mask(x, mask, epsilon=1e-8, use_layernorm=True):














    

    has_valid = mask.any(dim=1)
    

    valid_mask = mask.unsqueeze(-1).expand_as(x)

    if use_layernorm:


        feature_dim = x.size(-1)
        layer_norm = nn.LayerNorm(feature_dim, eps=epsilon).to(x.device)
        


        norm_x = torch.zeros_like(x, dtype=x.dtype)
        for i in range(x.size(0)):
            if has_valid[i]:
                valid_indices = mask[i]
                if valid_indices.any():

                    valid_features = x[i][valid_indices]
                    normalized_features = layer_norm(valid_features)

                    norm_x[i][valid_indices] = normalized_features.to(norm_x.dtype)
        

        norm_x = norm_x * valid_mask
    else:


        valid_x = x * valid_mask


        count = valid_mask.sum(dim=1).clamp(min=1)
        

        mean = valid_x.sum(dim=1) / count
        std = torch.sqrt(
            ((valid_x - mean.unsqueeze(1)) ** 2).sum(dim=1) / count + epsilon)


        std = torch.clamp(std, min=epsilon)
        std = torch.where(torch.isnan(std), torch.ones_like(std, dtype=std.dtype) * epsilon, std)


        norm_x = (x - mean.unsqueeze(1)) / std.unsqueeze(1)


        norm_x = norm_x * valid_mask
        

        norm_x = torch.where(
            has_valid.unsqueeze(-1).unsqueeze(-1),
            norm_x,
            torch.zeros_like(norm_x, dtype=norm_x.dtype)
        )
    

    norm_x = torch.where(torch.isnan(norm_x), torch.zeros_like(norm_x, dtype=norm_x.dtype), norm_x)

    return norm_x

class DMPNNEncoder(nn.Module):
    def __init__(self, node_output_dim, edge_output_dim, node_feat_dim=109, edge_feat_dim=13, num_rounds=3,
                 dropout_rate=0.1, activation_type="leakyrelu"):
        super(DMPNNEncoder, self).__init__()
        self.activation_type = activation_type
        self.activation = ActivationFunction[activation_type].value()
        self.dropout = nn.Dropout(dropout_rate)
        self.edge_mlp_substrate = nn.Sequential(nn.Linear(node_feat_dim + edge_feat_dim, edge_output_dim, bias= False), ActivationFunction[activation_type].value())

        self.edge_update_mlp_substrate = nn.Sequential(nn.Linear(edge_output_dim, edge_output_dim, bias=False))

        self.node_mlp_substrate = nn.Sequential(nn.Linear(node_feat_dim + edge_output_dim, node_output_dim, bias=True), ActivationFunction[activation_type].value(), self.dropout)




        self.num_rounds = num_rounds

    def forward(self, batched_substrate_graph):

        batched_substrate_graph.edata['h0'] = self.initialize_edge_features(batched_substrate_graph)
        batched_substrate_graph.edata['h'] = batched_substrate_graph.edata['h0']

        self.setup_reverse_edges(batched_substrate_graph)

        for _ in range(self.num_rounds):
            self.message_passing(batched_substrate_graph)


        batched_substrate_graph.update_all(self.message_func_sum, self.reduce_func_sum)

        new_node_feats = torch.cat([batched_substrate_graph.ndata['feat'], batched_substrate_graph.ndata['m']], dim=1)
        batched_substrate_graph.ndata['h'] = self.node_mlp_substrate(new_node_feats)


        num_nodes_per_graph = batched_substrate_graph.batch_num_nodes()
        h_padded = dgl.backend.pad_packed_tensor(
            batched_substrate_graph.ndata['h'],
            num_nodes_per_graph,
            0.0
        )




        max_nodes = int(num_nodes_per_graph.max())
        mask = (torch.arange(max_nodes, device=h_padded.device)[None, :] <
                num_nodes_per_graph[:, None])



        return h_padded, mask


    def initialize_edge_features(self, g):
        edge_features = torch.cat([g.ndata['feat'][g.edges()[0]], g.edata['feat']], dim=1)
        return self.edge_mlp_substrate(edge_features)

    def setup_reverse_edges(self, g):
        src, dst = g.edges()
        g.edata['reverse_edge'] = g.edge_ids(dst, src)
    def message_passing(self, g):
        g.update_all(self.message_func, self.reduce_func)
        g.apply_edges(self.apply_edges_func)

    def message_func(self, edges):
        return {'m': edges.data['h']}
    def reduce_func(self, nodes):
        return {'sum0': torch.sum(nodes.mailbox['m'], dim=1)}
    def apply_edges_func(self, edges):
        edges.data['sum'] = edges.src['sum0']
        edges.data['m'] = edges.data['sum'] - edges.data['h'][edges.data['reverse_edge']]
        weighted_m = self.edge_update_mlp_substrate(edges.data['m'])
        edges.data['h'] = self.activation(weighted_m + edges.data['h0'])
        edges.data['h'] = self.dropout(edges.data['h'])
        return {'h': edges.data['h']}

    def message_func_sum(self, edges):
        return {'m2': edges.data['h']}
    def reduce_func_sum(self, nodes):
        return {'m': torch.sum(nodes.mailbox['m2'], dim=1)}


class AttentionPooling(nn.Module):




    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, x, mask):






        attn_scores = self.attention(x)
        

        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        

        attn_weights = F.softmax(attn_scores, dim=1)
        

        pooled = (x * attn_weights).sum(dim=1)
        
        return pooled



class bimodal_regression_embed(nn.Module):
    def __init__(self, x1_dim, x2_dim, hid_dim, dim_q, dim_kv, dim_out, num_heads, attn_dropout, ffn_mult, activation_type, use_cross_attn=True):
        super(bimodal_regression_embed, self).__init__()
        self.activation_type = activation_type
        self.use_cross_attn = use_cross_attn
        

        if self.use_cross_attn:
            self.cross_attn = CrossAttentionBlock(dim_q, dim_kv, dim_out,
                     num_heads, attn_dropout, ffn_mult, activation_type)
        


        pooling_dim_x1 = hid_dim if self.use_cross_attn else x1_dim
        pooling_dim_x2 = hid_dim if self.use_cross_attn else x2_dim
        self.x1_pooling = AttentionPooling(pooling_dim_x1)
        self.x2_pooling = AttentionPooling(pooling_dim_x2)
        


        fusion_input_dim = hid_dim * 2 if self.use_cross_attn else (x1_dim + x2_dim)
        self.fusion_layer = FusionLayer(fusion_input_dim, hid_dim * 2, dropout=attn_dropout, activation_type=activation_type)
        




    def forward(self, padded_x1, padded_x2, mask_x1, mask_x2):









        padded_x1 = normalize_with_mask(padded_x1, mask_x1)
        padded_x2 = normalize_with_mask(padded_x2, mask_x2)
        

        if self.use_cross_attn:
            x1_attn, x2_attn = self.cross_attn(padded_x1, padded_x2, mask_x1, mask_x2)  


        else:

            x1_attn = padded_x1
            x2_attn = padded_x2
        

        x1_out = self.x1_pooling(x1_attn, mask_x1)
        x2_out = self.x2_pooling(x2_attn, mask_x2)
        

        out = self.fusion_layer(x1_out, x2_out)

        
        return out


class moe(nn.Module):
    def __init__(self, num_experts, drop_r, x1_dim, x2_dim, hid_dim, node_output_dim, edge_output_dim,
                 node_feat_dim, edge_feat_dim, num_rounds, dropout_rate, activation_type, dim_q, dim_kv, dim_out,
                 num_heads, attn_dropout, ffn_mult, use_cross_attn=True, use_moe=True):
        super(moe, self).__init__()

        self.activation_type = activation_type
        self.use_cross_attn = use_cross_attn
        self.use_moe = use_moe


        self.dmpnn = DMPNNEncoder(
            node_output_dim=node_output_dim,
            edge_output_dim=edge_output_dim,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            num_rounds=num_rounds,
            dropout_rate=dropout_rate,
            activation_type=activation_type
        )


        self.embed = bimodal_regression_embed(
            x1_dim=x1_dim,
            x2_dim=x2_dim,
            hid_dim=hid_dim,
            dim_q=dim_q,
            dim_kv=dim_kv,
            dim_out=dim_out,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            ffn_mult=ffn_mult,
            activation_type=activation_type,
            use_cross_attn=use_cross_attn
        )

        if self.use_moe:

            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * hid_dim, hid_dim),
                    ActivationFunction[self.activation_type].value(),
                    nn.Dropout(drop_r),
                    nn.Linear(hid_dim, hid_dim),
                    ActivationFunction[self.activation_type].value(),
                    nn.Dropout(drop_r),
                    nn.Linear(hid_dim, hid_dim),
                    ActivationFunction[self.activation_type].value(),
                    nn.Dropout(drop_r),
                    nn.Linear(hid_dim, 1)
                )
                for _ in range(num_experts)
            ])


            self.gating = nn.Sequential(
                nn.Linear(2 * hid_dim, hid_dim),
                ActivationFunction[self.activation_type].value(),
                nn.Linear(hid_dim, hid_dim),
                ActivationFunction[self.activation_type].value(),
                nn.Linear(hid_dim, hid_dim),
                ActivationFunction[self.activation_type].value(),
                nn.Linear(hid_dim, num_experts),
                nn.Softmax(dim=1)
            )
        else:

            self.simple_ann = nn.Sequential(
                nn.Linear(2 * hid_dim, hid_dim),
                ActivationFunction[self.activation_type].value(),
                nn.Dropout(drop_r),
                nn.Linear(hid_dim, hid_dim),
                ActivationFunction[self.activation_type].value(),
                nn.Dropout(drop_r),
                nn.Linear(hid_dim, hid_dim),
                ActivationFunction[self.activation_type].value(),
                nn.Dropout(drop_r),
                nn.Linear(hid_dim, 1)
            )

    def forward(self, mols, padded_x2, mask_x2):







        padded_x1, mask_x1 = self.dmpnn(mols)


        out = self.embed(padded_x1, padded_x2, mask_x1, mask_x2)

        if self.use_moe:

            expert_outputs = [expert(out) for expert in self.experts]
            expert_outputs = torch.stack(expert_outputs, dim=1)

            gating_weights = self.gating(out)

            final_output = (expert_outputs.squeeze(-1) * gating_weights).sum(dim=1, keepdim=True)
        else:

            final_output = self.simple_ann(out)

        return final_output



