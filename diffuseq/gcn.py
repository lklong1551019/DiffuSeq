import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GCNConv, RGCNConv
except ImportError:
    # Fallback for environment without torch-geometric during development
    # In production, this dependency should be installed.
    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.lin = nn.Linear(in_channels, out_channels)
        def forward(self, x, edge_index):
            return self.lin(x)
            
    class RGCNConv(nn.Module):
        def __init__(self, in_channels, out_channels, num_relations):
            super().__init__()
            self.lin = nn.Linear(in_channels, out_channels)
        def forward(self, x, edge_index, edge_type):
            return self.lin(x)

class GCNModel(nn.Module):
    """
    Graph Convolutional Network module that can switch between standard GCN and Relational GCN.
    
    Architecture:
    This module acts as a wrapper around torch_geometric layers (GCNConv or RGCNConv).
    Since DiffuSeq uses 3D tensors [Batch, SeqLen, HiddenDim], and torch_geometric 
    layers expect a flattened graph [N_total, HiddenDim], this module:
    1. Flattens the input sequence into a single massive disjoint graph.
    2. Shifts the edge indices for each sample in the batch to point to the correct
       offsets in the flattened sequence.
    3. Processes the batch in a single forward pass through the GCN.
    4. Reshapes the output back to [Batch, SeqLen, HiddenDim].
    """
    def __init__(self, hidden_dim, use_relational_gcn=False, num_relations=None):
        super().__init__()
        self.use_relational_gcn = use_relational_gcn
        self.hidden_dim = hidden_dim
        
        if use_relational_gcn:
            # num_relations is the number of AMR relation types
            self.gcn = RGCNConv(hidden_dim, hidden_dim, num_relations)
        else:
            self.gcn = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_type=None):
        """
        Args:
            x: [Batch, SeqLen, HiddenDim]
            edge_index: [2, E] or list of [2, Ei] for each batch
            edge_type: [E] or list of [Ei] for each batch
        """
        batch_size, seq_len, _ = x.shape
        
        # Flatten batch for torch-geometric
        # x_flat: [Batch * SeqLen, HiddenDim]
        x_flat = x.view(-1, self.hidden_dim)
        
        # If edge_index is a list (from DataLoader), we need to collate it
        if isinstance(edge_index, list):
            collated_edge_index = []
            collated_edge_type = []
            for i, ei in enumerate(edge_index):
                if ei.shape[1] == 0: continue
                # Shift indices by i * seq_len
                shifted_ei = ei + i * seq_len
                collated_edge_index.append(shifted_ei)
                if edge_type is not None:
                    collated_edge_type.append(edge_type[i])
            
            if not collated_edge_index:
                # No edges in the whole batch
                return x
                
            edge_index = torch.cat(collated_edge_index, dim=1)
            if edge_type is not None:
                edge_type = torch.cat(collated_edge_type, dim=0)

        # Apply GCN
        if self.use_relational_gcn:
            out = self.gcn(x_flat, edge_index, edge_type)
        else:
            out = self.gcn(x_flat, edge_index)
            
        # Reshape back
        out = out.view(batch_size, seq_len, self.hidden_dim)
        return out
