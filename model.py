import torch
import torch.nn as nn

class TempEncoder(nn.Module):
    """
    Encodes temporal information AND static node features (lat/long).
    """
    def __init__(self, input_len, input_dim, static_dim, embed_dim):
        super(TempEncoder, self).__init__()
        # Input dim = (Time * Features) + Static_Features
        self.mlp = nn.Sequential(
            nn.Linear((input_len * input_dim) + static_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, static_feat):
        # x: (Batch, Input_Len, Nodes, Input_Dim)
        # static_feat: (Batch, Nodes, Static_Dim)
        B, T, N, C = x.shape
        
        # Flatten Time: (Batch, Nodes, T*C)
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, T * C)
        
        # Concatenate Static Features (Lat/Long) to every node
        # static_feat is already (Batch, Nodes, 2)
        combined = torch.cat([x_flat, static_feat], dim=-1)
        
        # Encode: (Batch, Nodes, Embed_Dim)
        return self.mlp(combined)

class STMixerLayer(nn.Module):
    """
    Standard ST-Mixer Layer (Unchanged)
    """
    def __init__(self, num_nodes, embed_dim):
        super(STMixerLayer, self).__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(num_nodes, num_nodes),
            nn.GELU(),
            nn.Linear(num_nodes, num_nodes)
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Spatial Mix
        y = x.permute(0, 2, 1) # (Batch, Embed, Nodes)
        y = self.spatial_mlp(y)
        y = y.permute(0, 2, 1) # (Batch, Nodes, Embed)
        x = self.norm1(x + y)
        
        # Channel Mix
        y = self.channel_mlp(x)
        x = self.norm2(x + y)
        return x

class STMLP(nn.Module):
    def __init__(self, num_nodes, input_len, input_dim, static_dim, embed_dim, output_len, num_layers=3):
        super(STMLP, self).__init__()
        self.temp_encoder = TempEncoder(input_len, input_dim, static_dim, embed_dim)
        self.mixers = nn.ModuleList([
            STMixerLayer(num_nodes, embed_dim) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, output_len),
            nn.ReLU()
        )

    def forward(self, x, static_feat):
        x = self.temp_encoder(x, static_feat)
        for mixer in self.mixers:
            x = mixer(x)
        out = self.decoder(x)
        return out.permute(0, 2, 1) # (Batch, Output_Len, Nodes)