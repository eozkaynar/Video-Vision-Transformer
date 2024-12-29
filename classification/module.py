import torch
import math
import torch.nn as nn

class TubeletEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super(TubeletEmbedding, self).__init__()
        self.projection     = nn.Conv3d(
            in_channels     = 1,
            out_channels    = embed_dim,
            kernel_size     = patch_size,
            stride          = patch_size,
            padding         = 0             # VALID padding
        )

    def forward(self, videos):
        # Conv 3D output dim: (B, embed_dim, T, H, W)
        # B: Batch size, C: # of channels  T: Frames H,W: Height, width 
        projected_patches   = self.projection(videos) # Conv 3D output dim: (B, embed_dim, T, H, W)
        B, C, T, H, W       = projected_patches.size()
        flattened_patches   = projected_patches.view(B, C, -1).permute(0, 2, 1)  # (B, N, embed_dim)
        # N = T x H x W , flatten the spatial and temporal dimensions into a single dimension N
        # Key argument -1 instructs PyTorch to automatically calculate this dimension based on 
        # the original tensor size and the provided dimensions
        # permute(0, 2, 1) changes the order of dimensions from (B, C, N) to (B, N, C) so it 
        # can be compatible with transformer architectures, where N represents the sequence 
        # of patches, and C represents their embedding dimensions.
        return flattened_patches
    
# class PositionalEncoder(nn.Module):
#     def __init__(self,embed_dim):
#         super(PositionalEncoder).__init__()
#         self.embed_dim          = embed_dim
#         self.position_embedding = None
#         self.positions          = None      

#     def build(self, num_tokens):

#         self.position_embedding = nn.Embedding(
#             num_embeddings      = num_tokens,
#             embedding_dim       = self.embed_dim
#         )
#         self.positions = torch.arange(0, num_tokens, dtype=torch.long) #Batch size için tekrar bak 

#     def forward(self,encoded_tokens):
#         if self.position_embedding is None or self.positions is None:
#             raise ValueError("PositionalEncoder must be built with the correct num_tokens before forward pass.")
#         encoded_positions       = self.position_embedding(self.positions.to(encoded_tokens.device))
#         encoded_tokens          = encoded_tokens + encoded_positions
#         return encoded_tokens

# class PositionalEncoder(nn.Module):
#     def __init__(self, embed_dim, num_tokens):
#         super(PositionalEncoder, self).__init__()
#         self.embed_dim = embed_dim
#         self.position_embedding = nn.Embedding(
#             num_embeddings=num_tokens,
#             embedding_dim=embed_dim
#         )

#     def forward(self, encoded_tokens):
#         # Pozisyon bilgilerini oluştur ve ekle
#         positions = torch.arange(encoded_tokens.size(1), device=encoded_tokens.device).unsqueeze(0)
#         encoded_positions = self.position_embedding(positions)
#         return encoded_tokens + encoded_positions

# Sinusoidal Positional Encoding
class PositionalEncoder(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.size()
        position = torch.arange(0, num_patches, dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32, device=x.device) * 
                             -(math.log(10000.0) / embed_dim))

        # Sin and Cosine Encoding
        pos_encoding = torch.zeros((num_patches, embed_dim), device=x.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        pos_encoding = pos_encoding.unsqueeze(0)  # Add batch dimension
        return x + pos_encoding
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim,num_heads,dropout=0.1)
        self.norm1     = nn.LayerNorm(embed_dim)
        self.norm2     = nn.LayerNorm(embed_dim)
        self.ff        = nn.Sequential(
            nn.Linear(embed_dim,ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim,embed_dim)
        )
    def forward(self,x):
        attention_output,_ = self.attention(x,x,x)
        x                  = self.norm1(x + attention_output)
        ff_output          = self.ff(x)
        x                  = self.norm2(x + ff_output)
        return x 
class ViViT(nn.Module):
    def __init__(
            self, 
            input_shape,
            embed_dim,
            num_heads,
            num_layers,
            num_classes, # KALDIR
            patch_size,
            layer_norm_eps = 1e-6
                ):
        super(ViViT, self).__init__()

        num_patches = (input_shape[1] // patch_size[0]) * (input_shape[2] // patch_size[1]) * (input_shape[3] // patch_size[2])
        self.tubelet_embedding  = TubeletEmbedding(embed_dim, patch_size)
        self.positional_encoder = PositionalEncoder(num_patches=num_patches, embed_dim=embed_dim) 

        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim       = embed_dim,
                    num_heads       = num_heads,
                    ff_dim          = embed_dim*4,
                )
                for _ in range(num_layers)
            ]
        )

        # self.layer_norm         = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        # self.classifier         = nn.Linear(embed_dim, num_classes) #KALDIR
        self.cls_head             = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, videos):
        patches                 = self.tubelet_embedding(videos) # (B, N, embed_dim)
        # self.positional_encoder.build(patches.size(1))
        encoded_patches         = self.positional_encoder(patches)

        for transformer in self.transformer_layers:
            encoded_patches     = transformer(encoded_patches)

        # representation          = self.layer_norm(encoded_patches.mean(dim=1))
        # outputs                 = self.classifier(representation)

        encoded_patches         = encoded_patches.mean(dim=1)
        return self.cls_head(encoded_patches)