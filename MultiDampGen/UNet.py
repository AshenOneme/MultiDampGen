import math
import torch
from torch import nn
from torch.nn import functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_time=128, d_cond=128, d_cond2=128):
        super().__init__()
        self.gn_feat = nn.GroupNorm(32, in_channels)
        self.cv_feat = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.lin_time = nn.Linear(d_time, out_channels)
        self.lin_cond = nn.Linear(d_cond, out_channels)
        self.lin_cond2 = nn.Linear(d_cond2, out_channels)
        self.gn_merged = nn.GroupNorm(32, out_channels)
        self.cv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feat, t, cond,cond2):
        # feat: (B, C_in, H, W), t: (B, d_time), cond: (B, d_cond)
        residual_ = feat
        feat = F.silu(self.gn_feat(feat))
        feat = F.silu(self.cv_feat(feat))  # (B, C_out, H, W)
        feat += self.lin_time(F.silu(t))[:, :, None, None]  # Add time embedding
        feat += self.lin_cond(F.silu(cond))[:, :, None, None]  # Add condition embedding
        feat += self.lin_cond(F.silu(cond2))[:, :, None, None]  # Add condition embedding
        feat = self.cv_merged(F.silu(self.gn_merged(feat)))  # (B, C_out, H, W)
        return feat + self.residual_layer(residual_)

class UNet_AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int, n_dims: int = None, n_groups: int = 32, d_time: int = 128, d_cond: int = 128,d_cond2: int=128):
        super().__init__()
        if n_dims is None:
            n_dims = n_channels
        self.gn = nn.GroupNorm(n_groups, n_channels)
        self.proj = nn.Linear(n_channels, n_dims * 3)
        self.output = nn.Linear(n_dims, n_channels)
        self.scale = n_dims ** -0.5
        self.n_heads = n_heads
        self.n_dims = n_dims

        # Time and Condition embeddings
        self.lin_time = nn.Linear(d_time, n_channels)
        self.lin_cond = nn.Linear(d_cond, n_channels)
        self.lin_cond2 = nn.Linear(d_cond2, n_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,cond2: torch.Tensor):
        b, c, h, w = x.shape
        x = self.gn(x)  # Apply GroupNorm first

        # Add time and condition embeddings
        time_embed = self.lin_time(F.silu(t))[:, :, None, None]  # (B, C, 1, 1)
        cond_embed = self.lin_cond(F.silu(cond))[:, :, None, None]  # (B, C, 1, 1)
        cond2_embed=self.lin_cond2(F.silu(cond2))[:, :, None, None]  # (B, C, 1, 1)
        x = x + time_embed + cond_embed + cond2_embed

        # Flatten spatial dimensions
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (B, C, H*W) -> (B, H*W, C)

        # Compute Q, K, V
        q, k, v = self.proj(x).view(b, h * w, self.n_heads, -1).chunk(3, dim=-1)

        # Compute scaled dot-product attention
        att = torch.einsum('blhd, bmhd -> blmh', q, k) * self.scale
        att = torch.softmax(att, dim=-2)
        x_ = torch.einsum('blmh, bmhd -> blhd', att, v)

        # Combine heads and project back to original dimensions
        x_ = self.output(x_.reshape(b, h * w, -1)) + x  # Use reshape instead of view
        return x_.permute(0, 2, 1).view(b, c, h, w)

class SwitchSequential(nn.Sequential):
    def forward(self, x, t, cond,cond2):
        for layer in self:
            if isinstance(layer, UNet_AttentionBlock) or isinstance(layer, UNet_ResidualBlock):
                x = layer(x, t, cond,cond2)
            else:
                x = layer(x)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int = 64):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.lin2(self.act(self.lin1(emb)))

#===================================================================================================================
class ClassConditionedUnet_M(nn.Module):
    def __init__(self, in_channels:int=1,out_channels:int=1,cond_dim: int = 128, cond_dim2: int=1):
        super().__init__()
        self.encoder = nn.ModuleList([
            SwitchSequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)),
            SwitchSequential(UNet_ResidualBlock(64, 64),UNet_ResidualBlock(64, 64), UNet_AttentionBlock(64, 8, 40), UNet_AttentionBlock(64, 8, 40)),
            SwitchSequential(UNet_ResidualBlock(64, 64),UNet_ResidualBlock(64, 64), UNet_AttentionBlock(64, 8, 40), UNet_AttentionBlock(64, 8, 40)),
            SwitchSequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNet_ResidualBlock(128, 128), UNet_ResidualBlock(128, 128), UNet_AttentionBlock(128, 8, 80), UNet_AttentionBlock(128, 8, 80)),
            SwitchSequential(UNet_ResidualBlock(128, 128),UNet_ResidualBlock(128, 128), UNet_AttentionBlock(128, 8, 80), UNet_AttentionBlock(128, 8, 80)),
            SwitchSequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNet_ResidualBlock(256, 256),UNet_ResidualBlock(256, 256), UNet_AttentionBlock(256, 8, 120), UNet_AttentionBlock(256, 8, 120)),
            SwitchSequential(UNet_ResidualBlock(256, 256),UNet_ResidualBlock(256, 256), UNet_AttentionBlock(256, 8, 120), UNet_AttentionBlock(256, 8, 120)),
            SwitchSequential(nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            SwitchSequential(UNet_ResidualBlock(256, 256),UNet_ResidualBlock(256, 256), UNet_AttentionBlock(256, 8, 120), UNet_AttentionBlock(256, 8, 120)),
            SwitchSequential(UNet_ResidualBlock(256, 256),UNet_ResidualBlock(256, 256), UNet_AttentionBlock(256, 8, 120), UNet_AttentionBlock(256, 8, 120))
        ])
        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(256, 256),
            UNet_ResidualBlock(256, 256),
            UNet_AttentionBlock(256, 8, 120),
            UNet_AttentionBlock(256, 8, 120),
        )
        self.decoder = nn.ModuleList([
            SwitchSequential(UNet_ResidualBlock(512, 256),UNet_ResidualBlock(256, 256), UNet_AttentionBlock(256, 8, 120), UNet_AttentionBlock(256, 8, 120)),
            SwitchSequential(UNet_ResidualBlock(512, 256),UNet_ResidualBlock(256, 256), UNet_AttentionBlock(256, 8, 120), UNet_AttentionBlock(256, 8, 120)),
            SwitchSequential(nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)),
            SwitchSequential(UNet_ResidualBlock(512, 256),UNet_ResidualBlock(256, 256), UNet_AttentionBlock(256, 8, 120), UNet_AttentionBlock(256, 8, 120)),
            SwitchSequential(UNet_ResidualBlock(512, 256),UNet_ResidualBlock(256, 256), UNet_AttentionBlock(256, 8, 120), UNet_AttentionBlock(256, 8, 120)),
            SwitchSequential(nn.ConvTranspose2d(512, 256, 4, 2, 1)),
            SwitchSequential(UNet_ResidualBlock(384, 128),UNet_ResidualBlock(128, 128), UNet_AttentionBlock(128, 8, 80), UNet_AttentionBlock(128, 8, 80)),
            SwitchSequential(UNet_ResidualBlock(256, 128),UNet_ResidualBlock(128, 128), UNet_AttentionBlock(128, 8, 80), UNet_AttentionBlock(128, 8, 80)),
            SwitchSequential(nn.ConvTranspose2d(256, 128, 3, 2, 1)),
            SwitchSequential(UNet_ResidualBlock(192, 64),UNet_ResidualBlock(64, 64), UNet_AttentionBlock(64, 8, 40), UNet_AttentionBlock(64, 8, 40)),
            SwitchSequential(UNet_ResidualBlock(128, 64),UNet_ResidualBlock(64, 64), UNet_AttentionBlock(64, 8, 40), UNet_AttentionBlock(64, 8, 40)),
            SwitchSequential(nn.ConvTranspose2d(128, out_channels, 3, padding=1))
        ])
        self.time_emb = TimeEmbedding(n_channels=128)
        self.cond_emb = nn.Linear(cond_dim, 128)
        self.cond_emb2= nn.Linear(cond_dim2, 128)

    def forward(self, x, t, cond,cond2):
        t = self.time_emb(t)
        cond = self.cond_emb(cond)
        cond2= self.cond_emb2(cond2)
        skip_cont = []
        for layer in self.encoder:
            x = layer(x, t, cond,cond2)
            skip_cont.append(x)
        x = self.bottleneck(x, t, cond,cond2)
        for layer in self.decoder:
            skip = skip_cont.pop()
            # Use interpolation to align feature sizes
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = layer(x, t, cond,cond2)
        return x
#===================================================================================================================
if __name__ == "__main__":
    model = ClassConditionedUnet_M(in_channels=1,out_channels=1,cond_dim=128,cond_dim2=1)
    x = torch.randn(8, 1, 28, 28)
    t = torch.randn(8)
    cond = torch.randn(8, 128)
    cond2= torch.randn(8, 1)
    output = model(x, t, cond,cond2)
    print(output.shape)