import math
import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads: torch.tensor, d_embed: torch.tensor):
        super().__init__()
        self.in_proj= nn.Linear(d_embed, 3* d_embed, bias= True)
        self.out_proj= nn.Linear(d_embed, d_embed)
        self.d_embed= d_embed
        self.n_heads= n_heads
    def forward(self, x: torch.tensor, causal_mask=False)-> torch.tensor:
        b, l, d= x.shape
        # x, (batch_size, len, dim)>> (batch_size, len, 3* dim)>> (batch_size, len, head, dim// head* 3)
        x_proj= self.in_proj(x).view(b, l, self.n_heads, -1)
        # q, k or v, (batch_size, len, head, dim// head)
        q, k, v= x_proj.chunk(3, dim= -1)
        # att, (batch_size, len, len, head)
        att= torch.einsum('blhd, bmhd->blmh', q, k)/ math.sqrt(self.d_embed// self.n_heads)
        # mask
        if causal_mask:
            mask= torch.ones_like(att, dtype=torch.bool).triu(1)
            att.masked_fill_(mask, -torch.inf)
        att= torch.softmax(att, dim= 2)
        opt= torch.einsum('blmh, blhd-> blhd', att, v).view(b, l, d)
        return self.out_proj(opt)

class CrossAttention(nn.Module):
    def __init__(self, n_heads: torch.tensor, d_embed: torch.tensor, d_cross: torch.tensor):
        super().__init__()
        self.q_proj= nn.Linear(d_embed, d_embed, bias= True)
        self.kv_proj= nn.Linear(d_cross, 2* d_embed, bias= True)
        self.out_proj= nn.Linear(d_embed, d_embed, bias= True)
        self.n_heads= n_heads
        self.d_embed= d_embed
    def forward(self, x: torch.tensor, y: torch.tensor)-> torch.tensor:
        b, l, d= x.shape
        # x, (batch_size, seq_len, d_embed)-> (b, l, h, d//h), y, (batch_size, seq_len, d_cross)
        q= self.q_proj(x).view(b, l, self.n_heads, -1)
        # y, (batch_size, seq_len, 2* d_embed)-> (b, l, h, 2* d// h)-> k, v, (b, l, h, d// h)
        k, v= self.kv_proj(y).view(b, y.shape[1], self.n_heads, -1).chunk(2, dim= -1)
        att= torch.softmax(torch.einsum('blhd, bmhd-> blmh', q, k)/ (self.d_embed// self.n_heads)** 0.5, dim= 2)
        return self.out_proj(torch.einsum('blmh, bmhd-> blhd', att, v).reshape(b, l, d))

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    def forward(self, x):
        # x: (Batch_Size, Features, Height, Width)
        residue= x
        x= self.groupnorm(x)
        n, c, h, w= x.shape
        x= x.view((n, c, h * w))
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features). Each pixel becomes a feature of size "Features", the sequence length is "Height * Width".
        x= x.transpose(-1, -2)
        x= self.attention(x)
        x= x.transpose(-1, -2)
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x= x.view((n, c, h, w))
        # (Batch_Size, Features, Height, Width)
        return x+ residue

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1= nn.GroupNorm(32, in_channels)
        self.conv_1= nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2= nn.GroupNorm(32, out_channels)
        self.conv_2= nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        # x: (Batch_Size, In_Channels, Height, Width)
        residue= x
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x= self.groupnorm_1(x)
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x= F.silu(x)
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x= self.conv_1(x)
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x= self.groupnorm_2(x)
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x= F.silu(x)
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x= self.conv_2(x)
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return x+ self.residual_layer(residue)

#===================================================================================================================
class VAE_Encoder_M(nn.Sequential):
    def __init__(self, device,in_channels,z_channels):
        super().__init__()
        self.encoder = nn.ModuleList([
            # Batch_size, B; Channel, C; Height, H; Weight, W.
            # (B, 1, H, W)-> (B, 64, H, W)
            nn.Conv2d(in_channels, 64, kernel_size= 3, padding= 1),
            VAE_ResidualBlock(64, 64),
            # (B, 64, H, W)-> (B, 128, H/ 2, W/ 2)
            nn.Conv2d(64, 128, kernel_size= 3, stride= 2),
            VAE_ResidualBlock(128, 128),
            # (B, 128, H/ 2, W/ 2)-> (B, 256, H/ 4, W/ 4)
            nn.Conv2d(128, 256, kernel_size= 3, stride= 2),
            VAE_AttentionBlock(256),
            VAE_ResidualBlock(256, 256),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, 64, kernel_size= 3, padding= 1),
            # output, (B, 8, H/ 4, W/ 4)
            nn.Conv2d(64, z_channels*2, kernel_size= 1, padding= 0)
        ])
        self.norm_dist= torch.distributions.Normal(0, 1)
        self.device= device
    def forward(self, x):
        # transform x into latent code.
        # x, (B, 1, H, W)-> (B, 8, H/ 4, W/ 4)
        for module in self.encoder:
            # confirm H/ 2
            if getattr(module, 'stride', None)== (2, 2):
                x= F.pad(x, (0, 1, 0, 1))
            x= module(x)
        # mean, (B, 4, H/ 4, W/ 4); log_var, (B, 4, H/ 4, W/ 4)
        mean, log_var= torch.chunk(x, 2, dim= 1)
        var_= torch.clamp(log_var, -30, 20).exp()
        stdev= var_** 0.5
        z= self.norm_dist.sample(mean.shape).to(self.device)* stdev+ mean
        z = torch.tanh(z)
        return mean, stdev, z

class VAE_Decoder_M(nn.Sequential):
    def __init__(self,out_channels,z_channels):
        super().__init__()
        self.decoder = nn.ModuleList([
            # (B, 4, H/ 4, W/ 4)-> (B, 256, H/ 4, W/ 4)
            nn.Conv2d(z_channels, 64, kernel_size= 1, padding= 0),
            nn.Conv2d(64, 256, kernel_size= 3, padding= 1),
            VAE_AttentionBlock(256),
            VAE_ResidualBlock(256, 256),
            # (B, 256, H/ 4, W/ 4)-> (B, 256, H/ 2, W/ 2)
            nn.Upsample(scale_factor= 2),
            # (B, 256, H/ 2, W/ 2)-> (B, 128, H/ 2, W/ 2)
            nn.Conv2d(256, 128, kernel_size= 3, padding= 1),
            VAE_ResidualBlock(128, 128),
            # (B, 128, H/ 2, W/ 2)-> (B, 128, H, W)
            nn.Upsample(scale_factor= 2),
            nn.Conv2d(128, 128, kernel_size= 3, padding= 1),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, kernel_size= 3, padding= 1),
            nn.Sigmoid()
        ])
    def forward(self, x):
        for module in self.decoder:
            x= module(x)
        return x

class VAE_M(nn.Module):
    def __init__(self, device,z_channels):
        super().__init__()
        self.device= device
        self.encoder= VAE_Encoder_M(device=device,in_channels=1,z_channels=z_channels).to(device)
        self.decoder= VAE_Decoder_M(out_channels=1,z_channels=z_channels).to(device)
    def loss_fuc(self, pred, label, mu, sga, balance_factor= 0.05, eps= 1e-10):
        recon_loss = F.binary_cross_entropy(pred, label,reduction='mean')
        return recon_loss+ balance_factor* (sga** 2+ mu** 2- torch.log(sga+ eps)- 0.5).mean()
    def forward(self, x):
        mu, sga, z= self.encoder(x)
        return mu, sga, self.decoder(z)
#===================================================================================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE_M(device=device,z_channels=3)
    x = torch.randn(8, 1, 128, 128).to(device)
    mu, sga, decoderz = model(x)
    print(mu.shape, sga.shape, decoderz.shape)