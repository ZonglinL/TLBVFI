# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange,repeat

from model.BrownianBridge.base.util import instantiate_from_config
from model.BrownianBridge.base.modules.attention import LinearAttention, SpatialCrossAttentionWithPosEmb
from model.BrownianBridge.base.modules.maxvit import SpatialCrossAttentionWithMax, MaxAttentionBlock
from cupy_module import dsepconv 

from VFI.archs.VFIformer_arch import VFIformer,FlowRefineNet_Multis,FlowRefineNet_Multis_our
from VFI.archs.warplayer import warp
import torch.nn.functional as F

def Rearrange(x,frames = 3, back = False):

        if back:
            x = x.permute(0,2,1,3,4) ## B C F H W --> B F C H W
            x = rearrange(x,'b f c h w -> (b f) c h w') ## BF C H W
        else:
            x = torch.chunk(x,frames) ## F    B C H W
            x = torch.stack(x,2) ## B C F H W
        return x



def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=4):
    ## change to 8 for new
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class IdentityWrapper(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x,ctx = None):
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x



class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)

        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)


        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
  
        if self.use_conv_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1)
        else:
            self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)

    def forward(self, x, temb):

        h = x ## BF C H W
        h = self.norm1(h)
        h = nonlinearity(h)

        
        h = self.conv1(h)


        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)    

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)


        h = h + x
        
        return h

class ResnetBlock_3D(nn.Module):
    ## stride = 2
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
       
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=(2,1,1),
                                     padding=(2,1,1))

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=(2,1,1),
                                     padding=(2,1,1))
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=(3,3,3),
                                                     stride=1,
                                                     padding=(1,1,1))
            else:

                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        x = Rearrange(x) ## B C F H W


        h = x ## BF C H W
        h = self.norm1(h)
        h = nonlinearity(h)

        
        h = self.conv1(h)
        h = Rearrange(h,back = True)  ## BF C H W

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)

        h = Rearrange(h) ## B C F H W
        h = self.conv2(h)


    

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)


        h = h + x
        h = Rearrange(h,back = True)  ## BF C H W
        
        return h



class ResnetBlock_Dec(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        self.norm1 = Normalize(in_channels)


        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=(1,1,1),
                                     padding=(1,1,1))


        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=(1,1,1),
                                     padding=(0,1,1))

        if self.use_conv_shortcut:

            self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=(3,3,3),
                                                    stride=1,
                                                    padding=(0,1,1))

        else:
            self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=(3,1,1),
                                                    stride=(1,1,1),
                                                    padding=(0,0,0))

    def forward(self, x, temb):
        
        x = Rearrange(x)


        h = x ## B C F H W
        h = self.norm1(h)
        h = nonlinearity(h)

        h = self.conv1(h)
        h = Rearrange(h,back = True)  ## BF C H W

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)

        h = Rearrange(h) ## B C F H W
        h = self.conv2(h)


        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        else:
            x = self.nin_shortcut(x)
        

        h = h + x

        h = h.squeeze(2) ## B C H W
        
        return h



class ResnetBlock_fusion(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        self.norm1 = Normalize(in_channels)

        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=(1,1,1),
                                     padding=(1,1,1))


        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=(1,1,1),
                                     padding=(1,1,1))

        if self.use_conv_shortcut:

            self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=(3,3,3),
                                                    stride=1,
                                                    padding=(1,1,1))

        else:
            self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):

        x = Rearrange(x) ## B C F H W

        h = x ## BF C H W
        h = self.norm1(h)
        h = nonlinearity(h)

        
        h = self.conv1(h)
        h = Rearrange(h,back = True)  ## BF C H W

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)

        h = Rearrange(h) ## B C F H W
        h = self.conv2(h)


    

        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        else:
            x = self.nin_shortcut(x)


        h = h + x
        h = Rearrange(h,back = True)  ## BF C H W
        
        return h

class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x): 
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_



class STAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x): 
        h_ = x ## B3 C H W
        #h_ = Rearrange(h_) ## B C F H W
        h_ = torch.chunk(h_,3) ## F    B C H W
        h_ = torch.stack(h_,2) ## B C F H W
        


        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_) ## B C F H W
        b,c,f,h,w = q.shape
        q = rearrange(q,'b c f h w -> b c (f h w)')
        q = q.permute(0,2,1)   # b,hw,c
        k = rearrange(k,'b c f h w -> b c (f h w)')
        w_ = torch.bmm(q,k)     # b,fhw,fhw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v,'b c f h w -> b c (f h w)')
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = rearrange(h_,'b c (f h w) -> b c f h w',f=f,h=h,w=w)

        h_ = self.proj_out(h_) ## B C F H W

        h_ = h_.permute(0,2,1,3,4) ## B F C H W

        h_ = rearrange(h_,'b f c h w -> (b f) c h w')
        
        #h_ = Rearrange(h_,back = True)

        return x+h_



class STCrossAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x): 
        h_ = x ## B3 C H W
        #h_ = Rearrange(h_) ## B C F H W
        
        h_ = torch.chunk(h_,3) ## F B C H W
        h_ = torch.stack(h_,2) ## B C F H W
        


        h_ = self.norm(h_)
        q = self.q(h_)[:,:,1:2] ## B C 1 H W
        skip = q.clone().squeeze(2)
        k = self.k(h_)
        v = self.v(h_) ## B C F H W
        b,c,f,h,w = q.shape
        q = rearrange(q,'b c f h w -> b c (f h w)')
        q = q.permute(0,2,1)   # b,hw,c
        k = rearrange(k,'b c f h w -> b c (f h w)')
        w_ = torch.bmm(q,k)     # b,fhw,fhw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v,'b c f h w -> b c (f h w)')
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = rearrange(h_,'b c (f h w) -> b c f h w',f=f,h=h,w=w)

        h_ = self.proj_out(h_)
        h_ = h_.squeeze(2) ## B C H W
        
        #h_ = Rearrange(h_,frames = 1,back = True)
        return skip+h_



def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none", 'max'], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    elif attn_type == 'max':
        return MaxAttentionBlock(in_channels, heads=1, dim_head=in_channels)
    else:
        return LinAttnBlock(in_channels)


def make_st_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none", 'max'], f'attn_type {attn_type} unknown'
    print(f"making spatial temporal attention of type vanilla with {in_channels} in_channels")
    return STAttnBlock(in_channels)


def make_st_cross_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none", 'max'], f'attn_type {attn_type} unknown'
    print(f"making spatial temporal attention of type vanilla with {in_channels} in_channels")
    return STCrossAttnBlock(in_channels)



class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



class Upsampler(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size//in_size))+1
        factor_up = 1.+ (out_size % in_size)
        print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, mid_channels=2*in_channels,
                                       out_channels=in_channels)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
                               attn_resolutions=[], in_channels=None, ch=in_channels,
                               ch_mult=[ch_mult for _ in range(num_blocks)])

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(nn.Module):
    def __init__(self, in_channels=None, learned=False, mode="bilinear"):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

    def forward(self, x, scale_factor=1.0):
        if scale_factor==1.0:
            return x
        else:
            x = torch.nn.functional.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x

class FirstStagePostProcessor(nn.Module):

    def __init__(self, ch_mult:list, in_channels,
                 pretrained_model:nn.Module=None,
                 reshape=False,
                 n_channels=None,
                 dropout=0.,
                 pretrained_config=None):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)

        self.do_reshape = reshape

        if n_channels is None:
            n_channels = self.pretrained_model.encoder.ch

        self.proj_norm = Normalize(in_channels,num_groups=in_channels//2)
        self.proj = nn.Conv2d(in_channels,n_channels,kernel_size=3,
                            stride=1,padding=1)

        blocks = []
        downs = []
        ch_in = n_channels
        for m in ch_mult:
            blocks.append(ResnetBlock(in_channels=ch_in,out_channels=m*n_channels,dropout=dropout))
            ch_in = m * n_channels
            downs.append(Downsample(ch_in, with_conv=False))

        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)


    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        # self.pretrained_model.train = False
        for param in self.pretrained_model.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def encode_with_pretrained(self,x):
        c = self.pretrained_model.encode(x)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
        return  c

    def forward(self,x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)

        for submodel, downmodel in zip(self.model,self.downsampler):
            z = submodel(z,temb=None)
            z = downmodel(z)

        if self.do_reshape:
            z = rearrange(z,'b c h w -> b (h w) c')
        return z

"""
class FIEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch # 128
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult) # 3
        self.num_res_blocks = num_res_blocks # 2
        self.resolution = resolution # 256
        self.in_channels = in_channels # 3

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution # 256
        in_ch_mult = (1,)+tuple(ch_mult) # (1,1,2,4)
        self.in_ch_mult = in_ch_mult # (1,1,2,4)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = int(ch*in_ch_mult[i_level])
            block_out = int(ch*ch_mult[i_level])
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock_3D(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            # if i_level != self.num_resolutions-1:
            down.downsample = Downsample(block_in, resamp_with_conv)
            curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels, # 3
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, ret_feature=False):
        # timestep embedding
        temb = None
        if x.min() < 0:
            x = x/2 + 0.5
        # downsampling
        hs = [self.conv_in(x)] ## B3 C H W
        phi_list_prev = []
        phi_list_next = []
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            # if i_level != self.num_resolutions-1:
            hs.append(self.down[i_level].downsample(hs[-1]))
            reshaped = Rearrange(hs[-1]) ## B C 3 H W 
            phi_list_prev.append(reshaped[:,:,0]) ## previous frame feature
            phi_list_next.append(reshaped[:,:,-1]) ## next frame feature

        # middle
        h = hs[-1] ## B3 C H W
        h = torch.chunk(h,3)[1] ## B C H W 2D representation of 3D videos
        h = self.mid.block_1(h, temb) ## video aware fusion
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)

        #h = Rearrange(h) ## B C 3 H W
        h = self.conv_out(h)
        #h = Rearrange(h,back=True) ## B3 C H W

        return h, phi_list_prev, phi_list_next

"""
class WaveletTransform3D(torch.nn.Module):
    def __init__(self):
        super(WaveletTransform3D, self).__init__()
        
        # Define Haar wavelet filters for low-pass and high-pass filtering
        self.low_filter = torch.tensor([1/2, 1/2], dtype=torch.float32).view(1, 1, -1)/torch.sqrt(torch.tensor(2))
        self.high_filter = torch.tensor([-1/2, 1/2], dtype=torch.float32).view(1, 1, -1)/torch.sqrt(torch.tensor(2))

    def conv1d_flat(self, x, filter, dim):
        """
        Apply Conv1d along a specified dimension (frames, height, or width) by flattening the tensor,
        applying the filter, and reshaping back to the original structure.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, frames, height, width)
            filter (torch.Tensor): 1D filter to apply
            dim (int): Dimension along which to apply the filter (0=frames, 1=height, 2=width)
        """
        # Ensure filter is on the same device as input tensor
        
        filter = filter.to(x.device)

        
        # Flatten the tensor along the specified dimension for Conv1d
        if dim == 0:  # frames
            b, c, f, h, w = x.shape
            x = x.permute(0, 1, 3, 4, 2).reshape(b * c * h * w, f)  # Flatten (b, c, h, w) -> (B, frames)

            x = F.conv1d(x.unsqueeze(1), filter, padding=0).squeeze(1)  # Apply Conv1d
            x = x.view(b, c, h, w, -1).permute(0, 1, 4, 2, 3)  # Unflatten back to original shape
        elif dim == 1:  # height
            b, c, f, h, w = x.shape
            x = x.permute(0, 1, 2, 4, 3).reshape(b * c * f * w, h)  # Flatten (b, c, f, w) -> (B, height)
            x = F.conv1d(x.unsqueeze(1), filter, padding="same",stride = 1).squeeze(1)  # Apply Conv1d
            x = x.view(b, c, f, w, -1).permute(0, 1, 2, 4, 3)  # Unflatten back to original shape
        elif dim == 2:  # width
            b, c, f, h, w = x.shape
            x = x.permute(0, 1, 2, 3, 4).reshape(b * c * f * h, w)  # Flatten (b, c, f, h) -> (B, width)
            x = F.conv1d(x.unsqueeze(1), filter, padding="same",stride = 1).squeeze(1)  # Apply Conv1d
            x = x.view(b, c, f, h, -1)  # Unflatten back to original shape
        return x

    def forward(self, x):
        """
        Apply 3D wavelet transform (Haar wavelet) to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, frames, height, width)
        Returns:
            Tuple[torch.Tensor]: Approximation coefficients and detail coefficients (horizontal, vertical, diagonal)
        """
        # Apply low-pass and high-pass filters along each axis
        L = self.conv1d_flat(x, self.low_filter, 0)  # Low-pass along frames
        H = self.conv1d_flat(x, self.high_filter, 0)  # High-pass along frames

        LL = self.conv1d_flat(L, self.low_filter, 1)  # Low-pass along frames and height
        LH = self.conv1d_flat(L, self.high_filter, 1)  # Low-pass frames, high-pass along height
        HL = self.conv1d_flat(H, self.low_filter, 1)  # High-pass frames, low-pass along height
        HH = self.conv1d_flat(H, self.high_filter, 1)  # High-pass both frames and height

        # Apply filters along width dimension to get final 3D wavelet decomposition
        LLL = self.conv1d_flat(LL, self.low_filter, 2)  # Low-pass all three dimensions
        LLH = self.conv1d_flat(LL, self.high_filter, 2)  # Low-pass frames and height, high-pass along width
        LHL = self.conv1d_flat(LH, self.low_filter, 2)  # Low-pass frames, high-pass height, low-pass width
        LHH = self.conv1d_flat(LH, self.high_filter, 2)  # Low-pass frames, high-pass both height and width
        HLL = self.conv1d_flat(HL, self.low_filter, 2)  # High-pass frames, low-pass height, low-pass width
        HLH = self.conv1d_flat(HL, self.high_filter, 2)  # High-pass frames, low-pass height, high-pass width
        HHL = self.conv1d_flat(HH, self.low_filter, 2)  # High-pass frames and height, low-pass along width
        HHH = self.conv1d_flat(HH, self.high_filter, 2)  # High-pass all three dimensions


        return LLL, torch.cat((LLH, LHL, LHH, HLL, HLH, HHL, HHH),dim = 2).squeeze(1) ## B C F H W, B C H W


class Frequency_block(nn.Module): ## 4x down
    def __init__(self,in_channels = 1, out_channels = 256):
        super().__init__()
        
        self.norm1 = Normalize(in_channels)

        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=5,
                                     stride=2,
                                     padding=2)
        
        self.norm2 = Normalize(out_channels)

        self.non_lin = nn.ReLU()
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=5,
                                     stride=2,
                                     padding=2)

        self.shortcut = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=5,
                                        stride=4,
                                        padding=1)
    def forward(self, x):

        h = self.norm1(x)
        h = self.non_lin(h)

        
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.non_lin(h)

        h = self.conv2(h)

        x = self.shortcut(x)

        return h + x

class Frequency_extractor(nn.Module):
    def __init__(self, out_channels = 256,num_blocks = 5):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.num_blocks = (num_blocks-1)//2
        

        self.conv_in = torch.nn.Conv2d(21,
                                     64,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1)

        Blocks = []
        for i in range(self.num_blocks-1):
            Blocks.append(Frequency_block(64,64))
        Blocks.append(Frequency_block(64,out_channels))

        self.Blocks = torch.nn.Sequential(*Blocks)

    def forward(self, x, temb):

        x = self.conv_in(x)

        x = self.Blocks(x)
        
        return x


class FIEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch # 128
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult) # 3
        self.num_res_blocks = num_res_blocks # 2
        self.resolution = resolution # 256
        self.in_channels = in_channels # 3

        # downsampling
        self.wavelet_transform = WaveletTransform3D()
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution # 256
        in_ch_mult = (1,)+tuple(ch_mult) # (1,1,2,4)
        self.in_ch_mult = in_ch_mult # (1,1,2,4)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = int(ch*in_ch_mult[i_level])
            block_out = int(ch*ch_mult[i_level])
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            # if i_level != self.num_resolutions-1:
            down.downsample = Downsample(block_in, resamp_with_conv)
            curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock_fusion(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_st_attn(block_in, attn_type=attn_type)
        #self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock_fusion(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.frequency_extractor = Frequency_extractor(block_in,self.num_resolutions)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        2*z_channels if double_z else z_channels, # 3
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, ret_feature=False):
        # timestep embedding
        temb = None
        if x.min() < 0:
            x = x/2 + 0.5
        # downsampling
        
        vid = Rearrange(x).clone().detach() ## B C F H W
        vid = (vid * torch.tensor([0.2989, 0.5870, 0.1140]).view(1,3,1,1,1).to(vid.device)).sum(dim = 1, keepdim = True)
        
        low_freq, high_freq_1 = self.wavelet_transform(vid) 
        low_freq, high_freq_2 = self.wavelet_transform(low_freq)
        high_freq = torch.cat([high_freq_1,high_freq_2],dim = 1)
        high_freq_fea = self.frequency_extractor(high_freq,None)
        phi_list = []
        hs = [self.conv_in(x)] ## B3 C H W

        reshaped = torch.chunk(hs[-1],3)
        phi_list.append(torch.cat([reshaped[0],reshaped[-1]])) ## batch concatenation of features
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            # if i_level != self.num_resolutions-1:
            hs.append(self.down[i_level].downsample(hs[-1]))
            #reshaped = Rearrange(hs[-1]) ## B C 3 H W
            #phi_list_prev.append(reshaped[:,:,0])
            #phi_list_next.append(reshaped[:,:,-1])
            reshaped = torch.chunk(hs[-1],3)
            phi_list.append(torch.cat([reshaped[0],reshaped[-1]])) ## batch concatenation of features

            #phi_list.append(hs[-1])

        # middle
        h = hs[-1] # ## B3 C H W
        
        h = Rearrange(h) 
        h = h  + torch.sigmoid(high_freq_fea.unsqueeze(2))*h ## B C F H W * B C 1 H W
        h = Rearrange(h,back = True)
        
        
        #h = torch.chunk(h,3)[1]
    
        h = self.mid.block_1(h, temb) ## video aware fusion
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)

        #h = Rearrange(h)[:,:,1] ## B C 3 H W
        h = Rearrange(h) ## B3 C H W --> B C 3 H W
        h = self.conv_out(h)
        h = Rearrange(h,back = True)
        #h = Rearrange(h,back=True,frames = 1) ## B3 C H W

        return h, phi_list

class FlowEncoder(FIEncoder):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0, resamp_with_conv=True, in_channels, resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla", **ignore_kwargs):
        super().__init__(
            ch=ch, 
            out_ch=out_ch,
            ch_mult=ch_mult, 
            num_res_blocks=num_res_blocks, 
            attn_resolutions=attn_resolutions, 
            dropout=dropout, 
            resamp_with_conv=resamp_with_conv, 
            in_channels=in_channels, 
            resolution=resolution, 
            z_channels=z_channels, 
            double_z=double_z, 
            use_linear_attn=use_linear_attn, 
            attn_type=attn_type, 
            **ignore_kwargs
        )


"""
class FlowDecoderWithResidual(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", num_head_channels=32, num_heads=1, cond_type=None,load_VFI = None,
                 **ignorekwargs):
        super().__init__()

        def MaskHead(c_in):
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    Normalize(64,num_groups = 16),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    Normalize(32,num_groups = 8),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
                    torch.nn.Sigmoid()
                )
        def ResidualHead(c_in):
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    Normalize(64,num_groups = 16),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    Normalize(32,num_groups = 8),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
                    torch.nn.Sigmoid()
                )
        
        self.load_VFI = load_VFI
        self.ch = ch # 128
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult) # 3
        self.num_res_blocks = num_res_blocks # 2
        self.resolution = resolution # 256
        self.in_channels = in_channels # 3
        self.give_pre_end = give_pre_end # False
        self.tanh_out = tanh_out # False

        vfi = VFIformer()
        if not self.load_VFI is None:
            print(f'loading VFIformer from {self.load_VFI}')
            vfi.load_state_dict(torch.load(self.load_VFI))
        self.flownet = vfi.flownet
        self.refinenet = vfi.refinenet
        for p in self.flownet.parameters():
            p.requires_grad = False
        for p in self.refinenet.parameters():
            p.requires_grad = False

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult) # (1,1,2,4)
        block_in = int(ch*ch_mult[self.num_resolutions-1]) # 512
        curr_res = resolution // 2**(self.num_resolutions-1) # 64
        self.z_shape = (1,z_channels,curr_res,curr_res) # (1,3,64,64)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)): # 2,1,0
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = int(ch*ch_mult[i_level])
            # ResBlocks
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            # CrossAttention
            if num_head_channels == -1:
                dim_head = block_in // num_heads
            else:
                num_heads = block_in // num_head_channels
                dim_head = num_head_channels # 32
            if cond_type == 'cross_attn':
                cross_attn = SpatialCrossAttentionWithPosEmb(in_channels=block_in, 
                                                             heads=num_heads,
                                                             dim_head=dim_head)
            elif cond_type == 'max_cross_attn':
                cross_attn = SpatialCrossAttentionWithMax(in_channels=block_in,
                                                          heads=num_heads,
                                                          dim_head=dim_head)
            elif cond_type == 'max_cross_attn_frame':
                cross_attn = SpatialCrossAttentionWithMax(in_channels=block_in,
                                                          heads=num_heads,
                                                          dim_head=dim_head,
                                                          ctx_dim=6)
            else:
                cross_attn = IdentityWrapper()

            up = nn.Module()
            up.block = block
            up.attn = attn
            up.cross_attn = cross_attn

            # Upsample
            # if i_level != self.num_resolutions-1: ## THIS IS ORIGINAL CODE
            # if i_level != 0:
            up.upsample = Upsample(block_in, resamp_with_conv)
            curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        block_in,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        self.moduleMask = MaskHead(c_in=block_in)
        self.moduleResidual = ResidualHead(c_in=block_in)

    def forward(self, z, cond_dict):
        self.flownet.eval()
        self.refinenet.eval()

        phi_prev_list = cond_dict['phi_prev_list']
        phi_next_list = cond_dict['phi_next_list']
        frame_prev = cond_dict['frame_prev']
        frame_next = cond_dict['frame_next']

        back = False
        if frame_prev.min() < 0:
            back = True
            frame_prev = frame_prev/2 + 0.5
            frame_next = frame_next/2 + 0.5

        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        B, _, H, W = frame_prev.size()
        imgs = torch.cat((frame_prev, frame_next), 1)

        flow, flow_list = self.flownet(imgs)
        flow, c0, c1 = self.refinenet(frame_prev, frame_next, flow) ## flow and warped features of refined flows, 1, 1/2,1/4,1/8
        warped_img0 = warp(frame_prev, flow[:, :2])
        warped_img1 = warp(frame_next, flow[:, 2:])
        
        phi_prev_list = self.refinenet.warp_fea(phi_prev_list,flow[:, :2])
        phi_next_list = self.refinenet.warp_fea(phi_next_list,flow[:,2:4])## warping features

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)): # [2,1,0]
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            ctx = None
            if phi_prev_list[i_level] is not None:
                ctx = torch.cat([phi_prev_list[i_level], phi_next_list[i_level]], dim=1)
            h = self.up[i_level].cross_attn(h, ctx)
            # if i_level != self.num_resolutions-1:
            # if i_level != 0:
            h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) 

        mask1 = self.moduleMask(h)
        mask2 = 1.0 - mask1
        res = self.moduleResidual(h)
        res = res*2 - 1 ## -1,1
        out = warped_img0*mask1 + warped_img1*mask2 + res
        if back:
            out = out.clamp_(min = 0, max = 1)
            out = out*2 - 1 ## -1,1
        else:
            out = out.clamp_(min=-1,max = 1)
        return out
"""


### 3D aware version
class FlowDecoderWithResidual(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", num_head_channels=32, num_heads=1, cond_type=None,load_VFI = None,
                 **ignorekwargs):
        super().__init__()

        def OutputHead(c_in):
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    Normalize(64,num_groups = 16),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    Normalize(32,num_groups = 8),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1)
                )
        
        self.load_VFI = load_VFI
        self.ch = ch # 128
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult) # 3
        self.num_res_blocks = num_res_blocks # 2
        self.resolution = resolution # 256
        self.in_channels = in_channels # 3
        self.give_pre_end = give_pre_end # False
        self.tanh_out = tanh_out # False

        vfi = VFIformer()
        if not self.load_VFI is None:
            print(f'loading VFIformer from {self.load_VFI}')
            vfi.load_state_dict(torch.load(self.load_VFI))
        self.flownet = vfi.flownet
        #self.refinenet = vfi.refinenet
    
        self.refinenet = FlowRefineNet_Multis_our(c = self.ch)
        
        for p in self.flownet.parameters():
             p.requires_grad = False

        
        
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult) # (1,1,2,4)
        block_in = int(ch*ch_mult[self.num_resolutions-1]) # 512
        curr_res = resolution // 2**(self.num_resolutions-1) # 64
        self.z_shape = (1,z_channels,curr_res,curr_res) # (1,3,64,64)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv3d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1) ## video aware

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock_fusion(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_st_attn(block_in, attn_type=attn_type)
        #self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock_Dec(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout) ## video aware
        


        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)): # 2,1,0
            block = nn.ModuleList()
            fusion = nn.ModuleList()
            attn = nn.ModuleList()

            block_out = int(ch*ch_mult[i_level])
            # ResBlocks
            if i_level > 2: ## if > 8x downsampling, do not warp
                scale = 3
            else:
                scale = 5
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            # CrossAttention
            
            
            if num_head_channels == -1:
                dim_head = block_in // num_heads
            else:
                num_heads = block_in // num_head_channels
                dim_head = num_head_channels # 32
            if cond_type == 'cross_attn':
                cross_attn = SpatialCrossAttentionWithPosEmb(in_channels=block_in, 
                                                             heads=num_heads,
                                                             dim_head=dim_head)
            elif cond_type == 'max_cross_attn':
                if i_level > 2: ## if > 8x downsampling, do not warp
                    cross_attn = SpatialCrossAttentionWithMax(in_channels=block_in,
                                                          heads=num_heads,
                                                          dim_head=dim_head,
                                                          ctx_dim = block_in*2)
                else:
                    cross_attn = SpatialCrossAttentionWithMax(in_channels=block_in,
                                                          heads=num_heads,
                                                          dim_head=dim_head,
                                                          ctx_dim = block_in*4)

            elif cond_type == 'max_cross_attn_frame':
                cross_attn = SpatialCrossAttentionWithMax(in_channels=block_in,
                                                          heads=num_heads,
                                                          dim_head=dim_head,
                                                          ctx_dim=6)
            else:
                cross_attn = IdentityWrapper()
            

            up = nn.Module()
            up.block = block
            up.attn = attn
            up.cross_attn = cross_attn
            up.fusion = fusion

            # Upsample
            # if i_level != self.num_resolutions-1: ## THIS IS ORIGINAL CODE
            # if i_level != 0:
            up.upsample = Upsample(block_in, resamp_with_conv)
            curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)

        self.conv_out = torch.nn.Conv2d(block_in,
                                        block_in,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        #self.moduleMask = MaskHead(c_in=block_in)
        #self.moduleResidual = ResidualHead(c_in=block_in)
        self.moduleout = OutputHead(c_in=block_in)

    def forward(self, z, cond_dict,flow = None):
        #self.flownet.eval()
        #self.refinenet.eval()

        phi_list = cond_dict['phi_list']
        frame_prev = cond_dict['frame_prev']
        frame_next = cond_dict['frame_next']



        back = False
        if frame_prev.min() < 0:
            back = True
            frame_prev = frame_prev/2 + 0.5
            frame_next = frame_next/2 + 0.5


        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        B, _, H, W = frame_prev.size()
        imgs = torch.cat((frame_prev, frame_next), 1)

        #phi_list = phi_list[1:]
        
        if flow is not None:
            _, c0, c1 = self.refinenet.get_context(phi_list[:-2], flow) ## flow and warped features of refined flows, 1, 1/2,1/4,1/8
            #c0,c1 =  self.refinenet.warp_batch_fea(phi_list, flow,B)
        else:
            flow, flow_list = self.flownet(imgs)
            flow, c0, c1 = self.refinenet(phi_list[:-2], flow)
            #flow, c0, c1 = self.refinenet(frame_prev, frame_next, flow)
            #c0, c1 =  self.refinenet.warp_batch_fea(phi_list, flow,B)
        
        ## flow : B 4 H W

        phi_list = phi_list[1:]
        c0, c1 = c0[1:], c1[1:] ## remove full size features
        #c0,c1 =  self.refinenet.warp_batch_fea(phi_list, flow,B)
        warped_img0 = warp(frame_prev, flow[:, :2])
        warped_img1 = warp(frame_next, flow[:, 2:])

        # z to block_in

        ## z: bf c h w
        z = Rearrange(z)
        h = self.conv_in(z) ## B C 3 H W
        h = Rearrange(h,back = True) ## B3 C H W

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)#.squeeze(2)
        h = self.mid.block_2(h, temb)

        #h = self.fusion(h,temb).squeeze(2)


        # upsampling
        
        for i_level in reversed(range(self.num_resolutions)): # [2,1,0]
            ctx = None
            if phi_list[i_level] is not None:
                if i_level > 2:
                    ctx = torch.cat([phi_list[i_level][:B],phi_list[i_level][B:]],dim =1) ## B 2C H W
                else:
                    ctx = torch.cat([phi_list[i_level][:B],phi_list[i_level][B:],c0[i_level],c1[i_level]], dim=1) ## B 4C H W
                #ctx = torch.cat([phi_prev_list[i_level], phi_next_list[i_level]], dim=3) ## B C H 2W
                #h = torch.cat([phi_prev_list[i_level], h, phi_next_list[i_level]],0)
                #h = self.up[i_level].fusion[i_block](h,temb)


            for i_block in range(self.num_res_blocks):

                #h = self.up[i_level].block[i_block](torch.cat([h,ctx],dim = 1), temb)
                h = self.up[i_level].block[i_block](h, temb)

                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            

            h = self.up[i_level].cross_attn(h, ctx) ## cross attention
            # if i_level != self.num_resolutions-1:
            # if i_level != 0:
            h = self.up[i_level].upsample(h) 

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h) ## B C H W
        h = nonlinearity(h)
        #h = Rearrange(h)
        h = self.conv_out(h)#.squeeze(2) ## B C 1 H W --> B C H W 
        """
        mask1 = self.moduleMask(h)
        mask2 = 1.0 - mask1
        """
        out = self.moduleout(h)
        mask1 = torch.sigmoid(out[:,3:4]) ## 0 - 1
        mask2 = 1 - mask1
        #res = self.moduleResidual(h)
        res = out[:,:3]
        res = torch.sigmoid(res)*2 - 1 ## -1,1
        out = warped_img0*mask1 + warped_img1*mask2 + res
        """
        back = False
        out = res
        """
        if back:
            out = out.clamp_(min = 0, max = 1)
            out = out*2 - 1 ## -1,1
        else:
            out = out.clamp_(min=-1,max = 1)

        return out

    def get_flow(self, img0, img1,feats):
        imgs = torch.cat((img0, img1), 1)
        flow, flow_list = self.flownet(imgs)
        flow, c0, c1 = self.refinenet(feats,flow)
        return flow
    
