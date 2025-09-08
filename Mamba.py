import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from einops import rearrange
from tqdm import tqdm
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
import math
import os
import urllib.request
from zipfile import ZipFile

from transformers import AutoTokenizer
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair
torch.autograd.set_detect_anomaly(True)
# Configuration flags and hyperparameters
USE_MAMBA = 1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_model = 8
state_size = 128  # Example state size
seq_len = 100  # Example sequence length
batch_size = 256  # Example batch size
last_batch_size = 81  # only for the very last batch of the dataset
current_batch_size = batch_size
different_batch_size = False
h_new = None
temp_buffer = None


class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(S6, self).__init__()

        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)

        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)

        # h [batch_size, seq_len, d_model, state_size]
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)

    def discretization(self):

        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)

        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))

        return self.dA, self.dB

    def forward(self, x):
        # Algorithm 2 MAMBA paper
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        self.discretization()

        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:

            global current_batch_size
            current_batch_size = x.shape[0]

            if self.h.shape[0] != current_batch_size:
                different_batch_size = True

                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x,
                                                                                                               "b l d -> b l d 1") * self.dB

            else:
                different_batch_size = False
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y [batch_size, seq_len, d_model]
            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()

            return self.y

        else:
            # h [batch_size, seq_len, d_model, state_size]
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
            y = torch.zeros_like(x)

            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y [batch_size, seq_len, d_model]
            y = torch.einsum('bln,bldn->bld', self.C, h)

            return y


class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(Mamba, self).__init__()
        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size, device)

    def forward(self, x):
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: str ='cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class Embeddings(nn.Module):
    # Construct the patch, position embeddings
    def __init__(self, config, patch_size, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        # img_size[0]=img
        # patch_size[0]=patch_size[1] [16, 8, 4, 2]
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):  #  x torch.Size([4, 64, 224, 224])
        if x is None:
            return None
        # print(x.size())
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2)) ([4, 64, 14, 14])
        # print(x.size())
        x = x.flatten(2)   # ([4, 64, 196])
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)  ([4, 196, 64])
        embeddings = x + self.position_embeddings  # ([4, 196, 64])
        embeddings = self.dropout(embeddings) # ([4, 196, 64])
        return embeddings

class ConvTransBN(nn.Module):  # (convolution => [BN] => ReLU)
    def __init__(self, in_channels, out_channels):
        super(ConvTransBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device, config, patch_size, img_size, channel_num, embed_dim):
        super(MambaBlock, self).__init__()
        self.embeddings = Embeddings(config=config, patch_size=patch_size, img_size=img_size, in_channels=channel_num)
        self.inp_proj = nn.Linear(d_model, 2*d_model, device=device)
        self.out_proj = nn.Linear(2*d_model, d_model, device=device)
        self.embed = embed_dim
         # For residual skip connection
        self.D = nn.Linear(d_model, 2*d_model, device=device)

         # Set _no_weight_decay attribute on bias
        self.out_proj.bias._no_weight_decay = True

         # Initialize bias to a small constant value
        nn.init.constant_(self.out_proj.bias, 1.0)

        self.S6 = S6(seq_len, 2*d_model, state_size, device)

         # Add 1D convolution with kernel size 3
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device)

         # Add linear layer for conv output
        self.conv_linear = nn.Linear(2*d_model, 2*d_model, device=device)

         # rmsnorm
        self.norm = RMSNorm(d_model, device=device)

        self.CTBN = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim//2)
        self.CTBN2 = ConvTransBN(in_channels=embed_dim*2, out_channels=embed_dim)
        self.CTBN3 = ConvTransBN(in_channels=10, out_channels=49)

    def forward(self, x, skip_x, text, reconstruct=False):  # 1 x和skipx torch.Size([4, 64, 224, 224]),text torch.Size([4, 10, 64])
                                                            # 2 x([4, 128, 112, 112]),skipx([4, 196, 64]) ,text torch.Size([4, 10, 128])
        """
        x_proj.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv_act.shape = torch.Size([batch_size, seq_len, 2*d_model])
        """
         # Refer to Figure 3 in the MAMBA paper
        # print(x.size())
        # print(skip_x.size())
        # print(text.size())
        if not reconstruct:
            x = self.embeddings(x)  # torch.Size([4, 196, 64])
            if self.embed == 64:
                x = x + self.CTBN3(text)
            else:
                x = x

            x = self.norm(x)

            x_proj = self.inp_proj(x)

             # Add 1D convolution with kernel size 3
            x_conv = self.conv(x_proj)

            x_conv_act = F.silu(x_conv)

             # Add linear layer for conv output
            x_conv_out = self.conv_linear(x_conv_act)

            x_ssm = self.S6(x_conv_out)
            x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)

             # residual skip connection with nonlinearity introduced by multiplication
            x_residual = F.silu(self.D(x))

            x_combined = x_act * x_residual

            x_out = self.out_proj(x_combined)
            # print("x_out",x_out.size())
        if self.embed == 64 and not reconstruct:
            # print("x_out2", x_out.size())
            return x_out
        elif self.embed == 512 and reconstruct:
            return x

        elif not reconstruct:
            # print("elif not reconstruct")
            x_out = x_out.transpose(1, 2)
            # print(x.size())
            x_out = self.CTBN(x_out)
            # print(x.size())
            x_out = x_out.transpose(1, 2)
            # print(x.size())
            y = torch.cat([x_out, skip_x], dim=2)
            # print(y.size())
            # print("y1", y.size())
            return y
        elif reconstruct:
            # print("elif reconstruct")
            skip_x = skip_x.transpose(1, 2)
            # print(skip_x.size())
            skip_x = self.CTBN2(skip_x)
            # print(skip_x.size())
            skip_x = skip_x.transpose(1, 2)
            # print(skip_x.size())
            y = x+skip_x
            # print(y.size())
            # print("y2", y.size())
            return y





