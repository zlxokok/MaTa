import torch
import torch.nn as nn
import torch.nn.functional as F

from .Vit import VisionTransformer, Reconstruct
from .pixlevel import PixLevelModule
from .Mamba import MambaBlock
from collections import OrderedDict


class Curvature(torch.nn.Module):
    def __init__(self, ratio=1):
        super(Curvature, self).__init__()
        weights = torch.tensor([[[[-1 / 16, 5 / 16, -1 / 16], [5 / 16, -1, 5 / 16], [-1 / 16, 5 / 16, -1 / 16]]]])
        self.weight = torch.nn.Parameter(weights).cuda()
        self.ratio = ratio

    def forward(self, x):
        B, C, H, W = x.size()
        x_origin = x
        x = x.reshape(B * C, 1, H, W)
        out = F.conv2d(x, self.weight)
        out = torch.abs(out)
        p = torch.sum(out, dim=-1)
        p = torch.sum(p, dim=-1)
        p = p.reshape(B, C)

        _, index = torch.topk(p, int(self.ratio * C), dim=1)
        selected = []
        for i in range(x_origin.shape[0]):
            selected.append(torch.index_select(x_origin[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)

        return selected

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)
class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 1, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class CTRGAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            gnconv(dim),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2*dim, 2*dim//factor, 1, bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor, kernel_size*kernel_size*dim, 1)
        )


    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size*self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = F.softmax(att, dim=-1)*v
        k2 = k2.view(bs, c, h, w)


        return k1 + k2


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN1(nn.Module):
    def __init__(self, channel, map_size, pad):
        super(CNN1, self).__init__()
        self.conv = nn.Conv2d(channel, channel,
                              kernel_size=map_size, padding=pad)
        self.norm = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.relu(out)

class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class MaTa(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(32, 32))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(4, 4))
        self.pix_module1 = PixLevelModule(64)
        self.pix_module2 = PixLevelModule(128)
        self.pix_module3 = PixLevelModule(256)
        self.pix_module4 = PixLevelModule(512)
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.mamba1_down = MambaBlock(49, 64, 4, "cuda",config, img_size=224, channel_num=64, patch_size=32,embed_dim=64)
        self.mamba2_down = MambaBlock(49, 128, 4, "cuda",config, img_size=112, channel_num=128, patch_size=16, embed_dim=128)
        self.mamba3_down = MambaBlock(49, 256, 4, "cuda", config, img_size=56, channel_num=256, patch_size=8, embed_dim=256)
        self.mamba4_down = MambaBlock(49, 512, 4, "cuda", config, img_size=28, channel_num=512, patch_size=4, embed_dim=512)
        self.mamba1_up = MambaBlock(49, 64, 4, "cuda",config, img_size=224, channel_num=64, patch_size=32,embed_dim=64)
        self.mamba2_up = MambaBlock(49, 128, 4, "cuda",config, img_size=112, channel_num=128, patch_size=16, embed_dim=128)
        self.mamba3_up = MambaBlock(49, 256, 4, "cuda", config, img_size=56, channel_num=256, patch_size=8, embed_dim=256)
        self.mamba4_up = MambaBlock(49, 512, 4, "cuda", config, img_size=28, channel_num=512, patch_size=4, embed_dim=512)
        self.CoDw64 = CTRGAttention(64)
        self.CoDw128 = CTRGAttention(128)
        self.CoDw256 = CTRGAttention(256)
        self.CoDw512 = CTRGAttention(512)
        self.mamba2 = MambaBlock(196, 128, 4, "cuda")
        self.mamba3 = MambaBlock(196, 256, 4, "cuda")
        self.mamba4 = MambaBlock(196, 512, 4, "cuda")

    def forward(self, x, text):
        x = x.float()  # x [4,3,224,224]
        x1 = self.inc(x)  # x1 [4, 64, 224, 224]
        # text = torch.rand(4, 10,768).to('cuda')
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2)
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)  # batch=8, text1([8, 10, 64]), x1=([8, 64, 224, 224])
        x1 = self.CoDw64(x1)
        y1 = self.mamba1_down(x1, x1, text1)  # y1 [8, 196, 64])

        x2 = self.down1(x1)
        y2 = self.mamba2_down(x2, y1, text2)  # y2 [8, 196, 128])
        x2 = self.CoDw128(x2)
        x3 = self.down2(x2)
        y3 = self.mamba3_down(x3, y2, text3)  # y3 [8, 196, 256])
        x3 = self.CoDw256(x3)
        x4 = self.down3(x3)
        y4 = self.mamba4_down(x4, y3, text4)  # y4 [8, 196, 512])
        x4 = self.CoDw512(x4)
        x5 = self.down4(x4)
        y4 = self.mamba4_up(y4, y4, text4, True)
        y3 = self.mamba3_up(y3, y4, text3, True)
        y2 = self.mamba2_up(y2, y3, text2, True)
        y1 = self.mamba1_up(y1, y2, text1, True)
        # print(x1.size())
        # print(self.reconstruct1(y1).size())
        x1 = self.reconstruct1(y1) + x1
        x2 = self.reconstruct2(y2) + x2
        x3 = self.reconstruct3(y3) + x3
        x4 = self.reconstruct4(y4) + x4
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
        return logits


