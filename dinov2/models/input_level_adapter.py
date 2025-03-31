# -*- coding: utf-8 -*-
"""Input_level_adapter.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/167em6SRMZhY0L7Lo-zY2ibkV0qQ7kMWJ
"""

import torch.nn as nn
import os
import torch
from torch import Tensor
from typing import Optional, Tuple, List
from torch.nn.functional import grid_sample, conv2d, interpolate, pad as torch_pad

class Kernel_Predictor(nn.Module):
    def __init__(self, dim, mode='low', num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        # Query Adaptive Learning (QAL)
        self.q = nn.Parameter(torch.rand((1, 4, dim)), requires_grad=True)

        self.kv_downsample = nn.Sequential(
            nn.Conv2d(3, dim // 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim),
        )
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.down = nn.Linear(dim, 1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Basic Parameters Number
        if mode == 'low':
            self.gain_base = nn.Parameter(torch.FloatTensor([3]), requires_grad=True)
        else:
            self.gain_base = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)


        self.r1_base = nn.Parameter(torch.FloatTensor([3]), requires_grad=False)
        self.r2_base = nn.Parameter(torch.FloatTensor([2]), requires_grad=False)

    def forward(self, x):
        d_x = self.kv_downsample(x).flatten(2).transpose(1, 2)
        B, N, C = d_x.shape
        k = self.k(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, 4, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = self.down(out).squeeze(-1)

        out = torch.unbind(out, 1)
        r1, r2, gain, sigma = out[0], out[1], out[2], out[3]
        r1 = 0.1 * r1 +  self.r1_base
        r2 = 0.1 * r2 +  self.r2_base

        gain =gain + self.gain_base

        return r1, r2, gain, self.sigmoid(sigma)

class Matrix_Predictor(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        # Query Adaptive Learning (QAL)
        self.q = nn.Parameter(torch.rand((1, 9 + 1, dim)), requires_grad=True)
        self.kv_downsample = nn.Sequential(
            nn.Conv2d(3, dim // 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim),
        )
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.down = nn.Linear(dim, 1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.ccm_base = nn.Parameter(torch.eye(3), requires_grad=False)

    def forward(self, x):
        d_x = self.kv_downsample(x).flatten(2).transpose(1, 2)
        B, N, C = d_x.shape
        k = self.k(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, 9 + 1, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = self.down(out)
        out, distance = out[:, :9, :], out[:, 9:, :].squeeze(-1)
        out = out.view(B, 3, 3)
        # print(self.ccm_base)
        # print(out)

        ccm_matrix = 0.1 * out + self.ccm_base
        distance = self.relu(distance) + 1

        return ccm_matrix, distance

class NILUT(nn.Module):
    """
    Simple residual coordinate-based neural network for fitting 3D LUTs
    Official code: https://github.com/mv-lab/nilut
    """
    def __init__(self, in_features=3, hidden_features=32, hidden_layers=3, out_features=3, res=True):
        super().__init__()

        self.res = res
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())

        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())

        self.net.append(nn.Linear(hidden_features, out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, intensity):
        output = self.net(intensity)
        if self.res:
            output = output + intensity
            output = torch.clamp(output, 0.,1.)
        return output

def _assert_image_tensor(img: Tensor) -> None:
    if not img.ndim >= 2:
        raise TypeError("Tensor is not a torch image.")

def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(sigma.device)
    #print(x.device)
    #print(sigma.device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d

def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype

def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img

def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    _assert_image_tensor(img)

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(
        img,
        [
            kernel.dtype,
        ],
    )

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img

def Gain_Denoise(I1, r1, r2, gain, sigma, k_size=3):  # [9, 9] in LOD dataset, [3, 3] in other dataset
    out = []
    for i in range(I1.shape[0]):
        I1_gain = gain[i] * I1[i,:,:,:]
        blur = gaussian_blur(I1_gain, \
                                [k_size, k_size], \
                                [r1[i], r2[i]])
        sharp = blur + sigma[i] * (I1[i,:,:,:] - blur)
        out.append(sharp)
    return torch.stack([out[i] for i in range(I1.shape[0])], dim=0)

# Shades of Gray and Colour Constancy (Graham D. Finlayson, Elisabetta Trezzi)
def SoG_algo(img, p=1):
    # https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/12/1/art00008
    img = img.permute(1,2,0)       # (C,H,W) --> (H,W,C)

    img_P = torch.pow(img, p)

    R_avg = torch.mean(img_P[:,:,0]) ** (1/p)
    G_avg = torch.mean(img_P[:,:,1]) ** (1/p)
    B_avg = torch.mean(img_P[:,:,2]) ** (1/p)

    Avg = torch.mean(img_P) ** (1/p)

    R_avg = R_avg / Avg
    G_avg = G_avg / Avg
    B_avg = B_avg / Avg

    img_out = torch.stack([img[:,:,0]/R_avg, img[:,:,1]/G_avg, img[:,:,2]/B_avg], dim=-1)

    return img_out

def WB_CCM(I2, ccm_matrix, distance):
    out_I3 = []
    out_I4 = []
    for i in range(I2.shape[0]):
        # SOG White Balance Algorithm
        I3 = SoG_algo(I2[i,:,:,:], distance[i])

        # Camera Color Matrix
        I4 = torch.tensordot(I3, ccm_matrix[i,:,:], dims=[[-1], [-1]])
        I4 = torch.clamp(I4, 1e-5, 1.0)

        out_I3.append(I3)
        out_I4.append(I4)

    return  torch.stack([out_I3[i] for i in range(I2.shape[0])], dim=0), \
            torch.stack([out_I4[i] for i in range(I2.shape[0])], dim=0)

class Input_level_Adapeter(nn.Module):
    def __init__(self, mode='normal', lut_dim=32, out='all', k_size=3, w_lut=True):
        super(Input_level_Adapeter, self).__init__()
        '''
        mode: normal (for normal & over-exposure conditions) or low (for low-light conditions)
        lut_dim: implicit neural look-up table dim number
        out: if all, return I1, I2, I3, I4, I5, if not all, only return I5
        k_size: denosing kernel size, must be odd number, we set it to 9 in LOD dataset and 3 in other dataset
        w_lut: with or without implicit 3D Look-up Table
        '''

        self.Predictor_K = Kernel_Predictor(dim=64, mode=mode)
        self.Predictor_M = Matrix_Predictor(dim=64)
        self.w_lut = w_lut
        if self.w_lut:
            self.LUT = NILUT(hidden_features=lut_dim)
        self.out = out
        self.k_size = k_size



    def forward(self, I1):
        # (1). I1 --> I2: Denoise & Enhancement & Sharpen
        r1, r2, gain, sigma = self.Predictor_K(I1)
        I2 = Gain_Denoise(I1, r1, r2, gain, sigma, k_size=self.k_size)  # (B,C,H,W)
        I2 = torch.clamp(I2, 1e-5, 1.0) # normal & over-exposure

        ccm_matrix, distance = self.Predictor_M(I2)

        # (2). I2 --> I3: White Balance, Shade of Gray
        # (3). I3 --> I4: Camera Colour Matrix Transformation
        I3, I4 = WB_CCM(I2, ccm_matrix, distance) # (B,H,W,C)

        if self.w_lut:
        # (4). I4 --> I5: Implicit Neural LUT
            I5 = self.LUT(I4).permute(0,3,1,2)

            if self.out == 'all':   # return all features
                return [I1, I2, I3.permute(0,3,1,2), I4.permute(0,3,1,2), I5]
            else:   # only return I5
                return [I5]

        else:
            if self.out == 'all':
                return [I1, I2, I3.permute(0,3,1,2), I4.permute(0,3,1,2)]
            else:
                return [I4.permute(0,3,1,2)]




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='4'

    input = torch.rand([4,3,512,512])
    net = Input_level_Adapeter(out='all', w_lut=False)
    out = net(input)

