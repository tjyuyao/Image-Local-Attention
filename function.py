import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from localAttention import (similar_forward,
                            similar_backward,
                            weighting_forward,
                            weighting_backward_ori,
                            weighting_backward_weight)

__all__ = ['f_similar', 'f_weighting', 'LocalAttention', 'TorchLocalAttention']


class similarFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW, dilate=1):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        ctx.dilate = dilate
        output = similar_forward(x_ori, x_loc, kH, kW, dilate)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = similar_backward(x_loc, grad_outputs, kH, kW, ctx.dilate, True)
        grad_loc = similar_backward(x_ori, grad_outputs, kH, kW, ctx.dilate, False)

        return grad_ori, grad_loc, None, None, None


class weightingFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW, dilate=1):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        ctx.dilate = dilate
        output = weighting_forward(x_ori, x_weight, kH, kW, dilate)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = weighting_backward_ori(x_weight, grad_outputs, kH, kW, ctx.dilate)
        grad_weight = weighting_backward_weight(x_ori, grad_outputs, kH, kW, ctx.dilate)

        return grad_ori, grad_weight, None, None, None


f_similar = similarFunction.apply
f_weighting = weightingFunction.apply


class LocalAttention(nn.Module):
    def __init__(self, inp_channels, out_channels, kH, kW, dilate=1):
        super(LocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.kH = kH
        self.kW = kW
        self.dilate = dilate

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        print("x1", x1.shape)
        print("x2", x2.shape)
        weight = f_similar(x1, x2, self.kH, self.kW, self.dilate)
        print("weight", weight.shape)
        weight = F.softmax(weight, -1)
        out = f_weighting(x3, weight, self.kH, self.kW, self.dilate)

        return out


class TorchLocalAttention(nn.Module):
    def __init__(self, inp_channels, out_channels, kH, kW, dilate=1):
        super(TorchLocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.kH = kH
        self.kW = kW
        self.dilate = dilate

    @staticmethod
    def f_similar(x_theta, x_phi, kh, kw, dilate=1):
        n, c, h, w = x_theta.size()  # (N, inter_channels, H, W)
        pad = (kh // 2 * dilate, kw // 2 * dilate)
        x_theta = x_theta.permute(0, 2, 3, 1).contiguous()
        x_theta = x_theta.view(n * h * w, 1, c)

        x_phi = F.unfold(x_phi, kernel_size=(kh, kw), stride=1, padding=pad, dilation=dilate)
        x_phi = x_phi.contiguous().view(n, c, kh * kw, h * w)
        x_phi = x_phi.permute(0, 3, 1, 2).contiguous()
        x_phi = x_phi.view(n * h * w, c, kh * kw)

        out = torch.matmul(x_theta, x_phi)
        out = out.view(n, h, w, kh * kw)

        return out

    @staticmethod
    def f_weighting(x_theta, x_phi, kh, kw, dilate=1):
        n, c, h, w = x_theta.size()  # (N, inter_channels, H, W)
        pad = (kh // 2 * dilate, kw // 2 * dilate)
        x_theta = F.unfold(x_theta, kernel_size=(kh, kw), stride=1, padding=pad, dilation=dilate)
        x_theta = x_theta.permute(0, 2, 1).contiguous()
        x_theta = x_theta.view(n * h * w, c, kh * kw)

        x_phi = x_phi.view(n * h * w, kh * kw, 1)

        out = torch.matmul(x_theta, x_phi)
        out = out.squeeze(-1)
        out = out.view(n, h, w, c)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        weight = self.f_similar(x1, x2, self.kH, self.kW, self.dilate)
        weight = F.softmax(weight, -1)
        out = self.f_weighting(x3, weight, self.kH, self.kW, self.dilate)

        return out
    
    
if __name__ == '__main__':
    b, c, h, w = 8, 3, 32, 32
    kH, kW = 5, 5
    x = torch.rand(b, c, h, w).cuda()
    m = LocalAttention(c, c, kH, kW)
    m.cuda()
    y = m(x)