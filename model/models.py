import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LightweightISP(nn.Module):
    """
    Lightweight 4ch RAW -> 3ch RGB model.

    Input:
      (B, 4, H, W)
    Output:
      (B, 3, 2H, 2W)
    """

    def __init__(self, width: int = 64, upsample_mode: str = "resize_conv", resize_mode: str = "bilinear"):
        super().__init__()
        if upsample_mode not in ("resize_conv", "pixelshuffle"):
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")
        if resize_mode not in ("nearest", "bilinear", "bicubic"):
            raise ValueError(f"Unsupported resize_mode: {resize_mode}")

        self.upsample_mode = upsample_mode
        self.resize_mode = resize_mode
        self.conv1 = nn.Conv2d(4, width, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(width, 12, kernel_size=3, stride=1, padding=1)
        if self.upsample_mode == "resize_conv":
            self.conv_up = nn.Conv2d(12, width, kernel_size=3, stride=1, padding=1)
            self.conv_out = nn.Conv2d(width, 3, kernel_size=3, stride=1, padding=1)
        else:
            self.conv_up = None
            self.conv_out = None

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(self.conv1(x))
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        if self.upsample_mode == "pixelshuffle":
            x = F.pixel_shuffle(x, upscale_factor=2)
            return x

        if self.resize_mode in ("bilinear", "bicubic"):
            x = F.interpolate(x, scale_factor=2, mode=self.resize_mode, align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.resize_mode)
        x = F.relu(self.conv_up(x), inplace=True)
        x = self.conv_out(x)
        return x


class UNetDiscriminatorSN(nn.Module):
    """Patch discriminator with spectral normalization."""

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, skip_connection: bool = True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = nn.utils.spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2

        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)
        return out
