# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class MSSSIMLoss(nn.Module):
#     def __init__(self, window_size=11, channel=3, size_average=True, weights=None):
#         super(MSSSIMLoss, self).__init__()
#         self.window_size = window_size
#         self.channel = channel
#         self.size_average = size_average
#         self.C1 = 0.01 ** 2
#         self.C2 = 0.03 ** 2
#         self.window = self.create_window(window_size, channel)

#         # Default weights for 5 scales as used in the original MS-SSIM paper
#         if weights is None:
#             self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
#         else:
#             self.weights = weights

#     def gaussian_window(self, window_size, sigma):
#         gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#         return gauss / gauss.sum()

#     def create_window(self, window_size, channel):
#         _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
#         _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#         window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#         return window

#     def ssim(self, X, Y, window, window_size, channel, size_average=True):
#         mu1 = F.conv2d(X, window, padding=window_size // 2, groups=channel)
#         mu2 = F.conv2d(Y, window, padding=window_size // 2, groups=channel)

#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1 * mu2

#         sigma1_sq = F.conv2d(X * X, window, padding=window_size // 2, groups=channel) - mu1_sq
#         sigma2_sq = F.conv2d(Y * Y, window, padding=window_size // 2, groups=channel) - mu2_sq
#         sigma12 = F.conv2d(X * Y, window, padding=window_size // 2, groups=channel) - mu1_mu2

#         ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

#         if size_average:
#             return ssim_map.mean()
#         else:
#             return ssim_map.mean([1, 2, 3])

#     def ms_ssim(self, X, Y):
#         # Create a list to hold SSIM values at each scale
#         msssim_vals = []
#         mcs_vals = []

#         # Apply SSIM at each scale, progressively downsampling
#         for weight in self.weights[:-1]:
#             ssim_val = self.ssim(X, Y, self.window, self.window_size, self.channel, self.size_average)
#             msssim_vals.append(ssim_val ** weight)

#             # Downsample images by 2
#             X = F.avg_pool2d(X, kernel_size=2)
#             Y = F.avg_pool2d(Y, kernel_size=2)

#         # Compute SSIM at the final scale (no downsampling here)
#         ssim_val = self.ssim(X, Y, self.window, self.window_size, self.channel, self.size_average)
#         msssim_vals.append(ssim_val ** self.weights[-1])

#         # Return the product of SSIM values across all scales
#         return torch.prod(torch.stack(msssim_vals))

#     def forward(self, X, Y):
#         window = self.window.type_as(X)
#         return 1 - self.ms_ssim(X, Y)

# # Example usage
# if __name__ == "__main__":
#     # Dummy images X and Y (batch of size 1, 3 channels, 256x256)
#     X = torch.rand(1, 3, 256, 256)
#     Y = torch.rand(1, 3, 256, 256)

#     # Initialize custom MS-SSIM loss
#     msssim_loss_fn = MSSSIMLoss()

#     # Calculate loss
#     loss = msssim_loss_fn(X, Y)

#     print(f'MS-SSIM Loss: {loss.item()}')



import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MSSSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=3, size_average=True, weights=None):
        super(MSSSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        # Default weights for 5 scales as used in the original MS-SSIM paper
        if weights is None:
            self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        else:
            self.weights = weights

    def gaussian_window(self, window_size, sigma):
        gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, X, Y, window, window_size, size_average=True):
        channel = X.shape[1]  # Dynamically get the number of channels from the input
        window = window.expand(channel, 1, window_size, window_size).contiguous()

        # Use dynamic input channels for group convolution
        mu1 = F.conv2d(X, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(Y, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(X * X, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(Y * Y, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(X * Y, window, padding=window_size // 2, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean([1, 2, 3])

    def ms_ssim(self, X, Y):
        # Create a list to hold SSIM values at each scale
        msssim_vals = []

        # Apply SSIM at each scale, progressively downsampling
        for weight in self.weights[:-1]:
            ssim_val = self.ssim(X, Y, self.window, self.window_size, self.size_average)
            msssim_vals.append(ssim_val ** weight)

            # Downsample images by 2
            X = F.avg_pool2d(X, kernel_size=2)
            Y = F.avg_pool2d(Y, kernel_size=2)

        # Compute SSIM at the final scale (no downsampling here)
        ssim_val = self.ssim(X, Y, self.window, self.window_size, self.size_average)
        msssim_vals.append(ssim_val ** self.weights[-1])

        # Return the product of SSIM values across all scales
        return torch.prod(torch.stack(msssim_vals))

    def forward(self, X, Y):
        # Ensure the window has the same type and number of channels as the input
        window = self.create_window(self.window_size, X.shape[1]).type_as(X)
        return 1 - self.ms_ssim(X, Y)
