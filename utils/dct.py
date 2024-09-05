import torch
import torch.fft
from torchjpeg import dct
from torch.nn import functional as F
import numpy as np
import matplotlib.pylab as plt


def dct_transform(x, chs_remove=None, chs_pad=False, del_num=0,
                  size=8, stride=8, pad=0, dilation=1, ratio=8):
    """
        Transform a spatial image into its frequency channels.
        Prune low-frequency channels if necessary.
    """

    # assert x is a (B, 3, H, W) RGB image
    assert x.shape[1] == 3

    # up-sample
    x = F.interpolate(x, scale_factor=ratio, mode='bilinear', align_corners=True)

    # convert to the YCbCr color domain, required by DCT
    x = x * 255
    x = dct.to_ycbcr(x)
    x = x - 128

    # perform block discrete cosine transform (BDCT)
    b, c, h, w = x.shape
    n_block = h // stride
    x = x.view(b * c, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.transpose(1, 2)
    x = x.view(b, c, -1, size, size)
    x_freq = dct.block_dct(x)
    x_freq = x_freq.view(b, c, n_block, n_block, size * size).permute(0, 1, 4, 2, 3)

    # prune channels
    if chs_remove is not None:
        channels = list(set([i for i in range(64)]) - set(chs_remove)) 
        if not chs_pad:
            if del_num > 0:
                channels = np.random.permutation(channels)[del_num:]
            # simply remove channels
            x_freq = x_freq[:, :, channels, :, :]
        else:
            # pad removed channels with zero, helpful for visualization
            if del_num > 0:
                chs_remove.extend( np.random.permutation(channels)[:del_num])
            x_freq[:, :, chs_remove] = 0

    # stack frequency channels from each color domain
    x_freq = x_freq.reshape(b, -1, n_block, n_block)

    return x_freq


def idct_transform(x, size=8, stride=8, pad=0, dilation=1, ratio=8):
    """
        The inverse of DCT transform.
        Transform frequency channels (must be 192 channels, can be padded with 0) back to the spatial image.
    """

    b, _, h, w = x.shape

    x = x.view(b, 3, 64, h, w)
    x = x.permute(0, 1, 3, 4, 2)
    x = x.view(b, 3, h * w, 8, 8)
    x = dct.block_idct(x)
    x = x.view(b * 3, h * w, 64)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(32 * ratio, 32 * ratio),
               kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.view(b, 3, 32 * ratio, 32 * ratio)
    x = x + 128
    x = dct.to_rgb(x)
    x = x / 255
    x = F.interpolate(x, scale_factor=1 / ratio, mode='bilinear', align_corners=True)
    x = x.clamp(min=0.0, max=1.0)
    return x


def create_ft_set(x, del_num = 0, random=True):
    """
        prune low-frequency channels from image and randomly delete a few channels, 
        turn back to spatial image afer randomizing the rest channels
    """
    # low-frequency channels to prune
    chs_prune = [0, 1, 2, 3, 8, 9, 10, 16, 17, 24]
    batches = x.shape[0] // 128
    x_list = []
    for batch in range(batches):
        x_freq = dct_transform(x[128 * batch: 128 * (batch + 1)], chs_remove=chs_prune, chs_pad=True, del_num=del_num)
        b, c, h, w = x_freq.shape
        if random:
            perm = np.random.permutation(c)
            x_freq = x_freq[:,perm]
        # turn back to spatial domain
        x_list.append(idct_transform(x_freq))
    
    if x.shape[0] % 128 != 0:
        x_freq = dct_transform(x[128 * batches:], chs_remove=chs_prune, chs_pad=True, del_num=del_num)
        b, c, h, w = x_freq.shape
        if random:
            perm = np.random.permutation(c)
            x_freq = x_freq[:,perm]
        x_list.append(idct_transform(x_freq))
    x = torch.cat(x_list, dim=0)
    return x


def visualize(dataname, x, px):
    import torchvision.transforms as t
    import os
    num = x.shape[0]
    perm = np.random.permutation(num)
    x = x[perm,]
    px = px[perm,]
    org, aft = x[:10], px[:10]
    org = F.interpolate(org, scale_factor=3, mode='bilinear', align_corners=True)
    aft = F.interpolate(aft, scale_factor=3, mode='bilinear', align_corners=True)
    fig, axes = plt.subplots(2, 10, tight_layout=True)
    # fig.subplots_adjust(hspace=0, wspace=0)
    
    for c in range(10):
        ax = axes[0][c]
        ax.imshow(t.ToPILImage()(org[c]))
        ax.axis('off')
    
    for c in range(10):
        ax = axes[1][c]
        ax.axis('off')
        ax.imshow(t.ToPILImage()(aft[c]))
    os.makedirs('./data', exist_ok=True)
    plt.savefig(f'./data/{dataname}.pdf')



if __name__ == '__main__':
    x = torch.rand(15, 3, 32, 32)
    y = torch.randint(10, (16,))

    x_freq = dct_transform(x)
    x_spat = idct_transform(x_freq)
    print(x_freq.shape, x_spat.shape)  # torch.Size([16, 192, 32, 32]) torch.Size([16, 3, 32, 32])

    x_aft = create_ft_set(x, 0, True)

    visualize('noisy', x, x_aft)