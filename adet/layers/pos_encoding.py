import torch
import numpy as np
import torch.nn as nn

from adet.utils.misc import NestedTensor

class PositionalEncoding1D(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.channels = num_pos_feats
        dim_t = torch.arange(0, self.channels, 2).float()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale
        self.normalize = normalize
        inv_freq = 1. / (temperature ** (dim_t / self.channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 2d tensor of size (len, c)
        :return: Positional Encoding Matrix of size (len, c)
        """
        if tensor.ndim != 2:
            raise RuntimeError("The input tensor has to be 2D!")
        x, orig_ch = tensor.shape
        pos_x = torch.arange(
            1, x + 1, device=tensor.device).type(self.inv_freq.type())

        if self.normalize:
            eps = 1e-6
            pos_x = pos_x / (pos_x[-1:] + eps) * self.scale

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels),
                          device=tensor.device).type(tensor.type())
        emb[:, :self.channels] = emb_x

        return emb[:, :orig_ch]


class PositionalEncoding2D(nn.Module):
    """
    This is a more standard version of the positional embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    Arguments:
        num_pos_feats: int. The number of positional embeddings per spatial location.
            Given that the spatial location embeddings are concatenated, the num_pos_feats
            is usually, the number of hidden dimensions of the transformer divided by 2.
        temperature: int. Temperature, larger values make embeddings more uniform initially.
        normalize: bool. Whether to normalize the embeddings.
        scale: float. Scale of the embeddings.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale

    def forward(self, tensors: NestedTensor) -> torch.Tensor:
        """
        Args:
            tensors: NestedTensor. A NestedTensor that contains a tensor of size (B, C, H, W) 
                and a mask of size (B, H, W).
        
        Returns:
            pos: torch.Tensor. A tensor of size (B, num_pos_feats, H, W).
        """
        x = tensors.tensors
        mask = tensors.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        # The normalization causes that for different sizes of images, the positional
        # embeddings will be of the same scale. i.e.: items in the corners will have
        # similar positional embedding values.
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
