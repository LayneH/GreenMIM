# Created by Lang Huang (laynehuang@outlook.com)
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import torch
import torch.nn as nn
import spconv.pytorch as spconv


def to_sparse_tensor(x: torch.Tensor, indexes: torch.Tensor, H: int, W: int):
    B, L, C = x.shape
    indexes = indexes.repeat(B, 1, 1).reshape(B*L, 2)
    batch_idx = torch.arange(B, device=x.device).unsqueeze(-1)
    batch_idx = batch_idx.repeat(1, L).reshape(B*L, 1)    # (B, L, 1)
    sparse_idx = torch.cat((batch_idx, indexes), dim=1).int() # (B*L, 3); (b_idx, h_idx, w_idx)
    x_s = spconv.SparseConvTensor(x.reshape(-1, C), sparse_idx, [H, W], B)
    return x_s

class SparseConv2d(spconv.SparseConv2d):
    '''SparseConv module that operates on torch.Tensor
    '''
    def forward(self, x: torch.Tensor, indexes: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        x_s = to_sparse_tensor(x, indexes, H, W)
        y_s = super().forward(x_s)
        # the number of elements might be different
        return y_s.features.reshape(B, -1, y_s.features.shape[-1])


class SubMConv2d(spconv.SubMConv2d):
    '''SubManifold Conv module that operates on torch.Tensor
    '''
    def forward(self, x: torch.Tensor, indexes: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        x_s = to_sparse_tensor(x, indexes, H, W)
        y_s = super().forward(x_s)
        return y_s.features.reshape(B, -1, y_s.features.shape[-1])

class SparseDWConv2d(nn.Conv2d):
    '''We need to first convert the sparse tensor to dense tensor and then
       perform DWConv because spconv does not support `groups` option.
    '''
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1,
                 padding: int = 0, bias: bool = True, groups: int = None, **kwargs):
        groups = groups or in_channels
        assert groups == in_channels
        assert stride == 1  # FIXME
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         bias=bias, groups=groups, **kwargs)
    
    def forward(self, x: torch.Tensor, indexes: torch.Tensor, mask: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        # sparse to dense
        x_s = to_sparse_tensor(x, indexes, H, W)
        x_d = x_s.dense()   # (B, C, H, W)
        y_d = super().forward(x_d).reshape(B, -1, H*W).transpose(2, 1)
        # dense to sparse
        return y_d[mask.expand(B, -1)].reshape(B, L, C)
