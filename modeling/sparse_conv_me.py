# Created by Lang Huang (laynehuang@outlook.com)
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import torch
import torch.nn as nn
import MinkowskiEngine as ME


def to_sparse_tensor(x: torch.Tensor, indexes: torch.Tensor):
    B, L, C = x.shape
    indexes = indexes.repeat(B, 1, 1).reshape(B*L, 2)
    batch_idx = torch.arange(B, device=x.device).unsqueeze(-1)
    batch_idx = batch_idx.repeat(1, L).reshape(B*L, 1)    # (B, L, 1)
    sparse_idx = torch.cat((batch_idx, indexes), dim=1).int() # (B*L, 3); (b_idx, h_idx, w_idx)
    x_s = ME.SparseTensor(feats=x.reshape(-1, C), coordinates=sparse_idx)
    return x_s

class SparseConv2d(ME.MinkowskiConvolution):
    '''SparseConv module that operates on torch.Tensor
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, dimension=2)
    
    def forward(self, x: torch.Tensor, indexes: torch.Tensor, H: int = None, W: int = None):
        B, L, C = x.shape
        x_s = to_sparse_tensor(x, indexes)
        y_s = super().forward(x_s)
        # the number of elements might be different
        return y_s.features.reshape(B, -1, y_s.features.shape[-1])

class SparseConv2d(ME.MinkowskiChannelwiseConvolution):
    '''SparseConv module that operates on torch.Tensor
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, dimension=2)
    
    def forward(self, x: torch.Tensor, indexes: torch.Tensor, H: int = None, W: int = None):
        B, L, C = x.shape
        x_s = to_sparse_tensor(x, indexes)
        y_s = super().forward(x_s)
        # the number of elements might be different
        return y_s.features.reshape(B, -1, y_s.features.shape[-1])

class SparseAvgPool2d(ME.MinkowskiAvgPooling):
    '''SparseConv module that operates on torch.Tensor
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, dimension=2)
    
    def forward(self, x: torch.Tensor, indexes: torch.Tensor, H: int = None, W: int = None):
        B, L, C = x.shape
        x_s = to_sparse_tensor(x, indexes)
        y_s = super().forward(x_s)
        # the number of elements might be different
        return y_s.features.reshape(B, -1, y_s.features.shape[-1])