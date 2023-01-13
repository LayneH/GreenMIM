# Created by Lang Huang (laynehuang@outlook.com)
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
from functools import partial
import torch.nn as nn

from .base_green_model import MaskedAutoencoder
from .green_swin_models import SwinTransformer
from .green_twins_models import ALTGVT as TwinsTransformer


def get_pretrain_model(model_str, norm_pix_loss=True):
    # NOTE: we might first check the model_str
    if model_str in globals().keys():
        model = globals()[model_str](norm_pix_loss=norm_pix_loss)
    else:
        raise KeyError(f"Model `{model_str}` is not suuported.")
    return model


# Swin Transformers
def green_mim_swin_base_patch4_dec512b1(**kwargs):
    encoder = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = MaskedAutoencoder(   
        encoder,
        embed_dim=1024,
        patch_size=32,
        in_chans=3,
        # common configs
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # decoder settings
        decoder_num_patches=49,
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=16,
        **kwargs)
    return model


def green_mim_swin_base_patch4_win14_dec512b1(**kwargs):
    encoder = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = MaskedAutoencoder(   
        encoder,
        embed_dim=1024,
        patch_size=32,
        in_chans=3,
        # common configs
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # decoder settings
        decoder_num_patches=49,
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=16,
        **kwargs)
    return model


def green_mim_swin_large_patch4_win14_dec512b1(**kwargs):
    encoder = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = MaskedAutoencoder(   
        encoder,
        embed_dim=1536,
        patch_size=32,
        in_chans=3,
        # common configs
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # decoder settings
        decoder_num_patches=49,
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=16,
        **kwargs)
    return model

# Twins Transformers
def green_mim_twins_large_patch4_dec512b1(**kwargs):
    encoder = TwinsTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dims=[128, 256, 512, 1024], 
        num_heads=[4, 8, 16, 32], 
        mlp_ratios=[4, 4, 4, 4], 
        depths=[2, 2, 18, 2], 
        wss=[7, 7, 7, 7], 
        sr_ratios=[8, 4, 2, 1],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = MaskedAutoencoder(   
        encoder,
        embed_dim=1024,
        patch_size=32,
        in_chans=3,
        # common configs
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # decoder settings
        decoder_num_patches=49,
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=16,
        **kwargs)
    return model