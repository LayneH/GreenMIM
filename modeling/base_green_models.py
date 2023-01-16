# Created by Lang Huang (laynehuang@outlook.com)
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed


class BaseGreenModel(nn.Module):
    
    def apply_mask(self, x, mask, patches_resolution):
        # mask out some patches according to the random mask
        B, N, C = x.shape
        H, W = patches_resolution
        mask = mask[:1].clone() # we use the same mask for the whole batch
        up_ratio = N // mask.shape[1]
        assert up_ratio * mask.shape[1] == N
        num_repeats = int(up_ratio**0.5)
        if up_ratio > 1:   # mask_size != patch_embed_size
            Mh, Mw = [sz // num_repeats for sz in patches_resolution]
            mask = mask.reshape(1, Mh, 1, Mw, 1)
            mask = mask.expand(-1, -1, num_repeats, -1, num_repeats)
            mask = mask.reshape(1, -1)
        
        # record the corresponding coordinates of visible patches
        coords_h = torch.arange(H, device=x.device)
        coords_w = torch.arange(W, device=x.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]), dim=-1)  # H W 2
        coords = coords.reshape(1, H*W, 2)

        # mask out patches
        vis_mask = ~mask    # ~mask means visible, (1, N_vis)
        x_vis = x[vis_mask.expand(B, -1)].reshape(B, -1, C)
        coords = coords[vis_mask].reshape(1, -1, 2) # (1 N_vis 2)

        return x_vis, coords, vis_mask

    def patchify(self, x):
        raise NotImplementedError()
    
    def forward_features(self, x, mask):
        raise NotImplementedError()
    
    def forward(self, x, mask):
        z_vis = self.forward_features(x, mask)
        return z_vis


class MaskedAutoencoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, encoder, embed_dim, patch_size, in_chans=3,
                 decoder_num_patches=196, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 block_cls=Block, mlp_ratio=4,
                 **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = encoder
        self.num_patches = decoder_num_patches
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.final_patch_size = patch_size
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, decoder_num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            block_cls(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # encoder to decoder
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        if hasattr(self.encoder, 'patch_embed'):
            w = self.encoder.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # 
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                w = m.weight.data
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, patch_size=None):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size or self.final_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = patch_size or self.final_patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        NOTE: Perform PER-BATCH random masking by per-sample shuffling.
        Per-batch shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L = 1, self.num_patches  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask.scatter_add_(1, ids_keep, torch.full([N, len_keep], fill_value=-1, dtype=mask.dtype, device=x.device))
        assert (mask.gather(1, ids_shuffle).gather(1, ids_restore) == mask).all()

        # repeat the mask
        ids_restore = ids_restore.repeat(x.shape[0], 1)
        mask = mask.repeat(x.shape[0], 1)

        return mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # generate random mask
        mask, ids_restore = self.random_masking(x, mask_ratio)

        # L -> L_vis
        latent = self.encoder(x, mask.bool())

        return latent, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x_ + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
