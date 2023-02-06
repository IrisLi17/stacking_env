import torch
import mvp
import torchvision.transforms
import numpy as np
from typing import Union


class ImageProcessor:
    def __init__(self, device) -> None:
        self.device = device
        self.mvp_model = mvp.load("vitb-mae-egosoup")
        self.mvp_model.to(self.device)
        self.mvp_model.freeze()
        # Norm should still be trainable, so I will not forward norm in this wrapper, and add a ln to policy
        self.image_transform = torchvision.transforms.Resize(224)
        self.im_mean = torch.Tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).to(self.device)
        self.im_std = torch.Tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)).to(self.device)
    
    def _normalize_obs(self, obs: Union[np.ndarray, torch.Tensor]):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        img = self.image_transform(obs)
        normed_img = (img / 255.0 - self.im_mean) / self.im_std
        return normed_img

    def _extract_patch_feat(self, x):
        B = x.shape[0]
        x = self.mvp_model.patch_embed(x)

        cls_tokens = self.mvp_model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.mvp_model.pos_embed

        for blk in self.mvp_model.blocks:
            x = blk(x)
        return x[:, 1:]  # (B, n_patch, feat_dim)
    
    def mvp_process_image(self, img: Union[np.ndarray, torch.Tensor], use_patch_feat=False, use_raw_img=False):
        if use_raw_img:
            scene_feat = torch.from_numpy(img).float().to(self.device).reshape(img.shape[0], -1)
        else:
            normed_img = self._normalize_obs(img)
            with torch.no_grad():
                if not use_patch_feat:
                    scene_feat = self.mvp_model.extract_feat(normed_img.float())
                else:
                    scene_feat = self._extract_patch_feat(normed_img.float())
                # scene_feat = self.mvp_model.forward_norm(scene_feat)
        return scene_feat
