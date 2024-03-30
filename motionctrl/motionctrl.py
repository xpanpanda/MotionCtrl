import torch.nn as nn
from einops import rearrange

from lvdm.models.ddpm3d import LatentDiffusion
from motionctrl.lvdm_modified_modules import (
    TemporalTransformer_forward, selfattn_forward_unet,
    spatial_forward_BasicTransformerBlock,
    temporal_selfattn_forward_BasicTransformerBlock)
from utils.utils import instantiate_from_config


class MotionCtrl(LatentDiffusion):
    def __init__(self,
                 omcm_config=None,
                 pose_dim=12,
                 context_dim=1024,
                 *args, 
                 **kwargs):
        super(MotionCtrl, self).__init__(*args, **kwargs)

        # object motion control module
        if omcm_config is not None:
            self.omcm = instantiate_from_config(omcm_config)
        else:
            self.omcm = None


        # camera motion control module
        bound_method = selfattn_forward_unet.__get__(
            self.model.diffusion_model,
            self.model.diffusion_model.__class__)
        # 使用 selfattn_forward_unet.__get__ 方法
        # 绑定 selfattn_forward_unet 函数到 diffusion_model 的 forward 方法。
        setattr(self.model.diffusion_model, 'forward', bound_method)

        for _name, _module in self.model.diffusion_model.named_modules():
            # 如果模块是 TemporalTransformer 类型，
            # 则绑定 TemporalTransformer_forward 方法到模块的 forward 方法。
            if _module.__class__.__name__ == 'TemporalTransformer':
                bound_method = TemporalTransformer_forward.__get__(
                    _module, _module.__class__)
                setattr(_module, 'forward', bound_method)
                
            # 如果模块是 BasicTransformerBlock 类型，
            # 则根据模块是否包含空间注意力（spatial_selfattn）和时间注意力（temporal_selfattn）来决定绑定哪种前向传播方法：
            if _module.__class__.__name__ == 'BasicTransformerBlock':
                # SpatialTransformer only
                if _module.attn2.to_k.in_features != context_dim: # TemporalTransformer without crossattn 
                    
                    bound_method = temporal_selfattn_forward_BasicTransformerBlock.__get__(
                        _module, _module.__class__)
                    setattr(_module, '_forward', bound_method)

                    cc_projection = nn.Linear(_module.attn2.to_k.in_features + pose_dim, _module.attn2.to_k.in_features)
                    nn.init.eye_(list(cc_projection.parameters())[0][:_module.attn2.to_k.in_features, :_module.attn2.to_k.in_features])
                    nn.init.zeros_(list(cc_projection.parameters())[1])
                    cc_projection.requires_grad_(True)

                    _module.add_module('cc_projection', cc_projection)
                    
                else:
                    bound_method = spatial_forward_BasicTransformerBlock.__get__(
                        _module, _module.__class__)
                    setattr(_module, '_forward', bound_method)

    def get_traj_features(self, extra_cond):
        b, c, t, h, w = extra_cond.shape
        ## process in 2D manner
        extra_cond = rearrange(extra_cond, 'b c t h w -> (b t) c h w')
        traj_features = self.omcm(extra_cond)
        traj_features = [rearrange(feature, '(b t) c h w -> b c t h w', b=b, t=t) for feature in traj_features]
        return traj_features
