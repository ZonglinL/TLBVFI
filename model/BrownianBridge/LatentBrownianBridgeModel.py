import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from einops import rearrange,repeat

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQFlowNetInterface


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQFlowNetInterface(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan ## VQGAN quantization
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams)) ## interpolation
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, y, z,  context=None):
        with torch.no_grad():
            gt,_ = self.encode(torch.cat([y,x,z],dim = 0).detach())
            gt = torch.stack(torch.chunk(gt,3),2)
            latent,_ = self.encode(torch.cat([y,torch.zeros_like(y),z],dim = 0))
            latent = torch.stack(torch.chunk(latent,3),2)
        
        return super().forward(gt, latent, context)

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            if self.condition_key == 'first_stage':
                context = self.encode(x_cond,cond = True)[0].detach()
            else:
                context = self.cond_stage_model(x_cond) 
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        model = self.vqgan
        if cond:
            x_latent,phi_list = model.encode(x,ret_feature=cond)
            return x_latent,phi_list
        else:
            x_latent = model.encode(x,ret_feature=cond) 
            return x_latent

    @torch.no_grad()
    def decode(self, x_latent, prev_img,next_img,phi_list,scale = 0.5):
        model = self.vqgan ## latent B C F H W
        x_latent = x_latent.permute(0,2,1,3,4) ## B C F H W --> B F C H W
        x_latent = rearrange(x_latent,'b f c h w -> (b f) c h w') ## BF C H W
        out = model.decode(x_latent,prev_img,next_img,phi_list,scale = scale)
        return out

    @torch.no_grad()
    def latent_p_sample_loop(self, latent, y, context, clip_denoised=True, sample_mid_step=False):
        ## latent: the latent video
        ## y:
        ## context y: end point y, spatial resclaer, first stage or none, same as z

        imgs,one_step_imgs = [y],[]
        for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):

            img, x0_recon = self.p_sample(x_t = imgs[-1],y=y,context = context,i = i)
            
            imgs.append(img)
            one_step_imgs.append(x0_recon) 

        return imgs,one_step_imgs



    @torch.no_grad()
    def sample(self, y, z, clip_denoised=False, sample_mid_step=False,scale = 0.5):

        x = torch.zeros_like(y)
        latent,phi_list = self.encode(torch.cat([y,x,z],0))

        """
        call the latent sample function
        """
        latent = torch.stack(torch.chunk(latent,3),2)
        context = latent

        imgs,one_step_imgs = self.latent_p_sample_loop(latent = latent,
                                            y = latent,
                                            context = latent,
                                            clip_denoised=clip_denoised,
                                            sample_mid_step=sample_mid_step)


        with torch.no_grad():
            out = self.decode(imgs[-1].detach(), y,z,phi_list,scale = scale)
        return out

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    # @torch.no_grad()
    # def reverse_sample(self, x, skip=False):
    #     x_ori_latent = self.vqgan.encoder(x)
    #     temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
    #     x_latent = temp[-1]
    #     x_latent = self.vqgan.quant_conv(x_latent)
    #     x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
    #     out = self.vqgan.decode(x_latent_quant)
    #     return out
    
    def get_flow(self, img0, img1,feats):
        return self.vqgan.get_flow(self,img0,img1,feats)