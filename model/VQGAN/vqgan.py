import pdb

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR,StepLR
import numpy as np
from packaging import version
from model.BrownianBridge.base.modules.ema import LitEma
from contextlib import contextmanager
import omegaconf

from model.VQGAN.model import Encoder, Decoder
from model.BrownianBridge.base.modules.diffusionmodules.model import *
from model.VQGAN.quantize import VectorQuantizer2 as VectorQuantizer
from model.VQGAN.quantize import GumbelQuantize
from model.BrownianBridge.base.util import instantiate_from_config
import argparse
#from utils import dict2namespace, namespace2dict
import importlib


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
            print("convertng to namespaces")
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    # pdb.set_trace()
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if config.__contains__('params'):
        return get_obj_from_str(config["target"])(**vars(config['params']))
    else:
        return get_obj_from_str(config["target"])()


class VQFlowNet(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()


        try:
            ddconfig = dict2namespace(ddconfig)
            lossconfig = dict2namespace(lossconfig)
        except:
            pass
        self.embed_dim = embed_dim # 3
        self.n_embed = n_embed # 8192 * 2
        self.image_key = image_key # 'image'
        self.encoder = FlowEncoder(**vars(ddconfig))
        self.decoder = FlowDecoderWithResidual(**vars(ddconfig))
        self.loss = instantiate_from_config(vars(lossconfig))
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig.z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig.z_channels, 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor
        self.h0 = None
        self.w0 = None
        self.h_padded = None
        self.w_padded = None
        self.pad_h = 0
        self.pad_w = 0

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x, ret_feature=False):
        '''
        Set ret_feature = True when encoding conditions in ddpm
        '''
        # Pad the input first so its size is deividable by 8.
        # this is to tolerate different f values, various size inputs, 
        # and some operations in the DDPM unet model.
        self.h0, self.w0 = x.shape[2:]
        # 8: window size for max vit
        # 2**(nr-1): f 
        # 4: factor of downsampling in DDPM unet
        min_side = 8 *2**(self.encoder.num_resolutions-1) * 4
        if self.h0 % min_side != 0:
            pad_h = min_side - (self.h0 % min_side)
            if pad_h == self.h0: # this is to avoid padding 256 patches
                pad_h = 0
            x = F.pad(x, (0, 0, 0, pad_h), mode='reflect')
            self.h_padded = True
            self.pad_h = pad_h

        if self.w0 % min_side != 0:
            pad_w = min_side - (self.w0 % min_side)
            if pad_w == self.w0:
                pad_w = 0
            x = F.pad(x, (0, pad_w, 0, 0), mode='reflect')
            self.w_padded = True
            self.pad_w = pad_w
        h, phi_list = self.encoder(x, ret_feature)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        

        return quant, emb_loss, info, phi_list


    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant, x_prev, x_next, phi_list=None):
        
        cond_dict = dict(
            phi_list = phi_list,
            frame_prev = F.pad(x_prev, (0, self.pad_w, 0, self.pad_h), mode='reflect'),
            frame_next = F.pad(x_next, (0, self.pad_w, 0, self.pad_h), mode='reflect')
        )
        
        """
        cond_dict = dict(
            phi_prev_list = self.encode(x_prev, ret_feature=True)[-1],
            phi_next_list = self.encode(x_next, ret_feature=True)[-1],
            frame_prev = F.pad(x_prev, (0, self.pad_w, 0, self.pad_h), mode='reflect'),
            frame_next = F.pad(x_next, (0, self.pad_w, 0, self.pad_h), mode='reflect')
        )
        """
        quant = self.post_quant_conv(quant)

        dec = self.decoder(quant, cond_dict)
        # check if image is padded and return the original part only
        if self.h_padded:
            dec = dec[:, :, 0:self.h0, :]
        if self.w_padded:
            dec = dec[:, :, :, 0:self.w0]


        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x, x_prev,x_next, return_pred_indices=False):
        
        inputs = torch.cat([x_prev,x,x_next],0) ## B3 C H W
        quant, diff, (_,_,ind), phi_list = self.encode(inputs)

        #quant= self.encode(input)

        dec = self.decode(quant, x_prev, x_next,phi_list)
        #dec = self.decode(quant, x_prev, x_next)
        if return_pred_indices:
            return dec, diff, ind

        return dec,diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    
    def training_step(self, batch, batch_idx, optimizer_idx):
        self.train()
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        x_prev = self.get_input(batch, 'prev_frame')
        x_next = self.get_input(batch, 'next_frame')



        xrec, qloss = self(x,x_prev,x_next)

        if optimizer_idx == 0:
            # autoencoder
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            flow_list = None,flow_gt=None,x_prev = x_prev,x_next = x_next)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",x_prev = x_prev,x_next = x_next)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
    



    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        self.eval()
        x = self.get_input(batch, self.image_key)
        x_prev = self.get_input(batch, 'prev_frame')
        x_next = self.get_input(batch, 'next_frame')
        xrec, qloss = self(x, x_prev, x_next)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))
        
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")

            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]

            return [opt_ae,opt_disc ], scheduler
        
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x_prev = self.get_input(batch, 'prev_frame')
        x_next = self.get_input(batch, 'next_frame')
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x, x_prev, x_next)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x, x_prev, x_next)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

    def get_flow(self, img0, img1,feats):

        return self.decoder.get_flow(img0,img1,feats)
    
class VQFlowNetInterface(VQFlowNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def encode(self, x, ret_feature=False):
        '''
        Set ret_feature = True when encoding conditions in ddpm
        '''
        # Pad the input first so its size is deividable by 8.
        # this is to tolerate different f values, various size inputs, 
        # and some operations in the DDPM unet model.
        self.h0, self.w0 = x.shape[2:]
        # 8: window size for max vit
        # 2**(nr-1): f 
        # 4: factor of downsampling in DDPM unet
        min_side = 8 * 2**(self.encoder.num_resolutions-1) * 4
        #min_side = 256
        if self.h0 % min_side != 0:
            pad_h = min_side - (self.h0 % min_side)
            if pad_h == self.h0: # this is to avoid padding 256 patches
                pad_h = 0
            x = F.pad(x, (0, 0, 0, pad_h), mode='reflect')
            self.h_padded = True
            self.pad_h = pad_h
        else:

            self.h_padded = False
            self.pad_h = 0
        if self.w0 % min_side != 0:
            pad_w = min_side - (self.w0 % min_side)
            if pad_w == self.w0:
                pad_w = 0
            x = F.pad(x, (0, pad_w, 0, 0), mode='reflect')
            self.w_padded = True
            self.pad_w = pad_w
        else:
            self.w_padded = False
            self.pad_w = 0          

        h, phi_list = self.encoder(x, ret_feature)
        h = self.quant_conv(h) ## before quantization
        

        return h, phi_list

    def decode(self, h, x_prev, x_next, phi_list, force_not_quantize=False,scale = 0.5):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        
        cond_dict = dict(
            phi_list = phi_list,
            frame_prev = F.pad(x_prev, (0, self.pad_w, 0, self.pad_h), mode='reflect'),
            frame_next = F.pad(x_next, (0, self.pad_w, 0, self.pad_h), mode='reflect')
        )
        quant = self.post_quant_conv(quant)

        tmp_list = []
        b,c,h,w = x_prev.shape

        with torch.no_grad():
            if scale < 1:
            
                b,c,h,w = F.interpolate(F.pad(x_prev, (0, self.pad_w, 0, self.pad_h), mode='reflect'), scale_factor=scale, mode="bilinear", align_corners=False).shape
                
                img0_down_ = F.interpolate(x_prev, scale_factor=scale, mode="bilinear", align_corners=False)
                img1_down_ = F.interpolate(x_next, scale_factor=scale, mode="bilinear", align_corners=False)
                _,_,h_,w_ = img0_down_.shape
                img0_down = torch.zeros(b,c,h,w).to(img0_down_.device)
                img1_down = torch.zeros(b,c,h,w).to(img1_down_.device)
                img0_down[:,:,:h_,:w_] = img0_down_
                img1_down[:,:,:h_,:w_] = img1_down_



                _,tmp_list = self.encoder(torch.cat([img0_down,torch.zeros_like(img0_down),img1_down]))
                flow_down = self.get_flow(img0_down, img1_down,tmp_list[:-2])
                flow = F.interpolate(flow_down, scale_factor=1/scale, mode="bilinear", align_corners=False) * 1/scale
            else:
                flow = None
            dec = self.decoder(quant, cond_dict,flow)
            # check if image is padded and return the original part only
            if self.h_padded:
                dec = dec[:, :, 0:self.h0, :]
            if self.w_padded:
                dec = dec[:, :, :, 0:self.w0]
            return dec

    
