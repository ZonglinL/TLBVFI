import torch
import torch.nn as nn
import torch.nn.functional as F

from model.VQGAN.lpips import LPIPS
#from lpips import LPIPS
from model.VQGAN.discriminator import NLayerDiscriminator
from runners.utils import weights_init
from torchvision.models.vgg import vgg19, VGG19_Weights

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class flow_loss(nn.Module):
    def __init__(self):
        super(flow_loss, self).__init__()

    def forward(self, flow, gt, mask = None):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return loss_map

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).float().reshape(1, 3, 1, 1)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).float().reshape(1, 3, 1, 1)
        self.alpha_l = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5] 
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.vgg_feats = nn.ModuleList([vgg.features[:4], vgg.features[4:9], vgg.features[9:14], vgg.features[14:23], vgg.features[23:32]])
        for p in self.vgg_feats.parameters():
            p.requires_grad = False
        self.vgg_feats.eval()
        self.vgg_weight = 10 * self.perceptual_weight
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    def get_vgg_features(self, x):
        # assuming the input is in the range of [-1, 1]
        x = (x + 1) / 2
        x = (x - self.vgg_mean.to(x.device)) / self.vgg_std.to(x.device)
        feat1_2 = self.vgg_feats[0](x)
        feat2_2 = self.vgg_feats[1](feat1_2)
        feat3_2 = self.vgg_feats[2](feat2_2)
        feat4_2 = self.vgg_feats[3](feat3_2)
        feat5_2 = self.vgg_feats[4](feat4_2)
        feats = [feat1_2, feat2_2, feat3_2, feat4_2, feat5_2]
        return feats
    def get_gram(self, x):
        if not isinstance(x, list):
            x = [x]
        grams = []
        for feat_lvl in x:
            grams.append(torch.einsum('b c h w, b d h w -> b c d', feat_lvl / 255., feat_lvl / 255.))
        return grams
    
    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",flow_list=None,flow_gt=None,x_prev = None,x_next = None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        #rec_loss = 0
        if x_prev is None:
            x_prev = torch.zeros_like(inputs)
            x_next = torch.zeros_like(inputs)

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        else:
            p_loss = torch.tensor([0.0])

        if self.vgg_weight > 0:
            
            x_feats = self.get_vgg_features(reconstructions)
            target_feats = self.get_vgg_features(inputs)
            x_grams = self.get_gram(x_feats)
            target_grams = self.get_gram(target_feats)
            v_loss = 0
            for i in range(len(x_grams)):
                x_gram_lvl = x_grams[i]
                target_gram_lvl = target_grams[i]
                v_loss = v_loss + ((x_gram_lvl - target_gram_lvl) ** 2).mean() * self.alpha_l[i]
        else:
            v_loss = torch.tensor([0.0]).to(inputs.device)  
        
        v_loss = v_loss * self.vgg_weight

        nll_loss = rec_loss + v_loss ## L1 LPIPS VGG
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update

            reconstructions = torch.stack([x_prev,reconstructions,x_next],dim = 2) ## B C H W --> B C 3 H W

            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)
            
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                
            if not self.training:
                d_weight = torch.tensor(0.0)
            
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()
            #loss = nll_loss + self.codebook_weight * codebook_loss.mean()
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            reconstructions = torch.stack([x_prev,reconstructions,x_next],dim = 2) ## B C H W --> B C 3 H W
            inputs = torch.stack([x_prev,inputs,x_next],dim = 2) ## B C H W --> B C 3 H W
            
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            
            
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                }
            return d_loss, log
