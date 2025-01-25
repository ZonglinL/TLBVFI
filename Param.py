import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from utils import dict2namespace, namespace2dict
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('-c', '--config', type=str, default='configs/Template-LBBDM-video.yaml', help='Path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results', help="The directory to save results")

    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--sample_to_eval', action='store_true', default=False, help='sample for evaluation')
    parser.add_argument('--sample_at_start', action='store_true', default=False, help='sample at start(for debug)')
    parser.add_argument('--save_top', action='store_true', default=False, help="save top loss checkpoint")

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')
    parser.add_argument('--port', type=str, default='12355', help='DDP master port')

    parser.add_argument('--resume_model', type=str, default="/home/zonglin/BBVFI/results/VQGAN/vimeo_unet.pth", help='model checkpoint')

    parser.add_argument('--frame0', type=str, default="/home/zonglin/benchmark/PerVFI/results/FILM_extreme/prev/100.png", help='previous frame')
    parser.add_argument('--frame1', type=str, default="/home/zonglin/benchmark/PerVFI/results/FILM_extreme/next/100.png", help='next frame')
    parser.add_argument('--frame', type=str, default="/home/zonglin/benchmark/PerVFI/results/FILM_extreme/gt/100.png", help='next frame')
    
    parser.add_argument('--resume_optim', type=str, default=None, help='optimizer checkpoint')

    parser.add_argument('--max_epoch', type=int, default=None, help='optimizer checkpoint') 
    parser.add_argument('--max_steps', type=int, default=None, help='optimizer checkpoint')

    args = parser.parse_args()

    args.resume_model = None

    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch
    if args.max_steps is not None:
        namespace_config.training.n_steps = args.max_steps

    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config

def encode(frame0,frame1,model,gt = None):
    with torch.no_grad():
        if gt is None:
            gt = torch.zeros_like(frame0)
        inputs = torch.cat([frame0,gt,frame1],0) 
        latent,phi_list= model.encode(inputs,cond = True)
    return latent


def unnorm(lst):
    out = []
    for a in lst:
        out.append(a/2 + 0.5)
    return out

def main():
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args
    model = LatentBrownianBridgeModel(nconfig.model)
    #state_dict_pth = args.resume_model
    
    frame1_path = args.frame1
    frame_gt_path = args.frame
    #model_states = torch.load(state_dict_pth, map_location='cpu')
    #model.load_state_dict(model_states['model'])
    model.eval()
    model = model.cpu()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    model_parameters = model.vqgan.parameters()
    params_ = sum([np.prod(p.size()) for p in model_parameters])
    print((params_ + params)/1000000)

if __name__ == "__main__":
    main()
 
