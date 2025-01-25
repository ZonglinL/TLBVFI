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

    parser.add_argument('--resume_model', type=str, default=None, help='model checkpoint')

    parser.add_argument('--frame0', type=str, default=None, help='previous frame')
    parser.add_argument('--frame1', type=str, default=None, help='next frame')
    
    parser.add_argument('--resume_optim', type=str, default=None, help='optimizer checkpoint')

    parser.add_argument('--max_epoch', type=int, default=None, help='optimizer checkpoint')
    parser.add_argument('--max_steps', type=int, default=None, help='optimizer checkpoint')

    args = parser.parse_args()

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

def interpolate(frame0,frame1,model):
    with torch.no_grad():

        out = model.sample(frame0,frame1)
        if isinstance(out, tuple): # using ddim
            out = out[0]
        out =  torch.clamp(out, min=-1., max=1.) # interpolated frame in [-1,1]
    return out


def unnorm(lst):
    out = []
    for a in lst:
        out.append(a/2 + 0.5)
    return out

def main():
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args
    model = LatentBrownianBridgeModel(nconfig.model)
    model.eval()

    interpolate(torch.zeros(1,3,1024,1536),torch.zeros(1,3,1024,1536),model)
    

if __name__ == "__main__":
    main()
 
