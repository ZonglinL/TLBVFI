import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import glob
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import dict2namespace, namespace2dict
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

    parser.add_argument('--resume_model', type=str, default="results/VQGAN/vimeo_unet.pth", help='model checkpoint')

    parser.add_argument('--data_dir', type=str, default="/home/zonglin/data/Xiph", help='previous frame')

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

def interpolate(frame0,frame1,model,gt = None):
    with torch.no_grad():
        if gt is None:
            gt = torch.zeros_like(frame0)
        inputs = torch.cat([frame0,gt,frame1],0) 
        print(inputs.shape)
        latent,phi_list= model.encode(inputs,cond = True)
        latent = torch.stack(torch.chunk(latent,3),2)
        
        out = model.decode(latent,frame0,frame1,phi_list) 
    return out


def unnorm(lst):
    out = []
    for a in lst:
        out.append(a/2 + 0.5)
    return out


class FILM(Dataset):
    ## interpolation dataset for SNU-FILM
    def __init__(self, image_size=(256, 256), flip=False, to_normal=True,mode = 'easy',root=None):
        self.image_size = image_size
        root = os.path.join(root,"SNU-FILM")
        self.root = root
        file = os.path.join(root,f'test-{mode}.txt')
        f = open(file, "r")
        im_list = f.read()
        self.image_dirs = im_list.split('\n')[:-1]

        self._length = len(self.image_dirs) ## folder of the images
        self.to_normal = to_normal # if normalize to [-1, 1] nor not
        self.flip = flip ## if flip or not
        self.image_dirs.sort()

    def __len__(self):

        return self._length

    def load_image(self,img_path,transform):


        root_out = os.path.split(self.root)[0] 
        root_out = os.path.split(root_out)[0]## directory that contains data/...
        img_path = os.path.join(root_out,img_path)

        try:
            image = Image.open(img_path)
        except:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)
        

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)


        return image

    def __getitem__(self, index):

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_path_first,img_path_target,img_path_second = self.image_dirs[index].split(' ') ## y,x,z

        x,y,z = self.load_image(img_path_target,transform),self.load_image(img_path_first,transform),self.load_image(img_path_second,transform)

        return x,y,z


def main():
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args
    model = LatentBrownianBridgeModel(nconfig.model)
    data_dir = args.data_dir
    model.eval()
    model = model.cuda()
    for mode in ['easy','medium','hard']:
        data = FILM(mode = mode,root = "/home/zonglin/data")
        to_pil = transforms.ToPILImage()
        counter = 0
        root = f"results/FILM_{mode}/LBBDM-f32/sample_to_eval"
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root,"10/0"), exist_ok=True)
        os.makedirs(os.path.join(root,"condition"), exist_ok=True)
        os.makedirs(os.path.join(root,"ground_truth"), exist_ok=True)


    for x,y,z in tqdm(data):
        x = x.unsqueeze(0).cuda()
        y = y.unsqueeze(0).cuda()
        z = z.unsqueeze(0).cuda()
        x_pred = interpolate(y,z,model)
        print(x_pred.min())



if __name__ == "__main__":
    main()
 
