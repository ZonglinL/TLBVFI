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

from utils import dict2namespace, namespace2dict
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from pytorch_ssim import ssim_matlab as ssim_
import math

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

    parser.add_argument('--data_dir', type=str, default="/home/zo258499/data/Xiph", help='previous frame')

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

def interpolate(frame0,frame1,model,gt = None,scale = 1):
    with torch.no_grad():
        if gt is None:
            gt = torch.zeros_like(frame0)
        #inputs = torch.cat([frame0,gt,frame1],0) 
        #latent,phi_list= model.encode(inputs,cond = True)
        #latent = torch.stack(torch.chunk(latent,3),2)
        out = model.sample(frame0,frame1, clip_denoised=False,scale = scale)
        
        #out = model.decode(latent,frame0,frame1,phi_list,scale = scale) 
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
    data_dir = args.data_dir
    model.eval()
    model = model.cuda()

    os.makedirs(name='./netflix', exist_ok=True)

    if len(glob.glob('./netflix/BoxingPractice-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_BoxingPractice_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/BoxingPractice-%03d.png')
    # end

    if len(glob.glob('./netflix/Crosswalk-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_Crosswalk_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/Crosswalk-%03d.png')
    # end

    if len(glob.glob('./netflix/DrivingPOV-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/Chimera/Netflix_DrivingPOV_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/DrivingPOV-%03d.png')
    # end

    if len(glob.glob('./netflix/FoodMarket-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/FoodMarket-%03d.png')
    # end

    if len(glob.glob('./netflix/FoodMarket2-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket2_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/FoodMarket2-%03d.png')
    # end

    if len(glob.glob('./netflix/RitualDance-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_RitualDance_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/RitualDance-%03d.png')
    # end

    if len(glob.glob('./netflix/SquareAndTimelapse-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_SquareAndTimelapse_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/SquareAndTimelapse-%03d.png')
    # end

    if len(glob.glob('./netflix/Tango-*.png')) != 100:
        os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_Tango_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/Tango-%03d.png')
    # end

    

    for strCategory in ['resized', 'cropped']:
        counter = 0
        if strCategory == 'cropped':
            scale = 0.5
            path = "results/Xiph_4K/LBBDM-f32/sample_to_eval"
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #outptu tensor in [-1,1]
        else:
            scale = 1
            path = "results/Xiph_2K/LBBDM-f32/sample_to_eval"
            transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((1080,2048)),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #outptu tensor in [-1,1]
        os.makedirs(path,exist_ok = True)
        os.makedirs(os.path.join(path,"10/0"),exist_ok = True)
        os.makedirs(os.path.join(path,"condition"),exist_ok = True)
        os.makedirs(os.path.join(path,"ground_truth"),exist_ok = True)
        
        PSNR = 0 
        SSIM = 0
        for strFile in ['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2', 'RitualDance', 'SquareAndTimelapse', 'Tango']:
            for intFrame in tqdm(range(2, 99, 2)):

                frame0_path = './netflix/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png'
                frame1_path = './netflix/' + strFile + '-' + str(intFrame + 1).zfill(3) + '.png'
                frame_gt_path = './netflix/' + strFile + '-' + str(intFrame).zfill(3) + '.png'

                frame0 = transform(Image.open(frame0_path)).cuda().unsqueeze(0)
                frame1 = transform(Image.open(frame1_path)).cuda().unsqueeze(0)
                frame_gt = transform(Image.open(frame_gt_path)).cuda().unsqueeze(0)
                if strCategory == 'cropped':
                    frame0 = frame0[:,:,540:-540, 1024:-1024]
                    frame1 = frame1[:,:,540:-540, 1024:-1024]
                    frame_gt = frame_gt[:,:,540:-540, 1024:-1024]


                I = interpolate(frame0,frame1,model,frame_gt,scale = scale)
                I = I/2 + 0.5
                frame0 = frame0/2 + 0.5
                frame1 = frame1/2 + 0.5
                frame_gt = frame_gt/2 + 0.5
                
                mse = ((frame_gt - I)**2).mean()
                PSNR += -10*math.log10(mse)
                
                SSIM += ssim_(frame_gt.cpu(), I.cpu().clamp_(min = 0,max = 1),val_range = 1)
                
                
                img_interp = Image.fromarray((I.squeeze().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
                img_interp.save(os.path.join(path,f"10/0/sample_from_next{counter}.png"))
                img_interp.save(os.path.join(path,f"10/0/sample_from_prev{counter}.png"))
                img0 = Image.fromarray((frame0.squeeze().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
                img1 = Image.fromarray((frame1.squeeze().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
                imgt = Image.fromarray((frame_gt.squeeze().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
                img0.save(os.path.join(path,f"condition/previous_frame{counter}.png"))
                img1.save(os.path.join(path,f"condition/next_frame{counter}.png"))
                imgt.save(os.path.join(path,f"ground_truth/GT_{counter}.png"))


                counter += 1
      
        print(f"PSNR = {PSNR/counter}")
        print(f"SSIM = {SSIM/counter}")


if __name__ == "__main__":
    main()
 
