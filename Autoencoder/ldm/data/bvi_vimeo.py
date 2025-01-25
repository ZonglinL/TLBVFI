import numpy as np
import random
from os import listdir
from os.path import join, isdir, split, getsize
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import ldm.data.vfitransforms as vt
from functools import partial

import cv2

class Vimeo90k_triplet(Dataset):
    def __init__(self, db_dir, train=True,  crop_sz=(256,256), augment_s=True, augment_t=True):
        seq_dir = join(db_dir, 'sequences')
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t
        self.train = train
        if train:
            seq_list_txt = join(db_dir, 'tri_trainlist.txt')
        else:
            seq_list_txt = join(db_dir, 'tri_testlist.txt')

        with open(seq_list_txt) as f:
            contents = f.readlines()
            seq_path = [line.strip() for line in contents if line != '\n']

        self.seq_path_list = [join(seq_dir, *line.split('/')) for line in seq_path]

    def aug(self, img0, gt, img1, flow_gt, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        flow_gt = flow_gt[x:x+h, y:y+w, :]
        return img0, gt, img1, flow_gt

    def random_resize(self,img0, imgt, img1, flow, p=0.1):
        if random.uniform(0, 1) < p:
            img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            imgt = cv2.resize(imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0
        return img0, imgt, img1, flow


    def random_reverse_channel(self,img0, imgt, img1, flow, p=0.5):
        if random.uniform(0, 1) < p:
            img0 = img0[:, :, ::-1]
            imgt = imgt[:, :, ::-1]
            img1 = img1[:, :, ::-1]
        return img0, imgt, img1, flow

    def random_vertical_flip(self,img0, imgt, img1, flow, p=0.5):
        if random.uniform(0, 1) < p:
            img0 = img0[::-1]
            imgt = imgt[::-1]
            img1 = img1[::-1]
            flow = flow[::-1]
            flow = np.concatenate((flow[:, :, 0:1], -flow[:, :, 1:2], flow[:, :, 2:3], -flow[:, :, 3:4]), 2)
        return img0, imgt, img1, flow

    def random_horizontal_flip(self,img0, imgt, img1, flow, p=0.5):
        if random.uniform(0, 1) < p:
            img0 = img0[:, ::-1]
            imgt = imgt[:, ::-1]
            img1 = img1[:, ::-1]
            flow = flow[:, ::-1]
            flow = np.concatenate((-flow[:, :, 0:1], flow[:, :, 1:2], -flow[:, :, 2:3], flow[:, :, 3:4]), 2)
        return img0, imgt, img1, flow

    def random_rotate(self,img0, imgt, img1, flow, p=0.):
        if random.uniform(0, 1) < p:
            img0 = img0.transpose((1, 0, 2))
            imgt = imgt.transpose((1, 0, 2))
            img1 = img1.transpose((1, 0, 2))
            flow = flow.transpose((1, 0, 2))
            flow = np.concatenate((flow[:, :, 1:2], flow[:, :, 0:1], flow[:, :, 3:4], flow[:, :, 2:3]), 2)
        return img0, imgt, img1, flow

    def random_reverse_time(self,img0, imgt, img1, flow, p=0.5):
        if random.uniform(0, 1) < p:
            tmp = img1
            img1 = img0
            img0 = tmp
            flow = np.concatenate((flow[:, :, 2:4], flow[:, :, 0:2]), 2)
        return img0, imgt, img1, flow

    def __getitem__(self, index):
        rawFrame3 = Image.open(join(self.seq_path_list[index],  "im1.png"))
        rawFrame4 = Image.open(join(self.seq_path_list[index],  "im2.png"))
        rawFrame5 = Image.open(join(self.seq_path_list[index],  "im3.png"))

        flo1 = np.load(join(self.seq_path_list[index],  "flo21.npy"))
        flo2 = np.load(join(self.seq_path_list[index],  "flo23.npy")) ## 2 256 488
        flow_gt = np.concatenate([flo1, flo2], axis=0).transpose(1, 2, 0) ## 4 256 488 --> 256 488 4

        to_array = partial(np.array, dtype=np.float32)
        frame3, frame4, frame5 = map(to_array, (rawFrame3, rawFrame4, rawFrame5)) #(256,488,3), 0-255

        if self.crop_sz is not None:
            frame3, frame4, frame5,flow_gt = self.aug(frame3, frame4, frame5,flow_gt,self.crop_sz[0],self.crop_sz[1])

        if self.augment_s:
            frame3, frame4, frame5,flow_gt = self.random_vertical_flip(frame3, frame4, frame5,flow_gt)
            frame3, frame4, frame5,flow_gt = self.random_horizontal_flip(frame3, frame4, frame5,flow_gt)
        
        if self.augment_t:
            frame3, frame4, frame5,flow_gt = self.random_reverse_time(frame3, frame4, frame5,flow_gt)
            #frame3, frame4, frame5,flow_gt = self.random_reverse_channel(frame3, frame4, frame5,flow_gt)

        frame3 = frame3/127.5 - 1.0
        frame4 = frame4/127.5 - 1.0
        frame5 = frame5/127.5 - 1.0

        return {'image': frame4, 'prev_frame': frame3, 'next_frame': frame5,'flow':flow_gt}

    def __len__(self):
        return len(self.seq_path_list)



class Vimeo90k_septuplet(Dataset):
    def __init__(self, db_dir, train=True,  crop_sz=(256,256), augment_s=True, augment_t=True):
        db_dir = db_dir.replace("triplet","septuplet")
        seq_dir = join(db_dir, 'sequences')
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t
        self.train = train
        if train:
            seq_list_txt = join(db_dir, 'sept_trainlist.txt')
        else:
            seq_list_txt = join(db_dir, 'sept_testlist.txt')

        with open(seq_list_txt) as f:
            contents = f.readlines()
            seq_path = [line.strip() for line in contents if line != '\n']

        self.seq_path_list = [join(seq_dir, *line.split('/')) for line in seq_path]

    def aug(self, img0, gt, img1, flow_gt, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        flow_gt = flow_gt[x:x+h, y:y+w, :]
        return img0, gt, img1, flow_gt

    def random_resize(self,img0, imgt, img1, flow, p=0.1):
        if random.uniform(0, 1) < p:
            img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            imgt = cv2.resize(imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0
        return img0, imgt, img1, flow


    def random_reverse_channel(self,img0, imgt, img1, flow, p=0.5):
        if random.uniform(0, 1) < p:
            img0 = img0[:, :, ::-1]
            imgt = imgt[:, :, ::-1]
            img1 = img1[:, :, ::-1]
        return img0, imgt, img1, flow

    def random_vertical_flip(self,img0, imgt, img1, flow, p=0.5):
        if random.uniform(0, 1) < p:
            img0 = img0[::-1]
            imgt = imgt[::-1]
            img1 = img1[::-1]
            flow = flow[::-1]
            flow = np.concatenate((flow[:, :, 0:1], -flow[:, :, 1:2], flow[:, :, 2:3], -flow[:, :, 3:4]), 2)
        return img0, imgt, img1, flow

    def random_horizontal_flip(self,img0, imgt, img1, flow, p=0.5):
        if random.uniform(0, 1) < p:
            img0 = img0[:, ::-1]
            imgt = imgt[:, ::-1]
            img1 = img1[:, ::-1]
            flow = flow[:, ::-1]
            flow = np.concatenate((-flow[:, :, 0:1], flow[:, :, 1:2], -flow[:, :, 2:3], flow[:, :, 3:4]), 2)
        return img0, imgt, img1, flow

    def random_rotate(self,img0, imgt, img1, flow, p=0.):
        if random.uniform(0, 1) < p:
            img0 = img0.transpose((1, 0, 2))
            imgt = imgt.transpose((1, 0, 2))
            img1 = img1.transpose((1, 0, 2))
            flow = flow.transpose((1, 0, 2))
            flow = np.concatenate((flow[:, :, 1:2], flow[:, :, 0:1], flow[:, :, 3:4], flow[:, :, 2:3]), 2)
        return img0, imgt, img1, flow

    def random_reverse_time(self,img0, imgt, img1, flow, p=0.5):
        if random.uniform(0, 1) < p:
            tmp = img1
            img1 = img0
            img0 = tmp
            flow = np.concatenate((flow[:, :, 2:4], flow[:, :, 0:2]), 2)
        return img0, imgt, img1, flow

    def __getitem__(self, index):

        if np.random.rand()<0.4:

            rawFrame3 = Image.open(join(self.seq_path_list[index],  "im2.png"))
            rawFrame4 = Image.open(join(self.seq_path_list[index],  "im4.png"))
            rawFrame5 = Image.open(join(self.seq_path_list[index],  "im6.png"))
        

            flo1 = np.load(join(self.seq_path_list[index],  "flo42.npy"))
            flo2 = np.load(join(self.seq_path_list[index],  "flo46.npy")) ## 2 256 488
            flow_gt = np.concatenate([flo1, flo2], axis=0).transpose(1, 2, 0) ## 4 256 488 --> 256 488 4
        else:
            rawFrame3 = Image.open(join(self.seq_path_list[index],  "im1.png"))
            rawFrame4 = Image.open(join(self.seq_path_list[index],  "im4.png"))
            rawFrame5 = Image.open(join(self.seq_path_list[index],  "im7.png"))
        

            flo1 = np.load(join(self.seq_path_list[index],  "flo41.npy"))
            flo2 = np.load(join(self.seq_path_list[index],  "flo47.npy")) ## 2 256 488
            flow_gt = np.concatenate([flo1, flo2], axis=0).transpose(1, 2, 0) ## 4 256 488 --> 256 488 4         

        to_array = partial(np.array, dtype=np.float32)
        frame3, frame4, frame5 = map(to_array, (rawFrame3, rawFrame4, rawFrame5)) #(256,488,3), 0-255

        if self.crop_sz is not None:
            frame3, frame4, frame5,flow_gt = self.aug(frame3, frame4, frame5,flow_gt,self.crop_sz[0],self.crop_sz[1])

        if self.augment_s:
            frame3, frame4, frame5,flow_gt = self.random_vertical_flip(frame3, frame4, frame5,flow_gt)
            frame3, frame4, frame5,flow_gt = self.random_horizontal_flip(frame3, frame4, frame5,flow_gt)
        
        if self.augment_t:
            frame3, frame4, frame5,flow_gt = self.random_reverse_time(frame3, frame4, frame5,flow_gt)
            #frame3, frame4, frame5,flow_gt = self.random_reverse_channel(frame3, frame4, frame5,flow_gt)

        frame3 = frame3/127.5 - 1.0
        frame4 = frame4/127.5 - 1.0
        frame5 = frame5/127.5 - 1.0

        return {'image': frame4, 'prev_frame': frame3, 'next_frame': frame5,'flow':flow_gt}

    def __len__(self):
        return len(self.seq_path_list)



class Sampler(Dataset):
    def __init__(self, datasets, p_datasets=None, iter=False, samples_per_epoch=1000):
        self.datasets = datasets
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter

        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

        self.samples_per_epoch = samples_per_epoch

        self.accum = [0,]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            # iterate through all datasets
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i-1].__getitem__(index-self.accum[i-1])
        else:
            # first sample a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            # sample a sequence from the dataset
            return dataset.__getitem__(random.randint(0,len(dataset)-1))
            

    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        else:
            return self.samples_per_epoch


class Vimeo_triplet(Dataset):
    def __init__(self, db_dir, crop_sz=[256,256], p_datasets=None, iter=False, samples_per_epoch=1000):
        tri_train = Vimeo90k_triplet(db_dir, train=True,  crop_sz=crop_sz)
        sept_train = Vimeo90k_septuplet(db_dir, crop_sz=crop_sz)

        self.datasets = [tri_train,sept_train]
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter

        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

        self.samples_per_epoch = samples_per_epoch

        self.accum = [0,]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            # iterate through all datasets
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i-1].__getitem__(index-self.accum[i-1])
        else:
            # first sample a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            # sample a sequence from the dataset
            return dataset.__getitem__(random.randint(0,len(dataset)-1))
            

    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        else:
            return self.samples_per_epoch