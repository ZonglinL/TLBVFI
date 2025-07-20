# [ICCV 2025] TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation

<div align="center">
  
[Zonglin Lyu](https://zonglinl.github.io/), [Chen Chen](https://www.crcv.ucf.edu/chenchen/)

[![Website shields.io](https://img.shields.io/website?url=http%3A//poco.is.tue.mpg.de)](https://zonglinl.github.io/tlbvfi_page/) [![YouTube Badge](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://youtu.be/LoJIVSiT5kE)  [![arXiv](https://img.shields.io/badge/arXiv-2507.04984-00ff00.svg)](https://arxiv.org/abs/2507.04984)

</div>

<p align="center">
<img src="images/visual1.png" width=95%>
<p>

## Overview
We takes advangtage of temporal information extraction in the pixel space (3D wavelet) and latent space (3D convolutino and attention) to improve the temporal consistentcy of our model. 

<p align="center">
<img src="images/overview.jpg" width=95%>
<p>

## Quantitative Results
Our method achieves state-of-the-art performance in LPIPS/FloLPIPS/FID among all recent SOTAs. 
<p align="center">
<img src="images/quant.png" width=95%>
<p>

## Qualitative Results
Our method achieves the best visual quality among all recent SOTAs. 
<p align="center">
<img src="images/visual3.png" width=95%>
<p>

For more visualizations, please refer to our <a href="https://zonglinl.github.io/tlbvfi_page/">project page</a>.

## Preparation

### Package Installation

To install necessary packages, run:

```
pip install pip==23.2
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

### Trained Model

The weights of our model are now available at <a href="https://huggingface.co/ucfzl/TLBVFI">huggingface</a>. vimeo_unet.pth is the full model, and vimeo_new.ckpt is the VQ Model (autoencoder).

We will keep the google drive link until July 31 2025. Full model <a href="https://drive.google.com/file/d/1e_v32r6dxRXzjQXo6XDALiO9PM-w6aJS/view?usp=sharing">here</a> and autoencoder <a href="https://drive.google.com/file/d/11HOW6LOwxOae2ET63Fqzs9Dzg3-F9pw9/view?usp=sharing"> here</a>.


## Inference

**Please leave the *model.VQGAN.params.dd_config.load_VFI* and *model.VQGAN.params.ckpt_path* in ```configs/Template-LBBDM-video.yaml``` as empty**, otherwise you need to download the model weights of VFIformer from <a href="https://drive.google.com/drive/folders/140bDl6LXPMlCqG8DZFAXB3IBCvZ7eWyv"> here</a> and our VQ Model. You need to change the path of *load_VFI* and *ckpt_path* to the path of downloaded VFIformer and our VQGAN respectively.

Please download our trained model.

Then run:

```
python interpolate.py --resume_model path_to_model_weights --frame0 path_to_the_previous_frame --frame1 path_to_the_next_frame
```
This will interpolate 7 frames in between, you may modify the code to interpolate different number of frames with a bisection like methods

```
python interpolate_one.py --resume_model path_to_model_weights --frame0 path_to_the_previous_frame --frame1 path_to_the_next_frame
```
This will interpolate 1 frame in between.


## Prepare datasets

### Training set
[[Vimeo-90K]](http://toflow.csail.mit.edu/) 

### Evaluation set

[[DAVIS]](https://drive.google.com/file/d/1tcOoF5DkxJcX7_tGaKgv1B1pQnS7b-xL/view) | [[SNU-FILM]](https://myungsub.github.io/CAIN/)

**Xiph is automatically downloaded when you run Xiph_eval.py**


The DAVIS dataset is preprocessed with the dataset code from [LDMVFI](https://github.com/danier97/LDMVFI/blob/main/ldm/data/testsets.py) and saved in a structured file. Please feel free to directly use it, or you may use the dataloader from LDMVFI.

Data should be in the following structure:

```
└──── <data directory>/
    ├──── DAVIS/
    |   ├──── bear/
    |   ├──── ...
    |   └──── walking/
    ├──── SNU-FILM/
    |   ├──── test-easy.txt
    |   ├──── ...
    |   └──── test/...
    └──── vimeo_triplet/
        ├──── sequences/
        ├──── tri_testlist.txt
        └──── tri_trainlist.txt
```

You can either rename folders to our structures, or change the the codes.

## Training and Evaluating




Please edit the configs file in ```configs/Template-LBBDM-video.yaml```! 

Change data.dataset_config.dataset_path to your path to dataset (the path until ```<data directory>``` above)

Change model.VQGAN.params.dd_config.load_VFI to your downloaded VFIformer weights

### Train your autoencoder

```
python3 Autoencoder/main.py --base configs/vqflow-f32.yaml -t --gpus 0,1,2,3 --resume "logs/...."
```
You may remove resume if you do not need. You can reduce number of gpus accordingly.

After training, you should move the saved VQModel at ```logs``` as ```results/VQGAN/vimeo_new.ckpt```. You are also free to change model.VQGAN.params.ckpt_path in ```configs/Template-LBBDM-video.yaml``` to fit your path of ckpt.

### Train the UNet

Make sure that model.VQGAN.params.ckpt_path in ```configs/Template-LBBDM-video.yaml``` is set correctly.

Please run:

```
python3 main.py --config configs/Template-LBBDM-video.yaml --train --save_top --gpu_ids 0
```

You may use ```--resume_model /path/to/ckpt``` to resume training. The model will be saved in ```results/dataset_name in configs file/model_name in configs file```. For simplicity, you can leave *dataset_name* and *model_name* unchanged as DAVIS and LBBDM-f32 during training.

### Evaluate

Please edit the configs file in ```configs/Template-LBBDM-video.yaml```! 

change data.eval and data.mode to decide which dataset you want to evaluate. eval is chosen from {"DAVIS","FILM"} and mode is from {"easy","medium","hard","extreme"}

Change data.dataset_name to create a folder to save sampled images. You will need to distinguish different difficulty level for SNU-FILM when you evaluating SNU-FILM. For example, in our implementation, we choose from {"DAVIS","FILM_{difficulty level}"}. The saved images will be in ```results/dataset_name```. Run:

```
python3 main.py --configs/Template-LBBDM-video.yaml --gpu_ids 0 --resume_model /path/to/vimeo_unet --sample_to_eval
```


**To evaluate Xiph dataset**

Run 

```
python3 Xiph_eval.py --resume_model 'path to vimeo_unet.pth'
```

**Above codes save sampled images and print out PSNR/SSIM**

Then, to get LPIPS/FloLPIPS/FID, run:

```

python3 batch_to_entire.py --latent --dataset dataset_name --step 10

python3 copy_GT.py --latent --dataset dataset_name

python3 eval.py --latent --dataset dataset_name --step 10
```
dataset_name is from 'DAVIS, FILM_{difficulty level}, Xiph_{4K/2K}'


## Acknowledgement

We greatfully appreaciate the source code from [BBDM](https://github.com/xuekt98/BBDM), [LDMVFI](https://github.com/danier97/LDMVFI), and [VFIformer](https://github.com/dvlab-research/VFIformer)

## Citation

If you find this repository helpful for your research, please cite:

```
@article{lyu2025tlbvfitemporalawarelatentbrownian,
      title={TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation}, 
      author={Zonglin Lyu and Chen Chen},
      year={2025},
      eprint={2507.04984},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```
