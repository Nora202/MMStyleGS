<<<<<<< HEAD
# [*Arxiv 2025*] M2StyleGS: Multi-Modality 3D Style Transfer with Gaussian Splatting

## [Project page](https://nora202.github.io/MMStyleGS/) |  [Paper](https://arxiv.org/abs/2403.07807)


---

## 1 Installation

We use [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to manage the environment for its quick speed, while Conda can also be used. 
```bash
mamba env create -f environment.yml -n m2stylegs
```

## 2 Inference Rendering

### 2.1 Style Transfer with Image

You can perform style transfer on a scene with a single style image by:
```bash
python render.py -m [model_path]  --style [style_image_path] 

# for example:
python render.py -m output/Caterpillar/artistic/default_5000 --style images/46.jpg 
```
where `model_path` is the path to the pre-trained model, named as `output/[scene_name]/artistic/[exp_name]`, and `style_image_path` is the path to the style image. The rendered stylized multi-view images will be saved in the `output/[scene_name]/artistic/[exp_name]/train` folder.

### 2.2 Style Trnasfer with Text
You can also perform style transfer on a scene with any text description by:
```bash
python render.py -m [model_path]  --text=[text reference] 

# for example:
python render.py -m output/flower/artistic/default_10000 --text=['Waterlily by Monet'] 
```
where `model_path` is the path to the pre-trained model, named as `output/[scene_name]/artistic/[exp_name]`, and `style_image_path` is the path to the style image. The rendered stylized multi-view images will be saved in the `output/[scene_name]/artistic/[exp_name]/train` folder.

## 3 Training

#### 1. Reconstruction Training
```bash
python train_reconstruction.py -s [dataset_path]

# for example:
python train_reconstruction.py -s datasets/train
```
The trained reconstruction model will be saved in the `output/[scene_name]/reconstruction` folder.

#### 2. Feature Embedding Training
```bash
python train_feature.py -s [dataset_path] --ply_path [ply_path]

# for example:
python train_feature.py -s datasets/train --ply_path output/train/reconstruction/default/point_cloud/iteration_30000/point_cloud.ply
```
where `dataset_path` is the path to the training dataset, and `ply_path` is the path to the 3D Gaussians reconstructed from the reconstruction training stage, name as `output/[scene_name]/reconstruction/[exp_name]/point_cloud/iteration_30000/point_cloud.ply`. The trained feature embedding model will be saved in the `output/[scene_name]/feature` folder.


#### 3. Style Transfer Training
```bash
python train_artistic.py -s [dataset_path] --wikiartdir [wikiart_path] --ckpt_path [feature_ckpt_path] --ode [flow number] --iters[training iters]

# for example:
python train_artistic.py -s datasets/Caterpillar --wikiartdir datasets/archive --ckpt_path output/Caterpillar/feature/default/chkpnt/feature.pth --style_weight 10 --ode 3  --iters 5000
```
where `dataset_path` is the path to the training dataset, `wikiart_path` is the path to the WikiArt dataset, `feature_ckpt_path` is the path to the checkpoint of the feature embedding model, named as `output/[scene_name]/feature/[exp_name]/chkpnt/feature.pth`, and `style_weight` is the weight for the style loss, `ode` is the flow segmentation number, `iters` is the training iters for each flow alignment. The trained style transfer model will be saved in the `output/[scene_name]/artistic/[exp_name]` folder. `--style_weight` is an optional argument.


## 4 Acknowledgements

Our work is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [StyleRF](https://github.com/Kunhao-Liu/StyleRF), [Conrf](https://github.com/xingy038/ConRF) and [StyleGaussian](https://github.com/Kunhao-Liu/StyleGaussian). We thank the authors for their great work and open-sourcing the code.

## 5 Citation


=======
We will release the code soon.
>>>>>>> 3602c8f19e55290810a58fca04b1c253235ad404
