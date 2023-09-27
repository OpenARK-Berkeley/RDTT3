# Towards Robust Mobile Digital-Twin Tracking via An RGBD-based Transformer Model and A Comprehensive Mobile Dataset
This repository is the implementation code of the paper "Towards Robust Mobile Digital-Twin Tracking via An RGBD-based Transformer Model and A Comprehensive Mobile Dataset" [ [arxiv](https://arxiv.org/abs/2309.13570), website, [code](https://github.com/OpenARK-Berkeley/DigitalTwin-6DPose/edit/), [dataset](https://github.com/OpenARK-Berkeley/DTTDv2-IPhoneLiDAR) ]. 

![ModelArch](https://github.com/OpenARK-Berkeley/DigitalTwin-6DPose/assets/106426767/3f78f335-2801-4822-934c-55bac10c543d)

In this work, we bridge the existing gap towards mobile AR object tracking scenarios in a dual approach. At the algorithm level, we introduced a novel Transformer-based 6DoF  pose estimator, specifically designed to navigate the complexities introduced by noisy depth data, which is a common issue in mobile AR environments. At the dataset level, on the other hand, we expanded the scope of our previous work [DTTD](https://arxiv.org/abs/2302.05991) by introducing an innovative RGBD dataset captured using the iPhone 14 Pro, thus broadening the applicability of our approach to include iPhone sensor data. Through extensive experiments and analysis, we examine the effectiveness of our method in being robust to erroneous depth data. Additionally, our research has brought to light new complexities associated with object tracking in dynamic AR environments. Future work includes the development of open-set 6DoF pose estimation algorithms tailored specifically for AR environments.


### Updates:
- [x] our dataset with IPhone LiDAR: DTTDv2 is released, please check our [repo here](https://github.com/OpenARK-Berkeley/DTTDv2-IPhoneLiDAR) for data collection and annotation.
- [x] our trained network checkpoints are released [here](https://drive.google.com/drive/folders/18laguqXN7b-WTFrHlRpbteqmE8oRF_8H?usp=drive_link).

### Installation

```
python=3.8
torch
torchvision
torchaudio
numpy
eniops
pillow
scipy
opencv_python
tensorboard
tqdm
```

For knn module used in ADD-S loss, install KNN_CUDA: https://github.com/pptrick/KNN_CUDA. (Install KNN_CUDA requires CUDA environment, ensure that your CUDA version >= 10.2. Also, It only supports torch v1.0+.)

### Estimation

### Train

### Load Pretrained Model

### Eval
[YCB-VideoToolBox](https://github.com/yuxng/YCB_Video_toolbox)

### Attention Visualization


### Citation
If our work is useful or relevant to your research, please kindly recognize our contributions by citing our papers:
```
@InProceedings{DTTDv1,
    author    = {Feng, Weiyu and Zhao, Seth Z. and Pan, Chuanyu and Chang, Adam and Chen, Yichen and Wang, Zekun and Yang, Allen Y.},
    title     = {Digital Twin Tracking Dataset (DTTD): A New RGB+Depth 3D Dataset for Longer-Range Object Tracking Applications},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {3288-3297}
}

@misc{DTTDv2,
      title={Towards Subcentimeter Accuracy Digital-Twin Tracking via An RGBD-based Transformer Model and A Comprehensive Mobile Dataset}, 
      author={Zixun Huang and Keling Yao and Seth Z. Zhao and Chuanyu Pan and Tianjian Xu and Weiyu Feng and Allen Y. Yang},
      year={2023},
      eprint={2309.13570},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


