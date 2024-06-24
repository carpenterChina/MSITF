# MSITF

[Space-Time Video Super-Resolution via Multi-Scale Feature Interpolation and Temporal Feature Fusion]([https://arxiv.org/abs/2104.08860](https://assets-eu.researchsquare.com/files/rs-4342774/v1_covered_541a56ba-e74d-4b11-88f1-bd8cdf8b6b32.pdf?c=1715322593))

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

The goal of Space-Time Video Super-Resolution (STVSR) is to simultaneously increase the spatial resolution and frame rate of low-resolution, low-frame-rate video. In response to the problem that the STVSR method does not fully consider the spatio-temporal correlation between successive video frames, which makes the video frame reconstruction results unsatisfactory, and the problem that the inference speed of large models is slow. This paper proposes a STVSR method based on Multi-Scale Feature Interpolation and Temporal Feature Fusion (MSITF). First, feature interpolation is performed in the low-resolution feature space to obtain the features corresponding to the missing frames. The feature is then enhanced using deformable convolution with the aim of obtaining a more accurate feature of the missing frames. Finally, the temporal alignment and global context learning of sequence frame features are performed by a temporal feature fusion module to fully extract and utilize the useful spatio-temporal information in adjacent frames, resulting in better quality of the reconstructed video frames. Extensive experiments on the benchmark datasets Vid4 and Vimeo-90k show that the proposed method achieves better qualitative and quantitative performance, with PSNR and SSIM on the Vid4 dataset improving by 0.8\% and 1.9\%, respectively, over the state-of-the-art two-stage method AdaCof+TTVSR, and MSITF improved by 1.2\% and 2.5\%, respectively, compared to single-stage method RSTT. The number of parameters decreased by 80.4\% and 8.2\% compared to the AdaCof+TTVSR and RSTT, respectively.We release our code at \href{https://github.com/carpenterChina/MSITF}{https://github.com/carpenterChina/MSITF.}

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/Dai-Wenxun/mmaction2/assets/58767402/f91fc927-d5f2-41dd-8198-def71d392991" width="800"/>
</div>

## Environment
python >= 3.6
Pytorch >= 1.7
torchvision >=1.10
opencv-python == 4.5.3.56
NVIDIA GPU + CUDA  [A100 CUDA 10.2]

## Data
### （1）train data：Vimeo-90K-T
The file structure is as follows:
--sequences
       --00001
           --0001
               img1.png
               img2.png
               img3.png
               img3.png
               img5.png
               img6.png
               img7.png
           --0002
           --0003
           ...
           --1000
       --00002
       ...
       --00096
### (2) Data downsampling
Use/ Perform BD downsampling on the dataset using the data_scripts/generateLR_Vimeo90K. m file.
The file structure after downsampling is consistent with the original data file structure



### (3) Vid4 testing video
The file structure is as follows:
--GT
--calendar
--00000001.png
--00000002.png
--00000003.png
...
--00000040.png
--city
--foliage
--walk

===Test data needs to delete even frames===


## 3. Training
For detailed configuration of model training parameters, please refer to "./options/train. yml"
```shell
python bd7train.py
```

## 4. Testing

```shell
python test.py
```


Configure the test video path test_dataset_folder and model path model_path in the code


## Table

![image](https://github.com/carpenterChina/MSITF/assets/103326359/ebb8195f-9377-49c9-b1e7-60026cc3e058)


### Performance comparison of evaluation indicators of different methods on datasets
![image](https://github.com/carpenterChina/MSITF/assets/103326359/2c758932-10c0-4405-9c4e-1df5f82f8aae)


![image](https://github.com/carpenterChina/MSITF/assets/103326359/7e2a7a71-2dd7-4374-b886-1f3ad622993f)


## Result

### Visualization comparison of different methods on the Vid4 dataset. 
Compared with other STVSR methods, the method proposed in this paper, MSITF, recovers video frames with more accurate structure and less motion blur, which is
consistent with the results in quantitative evaluation.
![可视化对比图](https://github.com/carpenterChina/MSITF/assets/103326359/f229655c-fdc9-4cf7-bae3-763fac41a0fa)


### Comparison of Model Performance on Challenging Scenarios. 
This figure illustrates the limitations of the proposed spatiotemporal video super-resolution model in three challenging scenarios. The specific issues highlighted are:high-speed rotational motion, densely packed small objects with similar colors, and human faces.
![故障案例](https://github.com/carpenterChina/MSITF/assets/103326359/436a3b40-bd7b-4e4d-b963-3cc301be7012)



