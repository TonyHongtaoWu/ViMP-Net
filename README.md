# Mask-Guided Progressive Network for Joint Raindrop and Rain Streak Removal in Videos (ACM MM 2023)
Hongtao Wu, Yijun Yang, Haoyu Chen, Jingjing Ren, Lei Zhu

This repo is the official Pytorch implementation of [Mask-Guided Progressive Network for Joint Raindrop and Rain Streak Removal in Videos](https://dl.acm.org/doi/abs/10.1145/3581783.3612001).
The first dataset and approach for video rain streaks and raindrops removal.


<hr />

> **Abstract:** *Videos captured in rainy weather are unavoidably corrupted by both rain streaks and raindrops in driving scenarios, and it is desirable and challenging to recover background details obscured by rain streaks and raindrops. However, existing video rain removal methods often address either video rain streak removal or video raindrop removal, thereby suffer from degraded performance when deal with both simultaneously. The bottleneck is a lack of a video dataset, where each video frame contains both rain streaks and raindrops. To address this issue, we in this work generate a synthesized dataset, namely VRDS, with 102 rainy videos from diverse scenarios, and each video frame has the corresponding rain streak map, raindrop mask, and the underlying rain-free clean image (ground truth). Moreover, we devise a mask-guided progressive video deraining network (ViMP-Net) to remove both rain streaks and raindrops of each video frame. Specifically, we develop an intensity-guided alignment block to predict the rain streak intensity map and remove the rain streaks of the input rainy video at the first stage. Then, we predict a raindrop mask and pass it into a devised mask-guided dual transformer block to learn inter-frame and intra-frame transformer features, which are then fed into a decoder for further eliminating raindrops. Experimental results demonstrate that our ViMP-Net outperforms state-of-the-art methods on our synthetic dataset and real-world rainy videos.*
<hr />




## Our Dataset
Our VRDS dataset can be downloaded [here](https://hkustgz-my.sharepoint.com/:f:/g/personal/hwu375_connect_hkust-gz_edu_cn/EmI_nfrnMyNAohEwNtnq50MB22RWxp-x_mtp264aVzOxlA?e=CjP3kO).




## Installation

This implementation is based on [MMEditing](https://github.com/open-mmlab/mmediting),
which is an open-source image and video editing toolbox.


Below are quick steps for installation.

**Step 1.**
Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

**Step 2.**
Install MMCV with [MIM](https://github.com/open-mmlab/mim).

```shell
pip3 install openmim
mim install mmcv-full
```

**Step 3.**
Install MAP-Net from source.

```shell
git clone https://github.com/TonyHongtaoWu/ViMP-Net.git
cd ViMP-Net
pip3 install -e .
```

Please refer to [MMEditing Installation](https://github.com/open-mmlab/mmediting/blob/master/docs/en/install.md) for more detailed instruction.


## Training and Testing
You may need to adjust the dataset path and dataloader before starting.

You can train ViMP-Net on VRDS dataset using the below command with 4 GPUs:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/derainers/ViMPNet/ViMPNet.py 4
```


You can use the following command with 1 GPU to test your trained model `xxx.pth`:

```shell
CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh configs/derainers/ViMPNet/ViMPNet.py "xxx.pth" 1 --save-path './save_path/'
```

You can find one model checkpoint trained on VRDS dataset [here](https://drive.google.com/drive/folders/1Iu_sxlN3nUpi99QUxWAnRP1a0mNNm2JU?usp=sharing).



## Our Results
The visual results of ViMP-Net can be downloaded in [Google Drive](https://drive.google.com/file/d/1yEFbQbh45hWOu2g4HR9-SUvZZpyJJd7l/view?usp=sharing) and [Outlook](https://hkustgz-my.sharepoint.com/:u:/g/personal/hwu375_connect_hkust-gz_edu_cn/EVM_XI3KcE9DgQaE9hbXvLQBjhnMP0rvQnSVcnOFcsMyTA?e=7tE2Kk).


## Contact
Should you have any question or suggestion, please contact hwu375@connect.hkust-gz.edu.cn.

## Acknowledgement
This code is based on [MMEditing](https://github.com/open-mmlab/mmagic) and [FuseFormer](https://github.com/ruiliu-ai/FuseFormer).

## Citation
If you find this repository helpful to your research, please consider citing the following:
```
@inproceedings{wu2023mask,
  title={Mask-Guided Progressive Network for Joint Raindrop and Rain Streak Removal in Videos},
  author={Wu, Hongtao and Yang, Yijun and Chen, Haoyu and Ren, Jingjing and Zhu, Lei},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={7216--7225},
  year={2023}
}
```
