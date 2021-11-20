[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ugc-vqa-benchmarking-blind-video-quality/video-quality-assessment-on-youtube-ugc)](https://paperswithcode.com/sota/video-quality-assessment-on-youtube-ugc?p=ugc-vqa-benchmarking-blind-video-quality)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ugc-vqa-benchmarking-blind-video-quality/video-quality-assessment-on-konvid-1k)](https://paperswithcode.com/sota/video-quality-assessment-on-konvid-1k?p=ugc-vqa-benchmarking-blind-video-quality)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ugc-vqa-benchmarking-blind-video-quality/video-quality-assessment-on-live-fb-lsvq)](https://paperswithcode.com/sota/video-quality-assessment-on-live-fb-lsvq?p=ugc-vqa-benchmarking-blind-video-quality)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ugc-vqa-benchmarking-blind-video-quality/video-quality-assessment-on-live-vqc)](https://paperswithcode.com/sota/video-quality-assessment-on-live-vqc?p=ugc-vqa-benchmarking-blind-video-quality)

# VIDEVAL
A MATLAB implementation of feature fused VIDeo quality EVALuator (VIDEVAL) proposed in [IEEE TIP2021] [UGC-VQA: Benchmarking blind video quality assessment for user generated content](https://arxiv.org/abs/2005.14354). [IEEEXplore](https://ieeexplore.ieee.org/document/9405420)

Check out our BVQA resource list and performance benchmark/leaderboard results in https://github.com/tu184044109/BVQA_Benchmark.

码云[Gitee](https://gitee.com/) mirror: https://gitee.com/zhengzhong-tu/VIDEVAL_release

The recommended system is Linux, than Windows. Mac is not supported though due to FRIQUEE [issue](https://github.com/vztu/VIDEVAL_release/issues/5).

Should you find any problems, please feel free to send an [issue](https://github.com/vztu/BVQA_Benchmark/issues) or email [me](mailto:zhengzhong.tu@utexas.edu).

## Updates

- [10-21-2021] All the features I used in the paper can be downloaded here: [Google Drive](https://drive.google.com/drive/folders/1_HFMO1KflNvlwkLC02ZWkxxiKZtFQqwV?usp=sharing) 
- [10-21-2021] Added the code for calibrating dataset MOSs. Check `inlsa/` for more details.
- :bug: [12-17-2020] Mac system is not supported since there was an error on calling the `mex` files [here](https://github.com/vztu/VIDEVAL_release/tree/master/include/matlabPyrTools/MEX). It also means that the FRIQUEE model cannot be run on Mac too. Thanks to @CXMANDTXW for finding this in [issue](https://github.com/vztu/VIDEVAL_release/issues/5).
- :sparkles: [12-20-2020] A light version `VIDEVAL_light` was provided as a speed-up version of vanilla VIDEVAL (scales better for high resolution and high fps), albeit the performance may drop. Please check [Performances](#performances) for the performance-speed tradeoff. Check [Demos](#demos) for the running of light VIDEVAL.
- :bug: [04-22-2021] Fixed nan bug by using nanmean(). Thanks to @Sissuire.


## Performances

### SRCC / PLCC

VIDEVAL means the original VIDEVAL in `demo_compute_VIDEVAL_feats.m`. It operates on the __original__ frame resolution sampled at __every second frame__.
VIDEVAL_light_{res}s_{fps}fps is the __light__ version of VIDEVAL where input video is spatially downscaled to {res} at a frame sampling rate of {fps} fps. Check `demo_compute_VIDEVAL_light_feats.m`. Note that speed-up parameters `[max_reso,frs_per_blk]` can be played with for specific application scenarios.

|    Methods   | KoNViD-1k | LIVE-VQC             | YouTube-UGC         | All-Combined |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| VIDEVAL      | 0.7832 / 0.7803 | 0.7522 / 0.7514  | 0.7787 / 0.7733 | 0.7960 / 0.7939  |
| VIDEVAL_light_720s_6fps | 0.7510 / 0.7510 | 0.7429 / 0.7453 | 0.7514 / 0.7477 | 0.7621 / 0.7689  | 
| VIDEVAL_light_720s_3fps | 0.7492 / 0.7508 | 0.7174 / 0.7225  | 0.7445 / 0.7413  | 0.7584 / 0.7666 |
| VIDEVAL_light_480s_3fps | 0.7281 / 0.7338 | 0.7144 / 0.7209  | 0.7140 / 0.7134 | 0.7462 / 0.7537  |

### Speed

Speed was evaluated on the whole `calc_VIDEVAL_feats.m` function. The unit is average `secs/video`. 

|    Methods   |  540p | 720p | 1080p | 4k@60 | scalability |
|:-----------:|:----:|:----:|:------:|:--------:|:------------:|
| VIDEVAL      |   61.9   |  146.5   |  354.5   | 1716.3  | :snail: :cursing_face: |
| VIDEVAL_light_720s_6fps | 29.9 | 68.2 | 72.6 | 205.2 | :bullettrain_front: :sweat_smile: |
| VIDEVAL_light_720s_3fps | 15.7 | 33.6 | 40.9 | 115.9 | :airplane:	:astonished:	
| VIDEVAL_light_480s_3fps | 12.2 | 16.6 | 20.4 | 77.9  | 	:rocket: :blush: |

Note:
- 540p: 540p@30fps@8sec in KoNViD-1k
- 720p: 720p@30fps@10sec in LIVE-VQC
- 1080p: 1080p@30fps@10sec in LIVE-VQC
- 4k@60: 4k@60fps@20s in YouTube-UGC

#### Our empirical observations

- Aggressive spatial downsampling will harm the performance on spatially-dominated datasets, KoNViD-1k, YouTube-UGC.
- Increasing frame sampling rate benefits the performance on temporal-distorted or motion-intensive videos, as those in LIVE-VQC.

## Installation

> [Note] Recommended system is Linux. Windows MATLAB users may suffer from `WARNING: You should compile the MEX version of "*.c"` and was slower from our tests.

* MATLAB
* FFmpeg
* python3
* sklearn

## Demos

#### Feature Extraction Only

```
demo_compute_VIDEVAL_feats.m
```

or light version:
```
demo_compute_VIDEVAL_light_feats.m
```
You need to specify the parameters

#### Quality Prediction with Pre-trained Model

This pre-trained model was trained on the combined dataset.

You need first extract features:
```
demo_compute_VIDEVAL_feats.m
```
or light version:
```
demo_compute_VIDEVAL_light_feats.m
```

Then run:
```
demo_pred_MOS_pretrained_VIDEVAL.py
```
or light version:
```
demo_pred_MOS_pretrained_VIDEVAL_light.py
```

#### Evaluation of BVQA Model on One Dataset

```
demo_eval_BVQA_feats_one_dataset.py
```
You need to specify the parameters

#### Evaluation of BVQA Model on All-Combined Dataset

```
demo_eval_BVQA_feats_all_combined.py
```
You need to specify the parameters

## Citation

If you use this code for your research, please cite our papers.

```
@article{tu2021ugc,
  title={UGC-VQA: Benchmarking blind video quality assessment for user generated content},
  author={Tu, Zhengzhong and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
```

## Contact
Zhengzhong TU, ```zhengzhong.tu@utexas.edu```
