# VIDEVAL_release
A MATLAB implementation of feature fused VIDeo quality EVALuator (VIDEVAL) proposed in [UGC-VQA: Benchmarking blind video quality assessment for user generated content](https://arxiv.org/abs/2005.14354).

Check out our BVQA resource list and performance benchmark/leaderboard results in https://github.com/tu184044109/BVQA_Benchmark.

码云[Gitee](https://gitee.com/) mirror: https://gitee.com/zhengzhong-tu/VIDEVAL_release

## Updates

- :bug: [12-17-2020] Mac system is not supported since there was an error on calling the `mex` files [here](https://github.com/vztu/VIDEVAL_release/tree/master/include/matlabPyrTools/MEX). It also means that the FRIQUEE model cannot be run on Mac too. Thanks to @CXMANDTXW for finding this in [issue](https://github.com/vztu/VIDEVAL_release/issues/5).
- :sparkles: [12-20-2020] `demo_compute_VIDEVAL_light_feats.m` was provided as a speed-up version of vanilla VIDEVAL, albeit the performance may drop. Please check [Performances](#performances) for the performance-speed tradeoff.


## Performances

### SRCC / PLCC

VIDEVAL means the original VIDEVAL in `demo_compute_VIDEVAL_feats.m`. It operates on the __original__ frame resolution sampled at __every second frame__.
VIDEVAL_light_\${res}s_\${fps}fps is the __light__ version of VIDEVAL where input video is spatially downscaled to \${res} at a frame sampling rate of \${fps} fps.

|    Methods   | KoNViD-1k             | LIVE-VQC             | YouTube-UGC         | All-Combined |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| VIDEVAL      | 0.7832 / 0.7803 | 0.7522 / 0.7514  | 0.7787 / 0.7733 | 0.7960 / 0.7939  |
| VIDEVAL_light_480s_3fps | 0.7281 / 0.7338 | 0.7144 / 0.7209  | 0.7140 / 0.7134 | 0.7462 / 0.7537  |

### Speed

Speed was evaluated on the whole `calc_VIDEVAL_feats.m` function. The unit is average `secs/video`. 

|    Methods   |  540p@30fps@8sec | 720p@30fps@10sec | 1080p@30fps@10sec | 4k@60fps@20s | scability |
|:-----------:|:----:|:----:|:------:|:--------:|
| VIDEVAL      |   61.9   |  146.5   |  354.5   |  6053.0   | :no_good_man::cursing_face: |	
| VIDEVAL_light_480s_3fps | 12.2 | 16.6 | 20.4 | 77.9  | :+1::blush: |

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
You need to specify the parameters

#### Quality Prediction with Pre-trained Model

This pre-trained model was trained on the combined dataset.

You need first extract features:
```
demo_compute_VIDEVAL_feats.m
```
Then run:
```
demo_pred_MOS_pretrained_VIDEVAL.py
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
@article{tu2020ugc,
  title={UGC-VQA: Benchmarking Blind Video Quality Assessment for User Generated Content},
  author={Tu, Zhengzhong and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C},
  journal={arXiv preprint arXiv:2005.14354},
  year={2020}
}
```

## Contact
Zhengzhong TU, ```zhengzhong.tu@utexas.edu```
