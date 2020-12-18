# VIDEVAL_release
A MATLAB implementation of feature fused VIDeo quality EVALuator (VIDEVAL) proposed in [UGC-VQA: Benchmarking blind video quality assessment for user generated content](https://arxiv.org/abs/2005.14354).

Check out our performance benchmark results in https://github.com/tu184044109/BVQA_Benchmark.

[Gitee](https://gitee.com/) mirror: https://gitee.com/zhengzhong-tu/VIDEVAL_release

## Updates

- :bug: [12-17-2020] Mac system is not supported since there was an error on calling the `mex` files [here](https://github.com/vztu/VIDEVAL_release/tree/master/include/matlabPyrTools/MEX). It also means that the FRIQUEE model cannot be run on Mac too. Thanks to @CXMANDTXW for finding this in [issue](https://github.com/vztu/VIDEVAL_release/issues/5).


## Installation

> [Note] Recommended system is Linux. Windows MATLAB users may suffer from `WARNING: You should compile the MEX version of "*.c"` and was extremely slow from our tests.

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
