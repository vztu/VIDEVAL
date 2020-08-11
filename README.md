# VIDEVAL_release
A MATLAB implementation of feature fused VIDeo quality EVALuator (VIDEVAL) proposed in [UGC-VQA: Benchmarking blind video quality assessment for user generated content](https://arxiv.org/abs/2005.14354).

Check out our performance benchmark results in https://github.com/tu184044109/BVQA_Benchmark.


## Installation

* MATLAB
* FFmpeg
* python 3.6.7
* sklearn 0.20.3

## Demos

#### Feature Extraction Only

```
demo_compute_VIDEVAL_feats.m
```
You need to specify the parameters

#### Quality Prediction with Pre-trained Model

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