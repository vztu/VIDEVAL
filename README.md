# VIDEVAL_release
A MATLAB implementation of feature fused VIDeo quality EVALuator (VIDEVAL)


## Installation

* MATLAB
* FFmpeg
* python 3.6 or higher
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