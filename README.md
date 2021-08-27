# Savlr Pruning on MLP-Mixer and Vision Transformer

**Follow [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models) codebase to install required packages and see List of  Models with Pretrained Weights**

[Available model list](model_list.txt)

---

## [MLP-Mixer](https://arxiv.org/pdf/2105.01601.pdf)

### Available models
![](MLP_Mixer_Model.png)

### Run Savlr pruning on MLP-Mixer Model

```python
### Step 1: run admm train
## change @config-file based on needed
python3 main_slr.py /data/imagenet --model mixer_b16_224 --pretrained --admm-train --config-file config_mlp_0.9 --batch-size 256

### Step 2: run maked retrain
## change @config-file based on needed
## change @admmtrain-acc based on model that get from admm-train (step 1)
python3 main_slr.py /data/imagenet --model mixer_b16_224 --pretrained --masked-retrain --admmtrain-acc 76.08 --config-file config_mlp_0.9
```


---
## [Vision Transformer](https://arxiv.org/pdf/2010.11929.pdf)

TBC# Vision_slr
