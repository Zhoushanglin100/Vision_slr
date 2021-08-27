### Step 1: run admm train
## change @config-file based on needed
python3 main_slr.py /data/imagenet --model mixer_b16_224 --pretrained --admm-train --config-file config_mlp_0.9 --batch-size 256

### Step 2: run maked retrain
## change @config-file based on needed
## change @admmtrain-acc based on model from admm train
python3 main_slr.py /data/imagenet --model mixer_b16_224 --pretrained --masked-retrain --admmtrain-acc 76.08 --config-file config_mlp_0.9 --batch-size 128 --ext _tmp3